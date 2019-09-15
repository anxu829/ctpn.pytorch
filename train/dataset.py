'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from coder import BoxCoder
from gt_converter import convert_boxes
from gt_heatmap_generator import gen_heatmap_image
from image_list import to_image_list


class ListDataset(data.Dataset):
    def __init__(self, train, transform, input_size_min, input_size_max, multi_scale=False
                 , train_path=None, test_path=None, debug_num=None, divisibility=0, step=16,
                 gen_heat_map=False
                 ):
        '''
        Args:
          root: (str) DB root ditectory.
          dataset: (str) Dataset name(dir).
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size. it is the min_size of short
          multi_scale: (bool) use multi-scale training or not.
        '''

        self.train = train
        self.transform = transform
        self.input_size_min = input_size_min
        self.input_size_max = input_size_max

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.multi_scale = multi_scale
        self.MULTI_SCALES = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960]  # step1, 2
        # self.MULTI_SCALES = [960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280] #step3

        self.path = train_path if self.train else test_path
        self.debug_num = debug_num

        self.get_My_dataset(train_path, test_path)

        print('init encoder...')
        self.encoder = BoxCoder()

        self.divisibility = divisibility
        self.step = step
        self.gen_heatmap = gen_heat_map

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) dataset index.

        Returns:
          image: (tensor) image array.
          boxes: (tensor) boxes array.
          labels: (tensor) labels array.
        '''
        # Load image, boxes and labels.
        fname = self.fnames[idx]

        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = self.boxes[idx].copy()
        labels = self.labels[idx]

        return {"image": img, "boxes": boxes, "labels": labels}

    def __len__(self):
        return self.num_samples

    def collate_fn(self, batch):
        '''bbox encode and make batch

        Args:
          batch: (dict list) images, boxes and labels

        Returns:
          batch_images, batch_loc, batch_cls
        '''
        # TODO  add multi size train support
        # size = self.input_size_min
        # if self.multi_scale:  # get random input_size for multi-scale traininig
        #     random_choice = random.randint(0, len(self.MULTI_SCALES) - 1)
        #     size = self.MULTI_SCALES[random_choice]

        imagelist = []
        d_cx_batch = []
        d_cy_batch = []
        d_h_batch = []
        d_dh_batch = []
        anchor_cls_batch = []
        edge_boxes_batch = []
        valid_anchor_batch, anchor_need_side_refinement_batch = [], []

        # 数据增强：
        # 保证前两个数据和后两个数据分别进行不同的旋转
        # 让每个batch的dh都能够合理的进行

        if len(batch) // 2 > 0:
            # 说明batch不止一个内容
            for n, data in enumerate(batch):
                deg_rand = 10
                if n < len(batch) // 2:
                    deg = deg_rand
                else:
                    deg = -deg_rand
                sample_transform = self.transform(size=(self.input_size_min, self.input_size_max), deg=deg)
                img, boxes, labels = sample_transform(data['image'], data['boxes'], data['labels'])
                batch[n] = {'image': img, 'boxes': boxes, 'labels': labels}
        else:
            # 如果一个sample只有一个样本，则不对这个样本进行旋转的数据增强
            deg = 0
            for n, data in enumerate(batch):
                sample_transform = self.transform(size=(self.input_size_min, self.input_size_max), deg=deg)
                img, boxes, labels = sample_transform(data['image'], data['boxes'], data['labels'])
                batch[n] = {'image': img, 'boxes': boxes, 'labels': labels}

        for n, data in enumerate(batch):
            img, boxes, labels = data['image'], data['boxes'], data['labels']
            imagelist.append(img)
        imagelist = to_image_list(imagelist, size_divisible=self.divisibility)

        template_img = imagelist.tensors[0]
        template_img = template_img.new(*template_img.shape).zero_()

        for n, data in enumerate(batch):
            img, boxes, labels = data['image'], data['boxes'], data['labels']
            # 把数据增强做在这里，好对数据增强进行控制
            gt, side_refinement = convert_boxes(boxes, img, self.step)
            d_cx, d_cy, d_h, d_dh, anchor_cls, valid_anchor, anchor_need_side_refinement, edge_boxes = self.encoder.encode_per_level(
                gt, side_refinement, template_img, step=self.step)
            d_cx_batch.append(d_cx)
            d_cy_batch.append(d_cy)
            d_h_batch.append(d_h)
            d_dh_batch.append(d_dh)
            anchor_cls_batch.append(anchor_cls)
            valid_anchor_batch.append(valid_anchor)
            anchor_need_side_refinement_batch.append(anchor_need_side_refinement)
            edge_boxes_batch.append(edge_boxes)

        d_cx_batch = torch.stack(d_cx_batch)
        d_cy_batch = torch.stack(d_cy_batch)
        d_h_batch = torch.stack(d_h_batch)
        d_dh_batch = torch.stack(d_dh_batch)
        anchor_cls_batch = torch.stack(anchor_cls_batch)
        valid_anchor_batch = torch.stack(valid_anchor_batch)
        anchor_need_side_refinement_batch = torch.stack(anchor_need_side_refinement_batch)

        batch_output = {"image": imagelist, "d_cx": d_cx_batch, "d_cy": d_cy_batch,
                        "d_h": d_h_batch, "d_dh": d_dh_batch, "cls": anchor_cls_batch,
                        'valid_anchor': valid_anchor_batch,
                        'side_refinement_anchor': anchor_need_side_refinement_batch
                        }

        # 如果配置中有heatmap 需求，则加入heatmap
        if self.gen_heatmap:
            heatmap_list = []
            corner_list = []
            for n, data in enumerate(batch):
                img, boxes, labels = data['image'], data['boxes'], data['labels']
                heatmap = gen_heatmap_image(img, boxes, method='CRAFT')
                corner_map = gen_heatmap_image(img, edge_boxes_batch[n], method='CRAFT')
                heatmap_list.append(torch.from_numpy(heatmap))
                corner_list.append(torch.from_numpy(corner_map))
            # TODO ,bugs here if image shape is not 16 x n
            # seq_targets =  to_image_list(heatmap_list, size_divisible=self.divisibility)
            seg_targets = torch.stack(heatmap_list, 0)
            corner_targets = torch.stack(corner_list,0)
            batch_output.update({"heatmap": seg_targets,'cornermap':corner_targets})
        return batch_output

    def get_My_dataset(self, train_path, test_path):

        dataset_list = os.listdir(self.path)
        dataset_list = [l[:-4] for l in dataset_list if "jpg" in l]

        if self.debug_num:
            # 在debug 模式中，只取debug_num 的数据
            dataset_list = np.random.choice(dataset_list, self.debug_num, replace=False)

        dataset_size = len(dataset_list)

        self.num_samples = dataset_size

        for i in tqdm(dataset_list):
            img_file = os.path.join(self.path, i + '.jpg')
            label_file = os.path.join(self.path, i + '.txt')
            label_file = open(label_file).readlines()

            _quad = []
            _classes = []

            for label in label_file:
                if len(label.split(',')) == 8:
                    _x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3 = label.split(",")[:8]
                else:
                    _x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3, txt = label.split(",")[:9]

                    if "###" in txt:
                        continue

                try:
                    _x0 = int(_x0)
                except:
                    _x0 = int(_x0[1:])

                _y0, _x1, _y1, _x2, _y2, _x3, _y3 = [int(p) for p in [_y0, _x1, _y1, _x2, _y2, _x3, _y3]]

                _quad.append([_x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3])
                _classes.append(1)

            if len(_quad) is 0:
                self.num_samples -= 1
                continue
            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))
