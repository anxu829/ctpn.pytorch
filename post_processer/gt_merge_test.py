"""
load gt ， and use my algo to recover gt
"""
import numpy as np
import torch

from coder_utils import recover_img
from post_processer.post_processer import Processer
from train.augmentations import Augmentation_traininig
from train.dataset import ListDataset

if __name__ == '__main__':
    trainset = ListDataset(train=True,
                           transform=Augmentation_traininig,
                           input_size_min=1024,
                           input_size_max=1600,
                           multi_scale=False,
                           train_path='test_data/image',
                           # train_path='/disk2/cwq/data/0704_huarui_rbox/image',
                           # train_path='/disk3/xuan/data/0829_medical_invoice_rbox/image',
                           debug_num=4,
                           divisibility=16,
                           step=8,
                           gen_heat_map=True
                           )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=2,
        shuffle=True, num_workers=1,
        collate_fn=trainset.collate_fn,
    )
    for data in trainloader:
        image, d_cx, d_cy, d_h, d_dh, cls, d_h_cls, d_cx_cls, seg_targets = data.values()
        loc_targets = torch.stack((d_cx, d_cy, d_h, d_dh), dim=2)
        cls_targets = cls

        post_process_test_image = recover_img(image.tensors[0])

        post_process_test_predloc = loc_targets[0]
        d_cx, d_cy, d_h, d_dh = post_process_test_predloc[:, 0], \
                                post_process_test_predloc[:, 1], \
                                post_process_test_predloc[:, 2], \
                                post_process_test_predloc[:, 3]
        # d_dh[:] = 0
        pred_loc = trainset.encoder.decode_per_level(post_process_test_image.transpose(2, 0, 1), 8, d_cx, d_cy, d_h,
                                                     d_dh, cls[0], thres=0.9, return_format='xyhd')

        post_process_test_seg_res = (seg_targets[0].data.numpy() > 0.5).astype(np.uint8)

        import time

        start = time.time()
        processer = Processer(
            post_process_test_image,
            pred_loc,
            post_process_test_seg_res
        )
        processer.process()
        end = time.time()
        print('infer time is ', end - start)
        break
