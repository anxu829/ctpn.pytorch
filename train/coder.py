import numpy as np
import torch

from coder_utils import meshgrid, convert, box_iou, change_box_order


class BoxCoder:
    def __init__(self):
        # self.box_height = torch.Tensor([11, 16, 22, 32, 46, 66, 94, 134, 191, 273])
        # self.box_height = torch.Tensor([8, 11, 16, 22, 32, 46, 66, 94, 134, 191,])
        self.box_height = torch.Tensor([8, 12, 16, 24, 32, 40, 48, 56, 64, 80])

    def _gen_anchor_per_level(self, img: torch.Tensor, step: int):
        """

        Args:
            输入一张图片
        Return:
            按照step织网格，使得每个网格的中心生成数据
        """
        if len(img.shape) == 4:
            input_height, input_width = img.shape[2], img.shape[3]
        else:
            input_height, input_width = img.shape[1], img.shape[2]

        # 在每个位置产生若干个anchor
        grid_size_x = np.floor(input_width / step).astype(int)
        grid_size_y = np.floor(input_height / step).astype(int)

        xy = meshgrid(grid_size_x, grid_size_y) + 0.5  # +0.5是为了把anchor的中心对到框的中心
        xy = (xy * step).view(grid_size_x, grid_size_y, 1, 2).expand(grid_size_x, grid_size_y, len(self.box_height), 2)
        h = self.box_height.view(1, 1, len(self.box_height), 1).expand(grid_size_x, grid_size_y, len(self.box_height),
                                                                       1)
        dh = torch.zeros_like(h)
        step_ = torch.ones_like(h) * step
        box = torch.cat([xy, h, dh, step_], 3)
        return box

    def encode_per_level(self, gt: torch.Tensor, side_refinement, img: torch.Tensor, step):
        """
        encode的作用为返回每个anchor 对应的回归偏移量，以及 每个 anchor是否真正的匹配到一个合理的gt
        :param gt: 只输入在这一层进行匹配的gt , gt的形式为 center_x ,center_y , h , dh ,step
                    在per_level中， gt的step应该是一致的,
        :param side_refinment : 考虑 side_refine ， 值不为 -1 的表示是有坐标修正的

        :param img:
        :param step:

        :return:
        """
        # gt 和 anchor 的表示 都是 cx,cy,h,dh,step的表示形式
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).type(torch.float)
        if isinstance(side_refinement, np.ndarray):
            side_refinement = torch.from_numpy(side_refinement).type(torch.float)

        # anchor:  H  x W x  A x 5 -> [ (h1 , w1 , a1 , 5),(h1 , w1 , a2 , 5)  ]
        anchor_xyhd_boxes = self._gen_anchor_per_level(img, step)
        # 数据的形式统一遵从 w ， h ， anchor的形式
        width, height, num_boxes, _ = anchor_xyhd_boxes.shape
        anchor_xyhd_boxes = anchor_xyhd_boxes.reshape(-1, 5)
        anchor_quad_boxes = change_box_order(anchor_xyhd_boxes, 'xyhd2quad')
        anchor_rect_boxes = change_box_order(anchor_quad_boxes, 'quad2xyxy')

        gt_xyhd_boxes = gt
        gt_quad_boxes = convert(gt_xyhd_boxes, mode='xyhd2quad')
        gt_rect_boxes = convert(gt_quad_boxes, mode='quad2xyxy')
        ious = box_iou(anchor_rect_boxes, gt_rect_boxes)
        max_ious, max_ids = ious.max(1)  # 计算出了每个box对应最大iou的gt
        max_match, _ = ious.max(0)
        # Each anchor box matches the largest iou with the gt box
        # gt_quad_boxes = gt_quad_boxes[max_ids]  # (num_gt_boxes, 8)
        # gt_rect_boxes = gt_rect_boxes[max_ids]  # (num_gt_boxes, 4)
        gt_xyhd_boxes = gt_xyhd_boxes[max_ids]
        # 计算每个anchor的回归目标

        # 注意，如果d_cx 使用如下公式计算出来，那么所有的pos sample的 d_cx都是0 ， 因为gt构造过程中都是和 x轴对齐的，
        # 真正的side_refinement信息记录在 side_refinement中
        # d_cx = (gt_xyhd_boxes[:, 0] - anchor_xyhd_boxes[:, 0]) / anchor_xyhd_boxes[:, 2]

        # d_cx 中等于-1的表明是不需要进行side_refinement的中间的框
        d_cx = side_refinement[max_ids] / step
        d_cy = (gt_xyhd_boxes[:, 1] - anchor_xyhd_boxes[:, 1]) / anchor_xyhd_boxes[:, 2]
        d_h = torch.log(gt_xyhd_boxes[:, 2] / anchor_xyhd_boxes[:, 2])
        d_dh = torch.atan(gt_xyhd_boxes[:, 3] / step)  # 求出对应的角度 ，然后划归到 0~1 之间

        #  获得pos_neg 标签，忽略掉中间的部分
        IOU_POS = 0.7
        IOU_NEG = 0.5
        anchor_cls = torch.zeros(anchor_xyhd_boxes.shape[0])
        anchor_cls[max_ious > IOU_POS] = 1
        anchor_cls[max_ious < IOU_POS] = -1
        anchor_cls[max_ious < IOU_NEG] = 0

        # 注意到有的gt，可能没有办法找到一个anchor，使得这个anchor直接的和他match，且iou大于0.7
        gt_matched = torch.unique(max_ids[anchor_cls == 1])
        if len(gt_matched) != gt.shape[0]:
            gt_unmatched = set(range(gt.shape[0])) - set(gt_matched.cpu().numpy())
            gt_unmatched = list(gt_unmatched)
            # 找到这些框的最大匹配anchor
            max_gt_ious, max_gt_ids = ious.max(0)
            max_gt_ids = max_gt_ids[gt_unmatched]
            anchor_cls[max_gt_ids] = 1

        # 在coder的过程中，要找到所有和gt的iou在0.5以内的计算高度的dh，用于计算回归的loss（cls 的iou设置为0.7，对于回归不是特别友好）
        valid_anchor = ((ious > 0.5).sum(1) > 0) + (anchor_cls == 1)
        valid_anchor = valid_anchor > 0
        # 还需要找出需要计算d_cx 的框
        anchor_need_side_refinement = (side_refinement[max_ids] != -999) * (anchor_cls == 1)

        edge_anchor_boxes = anchor_quad_boxes[anchor_need_side_refinement]

        return d_cx, d_cy, d_h, d_dh, anchor_cls, valid_anchor, anchor_need_side_refinement, edge_anchor_boxes

    def decode_per_level(self, img, step, d_cx, d_cy, d_h, d_dh, anchor_cls, thres=0.8, decode_mode='xyhd2quad',
                         return_format='quad', return_prob=False):
        """

        :param img: channel x h x w image
        :param step:  stride of anchor
        :param d_cx: pred cx
        :param d_cy: pred cy
        :param d_h: pred h
        :param d_dh: pred dh
        :param anchor_cls:  pred cls score
        :param thres: cls thres
        :return:
        """
        assert decode_mode in ['xyhd2quad']
        assert return_format in ['quad', 'xyhd']
        anchor_xyhd_boxes = self._gen_anchor_per_level(img, step)
        width, height, num_boxes, _ = anchor_xyhd_boxes.shape
        anchor_xyhd_boxes = anchor_xyhd_boxes.reshape(-1, 5)
        pred_xyhd_boxes = torch.zeros_like(anchor_xyhd_boxes)

        # do not need to use side_refinement directly!
        pred_xyhd_boxes[:, 0] = anchor_xyhd_boxes[:, 0]
        pred_xyhd_boxes[:, 1] = d_cy * anchor_xyhd_boxes[:, 2] + anchor_xyhd_boxes[:, 1]
        pred_xyhd_boxes[:, 2] = torch.exp(d_h) * anchor_xyhd_boxes[:, 2]
        pred_xyhd_boxes[:, 3] = torch.tan(d_dh) * step
        pred_xyhd_boxes[:, 4] = step

        if return_format == 'quad':
            pos_idx = anchor_cls > thres
            pred_xyhd_boxes = pred_xyhd_boxes[pos_idx]
            pred_quad_boxes = change_box_order(pred_xyhd_boxes, decode_mode)
            pred_cls_prob = anchor_cls[pos_idx]
            pred = pred_quad_boxes

        if return_format == 'xyhd':
            pos_idx = anchor_cls > thres
            pred_xyhd_boxes = pred_xyhd_boxes[pos_idx]
            pred_cls_prob = anchor_cls[pos_idx]
            pred = pred_xyhd_boxes

        if return_prob:
            return pred, pred_cls_prob
        else:
            return pred
