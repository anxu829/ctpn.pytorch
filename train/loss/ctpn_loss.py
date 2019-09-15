# 分类使用 FOCAL LOSS
# 回归使用smooth L1 loss
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import Module


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''

    y = torch.eye(num_classes)  # [D,D]
    return y[labels.type(torch.LongTensor)]  # [N,D]


class FocalLoss(Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.num_classes = 1

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''

        alpha = 0.25
        t = one_hot_embedding(y.data.cpu(), self.num_classes + 1)
        t = t[:, 1:]
        t = Variable(t).cuda()
        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (2 * xt + 1).sigmoid()
        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()

    def forward(self,
                loc_preds,
                loc_targets,
                cls_preds,
                cls_targets
                ):
        # loc_preds: n x anchors_num x 4 （cx ， cy ，h，dh）
        # loc_label : n x anchors_num x 4 （cx ， cy ，h，dh）
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.float().sum()
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]

        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, reduction='sum')
        # loc_loss *= 0.5  # TextBoxes++ has 8-loc offset

        pos_neg = cls_targets > -1
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])
        return loc_loss / num_pos, cls_loss / num_pos


class BalanceSampler:
    def __init__(self, pos_ratio=0.5, batch_size=256):
        self.pos_ratio = pos_ratio
        self.batch_size = batch_size
        self.pos_num = int(self.batch_size * self.pos_ratio)
        self.neg_num = self.batch_size - self.pos_num

    def sampler(self, mask, cls_preds, cls_targets, ):
        sample_per_image = []
        pos_per_image = []
        for i in range(cls_preds.shape[0]):
            cls_targets_ = cls_targets[i]
            mask_per_image = mask[i].reshape(-1)
            # TODO 如何合理的使用mask，需要将mask的维度缩水到特征图的维度
            # cls_targets_[mask_per_image == 0] = -1

            pos_idx = (cls_targets_ == 1).nonzero()
            neg_idx = (cls_targets_ == 0).nonzero()

            if len(pos_idx) >= self.pos_num:
                pos_idx = pos_idx[np.random.choice(range(len(pos_idx)), self.pos_num, replace=False)]
                neg_idx = neg_idx[np.random.choice(range(len(neg_idx)), self.neg_num, replace=False)]
            else:
                pos_idx = pos_idx
                self.neg_num = self.batch_size - len(pos_idx)
                neg_idx = neg_idx[np.random.choice(range(len(neg_idx)), self.neg_num, replace=False)]
            sample_per_image.append((neg_idx, pos_idx))
            pos_per_image.append(pos_idx)
        return sample_per_image  , pos_per_image


class CTPNLoss(Module):

    def __init__(self):
        super(CTPNLoss, self).__init__()
        self.num_classes = 1
        self.sampler = BalanceSampler()
        self.cls_loss = nn.CrossEntropyLoss()
        self.y_loss = nn.SmoothL1Loss()
        self.o_loss = nn.SmoothL1Loss()

    def forward(self,
                mask,
                loc_preds,
                loc_targets,
                cls_preds,
                cls_targets,
                d_h_cls,
                d_cx_cls,
                ):

        sample_per_image ,pos_per_image = self.sampler.sampler(mask, cls_preds, cls_targets, )
        batch_num, *_ = cls_preds.shape

        cls_loss = 0
        for i in range(batch_num):
            cls_preds_per_image = cls_preds[i]
            cls_target_per_image = cls_targets[i]
            selected_sample = torch.cat(sample_per_image[i])
            sample_pred = cls_preds_per_image[selected_sample.reshape(-1)]
            sample_target = cls_target_per_image[selected_sample.reshape(-1)].type(torch.cuda.LongTensor)
            cls_loss += self.cls_loss(sample_pred, sample_target)

        # target_loss 如何确定参考论文和别人的代码
        v_loss = 0
        for i in range(batch_num):
            v_pred = loc_preds[i][d_h_cls[i] == 1][:, 1:3]
            v_target = loc_targets[i][d_h_cls[i] == 1][:, 1:3]
            v_loss += self.y_loss(v_pred, v_target.cuda())

        # 对dh 添加loss:
        # 尝试使用正例样本的sample来学习dh_loss
        # dh_loss = 0
        # for i in range(batch_num):
        #     selected_pos = pos_per_image[i].squeeze(1)
        #     # 用这些正例样本的 dh 去做loss
        #     # 认为只有正例样本的anchor的信息才能够有效的去预测角度信息
        #     dh_pred = loc_preds[i][selected_pos, 3]
        #     dh_target = loc_targets[i][selected_pos,3]
        #     dh_loss += (1 - torch.cos(dh_pred - dh_target)).mean()
        #

        dh_loss = 0
        for i in range(batch_num):
            dh_pred = loc_preds[i][d_h_cls[i] == 1][:, 3]
            dh_target = loc_targets[i][d_h_cls[i] == 1][:, 3]
            dh_loss += (1- torch.cos(dh_pred - dh_target.cuda())).mean()


        # o_loss = 0
        # for i in range(batch_num):
        #     o_pred = loc_preds[i][d_cx_cls[i] == 1][:, 0]
        #     o_target = loc_targets[i][d_cx_cls[i] == 1][:, 0]
        #     o_loss += self.o_loss(o_pred, o_target)

        loc_loss = v_loss + 2 * dh_loss

        return loc_loss / batch_num, cls_loss / batch_num, dh_loss , sample_per_image
