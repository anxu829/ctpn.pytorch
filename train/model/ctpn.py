import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from model import feature_extractor
from .rnn import Im2col, BLSTM


# CPTN 网络结构
class AnchorHead(nn.Module):
    def __init__(self, cfg):
        super(AnchorHead, self).__init__()
        self.num_classes = cfg.ANCHOR_HEAD.NUM_CLASS
        if cfg.MODEL.USE_RNN:
            input_plane = cfg.ANCHOR_HEAD.RNN_OUTPUT_CHANNEL
        else:
            input_plane = cfg.ANCHOR_HEAD.CNN_OUTPUT_CHANNEL
        self.FC = nn.Conv2d(input_plane, 512, 1)
        self.vertical_coordinate_cy = nn.Conv2d(512, 10, 1)
        self.h_prediction = nn.Conv2d(512, 10, 1)
        self.score = nn.Conv2d(512, self.num_classes * 10, 1)
        self.side_refinement = nn.Conv2d(512, 10, 1)
        self.dh_prediction = nn.Sequential(
            nn.Conv2d(512, 10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.FC(x)
        B, C, H, W = x.shape
        x = F.relu(x, inplace=True)
        vertical_pred_cy = self.vertical_coordinate_cy(x)
        h_prediction = self.h_prediction(x)
        score = self.score(x)
        side_refinement = self.side_refinement(x)
        dh_prediction = (self.dh_prediction(x) - 0.5) * np.pi
        offset = []
        for output in [side_refinement, vertical_pred_cy, h_prediction, dh_prediction]:
            # 预测数据遵从w，h，anchorz
            output = output.permute(0, 2, 3, 1)
            offset.append(output.reshape(B, -1, 1))
        offset = torch.cat(offset, dim=2)
        score = score.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)

        # output shape : Batch x anchor_num x offset_num
        return offset, score


class SegHead(nn.Module):
    def __init__(self, num_classes=1, scale=1):
        super(SegHead, self).__init__()

        self.conv2 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)

        self.scale = scale

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, crnn_feature):
        x = crnn_feature['x']
        p2 = crnn_feature['P2']
        p3 = crnn_feature['P3']
        p4 = crnn_feature['P4']
        p5 = crnn_feature['P5']

        p3_ = self._upsample(p3, p2)
        p4_ = self._upsample(p4, p2)
        p5_ = self._upsample(p5, p2)

        out = torch.cat((p2, p3_, p4_, p5_), 1)
        out = self.conv2(out)
        out = self.relu2(self.bn2(out))
        out = self.conv3(out)
        out = self._upsample(out, x, scale=self.scale)
        # TODO  now only consider one class segmentation
        return out.squeeze(1)


class RNNExtractor(nn.Module):
    def __init__(self, cfg):
        super(RNNExtractor, self).__init__()
        self.rnn = nn.Sequential()
        self.rnn.add_module('im2col', Im2col((3, 3), (1, 1), (1, 1)))
        self.rnn.add_module('blstm', BLSTM(3 * 3 * cfg.ANCHOR_HEAD.CNN_OUTPUT_CHANNEL, 128))

    def forward(self, x):
        return self.rnn(x)


class CTPN(nn.Module):
    def __init__(self, cfg, ):
        super(CTPN, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.ANCHOR_HEAD.NUM_CLASS
        self.cnn = nn.Sequential()
        crnn_extractor = getattr(feature_extractor, cfg.ANCHOR_HEAD.CNN_NAME)
        self.cnn.add_module('cnn_extractor', crnn_extractor(pretrained=cfg.MODEL.PRETRAINED))
        # 注意到rnn是全卷积结构，所以理论上，cnn 的部分可以替换成任意的stride
        if self.cfg.MODEL.USE_RNN:
            self.rnn = RNNExtractor(cfg)
        if self.cfg.PIXEL_HEAD.PIXEL_PREDICTION:
            if self.cfg.PIXEL_HEAD.CORNER_PREDICTION:
                self.seg = SegHead(num_classes=2)
            else:
                self.seg = SegHead(num_classes=1)
        self.anchor_head = AnchorHead(cfg)
        self.weight_init()

    def weight_init(self):
        if self.cfg.MODEL.USE_RNN and self.cfg.MODEL.INIT_RNN:
            init_part = [self.rnn, self.anchor_head]
        else:
            init_part = [self.anchor_head]
        for module in init_part:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x, ):
        crnn_feature = self.cnn(x)
        x = crnn_feature[self.cfg.ANCHOR_HEAD.ANCHOR_EXTRACTOR]
        if self.cfg.MODEL.USE_RNN:
            x = self.rnn(x)
        offset, score = self.anchor_head(x)

        seg = None
        if self.cfg.PIXEL_HEAD.PIXEL_PREDICTION:
            seg = self.seg(crnn_feature)
        return {"anchor_head": [offset, score],
                "seg_head": seg
                }

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


def ctpn_test():
    tensor = torch.Tensor(3, 3, 600, 600)
    model = CTPN(1)
    result = model(tensor)
    print(result[0].shape)

# ctpn_test()
