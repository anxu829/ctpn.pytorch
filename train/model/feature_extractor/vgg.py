import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG_16(nn.Module):
    """
    VGG-16 without pooling layer before fc layer
    """

    def __init__(self):
        super(VGG_16, self).__init__()
        self.convolution1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.convolution1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.convolution2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.convolution2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.convolution3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.convolution3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convolution3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pooling3 = nn.MaxPool2d(2, stride=2)
        self.convolution4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.convolution4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pooling4 = nn.MaxPool2d(2, stride=2)
        self.convolution5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution5_3 = nn.Conv2d(512, 512, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.convolution1_1(x), inplace=True)
        x = F.relu(self.convolution1_2(x), inplace=True)
        x = self.pooling1(x)
        x = F.relu(self.convolution2_1(x), inplace=True)
        x = F.relu(self.convolution2_2(x), inplace=True)
        x = self.pooling2(x)

        x = F.relu(self.convolution3_1(x), inplace=True)
        x = F.relu(self.convolution3_2(x), inplace=True)
        c2 = F.relu(self.convolution3_3(x), inplace=True)

        x = self.pooling3(c2)
        x = F.relu(self.convolution4_1(x), inplace=True)
        x = F.relu(self.convolution4_2(x), inplace=True)
        c3 = F.relu(self.convolution4_3(x), inplace=True)

        x = self.pooling4(c3)
        x = F.relu(self.convolution5_1(x), inplace=True)
        x = F.relu(self.convolution5_2(x), inplace=True)
        c4 = F.relu(self.convolution5_3(x), inplace=True)

        return {
            "C2": c2,
            "C3": c3,
            "C4": c4,
        }


def vgg_16(pretrained=False):
    net = VGG_16()
    if pretrained:
        # 只读取cnn，rnn的预先训练的参数
        state_dict = torch.load('./vgg16.model')
        state_dict_base = {key: v for key, v in state_dict.items() if 'cnn' in key or 'rnn' in key}
        net_state_dict = net.state_dict()
        for key in net_state_dict:
            if key in state_dict_base:
                net_state_dict[key] = state_dict_base[key]
    return net
