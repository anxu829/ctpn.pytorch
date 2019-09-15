
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Im2col(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(Im2col, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        height = x.shape[2]
        x = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        x = x.reshape((x.shape[0], x.shape[1], height, -1))
        return x



class BLSTM(nn.Module):
    def __init__(self, channel, hidden_unit, bidirectional=True):
        """
        :param channel: lstm input channel num
        :param hidden_unit: lstm hidden unit
        :param bidirectional:
        """
        super(BLSTM, self).__init__()
        self.hidden_unit = hidden_unit
        self.lstm = nn.LSTM(channel, hidden_unit, bidirectional=bidirectional)

    def forward(self, x):
        """
        WARNING: The batch size of x must be 1.
        """
        x = x.transpose(1, 3)
        batch, width, height, T = x.shape
        recurrents = []
        for i in range(batch):
            recurrent, _ = self.lstm(x[i])
            recurrent = recurrent.unsqueeze(0)
            recurrent = recurrent.transpose(1, 3)
            recurrents.append(recurrent)
        recurrents = torch.cat(recurrents, dim=0)
        return recurrents
