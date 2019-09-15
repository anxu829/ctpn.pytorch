import torch
from train.model.feature_extractor import retina_fpn


def net_test():
    """
    注意， VGG ， PSENET 的网络结构都是默认只返回一层的，但是 RETINA_FPN 是返回每一个layer的信息
    :return:
    """
    x = torch.randn(1, 3, 608, 608)
    model = retina_fpn.resnet101()
    y = model(x)
    for hidden in y['C']:
        print(hidden.shape)
    for hidden in y['P']:
        print(hidden.shape)



net_test()