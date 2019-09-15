# ctpn.pytorch
an pytorch implementation of CTPN


# this code support :
# multi GPU training
# tensorboardX visulizing training detail
# add an psenet segmentation which can predict center of text , make convergence fast



# TODO

## add support for presize gt_converter [done]
## add support for side-refinment matcher [done]
## add self-defined loss [done]
## add sampler in loss calculator [done]
## add valid anchor for l1_loss in train [done]
## OHEM
## TEXTBOXES , add shift box
## GT split algo [done]
## imagelist add  train mask [aborted]
## check anchor reshape is match to res [done]
## fit anchor height to data [done]
## change resize method [done]
## only train for cls [done]

# many details

# use torch tensor to slice in numpy will cause strance error
# train must with np.random seed set
# iou calculate must consider axis align



# 控制rotate 方向，保证训练的合理性[done]
# 添加对 dh 支持(使用了 EAST 的角度预测方法)[done]
# 添加对side refinment 支持（分类）
# 添加对textboxes offset 支持
# 添加对stride = 8支持
# 添加对resnet/ pseNet 网络结构支持[done]
# ...
# only send used data to cuda when train (enlarge batch size)
# 预训练模型的导入[done]