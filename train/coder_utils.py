import cv2
import numpy as np
import torch


def meshgrid(x, y, row_major=True):
    '''Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    # a = torch.arange(0,x * 1.0) / 2.0  #v3
    a = torch.arange(0, x * 1.0)
    b = torch.arange(0, y * 1.0)  # v3

    xx = a.repeat(y).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, x).view(-1, 1)
    return torch.cat([xx, yy], 1).float() if row_major else torch.cat([yy, xx], 1).float()


def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.

    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) anchor_rect_boxes, sized [N,4]. xyxy
      box2: (tensor) gt_rect_boxes, sized [M,4]. xyxy

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:, None, :2].type(torch.int), box2[:, :2].type(torch.int))  # [N,M,2]
    rb = torch.min(box1[:, None, 2:].type(torch.int), box2[:, 2:].type(torch.int))  # [N,M,2]

    wh = (rb - lt + 1).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)  # [N,]
    area2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)  # [M,]

    area1 = area1.type(torch.float)
    area2 = area2.type(torch.float)
    inter = inter.type(torch.float)

    iou = inter / (area1[:, None] + area2 - inter)

    return iou


def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4] or [N, 8] .
      order: (str) one of ['xyxy2xywh','xywh2xyxy', 'xywh2quad', 'quad2xyxy']

    Returns:
      (tensor) converted bounding boxes, sized [N,4] or [N,8].
    '''
    assert order in ['xyxy2xywh', 'xywh2xyxy', 'xywh2quad', 'quad2xyxy', 'xyhd2quad']
    if isinstance(boxes, np.ndarray):
        anchor = torch.from_numpy(boxes)

    if order is 'xyxy2xywh':
        a = boxes[:, :2]
        b = boxes[:, 2:]
        new_boxes = torch.cat([(a + b) / 2, b - a + 1], 1)

    elif order is 'xywh2xyxy':
        a = boxes[:, :2]
        b = boxes[:, 2:]
        new_boxes = torch.cat([a - b / 2, a + b / 2], 1)

    elif order is 'xywh2quad':
        x0, y0, w0, h0 = torch.split(boxes, 1, dim=1)

        new_boxes = torch.cat([x0 - w0 / 2, y0 - h0 / 2,
                               x0 + w0 / 2, y0 - h0 / 2,
                               x0 + w0 / 2, y0 + h0 / 2,
                               x0 - w0 / 2, y0 + h0 / 2], dim=1)

    elif order is "quad2xyxy":
        """quad : [num_boxes, 8] / rect : [num_boxes, 4] #yxyx"""
        boxes = boxes.view((-1, 4, 2))

        new_boxes = torch.cat([torch.min(boxes[:, :, 0:1], dim=1)[0],
                               torch.min(boxes[:, :, 1:2], dim=1)[0],
                               torch.max(boxes[:, :, 0:1], dim=1)[0],
                               torch.max(boxes[:, :, 1:2], dim=1)[0]], dim=1)
    elif order is "xyhd2quad":
        num_anchor = len(boxes)
        x_center, y_center, h, dh, step = \
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        x_left = x_center - step / 2
        x_right = x_center + step / 2 - 1
        box_height = h + abs(dh)
        y_left_down = torch.where(dh < 0, y_center - box_height / 2, y_center - box_height / 2 + abs(dh))
        y_right_down = torch.where(dh < 0, y_center - box_height / 2 + abs(dh), y_center - box_height / 2)
        y_left_up = torch.where(dh < 0, y_center + box_height / 2 - abs(dh), y_center + box_height / 2)
        y_right_up = torch.where(dh < 0, y_center + box_height / 2, y_center + box_height / 2 - abs(dh))
        quad = [x_left, y_left_down, x_right, y_right_down, x_right, y_right_up - 1, x_left, y_left_up - 1]
        for idx in range(len(quad)):
            quad[idx] = quad[idx].reshape(1, -1)
        new_boxes = torch.cat(quad).transpose(0, 1)

    return new_boxes


def convert(anchor, mode='xyhd2quad'):
    """

    :param anchor: shape of n x 5 (num x (x , y , h ,dh , step) )
    :param mode:
    :return:
    """
    if isinstance(anchor, np.ndarray):
        anchor = torch.from_numpy(anchor)
    org_shape = list(anchor.shape)
    if len(org_shape) != 2:
        anchor = anchor.reshape(-1, org_shape[-1])
    new_boxes = change_box_order(anchor, mode)
    if len(org_shape) != 2:
        trans_shape = org_shape
        if mode[-4:] == 'xyxy':
            trans_shape[-1] = 4
        if mode[-4:] == 'quad':
            trans_shape[-1] = 8
        new_boxes = new_boxes.reshape(trans_shape)
    return new_boxes


def draw_four_vectors(img, line, color=(0, 255, 0)):
    """
    :param line: (x1,y1,x2,y2,x3,y3,x4,y4)
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    """
    if isinstance(line, tuple):
        line = list(line)
    if isinstance(line, np.ndarray):
        line = list(line)
    line = [int(l) for l in line]
    img = cv2.line(img, (line[0], line[1]), (line[2], line[3]), color)
    img = cv2.line(img, (line[2], line[3]), (line[4], line[5]), color)
    img = cv2.line(img, (line[4], line[5]), (line[6], line[7]), color)
    img = cv2.line(img, (line[6], line[7]), (line[0], line[1]), color)
    return img


def recover_img(img):
    if isinstance(img,torch.Tensor):
        img = img.clone()
    if isinstance(img,np.ndarray):
        img = img.copy()
    mean = (0.485, 0.456, 0.406)
    var = (0.229, 0.224, 0.225)
    infer_img = np.transpose(img.cpu().numpy(), (1, 2, 0))
    infer_img *= var
    infer_img += mean
    infer_img *= 255.
    infer_img = np.clip(infer_img, 0, 255)
    infer_img = infer_img.astype(np.uint8)
    return infer_img


def test_meshgrid():
    res = meshgrid(4, 4)
    print(res)

# test_meshgrid()
