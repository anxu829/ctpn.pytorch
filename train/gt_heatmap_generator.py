import cv2
import numpy as np
import torch


# 整体流程为：
# 首先利用opencv的draw函数确定框内的点
# 其次计算中心点位置
# 随后对这些内容计算alpha 、beta
# 然后对alpha  、 beta 取出正确的内容
# 然后绘制在特征图当中

def get_wh(image):
    if isinstance(image, torch.Tensor):
        w, h = image.shape[2], image.shape[1]
    else:
        w, h = image.shape[1], image.shape[0]
    return w, h


def four_point_transform(image, pts, w, h):
    # max_x, max_y = np.max(pts[:, 0]).astype(np.int32), np.max(pts[:, 1]).astype(np.int32)

    dst = np.array([
        [0, 0],
        [image.shape[1] - 1, 0],
        [image.shape[1] - 1, image.shape[0] - 1],
        [0, image.shape[0] - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(dst, pts)
    # warped = cv2.warpPerspective(image, M, (max_x, max_y))
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped


def get_gaussian_heatmap(sigma=10, spread=3):
    extent = int(spread * sigma)
    gaussian_heatmap = np.zeros([2 * extent, 2 * extent], dtype=np.float32)
    extent = int(spread * sigma)
    for i in range(2 * extent):
        for j in range(2 * extent):
            gaussian_heatmap[i, j] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                -1 / 2 * ((i - spread * sigma - 0.5) ** 2 + (j - spread * sigma - 0.5) ** 2) / (sigma ** 2))

    gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255).astype(np.uint8)
    return gaussian_heatmap


def gen_PMTD_heatmap_boxes(img, box):
    """
    input:
        input an ndarray with [h , w , c]
        box : [1,8]
    output:
        heap_map like input shape
    """
    # create heatmap with same shape of img
    x1, y1, x2, y2, x3, y3, x4, y4 = box

    w, h = get_wh(img)
    heatmap = np.zeros(shape=(h, w))

    poly = box.reshape(4, 2).astype(np.uint64)

    cv2.fillPoly(heatmap, [poly], 1)
    loc_in_box = np.argwhere(heatmap == 1)
    center_x = box[[0, 2, 4, 6]].min() + (box[[0, 2, 4, 6]].max() - box[[0, 2, 4, 6]].min()) / 2
    center_y = box[[1, 3, 5, 7]].min() + (box[[1, 3, 5, 7]].max() - box[[1, 3, 5, 7]].min()) / 2
    center = np.array([center_x, center_y])
    corner_comb = np.array([
        [[x1, x2], [y1, y2]],
        [[x2, x3], [y2, y3]],
        [[x3, x4], [y3, y4]],
        [[x4, x1], [y4, y1]],
    ])
    alpha_all = []
    beta_all = []
    try:
        for comb in corner_comb:
            comb_ = comb - center.reshape(2, 1)
            comb_inv = np.linalg.inv(comb_)
            xy_info = loc_in_box[:, [1, 0]] - center
            alpha, beta = np.matmul(comb_inv, xy_info.T)
            alpha_all.append(alpha)
            beta_all.append(beta)

        alpha_all = np.stack(alpha_all).T
        beta_all = np.stack(beta_all).T

        score = 1 - (alpha_all + beta_all).max(1)

        score = np.where(score > 0, score, 0)
        heatmap[loc_in_box[:, 0], loc_in_box[:, 1]] = score
        return heatmap
    except:
        print('invalid box , box is {} , img shape is {}'.format(box, img.shape))
        return heatmap


def gen_CRAFT_heatmap_boxes(img, box):
    # 对每个box使用透视变换
    w, h = get_wh(img)

    heatmap = np.zeros(shape=(h, w))

    box = box.reshape(4, 2)
    top_left = np.array([np.min(box[:, 0]), np.min(box[:, 1])]).astype(np.int32)
    bot_right = np.array([np.max(box[:, 0]), np.max(box[:, 1])]).astype(np.int32)

    if top_left[1] > w or top_left[0] > h:
        # This means there is some bug in the character bbox
        # Will have to look into more depth to understand this
        return heatmap

    origin = (0, 0)
    box_offset = [None] * 4
    image_offset = [None] * 4
    if top_left[0] < 0:
        box_offset[0] = 0
        image_offset[0] = abs(top_left[0])
    else:
        box_offset[0] = top_left[0]
        image_offset[0] = 0

    if top_left[1] < 0:
        box_offset[1] = 0
        image_offset[1] = abs(top_left[1])
    else:
        box_offset[1] = top_left[1]
        image_offset[1] = 0
    image_offset[2] = image_offset[0] + w
    image_offset[3] = image_offset[1] + h
    box_offset[2] = box_offset[0] + bot_right[0] - top_left[0]
    box_offset[3] = box_offset[1] + bot_right[1] - top_left[1]
    mask_w = max(box_offset[2], image_offset[2])
    mask_h = max(box_offset[3], image_offset[3])

    mask = np.zeros(shape=(mask_h, mask_w))

    xmin = box[:, 0].min()
    ymin = box[:, 1].min()
    box_ = box.copy()
    box_[:, 0] = box_[:, 0] - xmin
    box_[:, 1] = box_[:, 1] - ymin
    gaussian = get_gaussian_heatmap()
    gaussian_ = four_point_transform(gaussian.copy(), box_.astype(np.float32), bot_right[0] - top_left[0],
                                     bot_right[1] - top_left[1])

    xmin, ymin, xmax, ymax = box_offset
    try:
        mask[ymin: ymax, xmin:xmax] = gaussian_

        img_xmin, img_ymin, img_xmax, img_ymax = image_offset
        heatmap = mask[img_ymin: img_ymax, img_xmin:img_xmax]
        # rempa to 0~1
        heatmap = heatmap / 255
        return heatmap
    except:
        print(
            'gaussian shape is {} ,box_offset {} , image_offset is {} , mask is {} '.format(gaussian_.shape, box_offset,
                                                                                            image_offset, mask))
        return heatmap


def gen_heatmap_image(img, boxes, method='PMTD'):
    assert method in ['PMTD', 'CRAFT']
    w, h = get_wh(img)
    if isinstance(boxes, torch.Tensor):
        boxes_ = boxes.cpu().data.numpy()
    heatmap = np.zeros(shape=(h, w))

    if method == 'PMTD':
        for box in boxes_:
            heatmap_box = gen_PMTD_heatmap_boxes(img, box)
            heatmap = np.where(heatmap_box > heatmap, heatmap_box, heatmap)
    elif method == 'CRAFT':
        for box in boxes_:
            heatmap_box = gen_CRAFT_heatmap_boxes(img, box)
            heatmap = np.where(heatmap_box > heatmap, heatmap_box, heatmap)
    return heatmap
