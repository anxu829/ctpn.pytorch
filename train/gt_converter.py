import math

import cv2
import numpy as np
import torch
from scipy.spatial import distance as dist

from coder_utils import recover_img

"""
在这里将每个文本的框转换到ctpn类型的小框，返回每个小框的四个坐标点信息
"""

"""
gt 样例
矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
x1,y1,x2,y2,x3,y3,x4,y4,language,text
language 可能有：
- Arabic
- Latin
- Chinese
- Korean
- Japanese
- Bangla
- Symbols
- None
text 可能有：
- 图片上对应的文字
- 无法识别的文字用 ### 表示
只转换 Latin 和 Chinese 语言，忽略 ### 的部分
"""

DEBUG = False
DEBUG_ALL = False

SCALE = 600
MAX_SCALE_LENGTH = 1200
step = 16

# resize 以后允许的最小高度
MIN_TEXT_HEIGHT = 5

# resize 以后允许的 anchor 数量
MIN_CONTINUE_ANCHORS = 0

# resize 以后允许text line区域最大的高宽比
MAX_HEIGHT_WIDTH_SCALE = 4

# 对文本行进行 split 后，允许最少的 anchor 数量
MIN_ANCHOR_COUNT = 10

global_k_is_none_count = 0


def get_clockwise(pnts):
    """
    :param pnts: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :return:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        left-top, right-top, right-bottom, left-bottom
    """
    out = []
    p = sorted(pnts, key=lambda x: x[1])
    if p[0][0] < p[1][0]:
        out.append(p[0])
        out.append(p[1])
    else:
        out.append(p[1])
        out.append(p[0])

    if p[2][0] > p[3][0]:
        out.append(p[2])
        out.append(p[3])
    else:
        out.append(p[3])
        out.append(p[2])

    return out


def order_points(pts):
    # https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def parse_line(pnts, im_scale):
    """
    :param pnts:
        "x1,y1,x2,y2,x3,y3,x4,y4,language,text"
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    :return:
        (x1,y1,x2,y2,x3,y3,x4,y4), language, text
    """
    splited_line = pnts.split(',')
    if len(splited_line) > 10:
        splited_line[-1] = ','.join(splited_line[10:])

    for i in range(8):
        splited_line[i] = int(int(splited_line[i]) * im_scale)

    pnts = (splited_line[0], splited_line[1],
            splited_line[2], splited_line[3],
            splited_line[4], splited_line[5],
            splited_line[6], splited_line[7])

    return pnts, splited_line[-2], splited_line[-1]


def get_ltrb(line):
    """
    :param line: (x1,y1,x2,y2,x3,y3,x4,y4)
    :return: (xmin, ymin, xmax, ymax)
    """
    xmin = min(line[0], line[6])
    ymin = min(line[1], line[3])
    xmax = max(line[2], line[4])
    ymax = max(line[5], line[7])

    return xmin, ymin, xmax, ymax


def get_img_scale(img, scale, max_scale):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(scale) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_scale:
        im_scale = float(max_scale) / float(im_size_max)

    return im_scale


def clip_line(line, size):
    """
    :param line: (xmin, ymin, xmax, ymax)
    :param size: (height, width)
    :return: (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = line

    if xmin < 0:
        xmin = 0
    if xmax > size[1] - 1:
        xmax = size[1] - 1
    if ymin < 0:
        ymin = 0
    if ymax > size[0] - 1:
        ymax = size[0] - 1

    return xmin, ymin, xmax, ymax


def split_text_line(line, step):
    """
    按照 Bounding box 对文字进行划分
    :param line: (xmin, ymin, xmax, ymax)
    :return: [(anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax)]
    """
    xmin, ymin, xmax, ymax = line
    width = xmax - xmin

    anchor_count = int(math.ceil(width / step))

    splited_lines = []

    for i in range(anchor_count):
        anchor_xmin = i * step + xmin
        anchor_ymin = ymin
        anchor_xmax = anchor_xmin + step - 1
        anchor_ymax = ymax

        splited_lines.append((anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax))

    return splited_lines


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)


class Line:
    def __init__(self, p0: Point, p1: Point):
        self.p0 = p0
        self.p1 = p1

        if p0.x - p1.x == 0:
            self.k = None
        else:
            self.k = (self.p0.y - self.p1.y) / (self.p0.x - self.p1.x)

        # f = ax+by+c = 0
        self.a = self.p0.y - self.p1.y
        self.b = self.p1.x - self.p0.x
        self.c = self.p0.x * self.p1.y - self.p1.x * self.p0.y

    def cross(self, line) -> Point:
        d = self.a * line.b - line.a * self.b
        if d == 0:
            return None

        x = (self.b * line.c - line.b * self.c) / d
        y = (line.a * self.c - self.a * line.c) / d

        return Point(x, y)

    def contain(self, p: Point) -> bool:
        if p is None:
            return False

        # 输入的点应该吃 cross(求出来的交点)
        # p 点是否落在 p0 和 p1 之间, 而不是延长线上
        if p.x > max(self.p1.x, self.p0.x):
            return False

        if p.x < min(self.p1.x, self.p0.x):
            return False

        if p.y > max(self.p1.y, self.p0.y):
            return False

        if p.y < min(self.p1.y, self.p0.y):
            return False

        return True


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


def draw_bounding_box(img, line, color=(255, 0, 0)):
    """
    :param line: (xmin, ymin, xmax, ymax)
    """
    img = cv2.line(img, (line[0], line[1]), (line[2], line[1]), color)
    img = cv2.line(img, (line[2], line[1]), (line[2], line[3]), color)
    img = cv2.line(img, (line[2], line[3]), (line[0], line[3]), color)
    img = cv2.line(img, (line[0], line[3]), (line[0], line[1]), color)
    return img


def split_text_line2(line, step, img=None):
    """
    按照 minAreaRect 对文本进行划分
    :param line:
        (x1,y1,x2,y2,x3,y3,x4,y4)
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    :return: [(anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax)]
    """
    global global_k_is_none_count
    if DEBUG:
        img = draw_four_vectors(img, line)

    xmin, ymin, xmax, ymax = get_ltrb(line)
    width = xmax - xmin
    height = ymax - ymin

    if height > MAX_HEIGHT_WIDTH_SCALE * width:
        return []

    anchor_count = int(math.ceil(width / step))

    if DEBUG:
        img = draw_bounding_box(img, (xmin, ymin, xmax, ymax))

    rect = cv2.minAreaRect(np.asarray([[line[0], line[1]],
                                       [line[2], line[3]],
                                       [line[4], line[5]],
                                       [line[6], line[7]]]))
    # 获得最小 rotate rect 的四个角点
    box = cv2.boxPoints(rect)
    box = get_clockwise(box)

    if DEBUG:
        img = draw_four_vectors(img, (box[0][0], box[0][1],
                                      box[1][0], box[1][1],
                                      box[2][0], box[2][1],
                                      box[3][0], box[3][1]), color=(255, 55, 55))

    p1 = Point(box[0][0], box[0][1])
    p2 = Point(box[1][0], box[1][1])
    p3 = Point(box[2][0], box[2][1])
    p4 = Point(box[3][0], box[3][1])

    l1 = Line(p1, p2)
    l2 = Line(p2, p3)
    l3 = Line(p3, p4)
    l4 = Line(p4, p1)
    lines = [l1, l2, l3, l4]

    if l1.k is None:
        global_k_is_none_count += 1
        print("l1 K is None")
        print(p1)
        print(p2)
        print(p3)
        print(p4)
        return []

    splited_lines = []
    for i in range(anchor_count):
        anchor_xmin = i * step + xmin
        anchor_xmax = anchor_xmin + step - 1
        anchor_ymin = ymin
        anchor_ymax = ymax

        # 垂直于 X 轴的线
        left_line = Line(Point(anchor_xmin, 0), Point(anchor_xmin, height))
        right_line = Line(Point(anchor_xmax, 0), Point(anchor_xmax, height))

        left_cross_pnts = [left_line.cross(l) for l in lines]
        right_cross_pnts = [right_line.cross(l) for l in lines]

        if l1.k < 0:
            if l1.contain(right_cross_pnts[0]):
                anchor_ymin = right_cross_pnts[0].y

            if l4.contain(right_cross_pnts[3]):
                anchor_ymax = right_cross_pnts[3].y

            if l3.contain(left_cross_pnts[2]):
                anchor_ymax = left_cross_pnts[2].y

            if l2.contain(left_cross_pnts[1]):
                anchor_ymin = left_cross_pnts[1].y

        if l1.k > 0:
            if l4.contain(right_cross_pnts[3]):
                anchor_ymin = right_cross_pnts[3].y

            if l3.contain(right_cross_pnts[2]):
                anchor_ymax = right_cross_pnts[2].y

            if l1.contain(left_cross_pnts[0]):
                anchor_ymin = left_cross_pnts[0].y

            if l2.contain(left_cross_pnts[1]):
                anchor_ymax = left_cross_pnts[1].y

        if anchor_ymax - anchor_ymin <= MIN_TEXT_HEIGHT:
            continue

        if DEBUG:
            img = draw_bounding_box(img, (anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax), (0, 0, 255))
            cv2.imshow('test', img)
            cv2.waitKey()

        splited_lines.append((anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax))

    if DEBUG:
        cv2.imshow('test', img)
        cv2.waitKey()

    return splited_lines


def split_text_line3(line, step, img, step_align=True):
    """
    按照 minAreaRect 对文本进行划分
    :param line:
        (x1,y1,x2,y2,x3,y3,x4,y4)
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    :return: [(anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax)]
    """
    if isinstance(line, torch.Tensor):
        line = line.cpu().data.numpy()
    global global_k_is_none_count
    if DEBUG:
        img = draw_four_vectors(img, line)

    # 首先，按照step切分img
    step_max = int(np.ceil(img.shape[1] / step))
    max_start_col = step_max - 1
    xmin, ymin, xmax, ymax = get_ltrb(line)
    width = xmax - xmin
    height = ymax - ymin

    if height > MAX_HEIGHT_WIDTH_SCALE * width:
        return []
    if xmax - xmin < step:
        # 过滤掉特别小的框

        return []
    anchor_count = int(math.ceil(width / step))

    if DEBUG:
        img = draw_bounding_box(img, (xmin, ymin, xmax, ymax))

    rect = cv2.minAreaRect(np.asarray([[line[0], line[1]],
                                       [line[2], line[3]],
                                       [line[4], line[5]],
                                       [line[6], line[7]]]))
    # 获得最小 rotate rect 的四个角点
    box = cv2.boxPoints(rect)
    box = order_points(box)

    if DEBUG:
        img = draw_four_vectors(img, (box[0][0], box[0][1],
                                      box[1][0], box[1][1],
                                      box[2][0], box[2][1],
                                      box[3][0], box[3][1]), color=(255, 55, 55))
    # 获取anchor的相关信息
    p1 = Point(box[0][0], box[0][1])
    p2 = Point(box[1][0], box[1][1])
    p3 = Point(box[2][0], box[2][1])
    p4 = Point(box[3][0], box[3][1])

    mid_p12 = Point((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
    mid_p34 = Point((box[2][0] + box[3][0]) / 2, (box[2][1] + box[3][1]) / 2)

    if mid_p12.y >= mid_p34.y:
        print('bugs happen , this line is not useful')
        return []

    l1 = Line(p1, p2)
    l2 = Line(p2, p3)
    l3 = Line(p3, p4)
    l4 = Line(p4, p1)
    lines = [l1, l2, l3, l4]

    if l1.k is None:
        global_k_is_none_count += 1
        print("l1 K is None")
        print(p1)
        print(p2)
        print(p3)
        print(p4)
        return []

    quad = []
    splited_lines = []
    side_refinement = []
    shift = ((np.ceil((xmax - xmin) / step)) * step - (xmax - xmin)) / 2
    if step_align:
        anchor_start = int(np.floor(xmin / step) * step)
        anchor_end = int((np.ceil(xmax / step)) * step)
        if abs(anchor_start - xmin) > (step // 2):
            anchor_start = anchor_start + step
        if abs(anchor_end - xmax) > (step // 2):
            anchor_end = anchor_end - step
    else:
        anchor_start = int(np.floor(xmin - shift))
        anchor_end = int(np.floor(xmax + shift))

    interval = int((anchor_end - anchor_start) / step)
    for start in range(interval):
        # 这里的down，up是按照y轴方向上的大小来定义的，靠近0的是down，即在上方的是down
        if anchor_start + start * step > max_start_col * step:
            continue
        grid_start = anchor_start + start * step
        grid_end = anchor_start + (start + 1) * step
        line_left_down = Point(grid_start, 0)
        line_left_up = Point(grid_start, height)
        line_right_down = Point(grid_end, 0)
        line_right_up = Point(grid_end, height)
        line_left = Line(line_left_down, line_left_up)
        line_right = Line(line_right_down, line_right_up)
        # 计算和 box的上下的line的交点
        left_down = line_left.cross(l1)
        left_up = line_left.cross(l3)
        right_down = line_right.cross(l1)
        right_up = line_right.cross(l3)

        center_y = (left_down.y + right_down.y) / 2 + (
                (left_up.y + right_up.y) / 2 - (left_down.y + right_down.y) / 2) / 2
        center_x = (grid_start + grid_end) / 2

        h = (left_up.y - left_down.y + right_up.y - right_down.y) / 2
        dh = (left_up.y - right_up.y + left_down.y - right_down.y) / 2  # dh 定义成左侧减去右侧
        splited_lines.append((center_x, center_y, h, dh, step))
        quad.append(
            (left_down.x, left_down.y, right_down.x, right_down.y, right_up.x, right_up.y, left_up.x, left_up.y))

        if DEBUG:
            img = draw_four_vectors(img, (left_down.x, left_down.y,
                                          right_down.x, right_down.y,
                                          right_up.x, right_up.y,
                                          left_up.x, left_up.y,
                                          ), color=(0, 255, 55))
            cv2.imshow('test', img)
            cv2.waitKey()
        # 考虑side refinement
        # if abs(center_x - xmin) < 8:
        #     side_refinement.append(center_x - xmin)
        # elif abs(center_x - xmax) < 8:
        #     side_refinement.append(center_x - xmax)
        # else:
        #     side_refinement.append(-999)

        if start == 0:
            side_refinement.append(center_x - xmin)
        elif start == interval - 1:
            side_refinement.append(center_x - xmax)
        else:
            side_refinement.append(-999)

    if DEBUG:
        cv2.imshow('test', img)
        cv2.waitKey()
    splited_lines = np.array(splited_lines)
    quad = np.array(quad)
    return quad, splited_lines, side_refinement


def convert_boxes(boxes, template_img, step=16):
    template_img_ = template_img.clone()
    template_img_ = recover_img(template_img_)
    split_gt_all = []
    quad_gt_all = []
    side_refinement_all = []
    for box in boxes:
        res = split_text_line3(box, step, template_img_)
        if not res:
            continue
        quad, anchor, side_refinement = res
        if quad is not None and quad.shape[0] > 0:
            split_gt_all.append(anchor)
            quad_gt_all.append(quad)
            side_refinement_all.append(side_refinement)
    gt = np.concatenate(split_gt_all)
    side_refinement_all = np.concatenate(side_refinement_all)
    return gt, side_refinement_all
