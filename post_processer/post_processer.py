import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import label

_DEBUG = True
_DEBUG_PATH = '/Volumes/新加卷/ocr/detection/ctpn.pytorch/test_data/debug_res/'
shutil.rmtree(_DEBUG_PATH)
os.mkdir(_DEBUG_PATH)


class AnchorProcesser:
    def __init__(self, pred_loc: np.array, step=8):
        create_time_start = time.time()
        self.step = step
        self.pred_loc = pd.DataFrame({
            "cx": pred_loc[:, 0],
            "cy": pred_loc[:, 1],
            "h": pred_loc[:, 2],
            "dh": pred_loc[:, 3],
        })

        if _DEBUG:
            print('pred loc create idx time: {}'.format((time.time() - create_time_start) * 1000))

        self.update_df()

    def update_df(self):
        # for spped ,dit it greedy
        # TODO ,add groupby func
        self.pred_loc.drop_duplicates(subset=['cx', 'cy'])
        # add uid to each anchor
        self.pred_loc.loc[:, 'uid'] = np.arange(self.pred_loc.shape[0])
        self.pred_loc.loc[:, 'top'] = self.pred_loc.cy - self.pred_loc.h / 2
        self.pred_loc.loc[:, 'bot'] = self.pred_loc.cy + self.pred_loc.h / 2

    def update_core_anchor_label(self, seg_processer):
        loc_info = self.pred_loc.loc[:, ['cy', 'cx']]
        # 对于中心点在mask中的，直接归入mask，ctpn训练的足够时，应该是可行的

        core_cls = seg_processer.label_mask[loc_info['cy'].astype(int), loc_info['cx'].astype(int)]

        self.pred_loc.loc[:, 'core_cls'] = core_cls
        self.pred_loc.loc[:, 'is_core'] = core_cls > 0


class Mask:
    def __init__(self, mask_id, mask, anchor_processer: AnchorProcesser, image):
        # mask : binary mask of that cls
        # init : get quad info of that masks
        self.mask_id = mask_id
        self.mask = mask
        self.image = image.copy()
        self.base_anchor, self.edge_uid = self.get_base_anchor(anchor_processer)

    def get_base_anchor(self, anchor_processer: AnchorProcesser):
        self.base_anchor_set = anchor_processer.pred_loc.loc[anchor_processer.pred_loc.core_cls == self.mask_id]
        base_anchor_info = self.base_anchor_set.groupby('cx', as_index=False).agg(
            {"top": {
                "top_min": "min",
                "top_idx": "idxmin"
            }
                ,
                "bot": {
                    "bot_max": "max",
                    "bot_idx": "idxmax"
                }
            }).sort_index()

        if base_anchor_info.shape[0] > 0:
            first = base_anchor_info.iloc[0].values
            last = base_anchor_info.iloc[-1].values

            left_top = first[0] - anchor_processer.step, first[1]
            left_top_uid = self.base_anchor_set.loc[first[2].astype(int)].uid

            left_bot = first[0] - anchor_processer.step, first[3]
            left_bot_uid = self.base_anchor_set.loc[first[4].astype(int)].uid

            right_top = last[0] + anchor_processer.step, last[1]
            right_top_uid = self.base_anchor_set.loc[last[2].astype(int)].uid

            right_bot = last[0] + anchor_processer.step, last[3]
            right_bot_uid = self.base_anchor_set.loc[last[4].astype(int)].uid

            anchor_base = np.array([left_top, right_top, right_bot, left_bot])
            # TODO ,now only condiser 2 uid ,when ctpn result is good ,this is enough
            edge_uid = [left_top_uid, right_bot_uid]
            if _DEBUG:
                viz_plot = cv2.drawContours(self.image.copy(), anchor_base.reshape(-1, 4, 2).astype(int), -1, 255, 1)
                cv2.imwrite(_DEBUG_PATH + 'id_{}_viz.jpg'.format(self.mask_id), viz_plot)


            return anchor_base, edge_uid
        return None,None


class SegProcesser:
    def __init__(self, image, mask):
        self.image = image
        self.label_mask = label(mask)
        # if _DEBUG:
        #     plt.imshow(self.label_mask)
        #     plt.show()
        self.mask_num = self.label_mask.max()

    def update_core_mask_label(self, anchor_processer: AnchorProcesser):
        # 利用core_anchor的信息，更新每个mask的坐标信息
        for i in range(1, self.mask_num + 1):
            mask = Mask(i, self.label_mask == i, anchor_processer, self.image)


class Processer:
    def __init__(self, image: np.ndarray, pred_loc: np.array, pred_seg: np.ndarray, step=8):
        self.image = image
        self.pred_loc = pred_loc
        self.pred_seg = pred_seg

        self.anchor_processer = AnchorProcesser(self.pred_loc, step, )
        self.seg_processer = SegProcesser(image, pred_seg)

    def process(self):
        # 第一轮，直接为anchor 附上center 的label
        self.anchor_processer.update_core_anchor_label(self.seg_processer)

        # 第二轮，更新core_mask的信息，（选出core_mask中，最左和最右的anchor）
        self.seg_processer.update_core_mask_label(self.anchor_processer)

        # 第三轮，greedy的遍历剩余的没有遍历过的anchor直到全部遍历过

