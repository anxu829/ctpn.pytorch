import pickle

import cv2
import numpy as np
import torch
import glob , os
from nms.nms import cpu_nms

from train.augmentations import Augmentation_inference
from train.coder_utils import change_box_order, recover_img
from train.model.ctpn import CTPN
from train.text_connector.text_proposal_connector_oriented import TextProposalConnector as TextProposalConnectorOriented


class Inference:
    def __init__(self, model_path, encoder_path):
        self.model = self.load_model(model_path)
        self.encoder = pickle.load(open(encoder_path, 'rb'))
        self.text_proposal_connector = TextProposalConnectorOriented()

    def load_model(self, save_path,cuda = True):
        model_weights = torch.load(save_path)['net']
        model = CTPN(2)
        model.load_state_dict(model_weights)
        model.eval()
        if cuda:
            return model.cuda()
        else:
            return model

    def load_image(self, img: np.ndarray , cuda = True):
        img_infer = Augmentation_inference(1200)(img)[0]
        if cuda:
            return img_infer.unsqueeze(0).cuda()
        else:
            return img_infer.unsqueeze(0)


    def decode(self, loc_preds, cls_preds, infer_img):
        loc_pred, cls_pred = loc_preds[0].detach().cpu(), cls_preds[0].softmax(1)[:, 1].detach().cpu()
        d_cx, d_cy, d_h, d_dh = loc_pred[:, 0], loc_pred[:, 1], loc_pred[:, 2], loc_pred[:, 3]
        pred_quad, pred_cls_prob = self.encoder.decode_per_level(infer_img[0], 16, d_cx, d_cy, d_h, d_dh, cls_pred,
                                                                 thres=0.9, return_prob=True)
        pred_box = change_box_order(pred_quad, "quad2xyxy")
        return pred_quad, pred_box, pred_cls_prob

    def infernce(self, img_path, return_text_box=True):
        img = cv2.imread(img_path)
        infer_img = self.load_image(img)
        loc_preds, cls_preds = self.model(infer_img)
        pred_quad, pred_box, cls_pred = self.decode(loc_preds, cls_preds, infer_img)
        pred_nms_format = torch.cat((pred_box, cls_pred.reshape(-1, 1)), 1).numpy()
        nms_idx = cpu_nms(pred_nms_format, 0.7)
        pred_quad = pred_quad[nms_idx]
        pred_box = pred_box[nms_idx]
        pred_prob = cls_pred[nms_idx]

        if return_text_box:
            merge_res = self.text_proposal_connector.get_text_lines(pred_box.detach().cpu().numpy(),
                                                                    pred_prob.detach().cpu().numpy(), (1200, 1200))
            pred_quad_text, pred_prob_text = merge_res[:, :8], merge_res[:, -1]
            return pred_quad_text, pred_prob_text
        else:
            return pred_quad, pred_prob

    def viz(self, img_path, pred_quad):
        if isinstance(pred_quad, torch.Tensor):
            pred_quad = pred_quad.detach().cpu().numpy()
        img = cv2.imread(img_path)
        infer_img = self.load_image(img)
        infer_img = recover_img(infer_img[0])
        viz = cv2.drawContours(infer_img, pred_quad.reshape(-1, 4, 2).astype(int), -1, (0, 255, 0), 1)
        return viz


if __name__ == "__main__":
    save_path = '/disk3/xuan/detection/ctpn/model/cptn_medical_best.pth'
    encoder_path = '/tmp/pycharm_project_782/eval/encoder.pkl'
    img_path = '/disk2/zjq/data/gt/gt/menzhen/'
    write_output = '/disk3/xuan/data/medical_test'
    inference = Inference(save_path, encoder_path)
    for im in glob.glob(os.path.join(img_path , '*.jpg')):
        print(im)
        pred_quad, pred_prob = inference.infernce(im,return_text_box = True)
        viz_plot = inference.viz(im, pred_quad)
        cv2.imwrite(os.path.join(write_output,os.path.basename(im)), viz_plot)

        pred_quad, pred_prob = inference.infernce(im, return_text_box=False)
        viz_plot = inference.viz(im, pred_quad)
        cv2.imwrite(os.path.join(write_output, os.path.basename(im).split('.jpg')[0] + '_block.jpg'), viz_plot)


