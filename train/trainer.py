from __future__ import print_function

import os
import pickle
import shutil
import time
from subprocess import Popen, PIPE

import cv2
import numpy as np
import torch
import torch.optim as optim
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter

from arg_parser import args
from augmentations import Augmentation_traininig
from coder_utils import recover_img, draw_four_vectors, change_box_order
from dataset import ListDataset
from loss.ctpn_loss import CTPNLoss
from loss.seg_loss import SegLoss
from model.ctpn import CTPN


def adjust_learning_rate(cur_lr, optimizer, gamma, step):
    lr = cur_lr * (gamma ** (step // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Trainer:
    def __init__(self, args):

        self.cfg = yaml.load(open('config.yaml', encoding='utf-8'))
        self.cfg = EasyDict(self.cfg)
        if args.model == 'vgg':
            self.cfg = self.cfg.VGGConfig
        if args.model == 'pse':
            self.cfg = self.cfg.PSENetConfig

        assert torch.cuda.is_available(), 'Error: CUDA not found!'
        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)

        if not os.path.exists(args.logdir):
            os.mkdir(args.logdir)

        log_dir = args.logdir
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            os.mkdir(log_dir)

        # Data
        print('==> Preparing data..')
        self.trainset = ListDataset(train=True,
                                    transform=Augmentation_traininig,
                                    input_size_min=args.input_size_min,
                                    input_size_max=args.input_size_max,
                                    multi_scale=args.multi_scale,
                                    train_path='./test_data/image',
                                    debug_num=4,
                                    divisibility=self.cfg.DATA.SIZE_DIVISIBILITY,
                                    step=self.cfg.ANCHOR_HEAD.STRIDE,
                                    gen_heat_map=self.cfg.PIXEL_HEAD.PIXEL_PREDICTION
                                    )

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            collate_fn=self.trainset.collate_fn,
        )
        print('==> Preparing data Done...')
        # set model (focal_loss vs OHEM_CE loss)

        self.ctpn_criterion = CTPNLoss()
        self.seg_criterion = SegLoss()
        self.num_classes = 1

        # Training Detail option\
        # if args.dataset in ["SynthText"] else (2000, 4000, 6000, 8000, 10000)
        self.stepvalues = (10000, 20000, 30000, 40000, 50000)
        self.best_loss = float('inf')  # best test loss
        self.start_epoch = 0  # start from epoch 0 or last epoch
        self.iteration = 0
        self.cur_lr = args.lr
        self.mean = (0.485, 0.456, 0.406)
        self.var = (0.229, 0.224, 0.225)
        self.step_index = 0
        self.pEval = None

        # Model
        # 注意： resnet中， downsample 部分会用 stride = (2,2)的 1x1 conv ，会使得 25 x 25 -> 13 x 13 ， 和 anchor 处理机制不符
        # 所以resnet 中，一定要对数据做 SIZE_DIVISIBILITY
        self.net = CTPN(self.cfg)  # plus one means the bg

        if args.resume:
            # 这里要做模型加载预训练模型，在finetune是使用
            print('==> Resuming from checkpoint..', args.resume)
            checkpoint = torch.load(args.resume)
            self.net.load_state_dict(checkpoint['net'])

        print("multi_scale : ", args.multi_scale)
        print("input_size : ", args.input_size_min)
        print("stepvalues : ", self.stepvalues)
        print("start_epoch : ", self.start_epoch)
        print("iteration : ", self.iteration)
        print("cur_lr : ", self.cur_lr)
        print("step_index : ", self.step_index)
        print("num_gpus : ", torch.cuda.device_count())

        self.net = torch.nn.DataParallel(self.net, device_ids=list(range(torch.cuda.device_count())))
        self.net.cuda()

        # Training
        self.net.train()
        # net.module.freeze_bn()  # you must freeze batchnorm
        # optimizer = optim.SGD(net.parameters(), lr=cur_lr, momentum=0.9, weight_decay=1e-4)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.cur_lr)

        # encoder = DataEncoder()
        self.encoder = self.trainset.encoder

        pickle.dump(
            self.encoder,
            open(os.path.join(args.save_folder, 'encoder.pkl'), 'wb')
        )
        # tensorboard visualize
        self.writer = SummaryWriter(log_dir=args.logdir)

    def train(self):
        t0 = time.time()
        for epoch in range(self.start_epoch, 10000):
            if self.iteration > args.max_iter:
                break

            for data in self.trainloader:
                if self.cfg.PIXEL_HEAD.PIXEL_PREDICTION:
                    if self.cfg.PIXEL_HEAD.CORNER_PREDICTION:
                        image, d_cx, d_cy, d_h, d_dh, cls, d_h_cls, d_cx_cls, seg_targets, corner_targets = data.values()
                    else:
                        image, d_cx, d_cy, d_h, d_dh, cls, d_h_cls, d_cx_cls, seg_targets = data.values()
                else:
                    image, d_cx, d_cy, d_h, d_dh, cls, d_h_cls, d_cx_cls = data.values()
                    seg_targets = None

                offset_label = torch.stack((d_cx, d_cy, d_h, d_dh), dim=2)

                # 这里，注意到因为是不需要全部内容都放进cuda的，可以考虑之后再放进去，减少显存消耗
                # loc_targets = Variable(offset_label.cuda())
                # cls_targets = Variable(cls.cuda())
                loc_targets = offset_label
                cls_targets = cls

                self.optimizer.zero_grad()

                net_output = self.net(image.tensors)
                loc_preds, cls_preds = net_output['anchor_head']
                seg_preds = net_output['seg_head']

                loc_loss, cls_loss, seg_loss = 0, 0, 0
                # 计算ctpn loss
                loc_loss, cls_loss, dh_loss, cls_sample_per_image = self.ctpn_criterion(image.masks,
                                                                                        loc_preds, loc_targets,
                                                                                        cls_preds,
                                                                                        cls_targets, d_h_cls,
                                                                                        d_cx_cls)

                # 计算pixel loss
                if self.cfg.PIXEL_HEAD.PIXEL_PREDICTION:
                    if self.cfg.PIXEL_HEAD.CORNER_PREDICTION:
                        seg_loss = self.seg_criterion(seg_preds,
                                                      seg_targets.type(torch.float).cuda(),
                                                      corner_targets.type(torch.float).cuda()
                                                      )
                    else:
                        seg_loss = self.seg_criterion(seg_preds, seg_targets.type(torch.float).cuda())
                loss = loc_loss + cls_loss + seg_loss
                loss.backward()

                self.optimizer.step()
                if self.iteration % 80 == 0:
                    t1 = time.time()
                    print('iter ' + repr(self.iteration) + ' (epoch ' + repr(
                        epoch) + ') || loss: %.4f || l loc_loss: %.4f || l cls_loss: %.4f || l seg_loss : %.4f || (Time : %.1f)' \
                          % (loss.sum().item(), loc_loss.sum().item(), cls_loss.sum().item(), seg_loss.sum().item(),
                             (t1 - t0)))

                    self.viz(loss, loc_loss, cls_loss, dh_loss, seg_loss,
                             loc_preds, cls_preds, loc_targets, cls_targets, seg_targets,
                             cls_sample_per_image, d_h_cls, d_cx_cls, seg_preds,
                             image)
                    t0 = time.time()
                if self.iteration % args.save_interval == 0 and self.iteration > 0:
                    print('Saving state, iter : ', self.iteration)
                    state = {
                        'net': self.net.module.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        'iteration': self.iteration,
                        'epoch': epoch,
                        'lr': self.cur_lr,
                        'step_index': self.step_index
                    }
                    model_file = args.save_folder + 'ckpt_' + repr(self.iteration) + '.pth'
                    torch.save(state, model_file)
                if self.iteration in self.stepvalues:
                    self.step_index += 1
                    self.cur_lr = adjust_learning_rate(self.cur_lr, self.optimizer, args.gamma, self.step_index)
                if self.iteration > args.max_iter:
                    break
                if args.evaluation and self.iteration % args.eval_step == 0:
                    try:
                        if pEval is None:
                            print("Evaluation started at iteration {} on IC15...".format(self.iteration))
                            eval_cmd = "CUDA_VISIBLE_DEVICES=" + str(args.eval_device) + \
                                       " python eval.py" + \
                                       " --tune_from=" + args.save_folder + 'ckpt_' + repr(self.iteration) + '.pth' + \
                                       " --input_size=1024" + \
                                       " --output_zip=result_temp1"

                            pEval = Popen(eval_cmd, shell=True, stdout=PIPE, stderr=PIPE)

                        elif pEval.poll() is not None:
                            (scorestring, stderrdata) = pEval.communicate()

                            hmean = float(str(scorestring).strip().split(":")[3].split(",")[0].split("}")[0].strip())

                            self.writer.add_scalar('test_hmean', hmean, self.iteration)

                            print("test_hmean for {}-th iter : {:.4f}".format(self.iteration, hmean))

                            if pEval is not None:
                                pEval.kill()
                            pEval = None

                    except Exception as e:
                        print("exception happened in evaluation ", e)
                        if pEval is not None:
                            pEval.kill()
                        pEval = None
                self.iteration += 1

    def viz(self, loss, loc_loss, cls_loss, dh_loss, seg_loss,
            loc_preds, cls_preds, loc_targets, cls_targets, seg_targets,
            cls_sample_per_image, d_h_cls, d_cx_cls, seg_preds,
            image):

        # show inference image in tensorboard
        infer_img = recover_img(image.tensors[0])
        infer_img = infer_img.astype(np.uint8)
        h, w, _ = infer_img.shape

        self.writer.add_scalar('loss/loc_loss', loc_loss.sum().item(), self.iteration)
        self.writer.add_scalar('loss/cls_loss', cls_loss.sum().item(), self.iteration)
        self.writer.add_scalar('loss/dh_loss', dh_loss.sum().item(), self.iteration)
        self.writer.add_scalar('loss/seg_loss', seg_loss.sum().item(), self.iteration)
        self.writer.add_scalar('loss/total_loss', loss.sum().item(), self.iteration)
        self.writer.add_scalar('input_size', h, self.iteration)
        self.writer.add_scalar('learning_rate', self.cur_lr, self.iteration)

        if seg_targets is not None and seg_preds is not None:
            viz_seg_target = seg_targets[0]
            viz_seg_pred = seg_preds[0]
            predict_res = torch.sigmoid(viz_seg_pred).detach().cpu()
            gt_res = viz_seg_target.detach().cpu()
            gt_res = np.stack([gt_res, gt_res, gt_res])
            self.writer.add_image('seg_gt', gt_res, self.iteration)
            self.writer.add_image('seg_pred', predict_res, self.iteration)

        # 首先，画的一张图是 所有匹配到gt的anchor
        cls = cls_targets[0].data.cpu()
        anchor_img = infer_img.copy()
        anchor = loc_targets[0].data.cpu()
        img_anchor = self.encoder._gen_anchor_per_level(anchor_img.transpose([2, 0, 1]), self.cfg.ANCHOR_HEAD.STRIDE)
        img_anchor_pos = img_anchor.reshape(-1, 5)[cls > 0]
        img_anchor_pos = change_box_order(img_anchor_pos, "xyhd2quad")
        anchor_img = cv2.polylines(anchor_img, img_anchor_pos.numpy().astype(int).reshape(-1, 4, 2), True,
                                   (0, 255, 0), 4)
        self.writer.add_image('choose_anchor', anchor_img.transpose([2, 0, 1]), self.iteration)

        # 其次画的是anchor 对应到的gt的图片
        gt_img = infer_img.copy()
        d_cx, d_cy, d_h, d_dh = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        # d_dh[:] = 0
        pred = self.encoder.decode_per_level(gt_img.transpose([2, 0, 1]), self.cfg.ANCHOR_HEAD.STRIDE, d_cx, d_cy, d_h,
                                             d_dh, cls, thres=0.9)
        pred = torch.unique(pred, dim=0)
        for line in pred:
            gt_img = draw_four_vectors(gt_img, line, color=(0, 255, 0))
        self.writer.add_image('cover_gt', gt_img.transpose([2, 0, 1]), self.iteration)

        # 其次是对预测结果的一个反应：
        pred_img = infer_img.copy()
        loc_pred = loc_preds[0].cpu()
        cls_pred = cls_preds[0].softmax(1)[:, 1]

        ## 绘制解码的结果
        d_cx, d_cy, d_h, d_dh = loc_pred[:, 0], loc_pred[:, 1], loc_pred[:, 2], loc_pred[:, 3]
        # d_dh[:] = 0
        pred = self.encoder.decode_per_level(pred_img.transpose([2, 0, 1]), self.cfg.ANCHOR_HEAD.STRIDE, d_cx, d_cy,
                                             d_h, d_dh, cls_pred,
                                             thres=0.6)
        for line in pred:
            pred_img = draw_four_vectors(pred_img, line, color=(0, 255, 0))
        self.writer.add_image('image/pred', pred_img.transpose([2, 0, 1]), self.iteration)

        # 画出被预测为pos的anchor们
        anchor_img_pos = infer_img.copy()
        cls_pred_pos = cls_pred.cpu() > 0.5
        anchor_pos = change_box_order(img_anchor.reshape(-1, 5)[cls_pred_pos], 'xyhd2quad')
        anchor_img_pos = cv2.polylines(anchor_img_pos, anchor_pos.numpy().astype(int).reshape(-1, 4, 2), True,
                                       (0, 255, 0), 4)
        self.writer.add_image('image/pred_pos', anchor_img_pos.transpose([2, 0, 1]), self.iteration)

        # 画出 认为可以用于进行y的坐标回归的anchor 们
        anchor_img_dh_pos = infer_img.copy()
        dh_pred_pos = (d_h_cls[0] > 0.5)
        anchor_dh_pos = change_box_order(img_anchor.reshape(-1, 5)[dh_pred_pos], 'xyhd2quad')
        anchor_img_dh_pos = cv2.polylines(anchor_img_dh_pos, anchor_dh_pos.numpy().astype(int).reshape(-1, 4, 2), True,
                                          (0, 255, 0), 4)
        self.writer.add_image('image/pred_dh_pos', anchor_img_dh_pos.transpose([2, 0, 1]), self.iteration)

        # 画出认为可以用于进行o坐标回归的anchor们
        anchor_img_do_pos = infer_img.copy()
        do_pred_pos = d_cx_cls[0] > 0.5
        anchor_do_pos = change_box_order(img_anchor.reshape(-1, 5)[do_pred_pos], 'xyhd2quad')
        anchor_img_do_pos = cv2.polylines(anchor_img_do_pos, anchor_do_pos.numpy().astype(int).reshape(-1, 4, 2), True,
                                          (0, 255, 0), 4)
        self.writer.add_image('image/pred_do_pos', anchor_img_do_pos.transpose([2, 0, 1]), self.iteration)

        # 在这里画sample_per_image相关的东西：
        sampler_img = infer_img.copy()
        # 一个是挑选的样本的anchor示意图
        select_anchor_pos = cls_sample_per_image[0][1].squeeze(1)
        select_anchor_neg = cls_sample_per_image[0][0].squeeze(1)
        img_anchor = self.encoder._gen_anchor_per_level(gt_img.transpose([2, 0, 1]),
                                                        self.cfg.ANCHOR_HEAD.STRIDE).reshape(-1, 5)
        img_anchor = change_box_order(img_anchor, "xyhd2quad")
        img_anchor_pos_select = img_anchor[select_anchor_pos]
        img_anchor_neg_select = img_anchor[select_anchor_neg]
        sampler_anchor_img = cv2.polylines(sampler_img.copy(),
                                           img_anchor_pos_select.numpy().astype(int).reshape(-1, 4, 2), True,
                                           (0, 255, 0), 4)
        sampler_anchor_img = cv2.polylines(sampler_anchor_img,
                                           img_anchor_neg_select.numpy().astype(int).reshape(-1, 4, 2), True,
                                           (255, 0, 0), 4)
        self.writer.add_image('sampler_anchor_img', sampler_anchor_img.transpose([2, 0, 1]), self.iteration)
        # 一个是挑选样本的预测示意图
        cls_pred = cls_preds[0].softmax(1)[:, 1]
        cls_pred_pos = cls_pred[select_anchor_pos].cpu().data.numpy()
        cls_pred_neg = cls_pred[select_anchor_neg].cpu().data.numpy()

        sampler_anchor_pos2pos = cv2.polylines(sampler_img.copy(),
                                               img_anchor_pos_select.cpu().numpy().astype(int).reshape(-1, 4, 2)[
                                                   cls_pred_pos > 0.5], True,
                                               (255, 0, 0), 4)
        sampler_anchor_pos2neg = cv2.polylines(sampler_img.copy(),
                                               img_anchor_pos_select.cpu().numpy().astype(int).reshape(-1, 4, 2)[
                                                   cls_pred_pos <= 0.5], True,
                                               (255, 0, 0), 4)

        sampler_anchor_neg2pos = cv2.polylines(sampler_img.copy(),
                                               img_anchor_neg_select.cpu().numpy().astype(int).reshape(-1, 4, 2)[
                                                   cls_pred_neg > 0.5], True,
                                               (255, 0, 0), 4)
        sampler_anchor_neg2neg = cv2.polylines(sampler_img.copy(),
                                               img_anchor_neg_select.cpu().numpy().astype(int).reshape(-1, 4, 2)[
                                                   cls_pred_neg <= 0.5], True,
                                               (255, 0, 0), 4)

        self.writer.add_image('pos/pos2pos', sampler_anchor_pos2pos.transpose([2, 0, 1]), self.iteration)
        self.writer.add_image('pos/pos2neg', sampler_anchor_pos2neg.transpose([2, 0, 1]), self.iteration)
        self.writer.add_image('neg/neg2pos', sampler_anchor_neg2pos.transpose([2, 0, 1]), self.iteration)
        self.writer.add_image('neg/neg2neg', sampler_anchor_neg2neg.transpose([2, 0, 1]), self.iteration)


if __name__ == '__main__':
    trainer = Trainer(args)
    trainer.train()
