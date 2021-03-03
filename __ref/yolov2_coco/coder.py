import numpy as np
import torch
from math import sqrt
from abc import ABCMeta, abstractmethod
from config import device
# from util.utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap
from collections import OrderedDict
from anchor import SSD_Anchor, YOLO_Anchor
import torch.nn.functional as F
from utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap


class Coder(metaclass=ABCMeta):

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass


class YOLO_Coder(Coder):

    def __init__(self, data_type):
        super().__init__()
        anchor = YOLO_Anchor()
        self.anchor_whs = anchor.anchor_whs
        self.center_anchor = anchor.create_anchors().to(device)                           # [13, 13, 5, 4]
        self.corner_anchor = cxcy_to_xy(self.center_anchor).view(13 * 13 * 5, 4)          # [845, 4]

        assert data_type in ['voc', 'coco']
        if data_type == 'voc':
            self.num_classes = 20
        elif data_type == 'coco':
            self.num_classes = 80

    def assign_anchors_to_device(self):
        self.center_anchor = self.center_anchor.to(device)

    def assign_anchors_to_cpu(self):
        self.center_anchor = self.center_anchor.to('cpu')

    def encode(self, gt_boxes, gt_labels, pred_xy, pred_wh):
        """
        gt 와 pred 가 들어왔을 때 loss 를 구할 수 있도록 변환해주는 함수
        :param gt_boxes:    (B, 4)
        :param gt_labels:   (B)
        :param pred_xy:     (B, 13, 13, 5, 2)
        :param pred_wh:     (B, 13, 13, 5, 2)

        :return: resp_mask :(B, 13, 13, 5) 그곳에 object 가 있는지 여부
                 gt_xy     :(B, 13, 13, 5, 2)
                 gt_wh     :(B, 13, 13, 5, 2)
                 gt_conf   :(B, 13, 13, 5)
                 gt_cls    :(B, 13, 13, 5, num_classes)
        """

        out_size = pred_xy.size(2)
        batch_size = pred_xy.size(0)
        resp_mask = torch.zeros([batch_size, out_size, out_size, 5])  # y, x, anchor, ~

        gt_xy = torch.zeros([batch_size, out_size, out_size, 5, 2])
        gt_wh = torch.zeros([batch_size, out_size, out_size, 5, 2])
        gt_conf = torch.zeros([batch_size, out_size, out_size, 5])
        gt_cls = torch.zeros([batch_size, out_size, out_size, 5, self.num_classes])

        # 1. make resp_mask
        for b in range(batch_size):

            label = gt_labels[b]
            corner_gt_box = gt_boxes[b]
            corner_gt_box_13 = corner_gt_box * float(out_size)

            center_gt_box = xy_to_cxcy(corner_gt_box)
            center_gt_box_13 = center_gt_box * float(out_size)

            bxby = center_gt_box_13[..., :2]  # [# obj, 2]
            x_y_ = bxby - bxby.floor()  # [# obj, 2], 0~1 scale
            bwbh = center_gt_box_13[..., 2:]

            iou_anchors_gt = find_jaccard_overlap(self.corner_anchor, corner_gt_box_13)  # [845, # obj]
            iou_anchors_gt = iou_anchors_gt.view(out_size, out_size, 5, -1)

            num_obj = corner_gt_box.size(0)

            for n_obj in range(num_obj):
                cx, cy = bxby[n_obj]
                cx = int(cx)
                cy = int(cy)

                _, max_idx = iou_anchors_gt[cy, cx, :, n_obj].max(0)  # which anchor has maximum iou?
                j = max_idx  # j is idx.
                # # j-th anchor
                resp_mask[b, cy, cx, j] = 1
                gt_xy[b, cy, cx, j, :] = x_y_[n_obj]
                w_h_ = bwbh[n_obj] / torch.FloatTensor(self.anchor_whs[j]).to(device)  # ratio
                gt_wh[b, cy, cx, j, :] = w_h_
                gt_cls[b, cy, cx, j, int(label[n_obj].item())] = 1

            pred_xy_ = pred_xy[b]
            pred_wh_ = pred_wh[b]

            center_pred_xy = self.center_anchor[..., :2].floor() + pred_xy_  # [845, 2] fix floor error
            center_pred_wh = self.center_anchor[..., 2:] * pred_wh_  # [845, 2]
            center_pred_bbox = torch.cat([center_pred_xy, center_pred_wh], dim=-1)
            corner_pred_bbox = cxcy_to_xy(center_pred_bbox).view(-1, 4)  # [845, 4]

            iou_pred_gt = find_jaccard_overlap(corner_pred_bbox, corner_gt_box_13)  # [845, # obj]
            iou_pred_gt = iou_pred_gt.view(out_size, out_size, 5, -1)

            gt_conf[b] = iou_pred_gt.max(-1)[0]  # each obj, maximum preds          # [13, 13, 5]

        return resp_mask, gt_xy, gt_wh, gt_conf, gt_cls

    def decode(self, pred):
        '''
        pred to
        :param preds:
        :param centor_anchors:
        :return: pred_bbox : [B, 845, 4]
                 pred_cls  : [B, 845, num_clsses]
                 pred_conf : [B, 845]
        '''

        pred_targets = pred.view(-1, 13, 13, 5, 5 + self.num_classes)
        pred_xy = pred_targets[..., :2].sigmoid()  # sigmoid(tx ty)  0, 1
        pred_wh = pred_targets[..., 2:4].exp()  # 2, 3

        pred_conf = pred_targets[..., 4].sigmoid()  # 4
        pred_cls = pred_targets[..., 5:]  # 80

        # pred_bbox
        cxcy_anchors = self.center_anchor  # cxcy anchors 0~1

        anchors_xy = cxcy_anchors[..., :2]  # torch.Size([13, 13, 5, 2])
        anchors_wh = cxcy_anchors[..., 2:]  # torch.Size([13, 13, 5, 2])

        pred_bbox_xy = anchors_xy.floor().expand_as(
            pred_xy) + pred_xy  # torch.Size([B, 13, 13, 5, 2])  # floor() is very
        pred_bbox_wh = anchors_wh.expand_as(pred_wh) * pred_wh
        pred_bbox = torch.cat([pred_bbox_xy, pred_bbox_wh], dim=-1)  # torch.Size([B, 13, 13, 5, 4])
        pred_bbox = pred_bbox.view(-1, 13 * 13 * 5, 4) / 13.  # rescale 0~1   # [B, 845, 4]  # center_coord.
        pred_cls = F.softmax(pred_cls, dim=-1).view(-1, 13 * 13 * 5, self.num_classes)  # [B, 845, 80]
        pred_conf = pred_conf.view(-1, 13 * 13 * 5)  # [B, 845]

        # [B, 845, 4]
        # [B, 845, 80]
        # [B, 845]

        return pred_bbox, pred_cls, pred_conf

    def post_processing(self, pred, is_demo=False):
        """
        yolo post processing for one batch
        return corner coord bbox and scores
        """

        if is_demo:
            self.assign_anchors_to_cpu()
            pred = pred.to('cpu')

        pred_bbox, pred_cls, pred_conf = self.decode(pred)

        # [1, 845, 4]
        # [1, 845, 80]
        # [1, 845]

        # yolo 에서 나온 bbox 는 center coord
        pred_bboxes = cxcy_to_xy(pred_bbox).squeeze()
        pred_scores = (pred_cls * pred_conf.unsqueeze(-1)).squeeze()

        return pred_bboxes, pred_scores


class SSD_Coder(Coder):
    def __init__(self, data_type):
        super().__init__()
        self.center_anchor = SSD_Anchor('ssd').create_anchors(data_type).to(device)

    def assign_anchors_to_device(self):
        self.center_anchor = self.center_anchor.to(device)

    def assign_anchors_to_cpu(self):
        self.center_anchor = self.center_anchor.to('cpu')

    def encode(self, cxcy):
        """
        for loss, gt(cxcy) to gcxcy
        """
        gcxcy = (cxcy[:, :2] - self.center_anchor[:, :2]) / self.center_anchor[:, 2:]
        gwh = torch.log(cxcy[:, 2:] / self.center_anchor[:, 2:])
        return torch.cat([gcxcy, gwh], dim=1)

    def decode(self, gcxgcy):
        """
        for test and demo, gcxcy to gt
        """
        cxcy = gcxgcy[:, :2] * self.center_anchor[:, 2:] + self.center_anchor[:, :2]
        wh = torch.exp(gcxgcy[:, 2:]) * self.center_anchor[:, 2:]
        return torch.cat([cxcy, wh], dim=1)

    def post_processing(self, pred, is_demo=False):
        """
        ssd post processing for one batch
        """

        if is_demo:
            self.assign_anchors_to_cpu()
            pred_loc = pred[0].to('cpu')
            pred_cls = pred[1].to('cpu')
        else:
            pred_loc = pred[0]
            pred_cls = pred[1]

        n_priors = self.center_anchor.size(0)
        pred_cls = F.softmax(pred_cls, dim=2)  # (8732, n_classes)
        assert n_priors == pred_loc.size(1) == pred_cls.size(1)

        pred_bboxes = cxcy_to_xy(self.decode(pred_loc.squeeze()))  # for batch 1, [8732, 4]
        pred_scores = pred_cls.squeeze()                           # for batch 1, [8732, 80]

        return pred_bboxes, pred_scores


if __name__ == '__main__':
    ssd_coder = SSD_Coder()
    ssd_coder.assign_anchors_to_device()
    print(ssd_coder.center_anchor)