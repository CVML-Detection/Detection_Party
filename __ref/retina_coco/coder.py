import numpy as np
import torch
from math import sqrt
from abc import ABCMeta, abstractmethod
from config import device
# from util.utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap
from collections import OrderedDict
from anchor import SSD_Anchor, RETINA_Anchor
import torch.nn.functional as F
from utils import cxcy_to_xy


class Coder(metaclass=ABCMeta):

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass


class SSD_Coder(Coder):
    def __init__(self, data_type):
        super().__init__()
        self.data_type = data_type
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
        return corner coord bbox and scores
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

        # yolo 에서 나온 bbox 는 center coord
        pred_bboxes = cxcy_to_xy(self.decode(pred_loc.squeeze()))  # for batch 1, [8732, 4]
        pred_scores = pred_cls.squeeze()                           # for batch 1, [8732, 80]

        return pred_bboxes, pred_scores


class RETINA_Coder(Coder):
    def __init__(self, data_type):
        super().__init__()
        self.data_type = data_type
        self.center_anchor = RETINA_Anchor('retina').create_anchors().to(device)

        assert data_type in ['voc', 'coco']
        if data_type == 'voc':
            self.num_classes = 20
        elif data_type == 'coco':
            self.num_classes = 80

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
        return corner coord bbox and scores
        """

        if is_demo:
            self.assign_anchors_to_cpu()
            pred_loc = pred[0].to('cpu')
            pred_cls = pred[1].to('cpu')
        else:
            pred_loc = pred[0]
            pred_cls = pred[1]

        n_priors = self.center_anchor.size(0)
        # pred_cls = F.softmax(pred_cls, dim=2)  # (1, 67995, n_classes)
        assert n_priors == pred_loc.size(1) == pred_cls.size(1)

        # decode 에서 나온 bbox 는 center coord
        pred_bboxes = cxcy_to_xy(self.decode(pred_loc.squeeze()))  # for batch 1, [67995, 4]
        pred_scores = pred_cls.squeeze()                           # for batch 1, [67995, num_classes]

        return pred_bboxes, pred_scores


if __name__ == '__main__':
    ssd_coder = RETINA_Coder()
    ssd_coder.assign_anchors_to_device()
    print(ssd_coder.center_anchor)