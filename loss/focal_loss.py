import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from util.utils import center_to_corner, corner_to_center, cxcy_to_gcxgcy, find_jaccard_overlap
from option import device


class Focal_Loss(nn.Module):
    def __init__(self, coder, threshold=0.5):
        super().__init__()

        self.coder = coder
        self.priors_cxcy = self.coder.center_anchor
        self.priors_xy = center_to_corner(self.priors_cxcy)

        self.bce_with_logit_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.threshold = threshold
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred, b_boxes, b_labels):
        """
        Forward propagation.
        :param pred_loc: predicted locations/boxes (N, 67995, 4)
        :param pred_cls: (N, 67995, 21)
        :param labels: true object labels, a list of N tensors
        """
        pred_loc = pred[0]
        pred_cls = pred[1]
        batch_size = pred_loc.size(0)
        n_priors = self.priors_xy.size(0) # anchor 개수 
        # pred_cls = torch.sigmoid(pred_cls)  # sigmoid for classification
        n_classes = pred_cls.size(2)

        assert n_priors == pred_loc.size(1) == pred_cls.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 67995, 4)
        true_classes = torch.ones((batch_size, n_priors, n_classes), dtype=torch.float).to(device)   # (N, 67995, 21)
        true_classes *= -1
        batch_postivie_default_box = torch.zeros((batch_size, n_priors), dtype=torch.bool).to(device)  # (N, 67995)

        cls_losses = []
        # batch 에 따라 cls, loc 가 모두 다르니 batch 로 나누어 준다.
        for i in range(batch_size):
            boxes = b_boxes[i]  # xy coord
            labels = b_labels[i]
            # ------------------------------------- my code -------------------------------------
            # step1 ) positive default box
            iou = find_jaccard_overlap(self.priors_xy, boxes)  # [anchors, objects]

            # condition 1 - maximum iou
            # https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/losses.py
            IoU_max, IoU_argmax = iou.max(dim=1)                             # [num_anchors]

            true_classes[i][torch.lt(IoU_max, 0.4), :] = 0    # make negative
            positive_indices = torch.ge(IoU_max, 0.5)         # iou 가 0.5 보다 큰 아이들 - [67995] # Positive

            batch_postivie_default_box[i] = positive_indices
            num_positive_anchors = positive_indices.sum()     # 갯수
            argmax_labels = labels[IoU_argmax]         # assigned_labels

            true_classes[i][positive_indices, :] = 0
            true_classes[i][positive_indices, argmax_labels[positive_indices].long()] = 1.  # objects
            targets = true_classes[i]
            # eg. only 70 positive indices is 1

            ##########################################
            # focal sigmoid loss --> https: // arxiv.org / pdf / 1708.02002.pdf
            ##########################################
            alpha = 0.25
            gamma = 2

            pred = pred_cls[i, :, :].sigmoid().clamp(1e-4, 1.0 - 1e-4)                    # sigmoid
            # alpha_factor = torch.ones(targets.shape).to(device) * alpha                   # container FIXME
            alpha_factor = (torch.ones(targets.shape)*alpha).to(device)
            a_t = torch.where((targets == 1), alpha_factor, 1. - alpha_factor)            # a_t
            p_t = torch.where(targets == 1, pred, 1 - pred)                               # p_t
            ce = -torch.log(p_t)                                                          # loss
            cls_loss = a_t * torch.pow(1 - p_t, gamma) * ce                               # focal loss

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(device))
            cls_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # step 3 ) loc label b
            # b = corner_to_center(boxes[max_prior_idx])
            true_locs_ = corner_to_center(boxes[IoU_argmax])
            # bbox
            true_locs_ = cxcy_to_gcxgcy(true_locs_, self.priors_cxcy)
            true_locs[i] = true_locs_

        # ------------------------------------------ loc loss ------------------------------------------
        # positive_priors = true_classes != 0  # (N, 8732)
        # print('equal : ? ', torch.equal(positive_priors, batch_postivie_default_box))  # 같은걸로 판명!
        positive_priors = batch_postivie_default_box  # B, 8732
        loc_loss = self.smooth_l1(pred_loc[positive_priors], true_locs[positive_priors])  # (), scalar
        # ------------------------------------------ cls loss ------------------------------------------

        conf_loss = torch.stack(cls_losses).mean()
        # TOTAL LOSS

        total_loss = (conf_loss + loc_loss)
        return total_loss, (loc_loss, conf_loss)
