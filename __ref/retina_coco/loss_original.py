import torch
import torch.nn as nn
from utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap
from config import device
from model import RetinaNet, Resnet_50
import torch.nn.functional as F


class Focal_Loss(nn.Module):
    def __init__(self, coder):
        super().__init__()

        self.coder = coder
        self.priors_cxcy = self.coder.center_anchor
        self.priors_xy = cxcy_to_xy(self.priors_cxcy)
        self.num_classes = self.coder.num_classes
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, b_boxes, b_labels):
        """
        Forward propagation.
        :param pred (loc, cls) prediction tuple (N, 67995, 4) / (N, 67995, num_classes)
        :param labels: true object labels, a list of N tensors
        """
        pred_loc = pred[0]
        pred_cls = pred[1]

        batch_size = pred_loc.size(0)
        n_priors = self.priors_xy.size(0)

        assert n_priors == pred_loc.size(1) == pred_cls.size(1)  # 67995

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)                       # (N, 67995, 4)
        true_classes = -1 * torch.ones((batch_size, n_priors, self.num_classes), dtype=torch.float).to(device) # (N, 67995, num_classes)
        batch_postivie_default_box = torch.zeros((batch_size, n_priors), dtype=torch.bool).to(device)          # (N, 67995)

        cls_losses = []
        for i in range(batch_size):
            boxes = b_boxes[i]  # xy coord
            labels = b_labels[i]

            ########################################
            #           match strategies
            ########################################

            # ------------------------------------- my code -------------------------------------
            # step1 ) positive default box
            iou = find_jaccard_overlap(self.priors_xy, boxes)  # [67995, num_objects]

            # condition 1 - maximum iou
            # https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/losses.py
            IoU_max, IoU_argmax = iou.max(dim=1)               # [67995]

            # torch.lt(IoU_max, 0.4).size() [67995]  얘는 IoU_max 가 0.4 보다 작니? boolean 으로 출력해주는 함수
            true_classes[i][torch.lt(IoU_max, 0.4), :] = 0    # make negative
            positive_indices = torch.ge(IoU_max, 0.5)         # iou 가 0.5 보다 큰 아이들 - [67995]

            batch_postivie_default_box[i] = positive_indices
            num_positive_anchors = positive_indices.sum()     # 갯수
            argmax_labels = labels[IoU_argmax]         # assigned_labels

            # class 를 0 or 1 그리고 -1 로 assign 하는 부분
            true_classes[i][positive_indices, :] = 0
            true_classes[i][positive_indices, argmax_labels[positive_indices].long()] = 1.  # objects
            targets = true_classes[i]
            # eg. only 70 positive indices is 1

            ##########################################
            # focal sigmoid loss --> https: // arxiv.org / pdf / 1708.02002.pdf
            ##########################################
            alpha = 0.25
            gamma = 2

            pred = pred_cls[i, :, :].clamp(1e-4, 1.0 - 1e-4)                              # sigmoid
            alpha_factor = torch.ones(targets.shape).to(device) * alpha                   # container
            a_t = torch.where((targets == 1), alpha_factor, 1. - alpha_factor)            # a_t
            p_t = torch.where(targets == 1, pred, 1 - pred)                               # p_t
            ce = -torch.log(p_t)                                                          # loss
            cls_loss = a_t * torch.pow(1 - p_t, gamma) * ce                               # focal loss

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(device))
            cls_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # step 3 ) loc label b
            # b = xy_to_cxcy(boxes[max_prior_idx])
            true_locs_ = xy_to_cxcy(boxes[IoU_argmax])
            # bbox
            true_locs_ = self.coder.encode(true_locs_)
            true_locs[i] = true_locs_

        # ------------------------------------------ loc loss ------------------------------------------
        # positive_priors = true_classes != 0  # (N, 8732)
        # print('equal : ? ', torch.equal(positive_priors, batch_postivie_default_box))  # 같은걸로 판명!
        positive_priors = batch_postivie_default_box  # B, 8732
        loc_loss = self.smooth_l1(pred_loc[positive_priors], true_locs[positive_priors])  # (), scalar
        # ------------------------------------------ cls loss ------------------------------------------

        conf_loss = torch.stack(cls_losses).mean()
        loc_loss /= positive_priors.sum()
        # TOTAL LOSS

        total_loss = (conf_loss + loc_loss)
        return total_loss, (loc_loss, conf_loss)


if __name__ == '__main__':
    from anchor import RETINA_Anchor
    from coder import RETINA_Coder

    test_image = torch.randn([2, 3, 600, 600]).to(device)
    model = RetinaNet(base=Resnet_50()).to(device)
    reg, cls = model(test_image)
    print("cls' size() :", cls.size())
    print("reg's size() :", reg.size())

    gt = [torch.Tensor([[0.426, 0.158, 0.788, 0.997], [0.0585, 0.1597, 0.8947, 0.8213]]).to(device),
          torch.Tensor([[0.002, 0.090, 0.998, 0.867], [0.3094, 0.4396, 0.4260, 0.5440]]).to(device)]

    label = [torch.Tensor([14, 15]).to(device),
             torch.Tensor([12, 14]).to(device)]

    coder = RETINA_Coder(data_type='voc')
    loss = Focal_Loss(coder=coder)
    print(loss((reg, cls), gt, label))