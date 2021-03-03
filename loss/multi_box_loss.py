import torch
import torch.nn as nn
from util.utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap
from option import device


class MultiBoxLoss(nn.Module):
    def __init__(self, coder):
        super(MultiBoxLoss, self).__init__()

        self.coder = coder
        self.priors_cxcy = self.coder.center_anchor
        self.priors_xy = cxcy_to_xy(self.priors_cxcy)

        # in paper https://arxiv.org/pdf/1512.02325.pdf, positive threshold is 0.5
        # and ratio of negative and positive 0.3 and alpha 1.0

        self.positive_threshold = 0.5
        self.neg_pos_ratio = 3
        self.alpha = 1.0

        # self.smooth_l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False, reduction='sum')

    def forward(self, pred, b_boxes, b_labels):
        """
        Forward propagation.
        :param pred (loc, cls) prediction tuple (N, 8732, 4) / (N, 8732, n + 1)
        :param b_boxes: true object boxes, a list of N tensors
        :param b_labels: true object labels, a list of N tensors (each tensor's size is number of objects)
        """

        pred_loc = pred[0]
        pred_cls = pred[1]

        batch_size = pred_loc.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = pred_cls.size(2)

        assert n_priors == pred_loc.size(1) == pred_cls.size(1)  # num of anchors

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)               # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.int64).to(device)               # (N, 8732)
        batch_postivie_default_box = torch.zeros((batch_size, n_priors), dtype=torch.bool).to(device)  # (N, 8732)

        # batch 에 따라 cls, loc 가 모두 다르니 batch 로 나누어 준다.
        for i in range(batch_size):
            boxes = b_boxes[i]
            labels = b_labels[i]

            # step1 ) -------------- get positive default box --------------
            # iou of maximum obj set 1.0 for satisfying both condition 1 and condition 2.
            iou = find_jaccard_overlap(boxes, self.priors_xy)   # [ num_obj, num_default_box ] -> [num_obj, 8732]

            # condition 1 - maximum iou
            _, obj_idx = iou.max(dim=1)  # [num_obj]

            # condition 2 - iou that higher than 0.5
            for obj in range(len(obj_idx)):
                iou[obj][obj_idx[obj]] = 1.

            positive_prior = iou >= self.positive_threshold

            # > 0.5 인 object 대비 anchor 가 하나라도 있다면, 그것은 positive 한 anchor
            positive_prior, _ = positive_prior.max(dim=0)
            batch_postivie_default_box[i] = positive_prior

            # step2 ) -------------- get cls label --------------
            _, max_prior_idx = iou.max(dim=0)

            # positive prior 로 back ground fore ground 나누는 부분
            positive_prior = positive_prior.type(torch.long)
            binary_positive_prior = torch.where(positive_prior == 0, torch.Tensor([-1]).to(device), torch.Tensor([1]).to(device))
            background_label = torch.full(size=[8732], fill_value=n_classes - 1, dtype=torch.float32).to(device)
            true_classes_ = torch.where(binary_positive_prior < 0, background_label, labels[max_prior_idx].type(torch.float32))
            true_classes[i] = true_classes_

            # step3 ) -------------- get loc label --------------
            true_locs_ = xy_to_cxcy(boxes[max_prior_idx])

            true_locs_ = self.coder.encode(true_locs_)
            true_locs[i] = true_locs_

        ###################################################
        # location loss
        ###################################################

        positive_priors = batch_postivie_default_box  # B, 8732 : for positive anchors, smooth l1
        n_positives = positive_priors.sum(dim=1)      # B       : each batches, num of positive sample (e.g) [2, 8]
        loc_loss = self.smooth_l1(pred_loc[positive_priors], true_locs[positive_priors])
        loc_loss = (self.alpha * loc_loss)

        ###################################################
        # classification loss
        ###################################################

        conf_loss_all = self.cross_entropy(pred_cls.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)  # positive bbox 를 위한 resize

        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))  # positive masking
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)  #  new allocated conf loss

        # hard negative mining
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        n_hard_negatives = self.neg_pos_ratio * n_positives  # make number of each batch of hard samples using ratio
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
        hard_negative_mask = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)

        # remains only top-k hard negative samples indices.
        conf_loss_hard_neg = conf_loss_neg[hard_negative_mask]  # it means a network knows zero is background
        conf_loss = conf_loss_hard_neg.sum() + conf_loss_pos.sum()   # pos + neg loss

        # TOTAL LOSS
        conf_loss = conf_loss / n_positives.sum()
        loc_loss = loc_loss / n_positives.sum()
        total_loss = (conf_loss + loc_loss)
        return total_loss, (loc_loss, conf_loss)


if __name__ == '__main__':

    from model.ssd_vgg_16 import SSD
    from model.vgg_16 import VGG
    vgg = VGG(pretrained=True).to(device)
    img = torch.FloatTensor(2, 3, 300, 300).to(device)

    ssd = SSD(vgg).to(device)
    img = torch.FloatTensor(2, 3, 300, 300).to(device)
    pred = ssd(img)

    print(pred[0].size())
    print(pred[1].size())

    gt = [torch.Tensor([[0.426, 0.158, 0.788, 0.997], [0.0585, 0.1597, 0.8947, 0.8213]]).to(device),
          torch.Tensor([[0.002, 0.090, 0.998, 0.867], [0.3094, 0.4396, 0.4260, 0.5440]]).to(device)]

    label = [torch.Tensor([14, 15]).to(device),
             torch.Tensor([12, 14]).to(device)]

    from coder.coder import SSD_Coder
    ssd_coder = SSD_Coder()
    loss = MultiBoxLoss(coder=ssd_coder)
    print(loss(pred, gt, label))