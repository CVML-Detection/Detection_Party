import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch.nn as nn
import torch.nn.functional as F
import torch
from option import device




class Yolo_Loss(nn.Module):
    def __init__(self, coder):
        super().__init__()

        self.coder = coder
        self.num_classes = self.coder.num_classes

    def forward(self, pred_targets, gt_boxes, gt_labels):
        """

        :param pred_targets: (B, 13, 13, 125)
        :param gt_boxes:     (B, 4)
        :param gt_labels:
        :return:
        """
        out_size = pred_targets.size(1)
        pred_targets = pred_targets.view(-1, out_size, out_size, 5, 5 + self.num_classes)
        pred_xy = pred_targets[..., :2].sigmoid()                  # sigmoid(tx ty)  0, 1
        pred_wh = pred_targets[..., 2:4].exp()                     # 2, 3
        pred_conf = pred_targets[..., 4].sigmoid()                 # 4
        pred_cls = pred_targets[..., 5:]                           # 20

        resp_mask, gt_xy, gt_wh, gt_conf, gt_cls = self.coder.encode(gt_boxes, gt_labels, pred_xy, pred_wh)

        # 1. xy sse
        # sse
        xy_loss = resp_mask.unsqueeze(-1).expand_as(gt_xy) * (gt_xy - pred_xy.cpu()) ** 2

        # 2. wh loss
        wh_loss = resp_mask.unsqueeze(-1).expand_as(gt_wh) * (torch.sqrt(gt_wh) - torch.sqrt(pred_wh.cpu())) ** 2

        # 3. conf loss
        conf_loss = resp_mask * (gt_conf - pred_conf.cpu()) ** 2

        # 4. no conf loss
        no_conf_loss = (1 - resp_mask).squeeze(-1) * (gt_conf - pred_conf.cpu()) ** 2

        # 5. classification loss
        pred_cls = F.softmax(pred_cls, dim=-1)  # [N*13*13*5,20]
        resp_cell = resp_mask.max(-1)[0].unsqueeze(-1).unsqueeze(-1).expand_as(gt_cls)  # [B, 13, 13, 5, 20]
        # cls_loss = resp_cell * (gt_cls - pred_cls.cpu()) ** 2       # original code
        cls_loss = resp_cell * (gt_cls * -1 * torch.log(pred_cls.cpu()))

        loss1 = 5 * xy_loss.sum()
        loss2 = 5 * wh_loss.sum()
        loss3 = 1 * conf_loss.sum()
        loss4 = 0.5 * no_conf_loss.sum()
        loss5 = 1 * cls_loss.sum()

        return loss1 + loss2 + loss3 + loss4 + loss5, (loss1, loss2, loss3, loss4, loss5)


if __name__ == '__main__':
    from model.yolo_vgg_16 import YOLO_VGG_16
    data_type = 'coco' # voc or coco

    if data_type=='coco':
        num_classes = 80
    elif data_type=='voc':
        num_classes = 20

    yolo = YOLO_VGG_16(num_classes=num_classes).to(device)

    image = torch.randn([2, 3, 416, 416]).to(device)
    pred = yolo(image)

    gt = [torch.Tensor([[0.426, 0.158, 0.788, 0.997], [0.0585, 0.1597, 0.8947, 0.8213]]).to(device),
          torch.Tensor([[0.002, 0.090, 0.998, 0.867], [0.3094, 0.4396, 0.4260, 0.5440]]).to(device)]

    label = [torch.Tensor([14, 15]).to(device),
             torch.Tensor([12, 14]).to(device)]

    from coder.coder import YOLO_Coder
    yolo_coder = YOLO_Coder(data_type=data_type)
    criterion = Yolo_Loss(yolo_coder)
    loss = criterion(pred, gt, label)
    print(loss)



