import torch
import torch.nn.functional as F
from torchvision.ops.boxes import nms as torchvision_nms
from anchor import make_center_anchors
from config import device
import numpy as np
import os

# for voc label
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

voc_label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
voc_label_map['background'] = 0
voc_rev_label_map = {v: k for k, v in voc_label_map.items()}  # Inverse mapping
np.random.seed(0)
voc_color_array = np.random.randint(256, size=(21, 3))

# for coco label
coco_labels = ('person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
               'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_label_map = {k: v + 1 for v, k in enumerate(coco_labels)}  # {1 ~ 80 : 'person' ~ 'toothbrush'}
coco_label_map['background'] = 0                                # {0 : 'background'}
coco_rev_label_map = {v: k for k, v in coco_label_map.items()}  # Inverse mapping
np.random.seed(1)
coco_color_array = np.random.randint(256, size=(81, 3))


def center_to_corner(cxcy):

    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=-1)


def corner_to_center(xy):

    cxcy = (xy[..., 2:] + xy[..., :2]) / 2
    wh = xy[..., 2:] - xy[..., :2]
    return torch.cat([cxcy, wh], dim=-1)


def find_jaccard_overlap(set_1, set_2, eps=1e-5):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection + eps  # (n1, n2)

    return intersection / union  # (n1, n2)


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)  # 0 혹은 양수로 만드는 부분
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)  # 둘다 양수인 부분만 존재하게됨!


def decode(preds, center_anchors, n_classes):
    '''
    pred to
    :param preds:
    :param centor_anchors:
    :return: pred_bbox : [B, 845, 4]
             pred_cls  : [B, 845, num_clsses]
             pred_conf : [B, 845]
    '''

    pred_targets = preds.view(-1, 13, 13, 5, 5 + n_classes)
    pred_xy = pred_targets[..., :2].sigmoid()  # sigmoid(tx ty)  0, 1
    pred_wh = pred_targets[..., 2:4].exp()  # 2, 3

    pred_conf = pred_targets[..., 4].sigmoid()  # 4
    pred_cls = pred_targets[..., 5:]  # 80

    # pred_bbox
    cxcy_anchors = center_anchors       # cxcy anchors 0~1

    anchors_xy = cxcy_anchors[..., :2]  # torch.Size([13, 13, 5, 2])
    anchors_wh = cxcy_anchors[..., 2:]  # torch.Size([13, 13, 5, 2])

    pred_bbox_xy = anchors_xy.floor().expand_as(pred_xy) + pred_xy  # torch.Size([B, 13, 13, 5, 2])  # floor() is very
    pred_bbox_wh = anchors_wh.expand_as(pred_wh) * pred_wh
    pred_bbox = torch.cat([pred_bbox_xy, pred_bbox_wh], dim=-1)  # torch.Size([B, 13, 13, 5, 4])
    pred_bbox = pred_bbox.view(-1, 13 * 13 * 5, 4) / 13.  # rescale 0~1   # [B, 845, 4]  # center_coord.
    pred_cls = F.softmax(pred_cls, dim=-1).view(-1, 13 * 13 * 5, n_classes)      # [B, 845, 80]
    pred_conf = pred_conf.view(-1, 13 * 13 * 5)                           # [B, 845]

    return pred_bbox, pred_cls, pred_conf


def detect(pred, min_score, center_anchors, top_k=100, n_classes=80):

    pred_bbox, pred_cls, pred_conf = decode(pred, center_anchors, n_classes)

    # Lists to store boxes and scores for this image
    image_boxes = list()
    image_labels = list()
    image_scores = list()

    # Check for each class
    for c in range(0, n_classes):

        class_scores = pred_cls[..., c]
        class_scores = class_scores * pred_conf

        idx = class_scores > min_score                               # 0.01 for evaluation
        if idx.sum() == 0:
            continue

        class_scores = class_scores[idx]                                  # (n_qualified), n_min_score <= 845
        class_bboxes = pred_bbox[idx]                                     # (n_qualified, 4)

        sorted_scores, idx_scores = class_scores.sort(descending=True)
        sorted_boxes = class_bboxes[idx_scores]
        sorted_boxes = center_to_corner(sorted_boxes).clamp(0, 1)

        num_boxes = len(sorted_boxes)
        keep_idx = torchvision_nms(boxes=sorted_boxes, scores=sorted_scores, iou_threshold=0.45)
        keep_ = torch.zeros(num_boxes, dtype=torch.bool)
        keep_[keep_idx] = 1  # int64 to bool
        keep = keep_

        image_boxes.append(sorted_boxes[keep])  # convert to corner coord ans scale 0~1
        image_labels.append(torch.LongTensor((keep).sum().item() * [c]).to(device))
        image_scores.append(sorted_scores[keep])

    if len(image_boxes) == 0:
        image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
        image_labels.append(torch.LongTensor([n_classes]).to(device))  # 임시방편!
        image_scores.append(torch.FloatTensor([0.]).to(device))

        # Concatenate into single tensors
    image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
    image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
    image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
    n_objects = image_scores.size(0)

    # Keep only the top k objects
    if n_objects > top_k:
        image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
        image_scores = image_scores[:top_k]  # (top_k)
        image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
        image_labels = image_labels[sort_ind][:top_k]  # (top_k)

    return image_boxes, image_labels, image_scores


def resume(model, optimizer, scheduler, opts):
    if opts.start_epoch != 0:
        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
                                .format(opts.start_epoch - 1))           # train
        model.load_state_dict(checkpoint['model_state_dict'])            # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    # load optim state dict
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])    # load sched state dict
        print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))
    else:
        print('\nNo check point to resume.. train from scratch.\n')