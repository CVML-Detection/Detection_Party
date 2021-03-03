import torch
import torch.nn.functional as F
from torchvision.ops.boxes import nms as torchvision_nms
from option import device
import numpy as np
import math

# for voc label
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
voc_label_map = {k: v for v, k in enumerate(voc_labels)}
voc_label_map['background'] = 20
voc_rev_label_map = {v: k for k, v in voc_label_map.items()}  # Inverse mapping
np.random.seed(0)
voc_color_array = np.random.randint(256, size=(21, 3)) / 255  # In plt, rgb color space's range from 0 to 1

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

coco_label_map = {k: v for v, k in enumerate(coco_labels)}  # {0 ~ 79 : 'person' ~ 'toothbrush'}
coco_label_map['background'] = 80                                # {80 : 'background'}
coco_rev_label_map = {v: k for k, v in coco_label_map.items()}  # Inverse mapping
np.random.seed(1)
coco_color_array = np.random.randint(256, size=(81, 3)) / 255  # In plt, rgb color space's range from 0 to 1


def bar_custom(current, total, width=30):
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d byte]" % (current / total * 100, percent_bar, current, total)
    return progress


def center_to_corner(cxcy):
    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=-1)

def corner_to_center(xy):
    cxcy = (xy[..., 2:] + xy[..., :2]) / 2
    wh = xy[..., 2:] - xy[..., :2]
    return torch.cat([cxcy, wh], dim=-1)


def cxcy_to_xy(cxcy):
    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=1)


def xy_to_cxcy(xy):
    cxcy = (xy[..., 2:] + xy[..., :2]) / 2
    wh = xy[..., 2:] - xy[..., :2]
    return torch.cat([cxcy, wh], dim=1)


# 인코더 디코더에 들어갈 것
def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    gcxcy = (cxcy[:, :2] - priors_cxcy[:, :2]) / priors_cxcy[:, 2:]
    gwh = torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:])
    return torch.cat([gcxcy, gwh], dim=1)


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    cxcy = gcxgcy[:, :2] * priors_cxcy[:, 2:] + priors_cxcy[:, :2]
    wh = torch.exp(gcxgcy[:, 2:]) * priors_cxcy[:, 2:]
    return torch.cat([cxcy, wh], dim=1)


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


def nms(boxes, scores, iou_threshold=0.5, top_k=200):

    # 1. num obj
    num_boxes = len(boxes)

    # 2. get sorted scores, boxes
    sorted_scores, idx_scores = scores.sort(descending=True)
    sorted_boxes = boxes[idx_scores]

    # 3. iou
    iou = find_jaccard_overlap(sorted_boxes, sorted_boxes)
    keep = torch.ones(num_boxes, dtype=torch.bool)

    # 4. suppress boxes except max boxes
    for each_box_idx, iou_for_each_box in enumerate(iou):
        if keep[each_box_idx] == 0:  # 이미 없는것
            continue

        # 압축조건
        suppress = iou_for_each_box > iou_threshold  # 없앨 아이들
        keep[suppress] = 0
        keep[each_box_idx] = 1  # 자기자신은 살린당.

    return keep, sorted_scores, sorted_boxes


def detect(pred, coder, min_score, n_classes, max_overlap=0.45, top_k=200, is_demo=False):
    """
    post processing of out of models
    batch 1 에 대한 prediction ([N, 8732, 4] ,[N, 8732, n + 1])을  pred boxes pred labels 와 pred scores 로 변환하는 함수
    :param pred (loc, cls) prediction tuple
    :param coder SSDcoder
    """
    pred_bboxes, pred_scores = coder.post_processing(pred, is_demo)

    # Lists to store boxes and scores for this image
    image_boxes = list()
    image_labels = list()
    image_scores = list()

    # Check for each class
    for c in range(0, n_classes - 1):
        # Keep only predicted boxes and scores where scores for this class are above the minimum score
        class_scores = pred_scores[:, c]  # (8732)
        idx = class_scores > min_score    # torch.uint8 (byte) tensor, for indexing

        if idx.sum() == 0:
            continue

        class_scores = class_scores[idx]
        class_bboxes = pred_bboxes[idx]

        sorted_scores, idx_scores = class_scores.sort(descending=True)
        sorted_boxes = class_bboxes[idx_scores]
        sorted_boxes = sorted_boxes.clamp(0, 1)  # 0 ~ 1 로 scaling 해줌 --> 조금 오르려나? 78.30 --> 78.45 로 오름!

        # NMS
        num_boxes = len(sorted_boxes)
        keep_idx = torchvision_nms(boxes=sorted_boxes, scores=sorted_scores, iou_threshold=max_overlap)
        keep_ = torch.zeros(num_boxes, dtype=torch.bool)
        keep_[keep_idx] = 1  # int64 to bool
        keep = keep_

        # Store only unsuppressed boxes for this class
        image_boxes.append(sorted_boxes[keep])
        image_labels.append(torch.LongTensor((keep).sum().item() * [c]).to(device))
        image_scores.append(sorted_scores[keep])

    # If no object in any class is found, store a placeholder for 'background'
    if len(image_boxes) == 0:
        image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
        image_labels.append(torch.LongTensor([n_classes - 1]).to(device))  # background
        image_scores.append(torch.FloatTensor([0.]).to(device))

    # Concatenate into single tensors
    image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
    image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
    image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
    n_objects = image_scores.size(0)

    # remain top k objects
    if n_objects > top_k:
        image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
        image_scores = image_scores[:top_k]  # (top_k)
        image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
        image_labels = image_labels[sort_ind][:top_k]  # (top_k)

    return image_boxes, image_labels, image_scores  # lists of length batch_size

# def detect_objects(priors_cxcy, predicted_locs, predicted_scores, min_score, max_overlap, top_k, n_classes=21):
#     """
#     batch 1 에 대한 boxes 와 labels 와 scores 를 찾는 함수
#     :param priors_cxcy: [8732, 4]
#     :param predicted_locs: [1, 8732, 4]
#     :param predicted_scores: [1, 8732, 21]
#     :return:
#     after nms, remnant object is num_objects <= 200
#     image_boxes: [num_objects, 4]
#     image_labels:[num_objects]
#     image_scores:[num_objects]
#     """
#
#     batch_size = predicted_locs.size(0)
#     n_priors = priors_cxcy.size(0)
#     predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
#
#     assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
#
#     # Decode object coordinates from the form we regressed predicted boxes to
#     decoded_locs = center_to_corner(
#         gcxgcy_to_cxcy(predicted_locs[0], priors_cxcy))  # (8732, 4), these are fractional pt. coordinates
#
#     # Lists to store boxes and scores for this image
#     image_boxes = list()
#     image_labels = list()
#     image_scores = list()
#
#     # Check for each class
#     for c in range(n_classes - 1):
#         # Keep only predicted boxes and scores where scores for this class are above the minimum score
#         class_scores = predicted_scores[0][:, c]  # (8732)
#         score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
#         n_above_min_score = score_above_min_score.sum().item()
#         if n_above_min_score == 0:
#             continue
#         class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
#         class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)
#
#         sorted_scores, idx_scores = class_scores.sort(descending=True)
#         sorted_boxes = class_decoded_locs[idx_scores]
#         sorted_boxes = sorted_boxes.clamp(0, 1)  # 0 ~ 1 로 scaling 해줌 --> 조금 오르려나? 78.30 --> 78.45 로 오름!
#
#         num_boxes = len(sorted_boxes)
#         keep_idx = torchvision_nms(boxes=sorted_boxes, scores=sorted_scores, iou_threshold=max_overlap)
#         keep_ = torch.zeros(num_boxes, dtype=torch.bool)
#         keep_[keep_idx] = 1  # int64 to bool
#         keep = keep_
#
#         # Store only unsuppressed boxes for this class
#         image_boxes.append(sorted_boxes[keep])
#         image_labels.append(torch.LongTensor((keep).sum().item() * [c]).to(device))
#         image_scores.append(sorted_scores[keep])
#
#     # If no object in any class is found, store a placeholder for 'background'
#     if len(image_boxes) == 0:
#         image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
#         image_labels.append(torch.LongTensor([n_classes - 1]).to(device))  # background
#         image_scores.append(torch.FloatTensor([0.]).to(device))
#
#     # Concatenate into single tensors
#     image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
#     image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
#     image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
#     n_objects = image_scores.size(0)
#
#     # Keep only the top k objects --> 다구하고 200 개를 자르는 것은 느리지 않은가?
#     if n_objects > top_k:
#         image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
#         image_scores = image_scores[:top_k]  # (top_k)
#         image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
#         image_labels = image_labels[sort_ind][:top_k]  # (top_k)
#
#     return image_boxes, image_labels, image_scores  # lists of length batch_size
#
#
# def detect_objects_retina(priors_cxcy, predicted_locs, predicted_scores, min_score, max_overlap, top_k, n_classes=21):
#
#     batch_size = predicted_locs.size(0)
#     n_priors = priors_cxcy.size(0)
#     predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
#
#     # Lists to store final predicted boxes, labels, and scores for all images
#     batch_images_boxes = list()
#     batch_images_labels = list()
#     batch_images_scores = list()
#
#     assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
#
#     for b in range(batch_size):
#         # Decode object coordinates from the form we regressed predicted boxes to
#         decoded_locs = cxcy_to_xy(
#             gcxgcy_to_cxcy(predicted_locs[b], priors_cxcy))  # (8732, 4), these are fractional pt. coordinates
#
#         # Lists to store boxes and scores for this image
#         image_boxes = list()
#         image_labels = list()
#         image_scores = list()
#
#         # Check for each class
#         for c in range(1, n_classes):
#             # Keep only predicted boxes and scores where scores for this class are above the minimum score
#             class_scores = predicted_scores[b][:, c]  # (8732)
#             score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
#             n_above_min_score = score_above_min_score.sum().item()
#             if n_above_min_score == 0:
#                 continue
#             class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
#             class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)
#
#             sorted_scores, idx_scores = class_scores.sort(descending=True)
#             sorted_boxes = class_decoded_locs[idx_scores]
#             sorted_boxes = sorted_boxes.clamp(0, 1)  # 0 ~ 1 로 scaling 해줌 --> 조금 오르려나? 78.30 --> 78.45 로 오름!
#
#             num_boxes = len(sorted_boxes)
#             keep_idx = torchvision_nms(boxes=sorted_boxes, scores=sorted_scores, iou_threshold=max_overlap)
#             keep_ = torch.zeros(num_boxes, dtype=torch.bool)
#             keep_[keep_idx] = 1  # int64 to bool
#             keep = keep_
#
#             # Store only unsuppressed boxes for this class
#             image_boxes.append(sorted_boxes[keep])
#             image_labels.append(torch.LongTensor((keep).sum().item() * [c]).to(device))
#             image_scores.append(sorted_scores[keep])
#
#         # If no object in any class is found, store a placeholder for 'background'
#         if len(image_boxes) == 0:
#             image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
#             image_labels.append(torch.LongTensor([0]).to(device))
#             image_scores.append(torch.FloatTensor([0.]).to(device))
#
#         # Concatenate into single tensors
#         image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
#         image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
#         image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
#         n_objects = image_scores.size(0)
#
#         # Keep only the top k objects --> 다구하고 200 개를 자르는 것은 느리지 않은가?
#         if n_objects > top_k:
#             image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
#             image_scores = image_scores[:top_k]  # (top_k)
#             image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
#             image_labels = image_labels[sort_ind][:top_k]  # (top_k)
#
#         # Append to lists that store predicted boxes and scores for all images
#         batch_images_boxes.append(image_boxes)
#         batch_images_labels.append(image_labels)
#         batch_images_scores.append(image_scores)
#
#     return batch_images_boxes, batch_images_labels, batch_images_scores  # lists of length batch_size
#
#
# def decode(preds, center_anchors, n_classes):
#     '''
#     pred to
#     :param preds:
#     :param centor_anchors:
#     :return: pred_bbox : [B, 845, 4]
#              pred_cls  : [B, 845, num_clsses]
#              pred_conf : [B, 845]
#     '''
#
#     pred_targets = preds.view(-1, 13, 13, 5, 5 + n_classes)
#     pred_xy = pred_targets[..., :2].sigmoid()  # sigmoid(tx ty)  0, 1
#     pred_wh = pred_targets[..., 2:4].exp()  # 2, 3
#
#     pred_conf = pred_targets[..., 4].sigmoid()  # 4
#     pred_cls = pred_targets[..., 5:]  # 80
#
#     # pred_bbox
#     cxcy_anchors = center_anchors       # cxcy anchors 0~1
#
#     anchors_xy = cxcy_anchors[..., :2]  # torch.Size([13, 13, 5, 2])
#     anchors_wh = cxcy_anchors[..., 2:]  # torch.Size([13, 13, 5, 2])
#
#     pred_bbox_xy = anchors_xy.floor().expand_as(pred_xy) + pred_xy  # torch.Size([B, 13, 13, 5, 2])  # floor() is very
#     pred_bbox_wh = anchors_wh.expand_as(pred_wh) * pred_wh
#     pred_bbox = torch.cat([pred_bbox_xy, pred_bbox_wh], dim=-1)  # torch.Size([B, 13, 13, 5, 4])
#     pred_bbox = pred_bbox.view(-1, 13 * 13 * 5, 4) / 13.  # rescale 0~1   # [B, 845, 4]  # center_coord.
#     pred_cls = F.softmax(pred_cls, dim=-1).view(-1, 13 * 13 * 5, n_classes)      # [B, 845, 80]
#     pred_conf = pred_conf.view(-1, 13 * 13 * 5)                           # [B, 845]
#
#     return pred_bbox, pred_cls, pred_conf
#
#
# def detect(pred, min_score, center_anchors, top_k=100, n_classes=80):
#
#     pred_bbox, pred_cls, pred_conf = decode(pred, center_anchors, n_classes)
#
#     # Lists to store boxes and scores for this image
#     image_boxes = list()
#     image_labels = list()
#     image_scores = list()
#
#     # Check for each class
#     for c in range(0, n_classes):
#
#         class_scores = pred_cls[..., c]
#         class_scores = class_scores * pred_conf
#
#         idx = class_scores > min_score                               # 0.01 for evaluation
#         if idx.sum() == 0:
#             continue
#
#         class_scores = class_scores[idx]                                  # (n_qualified), n_min_score <= 845
#         class_bboxes = pred_bbox[idx]                                     # (n_qualified, 4)
#
#         sorted_scores, idx_scores = class_scores.sort(descending=True)
#         sorted_boxes = class_bboxes[idx_scores]
#         sorted_boxes = center_to_corner(sorted_boxes).clamp(0, 1)
#
#         num_boxes = len(sorted_boxes)
#         keep_idx = torchvision_nms(boxes=sorted_boxes, scores=sorted_scores, iou_threshold=0.45)
#         keep_ = torch.zeros(num_boxes, dtype=torch.bool)
#         keep_[keep_idx] = 1  # int64 to bool
#         keep = keep_
#
#         image_boxes.append(sorted_boxes[keep])  # convert to corner coord ans scale 0~1
#         image_labels.append(torch.LongTensor((keep).sum().item() * [c]).to(device))
#         image_scores.append(sorted_scores[keep])
#
#     if len(image_boxes) == 0:
#         image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
#         image_labels.append(torch.LongTensor([n_classes]).to(device))  # 임시방편!
#         image_scores.append(torch.FloatTensor([0.]).to(device))
#
#         # Concatenate into single tensors
#     image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
#     image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
#     image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
#     n_objects = image_scores.size(0)
#
#     # Keep only the top k objects
#     if n_objects > top_k:
#         image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
#         image_scores = image_scores[:top_k]  # (top_k)
#         image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
#         image_labels = image_labels[sort_ind][:top_k]  # (top_k)
#
#     return image_boxes, image_labels, image_scores
#
