import torch
import torch.nn.functional as F
from torchvision.ops.boxes import nms as torchvision_nms
from config import device
import numpy as np
import os
import math

# for voc label
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

voc_label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
voc_label_map['background'] = 0
voc_rev_label_map = {v: k for k, v in voc_label_map.items()}  # Inverse mapping
np.random.seed(0)
voc_color_array = np.random.randint(256, size=(21, 3)) / 255

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
coco_color_array = np.random.randint(256, size=(81, 3)) / 255


def bar_custom(current, total, width=30):
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d byte]" % (current / total * 100, percent_bar, current, total)
    return progress


def cxcy_to_xy(cxcy):

    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=-1)


def xy_to_cxcy(xy):

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


def detect(pred, coder, min_score, n_classes, max_overlap=0.45, top_k=200, is_demo=False):
    """
    post processing of out of models
    batch 1 에 대한 prediction ([N, 8732, 4] ,[N, 8732, n + 1])을  pred boxes pred labels 와 pred scores 로 변환하는 함수
    :param pred (loc, cls) prediction tuple
    :param coder SSDcoder
    """
    pred_bbox, pred_scores = coder.post_processing(pred, is_demo)

    # Lists to store boxes and scores for this image
    image_boxes = list()
    image_labels = list()
    image_scores = list()

    # Check for each class
    for c in range(0, n_classes):

        class_scores = pred_scores[:, c]  # (8732)
        idx = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing

        # idx = class_scores > min_score                               # 0.01 for evaluation
        if idx.sum() == 0:
            continue

        class_scores = class_scores[idx]                                  # (n_qualified), n_min_score <= 845
        class_bboxes = pred_bbox[idx]                                     # (n_qualified, 4)

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
        image_labels.append(torch.LongTensor([n_classes]).to(device))  # background
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

    return image_boxes, image_labels, image_scores  # lists of length batch_size


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