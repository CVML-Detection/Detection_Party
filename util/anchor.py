import numpy as np
import torch
from math import sqrt
from collections import OrderedDict

from option import device


def make_center_anchors(anchor_whs, grid_size=13):
    grid_arange = np.arange(grid_size)
    xx, yy = np.meshgrid(grid_arange, grid_arange)  # + 0.5  # grid center, [fmsize*fmsize,2]
    m_grid = np.concatenate([np.expand_dims(xx, axis=-1), np.expand_dims(yy, -1)], axis=-1) + 0.5
    m_grid = m_grid
    xy = torch.from_numpy(m_grid)

    anchor_whs = np.array(anchor_whs)  # numpy 로 변경
    wh = torch.from_numpy(anchor_whs)

    xy = xy.view(grid_size, grid_size, 1, 2).expand(grid_size, grid_size, 5, 2).type(torch.float32)  # centor
    wh = wh.view(1, 1, 5, 2).expand(grid_size, grid_size, 5, 2).type(torch.float32)  # w, h
    center_anchors = torch.cat([xy, wh], dim=3).to(device)
    # cy cx w h

    """
    center_anchors[0][0]
    tensor([[ 0.5000,  0.5000,  1.3221,  1.7314],
            [ 0.5000,  0.5000,  3.1927,  4.0094],
            [ 0.5000,  0.5000,  5.0559,  8.0989],
            [ 0.5000,  0.5000,  9.4711,  4.8405],
            [ 0.5000,  0.5000, 11.2364, 10.0071]], device='cuda:0')

    center_anchors[0][1]
    tensor([[ 1.5000,  0.5000,  1.3221,  1.7314],
            [ 1.5000,  0.5000,  3.1927,  4.0094],
            [ 1.5000,  0.5000,  5.0559,  8.0989],
            [ 1.5000,  0.5000,  9.4711,  4.8405],
            [ 1.5000,  0.5000, 11.2364, 10.0071]], device='cuda:0')

    center_anchors[1][0]
    tensor([[ 0.5000,  1.5000,  1.3221,  1.7314],
            [ 0.5000,  1.5000,  3.1927,  4.0094],
            [ 0.5000,  1.5000,  5.0559,  8.0989],
            [ 0.5000,  1.5000,  9.4711,  4.8405],
            [ 0.5000,  1.5000, 11.2364, 10.0071]], device='cuda:0')

    pytorch view has reverse index
    """

    return center_anchors


def create_anchor_boxes():
    """
    Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
    :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
    """
    fmap_dims = {'conv4_3': 38,
                 'conv7': 19,
                 'conv8_2': 10,
                 'conv9_2': 5,
                 'conv10_2': 3,
                 'conv11_2': 1}
    fmap_dims = OrderedDict(sorted(fmap_dims.items(), key=lambda t: t[1], reverse=True))  # 내림차순
    # value 를 기준으로 sorted 함!

    obj_scales = {'conv4_3': 0.1,
                  'conv7': 0.2,
                  'conv8_2': 0.375,
                  'conv9_2': 0.55,
                  'conv10_2': 0.725,
                  'conv11_2': 0.9}

    aspect_ratios = {'conv4_3': [1., 2., 0.5],
                     'conv7': [1., 2., 3., 0.5, .333],
                     'conv8_2': [1., 2., 3., 0.5, .333],
                     'conv9_2': [1., 2., 3., 0.5, .333],
                     'conv10_2': [1., 2., 0.5],
                     'conv11_2': [1., 2., 0.5]}

    fmaps = list(fmap_dims.keys())
    prior_boxes = []

    for k, fmap in enumerate(fmaps):            # conv4_3 = fmap
        for i in range(fmap_dims[fmap]):            # 38
            for j in range(fmap_dims[fmap]):        # 38   -> 38 x 38에 대하여
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                    # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                    # scale of the current feature map and the scale of the next feature map
                    if ratio == 1.:
                        try:
                            additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
    prior_boxes.clamp_(0, 1)  # (8732, 4)
    return prior_boxes


def create_retina_anchors(img_size=600):
    pyramid_levels = np.array([3, 4, 5, 6, 7])
    feature_maps = [(img_size + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]  # [75, 38, 19, 10, 5]
    areas = [32, 64, 128, 256, 512]
    aspect_ratios = np.array([0.5, 1, 2])
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    strides = [2 ** x for x in pyramid_levels]

    anchors = []

    for f_map, area, stride in zip(feature_maps, areas, strides):
        for i in range(f_map):
            for j in range(f_map):
                c_x = (j + 0.5) / f_map
                c_y = (i + 0.5) / f_map
                for aspect_ratio in aspect_ratios:
                    for scale in scales:
                        w = (area / img_size) * np.sqrt(aspect_ratio) * scale
                        h = (area / img_size) / np.sqrt(aspect_ratio) * scale

                        anchor = [c_x,
                                  c_y,
                                  w,
                                  h]  # shift 의 111 번 때문에!!

                        anchors.append(anchor)

    anchors = np.array(anchors).astype(np.float32)
    anchors = torch.FloatTensor(anchors).to(device)
    # anchors = torch.clamp(anchors, 0, 1)

    return anchors
    

if __name__ == "__main__":
    de_box = create_anchor_boxes()
    de_b = create_anchor_boxes()
    print(de_box.size())
    for db in de_box:
        print(db)

    print(torch.equal(de_box, de_b))