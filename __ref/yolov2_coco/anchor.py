import numpy as np
import torch
from math import sqrt
from abc import ABCMeta, abstractmethod
from config import device
from collections import OrderedDict


class Anchor(metaclass=ABCMeta):
    def __init__(self, model_name='yolo'):
        self.model_name = model_name.lower()
        assert model_name in ['yolo', 'ssd', 'retina']

    @abstractmethod
    def create_anchors(self):
        pass


class YOLO_Anchor(Anchor):

    def __init__(self):
        super().__init__()
        self.anchor_whs = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]

    def create_anchors(self):
        """
            Create the 835 prior (default) boxes for the YOLOv2
            :return: prior boxes in center-size coordinates, a tensor of dimensions (835, 4)
            """
        print('make yolo anchor')
        grid_size = 13
        grid_arange = np.arange(grid_size)
        xx, yy = np.meshgrid(grid_arange, grid_arange)  # + 0.5  # grid center, [fmsize*fmsize,2]
        m_grid = np.concatenate([np.expand_dims(xx, axis=-1), np.expand_dims(yy, -1)], axis=-1) + 0.5
        m_grid = m_grid
        xy = torch.from_numpy(m_grid)

        anchor_whs = np.array(self.anchor_whs)  # numpy 로 변경
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


class SSD_Anchor(Anchor):
    def create_anchors(self, data_type='voc'):
        """
            Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
            :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
            """
        assert data_type in ['voc', 'coco']
        print('make ssd anchor for {}'.format(data_type))

        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        fmap_dims = OrderedDict(sorted(fmap_dims.items(), key=lambda t: t[1], reverse=True))  # 내림차순

        # voc
        if data_type == 'voc':
            # voc
            obj_scales = {'conv4_3': 0.1,
                          'conv7': 0.2,
                          'conv8_2': 0.375,
                          'conv9_2': 0.55,
                          'conv10_2': 0.725,
                          'conv11_2': 0.9}

        elif data_type == 'coco':
            # coco
            obj_scales = {'conv4_3': 0.07,
                          'conv7': 0.15,
                          'conv8_2': 0.335,
                          'conv9_2': 0.525,
                          'conv10_2': 0.7125,
                          'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())
        center_anchors = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        center_anchors.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            center_anchors.append([cx, cy, additional_scale, additional_scale])

        center_anchors = torch.FloatTensor(center_anchors).to(device)  # (8732, 4)
        center_anchors.clamp_(0, 1)  # (8732, 4) 0 ~ 1
        return center_anchors

if __name__ == '__main__':
    yolo_anchor = YOLO_Anchor(model_name='yolo')
    ssd_anchor = SSD_Anchor(model_name='ssd')
    retina_anchor = RETINA_Anchor(model_name='retina')

    # print(yolo_anchor.create_anchors().size())
    # print(ssd_anchor.create_anchors().size())
    print(ssd_anchor.create_anchors(data_type='voc').size())
    # print(retina_anchor.create_anchors(img_size=600).size())


