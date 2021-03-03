import os
import sys
import glob
import numpy as np
import zipfile
import platform
import wget
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dataset.transform import transform
from util.utils import bar_custom, coco_color_array
from option import device


def download_coco(root_dir='D:\data\\coco', remove_compressed_file=True):
    # for coco 2014
    coco_2014_train_url = 'http://images.cocodataset.org/zips/train2014.zip'
    coco_2014_val_url = 'http://images.cocodataset.org/zips/val2014.zip'
    # coco_2014_test_url = 'http://images.cocodataset.org/zips/test2014.zip'
    coco_2014_trainval_anno_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'

    # for coco 35k
    coco_2014_minival_anno_url = 'https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip'
    coco_2014_valminusminival_anno_url = 'https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip'

    # for coco 2017
    coco_2017_train_url = 'http://images.cocodataset.org/zips/train2017.zip'
    coco_2017_val_url = 'http://images.cocodataset.org/zips/val2017.zip'
    # coco_2017_test_url = 'http://images.cocodataset.org/zips/test2017.zip'
    coco_2017_trainval_anno_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    os.makedirs(root_dir, exist_ok=True)

    img_dir = os.path.join(root_dir, 'images')
    anno_dir = os.path.join(root_dir, 'annotations')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    """Download the COCO data if it doesn't exit in processed_folder already."""
    if (os.path.exists(os.path.join(img_dir, 'val2014')) and
            os.path.exists(os.path.join(img_dir, 'val2017')) and
            os.path.exists(os.path.join(img_dir, 'train2014')) and
            os.path.exists(os.path.join(img_dir, 'train2017'))):
        print("COCO Data Already exist!")
        return

    print("Download COCO Data...")

    # image download
    wget.download(url=coco_2014_train_url, out=img_dir, bar=bar_custom)
    print('')
    wget.download(url=coco_2014_val_url, out=img_dir, bar=bar_custom)
    print('')
    wget.download(url=coco_2017_train_url, out=img_dir, bar=bar_custom)
    print('')
    wget.download(url=coco_2017_val_url, out=img_dir, bar=bar_custom)
    print('')

    # annotation download
    wget.download(coco_2014_trainval_anno_url, out=root_dir, bar=bar_custom)
    print('')
    wget.download(coco_2014_minival_anno_url, out=root_dir, bar=bar_custom)
    print('')
    wget.download(coco_2014_valminusminival_anno_url, out=root_dir, bar=bar_custom)
    print('')
    wget.download(coco_2017_trainval_anno_url, out=root_dir, bar=bar_custom)
    print('')

    print("Extract...")

    # image extract
    with zipfile.ZipFile(os.path.join(img_dir, 'train2014.zip')) as unzip:
        unzip.extractall(os.path.join(img_dir))
    with zipfile.ZipFile(os.path.join(img_dir, 'val2014.zip')) as unzip:
        unzip.extractall(os.path.join(img_dir))
    with zipfile.ZipFile(os.path.join(img_dir, 'train2017.zip')) as unzip:
        unzip.extractall(os.path.join(img_dir))
    with zipfile.ZipFile(os.path.join(img_dir, 'val2017.zip')) as unzip:
        unzip.extractall(os.path.join(img_dir))

    # annotation extract
    with zipfile.ZipFile(os.path.join(root_dir, 'annotations_trainval2017.zip')) as unzip:
        unzip.extractall(os.path.join(root_dir))
    with zipfile.ZipFile(os.path.join(root_dir, 'instances_minival2014.json.zip')) as unzip:
        unzip.extractall(os.path.join(anno_dir))
    with zipfile.ZipFile(os.path.join(root_dir, 'instances_valminusminival2014.json.zip')) as unzip:
        unzip.extractall(os.path.join(anno_dir))
    with zipfile.ZipFile(os.path.join(root_dir, 'annotations_trainval2014.zip')) as unzip:
        unzip.extractall(os.path.join(root_dir))

    # remove zips
    if remove_compressed_file:
        root_zip_list = glob.glob(os.path.join(root_dir, '*.zip'))  # in root_dir remove *.zip
        for anno_zip in root_zip_list:
            os.remove(anno_zip)

        img_zip_list = glob.glob(os.path.join(img_dir, '*.zip'))   # in img_dir remove *.zip
        for img_zip in img_zip_list:
            os.remove(img_zip)
        print("Remove *.zips")

    print("Done!")


class COCO_Dataset(Dataset):
    def __init__(self, root='D:\Data', set_name='train2017', split='TRAIN', download=True, resize=300):
        super().__init__()

        self.root = root
        if platform.system() == 'Windows':
            matplotlib.use('TkAgg')  # for window
        # -------------------------- download --------------------------
        self.download = download
        if self.download:
            download_coco(root_dir=self.root)

        self.resize = resize
        self.set_name = set_name
        if set_name == 'minival2014' or set_name == 'valminusminival2014':
            self.set_name = 'val2014'
        self.split = split

        self.img_path = glob.glob(os.path.join(self.root, 'images', self.set_name, '*.jpg'))        # val2014 (*안씀*)
        self.coco = COCO(os.path.join(self.root, 'annotations', 'instances_' + set_name + '.json')) # TRAIN: valminusminival2014 (35,158개 이미지)
                                                                                                    # TEST : minival2014 (4,952개 이미지)
        self.img_id = list(self.coco.imgToAnns.keys())                                              # 이미지 ID List ([35,158]) 랜덤으로 (Segmentation만 있는거 제외)
        # self.ids = self.coco.getImgIds()                                          ###### *** 이걸로 가져오면 35,504개 ***

        self.coco_ids = sorted(self.coco.getCatIds())  # list of coco labels [1, ...11, 13, ... 90]  # 0 ~ 79 to 1 ~ 90     # 카테고리(label) get
        self.coco_ids_to_continuous_ids = {coco_id: i for i, coco_id in enumerate(self.coco_ids)}  # 1 ~ 90 to 0 ~ 79       # 카테고리 정렬
        # int to int

        # self.continuous_id_to_coco_id = {v: k for k, v in self.coco_id_to_continuous_id.items()}      # 1 ~ 80 to 1 ~ 90
        # int to int for ssd

        self.coco_ids_to_class_names = {category['id']: category['name'] for category in self.coco.loadCats(self.coco_ids)}     # len 80
        # int to string
        # {1 : 'person', 2: 'bicycle', ...}
        '''
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        '''

    def __getitem__(self, index):

        visualize = False

        # 1. image_id
        img_id = self.img_id[index]

        # 2. load image
        img_coco = self.coco.loadImgs(ids=img_id)[0]
        file_name = img_coco['file_name']
        img_width = img_coco['width']
        img_height = img_coco['height']
        file_path = os.path.join(self.root, 'images', self.set_name, file_name)
        # eg. 'D:\\Data\\coco\\images\\val2017\\000000289343.jpg'
        image = Image.open(file_path).convert('RGB')

        # 3. load anno
        anno_ids = self.coco.getAnnIds(imgIds=img_id)   # img id 에 해당하는 anno id 를 가져온다.
        anno = self.coco.loadAnns(ids=anno_ids)         # anno id 에 해당하는 annotation 을 가져온다.

        det_anno = self.make_det_annos(anno)            # anno -> [x1, y1, x2, y2, c] numpy 배열로 

        boxes = torch.FloatTensor(det_anno[:, :4])      # numpy to Tensor
        labels = torch.LongTensor(det_anno[:, 4])

        image, boxes, labels = transform(image, boxes, labels, self.split, new_size=self.resize)
        boxes = torch.clamp(boxes, 1e-3, 1 - 1e-3)      # boxes 값이 1e-3, 혹은 1-(1e-3)을 벗어날 경우, 각 값을 리턴 

        if visualize:

            # ----------------- visualization -----------------
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # tensor to img
            img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
            img_vis *= std
            img_vis += mean
            img_vis = np.clip(img_vis, 0, 1)

            plt.figure('input')
            plt.imshow(img_vis)

            for i in range(len(boxes)):
                x1 = boxes[i][0] * self.resize
                y1 = boxes[i][1] * self.resize
                x2 = boxes[i][2] * self.resize
                y2 = boxes[i][3] * self.resize

                print(boxes[i], ':', self.coco_ids_to_class_names[self.coco_ids[labels[i]]])

                # class
                plt.text(x=x1 - 5,
                         y=y1 - 5,
                         s=str(self.coco_ids_to_class_names[self.coco_ids[labels[i]]]),
                         bbox=dict(boxstyle='round4',
                                   facecolor=coco_color_array[labels[i]],
                                   alpha=0.9))

                # bounding box
                plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                              width=x2 - x1,
                                              height=y2 - y1,
                                              linewidth=1,
                                              edgecolor=coco_color_array[labels[i]],
                                              facecolor='none'))

            plt.show()

        return image, boxes, labels

    def make_det_annos(self, anno):

        annotations = np.zeros((0, 5))
        for idx, anno_dict in enumerate(anno):

            # anno_dict is annotations's dictionary
            # anno_dict is about a object.
            # some annotations have basically no width / height, skip them
            if anno_dict['bbox'][2] < 1 or anno_dict['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = anno_dict['bbox']
            # print(self.class_ids.index(anno_dict["category_id"]))
            # print(self.coco_id_to_continuous_id[anno_dict['category_id']])

            annotation[0, 4] = self.coco_ids_to_continuous_ids[anno_dict['category_id']]    # 원래 category_id가 18이면 들어가는 값은 16
            annotations = np.append(annotations, annotation, axis=0)                        # np.shape()

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, difficulties, img_name and
        additional_info
        """
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)
        return images, boxes, labels

    def __len__(self):
        return len(self.img_id)


if __name__ == '__main__':
    coco_dataset = COCO_Dataset(root="D:/data/coco", set_name='train2017', split='train', download=True, resize=300)
    train_loader = torch.utils.data.DataLoader(coco_dataset,
                                               batch_size=1,
                                               collate_fn=coco_dataset.collate_fn,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    for i, data in enumerate(train_loader):

        images = data[0]
        boxes = data[1]
        labels = data[2]

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        print(labels)
