import torch
import argparse
import visdom

from dataset.voc_dataset import VOC_Dataset
from dataset.coco_dataset import CoCoDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model import YOLO_VGG_16
from loss import Yolo_Loss
import os
from torch.optim.lr_scheduler import StepLR

import time
from config import device
from anchor import make_center_anchors
from utils import decode, detect, resume

from train import train
from test import test

from option import opts

# from test_for_coco import test

def main():

    # 3. visdom
    vis = visdom.Visdom(port='8097')

    # 4. dataset
    if opts.dataset_type == 'coco':
        train_set = CoCoDataset(root=opts.data_root, set_name='valminusminival2014', split='TRAIN')
        test_set = CoCoDataset(root=opts.data_root, set_name='minival2014', split='TEST')
    elif opts.dataset_type == 'voc':
        train_set = VOC_Dataset(root=opts.data_root, split='TRAIN')
        test_set = VOC_Dataset(root=opts.data_root, split='TEST')

    # 5. dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=opts.batch_size,
                              collate_fn=train_set.collate_fn,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=opts.num_workers)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             collate_fn=test_set.collate_fn,
                             shuffle=False)

    # 6. model
    model = YOLO_VGG_16(num_classes=opts.num_classes).to(device)

    # 7. criterion
    criterion = Yolo_Loss(num_classes=opts.num_classes)

    # 8. optimizer
    optimizer = optim.SGD(params=model.parameters(),
                          lr=opts.lr,
                          momentum=0.9,
                          weight_decay=5e-4)

    # 9. scheduler
    scheduler = StepLR(optimizer=optimizer, step_size=150, gamma=0.1)

    # 10. resume
    resume(model, optimizer, scheduler, opts)

    # 11. train & test
    for epoch in range(opts.start_epoch, opts.epochs):

        # training
        train(epoch, vis, train_loader, model, criterion, optimizer, scheduler, opts)

        # testing
        test(epoch, vis, test_loader, model, criterion, opts)


if __name__ == '__main__':
    main()
