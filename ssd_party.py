import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from party import Party
from model.ssd_vgg_16 import SSD, VGG
from loss.multi_box_loss import MultiBoxLoss
from coder.coder import SSD_Coder
from util.anchor import create_anchor_boxes

from option import opts
from option import device


class SSD_PARTY(Party):
    def __init__(self):
        # 1. Dataset & Dataloader
        self.train_loader, self.test_loader = super().getDataLoader(
            opts.dataset_type, opts.dataset_root, opts.batch_size, opts.num_workers)

        # 2. Model
        self.model = SSD(VGG(pretrained=True), n_classes=self.num_classes, loss_type='multi').to(device)

        # 3. Coder - if a model uses anchors
        self.coder = SSD_Coder(data_type=opts.dataset_type)

        # 4. criterion
        self.criterion = MultiBoxLoss(coder=self.coder)

        # 5. Optimizer
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=opts.lr,
                                   momentum=opts.momentum,
                                   weight_decay=opts.weight_decay)
        
        # 9 : Scheduler
        self.scheduler = MultiStepLR(optimizer=self.optimizer, milestones=[120, 150], gamma=0.1)  # 115, 144

        # 10 : Resume on Training
        if not opts.test and opts.epochs_start != 0:
            super().resume()
        else:
            print('...Training from Scratch...')

    def training(self):
        print('== training SSD ==')
        for epoch in range(self.epochs_start, self.epochs):
            super().train(epoch)
            super().test(epoch)

    def testing(self, epoch_num):
        print('== testing SSD ==')
        # Testing for specific epoch num : 특정 에폭에 학습된 값 테스팅
        super().test(epoch=epoch_num)
