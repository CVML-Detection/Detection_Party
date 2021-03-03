import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from party import Party

from option import opts, device

from model.resnet import ResNet_50
from model.retinanet import RetinaNet
from coder.coder import Retina_Coder

from util.anchor import create_retina_anchors
from loss.focal_loss import Focal_Loss
from tqdm import tqdm


class RetinaNet_PARTY(Party):
    def __init__(self):
        # 1 : Dataset & Dataloader
        self.train_loader, self.test_loader = super().getDataLoader(
            opts.dataset_type, opts.dataset_root, opts.batch_size, opts.num_workers)
        # 2 : Model
        self.model = RetinaNet(base=ResNet_50(pretrained=True), n_classes=self.num_classes).to(device)
        
        # 3 : Coder
        self.coder = Retina_Coder(self.image_resize)

        # 7 : criterion (Loss)
        self.criterion = Focal_Loss(coder=self.coder)
        
        # 8 : Optimizer
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
            print('...Training from Scratch...');

    def training(self):
        print('== training RetinaNet ==')
        for epoch in range(self.epochs_start, self.epochs):
            super().train(epoch)
            super().test(epoch)


    def testing(self, epoch_num):
        print('== testing RetinaNet ==')
        # Testing for specific epoch num : 특정 에폭에 학습된 값 테스팅
        super().test(epoch = epoch_num)


if __name__ == "__main__":
    party_retinanet = RetinaNet_PARTY()