import torch
import visdom
import argparse
from dataset.voc_dataset import VOC_Dataset
from dataset.coco_dataset import COCO_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model import YOLO_VGG_16
from loss import Yolo_Loss
from torch.optim.lr_scheduler import StepLR
from coder import YOLO_Coder

from config import device, device_ids
from utils import resume

from train import train
from test import test

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def main():
    # 1. arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)  # 173
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='yolov2_vgg_16_voc')           # FIXME

    parser.add_argument('--conf_thres', type=float, default=0.01)
    parser.add_argument('--start_epoch', type=int, default=0)  # to resume

    parser.add_argument('--data_root', type=str, default='D:\data\\voc')
    # parser.add_argument('--data_root', type=str, default='D:\data\coco')
    # parser.add_argument('--data_root', type=str, default='/home/cvmlserver3/Sungmin/data/voc')
    # parser.add_argument('--data_root', type=str, default='/home/cvmlserver3/Sungmin/data/coco')          # FIXME

    parser.add_argument('--data_type', type=str, default='voc', help='choose voc or coco')              # FIXME
    parser.add_argument('--num_classes', type=int, default=20)

    opts = parser.parse_args()
    print(opts)

    # 3. visdom
    vis = visdom.Visdom()

    # 4. data set
    train_set = None
    test_set = None

    if opts.data_type == 'coco':
        train_set = COCO_Dataset(root=opts.data_root, set_name='train2017', split='train', download=True, resize=416)
        test_set = COCO_Dataset(root=opts.data_root, set_name='val2017', split='test', download=True, resize=416)

    elif opts.data_type == 'voc':
        train_set = VOC_Dataset(root=opts.data_root, split='train', download=True, resize=416)
        test_set = VOC_Dataset(root=opts.data_root, split='test', download=True, resize=416)

    # 5. dataloader
    train_loader = DataLoader(train_set,
                              batch_size=opts.batch_size,
                              collate_fn=train_set.collate_fn,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=1,
                             collate_fn=test_set.collate_fn,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)

    # 6. model
    model = YOLO_VGG_16(num_classes=opts.num_classes).to(device)
    model = torch.nn.DataParallel(module=model, device_ids=device_ids)
    yolo_coder = YOLO_Coder(data_type=opts.data_type)

    # 7. loss
    criterion = Yolo_Loss(yolo_coder)

    # 8. optimizer
    optimizer = optim.SGD(params=model.parameters(),
                          lr=opts.lr,
                          momentum=opts.momentum,
                          weight_decay=opts.weight_decay)

    # 9. scheduler
    scheduler = StepLR(optimizer=optimizer, step_size=150, gamma=0.1)

    # 10. resume
    resume(model, optimizer, scheduler, opts)

    # 11. train & test
    for epoch in range(opts.start_epoch, opts.epoch):

        # training
        train(epoch, vis, train_loader, model, criterion, optimizer, scheduler, opts)

        # testing
        if epoch % 10 == 0 or epoch >= 150:
            test(epoch, vis, test_loader, model, criterion, yolo_coder, opts)

        # scheduling
        scheduler.step()


if __name__ == '__main__':
    main()
