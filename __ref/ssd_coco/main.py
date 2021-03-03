import torch
from dataset.voc_dataset import VOC_Dataset
from dataset.coco_dataset import COCO_Dataset
from loss import MultiBoxLoss
import visdom
import argparse
from train import train
from test import test
from torch.optim.lr_scheduler import MultiStepLR
from model import SSD, VGG
import os
from config import device, device_ids
from coder import SSD_Coder

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def main():
    # 1. arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)  # 173
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='ssd_vgg_16_coco')                         # FIXME

    parser.add_argument('--conf_thres', type=float, default=0.01)
    parser.add_argument('--start_epoch', type=int, default=0)

    # parser.add_argument('--data_root', type=str, default='D:\data\\voc')
    # parser.add_argument('--data_root', type=str, default='D:\data\coco')
    # parser.add_argument('--data_root', type=str, default='/home/cvmlserver3/Sungmin/data/voc')
    parser.add_argument('--data_root', type=str, default='/home/cvmlserver3/Sungmin/data/coco')          # FIXME

    parser.add_argument('--data_type', type=str, default='coco', help='choose voc or coco')              # FIXME
    parser.add_argument('--n_classes', type=int, default=81)

    opts = parser.parse_args()
    print(opts)

    # 2. device
    #device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = visdom.Visdom(port='8098')

    # 4. data set
    train_set = None
    test_set = None

    if opts.data_type == 'voc':
        train_set = VOC_Dataset(root=opts.data_root, split='train')
        test_set = VOC_Dataset(root=opts.data_root, split='test')
        opts.n_classes = 21

    elif opts.data_type == 'coco':
        train_set = COCO_Dataset(root=opts.data_root, set_name='train2017', split='train')
        test_set = COCO_Dataset(root=opts.data_root, set_name='val2017', split='test')
        opts.n_classes = 81

    # 5. data loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=opts.batch_size,
                                               collate_fn=train_set.collate_fn,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True)

    # 6. network
    model = SSD(VGG(pretrained=True), n_classes=opts.n_classes).to(device)
    model = torch.nn.DataParallel(module=model, device_ids=device_ids)
    ssd_coder = SSD_Coder(data_type=opts.data_type)  # 내부에 center_anchor 존재.

    # 7. loss
    criterion = MultiBoxLoss(coder=ssd_coder)

    # 8. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr,
                                momentum=opts.momentum,
                                weight_decay=opts.weight_decay)

    # 9. scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[150, 200], gamma=0.1)

    # 10. resume
    if opts.start_epoch != 0:

        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
                                .format(opts.start_epoch - 1), map_location=device)        # 하나 적은걸 가져와서 train
        model.load_state_dict(checkpoint['model_state_dict'])                              # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                      # load optim state dict
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])                      # load sched state dict
        print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))

    else:

        print('\nNo check point to resume.. train from scratch.\n')

    # for statement
    for epoch in range(opts.start_epoch, opts.epoch):

        # 11. train
        train(epoch=epoch,
              vis=vis,
              train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=scheduler,
              opts=opts)

        ## 12. test
        # test(epoch=epoch,
        #      vis=vis,
        #      test_loader=test_loader,
        #      model=model,
        #      criterion=criterion,
        #      opts=opts)

        # testing condition
        if epoch % 10 == 0 or epoch >= 150:

            # 12. test
            test(epoch=epoch,
                 vis=vis,
                 test_loader=test_loader,
                 model=model,
                 criterion=criterion,
                 coder=ssd_coder,
                 opts=opts)

        scheduler.step()


if __name__ == "__main__":
    main()



