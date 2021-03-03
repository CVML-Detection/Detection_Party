import os
import time
import torch
from utils import detect
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from coder import YOLO_Coder
from evaluator import Evaluator
from config import device, device_ids


def test(epoch, vis, test_loader, model, criterion, coder, opts):

    # testing
    model.eval()
    check_point = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch),
                             map_location=device)
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict, strict=True)

    tic = time.time()
    sum_loss = 0

    is_coco = hasattr(test_loader.dataset, 'coco')  # if True the set is COCO else VOC
    if is_coco:
        print('COCO dataset evaluation...')
    else:
        print('VOC dataset evaluation...')

    evaluator = Evaluator(data_type=opts.data_type)

    with torch.no_grad():

        for idx, datas in enumerate(test_loader):

            images = datas[0]
            boxes = datas[1]
            labels = datas[2]

            # assign to cuda
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # feed forward
            pred = model(images)

            # loss
            loss, losses = criterion(pred, boxes, labels)
            sum_loss += loss.item()

            # eval
            pred_boxes, pred_labels, pred_scores = detect(pred=pred,
                                                          coder=coder,
                                                          min_score=opts.conf_thres,
                                                          n_classes=opts.num_classes)

            if opts.data_type == 'voc':
                img_name = datas[3][0]
                img_info = datas[4][0]
                info = (pred_boxes, pred_labels, pred_scores, img_name, img_info)

            elif opts.data_type == 'coco':
                img_id = test_loader.dataset.img_id[idx]
                img_info = test_loader.dataset.coco.loadImgs(ids=img_id)[0]
                coco_ids = test_loader.dataset.coco_ids
                info = (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids)

            evaluator.get_info(info)

            # visualization
            # if opts.dataset_type == 'coco':
            #     label_map = coco_label_map
            # elif opts.dataset_type == 'voc':
            #     label_map = voc_label_map
            # visualize(images, pred_boxes, pred_labels, pred_scores, label_map)

            toc = time.time()

            if idx % 1000 == 0 or idx == len(test_loader) - 1:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Test Time : {time:.4f}\t'
                      .format(epoch,
                              idx,
                              len(test_loader),
                              loss=loss,
                              time=toc - tic))

        mAP = evaluator.evaluate(test_loader.dataset)
        mean_loss = sum_loss / len(test_loader)

        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([mean_loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss',
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test loss',
                               legend=['test Loss', 'mAP']))


def visualize(images, bbox, cls, scores, label_dict):
    label_array = list(label_dict.keys())

    # 0. permute
    images = images.cpu()
    images = images.squeeze(0).permute(1, 2, 0)  # B, C, H, W --> H, W, C

    # 1. un normalization
    images *= torch.Tensor([0.229, 0.224, 0.225])
    images += torch.Tensor([0.485, 0.456, 0.406])

    # 2. RGB to BGR
    image_np = images.numpy()

    # 3. box scaling
    bbox *= 416

    plt.figure('result')
    plt.imshow(image_np)

    for i in range(len(bbox)):

        x1 = bbox[i][0]
        y1 = bbox[i][1]
        x2 = bbox[i][2]
        y2 = bbox[i][3]

        print(cls[i])

        plt.text(x=x1,
                 y=y1,
                 s=label_array[int(cls[i].item())] + str(scores[i].item()),
                 fontsize=10,
                 bbox=dict(facecolor='red', alpha=0.5))

        plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                      width=x2 - x1,
                                      height=y2 - y1,
                                      linewidth=1,
                                      edgecolor='r',
                                      facecolor='none'))
    plt.show()


if __name__ == '__main__':

    from dataset.voc_dataset import VOC_Dataset
    from dataset.coco_dataset import COCO_Dataset
    from loss import Yolo_Loss
    from model import YOLO_VGG_16
    import argparse

    # 1. argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='yolov2_vgg_16_coco')
    parser.add_argument('--conf_thres', type=float, default=0.01)

    # parser.add_argument('--data_root', type=str, default='D:\data\\voc')
    # parser.add_argument('--data_root', type=str, default='D:\data\coco')
    # parser.add_argument('--data_root', type=str, default='/home/cvmlserver3/Sungmin/data/voc')
    parser.add_argument('--data_root', type=str, default='/home/cvmlserver3/Sungmin/data/coco')

    parser.add_argument('--data_type', type=str, default='coco', help='choose voc or coco')
    parser.add_argument('--num_classes', type=int, default=80)

    test_opts = parser.parse_args()
    print(test_opts)

    epoch = test_opts.test_epoch

    # 2. device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = None

    # 4. data set
    if test_opts.data_type == 'voc':
        test_set = VOC_Dataset(root=test_opts.data_root, split='test')
        test_opts.n_classes = 20

    if test_opts.data_type == 'coco':
        test_set = COCO_Dataset(root=test_opts.data_root, set_name='val2017', split='test')
        test_opts.n_classes = 80

    # 5. data loader
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=0)

    # 6. network
    model = YOLO_VGG_16(num_classes=test_opts.num_classes).to(device)
    model = torch.nn.DataParallel(module=model, device_ids=device_ids)
    yolo_coder = YOLO_Coder(test_opts.data_type)

    # 7. loss
    criterion = Yolo_Loss(yolo_coder)

    test(epoch=epoch,
         vis=vis,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         coder=yolo_coder,
         opts=test_opts,
         )
