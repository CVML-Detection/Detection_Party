import os
import time
import torch
from config import device
from utils import detect
import tempfile
import json
from voc_eval import voc_eval
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from anchor import make_center_anchors
from utils import voc_label_map, coco_label_map


def test(epoch, vis, test_loader, model, criterion, opts):
    # testing
    model.eval()
    check_point = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch))
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    tic = time.time()

    with torch.no_grad():

        # voc results lists
        det_img_name = []
        det_additional = []
        det_boxes = []
        det_labels = []
        det_scores = []

        # coco results lists
        img_ids = []
        results = []

        for idx, datas in enumerate(test_loader):

            images = datas[0]
            boxes = datas[1]
            labels = datas[2]

            # assign to cuda
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # feed forward
            preds = model(images)
            preds = preds.permute(0, 2, 3, 1)  # B, 13, 13, 425

            # loss
            loss, losses = criterion(preds, boxes, labels)
            pred_boxes, pred_labels, pred_scores = detect(pred=preds, min_score=opts.conf_thres,
                                                          center_anchors=model.center_anchors,
                                                          top_k=100, n_classes=opts.num_classes)

            # visualization
            # if opts.dataset_type == 'coco':
            #     label_map = coco_label_map
            # elif opts.dataset_type == 'voc':
            #     label_map = voc_label_map
            # visualize(images, pred_boxes, pred_labels, pred_scores, label_map)
            toc = time.time()

            if idx % 1000 == 0:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Test Time : {time:.4f}\t'
                      .format(epoch,
                              idx,
                              len(test_loader),
                              time=toc - tic))
            if opts.dataset_type == 'coco':

                # -------------------- coco evaluation ----------------------
                # 1. get img_id list
                img_id = test_loader.dataset.img_id[idx]
                img_ids.append(img_id)

                # 2. coco_results
                pred_boxes[:, 2] -= pred_boxes[:, 0]  # x2 to w
                pred_boxes[:, 3] -= pred_boxes[:, 1]  # y2 to h

                image_info = test_loader.dataset.coco.loadImgs(ids=img_id)[0]
                w = image_info['width']
                h = image_info['height']

                pred_boxes[:, 0] *= w
                pred_boxes[:, 2] *= w
                pred_boxes[:, 1] *= h
                pred_boxes[:, 3] *= h

                for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                    if int(pred_label) == opts.num_classes:
                        print('wrong label :', int(pred_label))
                        continue

                    coco_result = {
                        'image_id': img_id,
                        'category_id': test_loader.dataset.coco_ids[int(pred_label)],
                        'score': float(pred_score),
                        'bbox': pred_box.tolist(),
                    }
                    results.append(coco_result)

            elif opts.dataset_type == 'voc':
                # -------------------- voc evaluation ----------------------
                img_names = datas[3][0]
                additional_info = datas[4][0]

                det_img_name.append(img_names)  # 4952 len list # [1] - img_ name
                det_additional.append(additional_info)  # 4952 len list # [2] -  w, h
                det_boxes.append(pred_boxes.cpu())  # 4952 len list # [obj, 4]
                det_labels.append(pred_labels.cpu())  # 4952 len list # [obj]
                det_scores.append(pred_scores.cpu())  # 4952 len list # [obj]

    if opts.dataset_type == 'coco':

        _, tmp = tempfile.mkstemp()
        json.dump(results, open(tmp, "w"))

        cocoGt = test_loader.dataset.coco
        cocoDt = cocoGt.loadRes(tmp)
        # https://github.com/argusswift/YOLOv4-pytorch/blob/master/eval/cocoapi_evaluator.py
        # workaround: temporarily write data to json file because pycocotools can't process dict in py36.

        coco_eval = COCOeval(cocoGt=cocoGt, cocoDt=cocoDt, iouType='bbox')
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP = coco_eval.stats[0]
        mAP_50 = coco_eval.stats[1]

    elif opts.dataset_type == 'voc':

        # --- for VOC --- (158~160)
        test_root = os.path.join(opts.data_root, 'TEST', 'VOC2007', 'Annotations')
        mAP = voc_eval(test_root, det_img_name, det_additional, det_boxes, det_scores, det_labels)

    if vis is not None:
        # loss plot
        vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                 Y=torch.Tensor([loss, mAP]).unsqueeze(0).cpu(),
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
    import argparse
    from config import device
    from dataset.voc_dataset import VOC_Dataset
    from model import YOLO_VGG_16
    from loss import Yolo_Loss

    # 1. argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='yolo_v2_vgg_16_voc')
    parser.add_argument('--conf_thres', type=float, default=0.01)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--dataset_type', type=str, default='voc')
    parser.add_argument('--data_root', type=str, default='D:\Data\VOC_ROOT')

    test_opts = parser.parse_args()
    print(test_opts)

    epoch = test_opts.test_epoch

    # 2. device
    device = device

    # 3. visdom
    vis = None

    # 4. data set
    test_set = VOC_Dataset(root=test_opts.data_root, split='TEST')
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False)
    # 6. network
    model = YOLO_VGG_16().to(device)

    # 7. loss
    criterion = Yolo_Loss(num_classes=test_opts.num_classes)

    test(epoch=epoch,
         vis=vis,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         opts=test_opts)
