import time
import os
from utils import detect_objects
import torch
from voc_eval import voc_eval
from config import device
import tempfile
import json


def test(epoch, device, vis, test_loader, model, criterion, opts, priors_cxcy=None, eval=False):

    # ---------- load ----------
    print('Validation of epoch [{}]'.format(epoch))
    model.eval()
    check_point = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch))
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict)

    tic = time.time()
    sum_loss = 0

    is_coco = hasattr(test_loader.dataset, 'coco')  # if True the set is COCO else VOC
    if is_coco:
        print('COCO dataset evaluation...')
    else:
        print('VOC dataset evaluation...')

    # for VOC evaluation
    # Lists to store detected and true boxes, labels, scores of whole test data
    # test_dataset 에 대하여 다 넣는 list
    det_img_name = list()
    det_additional = list()
    det_boxes = list()
    det_labels = list()
    det_scores = list()

    # for COCO evaluation
    results = []
    image_ids = []

    with torch.no_grad():

        # for idx, (images, boxes, labels, difficulties, additional_info) in enumerate(test_loader):
        for idx, datas in enumerate(test_loader):
            images = datas[0]
            boxes = datas[1]
            labels = datas[2]

            # ---------- cuda ----------
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # ---------- loss ----------
            predicted_locs, predicted_scores = model(images)
            loss, (loc, cls) = criterion(predicted_locs, predicted_scores, boxes, labels)
            # loss = torch.zeros()

            sum_loss += loss.item()

            # ---------- eval ----------
            if eval:
                pred_boxes, pred_labels, pred_scores = detect_objects(priors_cxcy,
                                                                      predicted_locs,
                                                                      predicted_scores,
                                                                      min_score=opts.conf_thres,
                                                                      max_overlap=0.45,
                                                                      top_k=100)

                # --- for VOC --- (68 ~ 71)
                img_names = datas[3]                                            # at dataset 'test' split,
                img_names = img_names[0]                                        # img_name,
                det_img_name.append(img_names)                                  # 4952 len list # [1] - img_ name

                additional_info = datas[4]
                additional_info = additional_info[0]                             # img_width, img_height
                det_additional.append(additional_info)                           # 4952 len list # [2] -  w, h

                det_boxes.append(pred_boxes.cpu())                               # 4952 len list # [obj, 4]
                det_labels.append(pred_labels.cpu())                             # 4952 len list # [obj]
                det_scores.append(pred_scores.cpu())                             # 4952 len list # [obj]

            toc = time.time() - tic
            # ---------- print ----------
            # for each steps
            if idx % 1000 == 0:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Time : {time:.4f}\t'
                      .format(epoch,
                              idx, len(test_loader),
                              loss=loss,
                              time=toc))

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


if __name__ == "__main__":

    from dataset.voc_dataset import VOC_Dataset
    from loss import MultiBoxLoss
    from model import VGG, SSD
    from test import test
    from anchor_boxes import create_anchor_boxes
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='ssd_vgg_16_voc')
    parser.add_argument('--conf_thres', type=float, default=0.01)
    parser.add_argument('--data_root', type=str, default='D:\Data\VOC_ROOT')
    parser.add_argument('--data_type', type=str, default='voc', help='choose voc or coco')
    # "/home/cvmlserver3/Sungmin/data/VOC_ROOT"
    test_opts = parser.parse_args()
    print(test_opts)

    # 1. epoch
    epoch = 0

    # 2. device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = None

    # 4. data set
    if test_opts.data_type == 'voc':
        test_set = VOC_Dataset(root=test_opts.data_root, split='TEST')
        n_classes = 21

    # 5. data loader
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=4)
    # 6. network
    model = SSD(VGG(pretrained=True), n_classes=n_classes).to(device)
    priors_cxcy = create_anchor_boxes()  # cx, cy, w, h - [8732, 4]

    # 7. loss
    criterion = MultiBoxLoss(priors_cxcy=priors_cxcy)

    test(epoch=epoch,
         device=device,
         vis=vis,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         priors_cxcy=priors_cxcy,
         eval=True,
         opts=test_opts,
         )







