import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import time
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
import visdom
import tempfile
import json

from dataset.coco_dataset import COCO_Dataset
from dataset.voc_dataset import VOC_Dataset
from util.utils import detect
# from util.utils import detect_objects
# from util.utils import detect_objects_retina
from util.voc_eval import voc_eval
from util.voc_eval_retina import voc_eval_retina

from option import opts, device
from tqdm import tqdm


class Party:
    vis = None                          # 3 : Use Visdom
    if opts.visdom:
        vis = visdom.Visdom(port=opts.visdom_port)
     
    num_classes = 0                     # 4, 5 : Dataset & Dataloader ( coco:80 voc:20 )
    train_loader = None
    test_loader = None
    if opts.model == 'yolov2_vgg_16':
        image_resize = 416
    elif opts.model == 'ssd_vgg_16':
        image_resize = 300
    elif opts.model == 'retinanet_resnet_50':
        image_resize = 600

    model = None                        # 6 : Model
    coder = None
    priors_cscy = None
    criterion = None                    # 7 : Criterion
    optimizer = None                    # 8 : Optimizer
    scheduler = None                    # 9 : Scheduler

    epochs_start = opts.epochs_start    # * : Training
    epochs = opts.epochs
    lr = opts.lr

    def getDataLoader(self, dataset_type, dataset_root, batch_size, num_workers):
        if dataset_type == 'coco':
            self.num_classes = 81
            train_set = COCO_Dataset(root=dataset_root, set_name='train2017', split='train', download=True, resize=self.image_resize)
            test_set = COCO_Dataset(root=dataset_root, set_name='minival2014', split='test', download=True, resize=self.image_resize)
        elif dataset_type == 'voc':
            self.num_classes = 21
            train_set = VOC_Dataset(root=dataset_root, split='train', download=True, resize=self.image_resize)
            test_set = VOC_Dataset(root=dataset_root, split='test', download=True, resize=self.image_resize)
        if opts.model == 'yolov2_vgg_16':
            self.num_classes = self.num_classes - 1

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  collate_fn=train_set.collate_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=num_workers)

        test_loader = DataLoader(dataset=test_set,
                                 batch_size=1,
                                 collate_fn=test_set.collate_fn,
                                 shuffle=False)

        return train_loader, test_loader

    def train(self, epoch):
        tic = time.time()
        self.model.train()

        # warm up training
        if opts.model == 'yolov2_vgg_16':
            if epoch < 5:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 1e-5
            elif epoch == 5:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 1e-4

        for idx, datas in enumerate(self.train_loader):
            images = datas[0]
            boxes = datas[1]
            labels = datas[2]
            # assign to cuda
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # FIXME ) 성능 테스트 용 RetinaNet
            if opts.model =='retinanet_resnet_50':
                loc, cls = self.model(images)
                loss, (loc, cls) = self.criterion(loc, cls, boxes, labels)
            # FIXME ) 원래 코드
            else:
                preds = self.model(images)
                loss, losses = self.criterion(preds, boxes, labels)

            '''
            if opts.model == 'yolov2_vgg_16':
                # feed forward
                preds = self.model(images)  # B, 425, 13, 13
                preds = preds.permute(0, 2, 3, 1)  # B, 13, 13, 425
                loss, losses = self.criterion(preds, boxes, labels)
            
            elif opts.model == 'ssd_vgg_16' or opts.model == 'retinanet_resnet_50':
                loc, cls = self.model(images)
                loss, (loc, cls) = self.criterion(loc, cls, boxes, labels)
            '''
            
            # backward and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            toc = time.time()

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']

            # for each steps (print for checking)
            if idx % 100 == 0:

                print('Epoch: [{0}]\t'
                       'Step: [{1}/{2}]\t'
                       'Loss: {loss:.4f}\t'
                       'Learning rate: {lr:.7f} s \t'
                       'Training Time : {time:.4f}\t'
                       .format(epoch, idx, len(self.train_loader),
                               loss=loss,
                               lr=lr,
                               time=toc - tic))

                # VISDOM
                if self.vis is not None:
                    self.vis.line(
                        X=torch.ones((1, 1)).cpu() * idx + epoch * self.train_loader.__len__(),  # step
                        Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                        win='train_loss',
                        update='append',
                        opts=dict(xlabel='step',
                                  ylabel='Loss',
                                  title='training loss',
                                  legend=['Total Loss']))

        # Save Checkpoint
        if not os.path.exists(opts.save_path):
            os.mkdir(opts.save_path)
            
        checkpoint = {'epoch': epoch,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict()}

        if self.scheduler is not None:
            checkpoint = {'epoch': epoch,
                          'model_state_dict': self.model.state_dict(),
                          'optimizer_state_dict': self.optimizer.state_dict(),
                          'scheduler_state_dict': self.scheduler.state_dict()}

        torch.save(checkpoint, os.path.join(opts.save_path, opts.save_filename) + '.{}.pth.tar'.format(epoch))

        if self.scheduler is not None:
            self.scheduler.step()

    def test(self, epoch):
        # @@@ Load Trained Model : 학습된 모델 불러오기 @@@
        print('Testing Start fo <', opts.model, '>  Using Dataset :<', opts.dataset_type, '>')
        self.model.eval()
        check_point = torch.load(os.path.join(opts.save_path, opts.save_filename) + '.{}.pth.tar'.format(epoch))
        state_dict = check_point['model_state_dict']
        self.model.load_state_dict(state_dict, strict=True)
        tic = time.time()

        # Evaluation Result 저장용 -> evaluator 만들기

        det_img_name = list()       # for VOC
        det_additional = list()
        det_boxes = list()
        det_labels = list()
        det_scores = list()

        img_ids = list()            # for COCO
        results = list()

        # @@@ Testing for test datasets @@@
        with torch.no_grad():

            for idx, datas in enumerate(self.test_loader):

                images = datas[0]
                boxes = datas[1]
                labels = datas[2]
                images = images.to(device)
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]

                # FIXME ) 성능 테스트 용 RetinaNet
                if opts.model =='retinanet_resnet_50':
                    loc, cls = self.model(images)
                    loss, (loc, cls) = self.criterion(loc, cls, boxes, labels)
                # FIXME ) 원래 코드
                else:
                    preds = self.model(images)
                    loss, _ = self.criterion(preds, boxes, labels)

                # # @@@@@ Get Loss : 로스 구하기 ! @@@@@
                # if opts.model == 'yolov2_vgg_16':
                #     # feed forward
                #     preds = self.model(images)
                #     preds = preds.permute(0, 2, 3, 1)  # B, 13, 13, 425
                #     loss, losses = self.criterion(preds, boxes, labels)
                #
                # elif opts.model == 'ssd_vgg_16' or opts.model == 'retinanet_resnet_50':
                #     # ---------- loss ----------
                #     predicted_locs, predicted_scores = self.model(images)
                #     loss, (loc, cls) = self.criterion(predicted_locs, predicted_scores, boxes, labels)
                    # loss = torch.zeros()

                # @@@@@ Get EVALUATION Results @@@@@
                if opts.test_eval:

                    pred_boxes, pred_labels, pred_scores = detect(pred=preds,
                                                                  coder=self.coder,
                                                                  min_score=opts.conf_thres,
                                                                  n_classes=self.num_classes)

                    # @@@ EVALUATION COCO @@@
                    if opts.dataset_type == 'coco':

                        # 1. get img_id list
                        img_id = self.test_loader.dataset.img_id[idx]
                        img_ids.append(img_id)

                        # 2. coco_results
                        pred_boxes[:, 2] -= pred_boxes[:, 0]  # x2 to w
                        pred_boxes[:, 3] -= pred_boxes[:, 1]  # y2 to h

                        image_info = self.test_loader.dataset.coco.loadImgs(ids=img_id)[0]
                        w = image_info['width']
                        h = image_info['height']

                        pred_boxes[:, 0] *= w
                        pred_boxes[:, 2] *= w
                        pred_boxes[:, 1] *= h
                        pred_boxes[:, 3] *= h

                        for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                            if int(pred_label) == self.num_classes - 1:
                                print('background label :', int(pred_label))
                                continue

                            coco_result = {
                                'image_id': img_id,
                                'category_id': self.test_loader.dataset.coco_ids[int(pred_label)],
                                'score': float(pred_score),
                                'bbox': pred_box.tolist(),
                            }
                            results.append(coco_result)

                    elif opts.dataset_type == 'voc':

                        img_names = datas[3][0]
                        additional_info = datas[4][0]

                        det_img_name.append(img_names)  # 4952 len list # [1] - img_ name
                        det_additional.append(additional_info)  # 4952 len list # [2] -  w, h

                        det_boxes.append(pred_boxes.cpu())  # 4952 len list # [obj, 4]
                        det_labels.append(pred_labels.cpu())  # 4952 len list # [obj]
                        det_scores.append(pred_scores.cpu())  # 4952 len list # [obj]

                toc = time.time()

                # @@@@@ Get Results @@@@@
                if idx % 10 == 0 or idx == len(self.test_loader) - 1:
                    print('Epoch: [{0}]\t'
                          'Step: [{1}/{2}]\t'
                          'Loss: {loss:.4f}\t'
                          'Time : {time:.4f}\t'
                          .format(epoch,
                                  idx, len(self.test_loader),
                                  loss=loss,
                                  time=toc-tic))

            # @@@ Evaluation Start @@@
            print('Start Evaluation...')
            if opts.dataset_type == 'coco':
                _, tmp = tempfile.mkstemp()
                json.dump(results, open(tmp, "w"))

                cocoGt = self.test_loader.dataset.coco
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
                test_root = os.path.join(opts.dataset_root, 'VOCtest_06-Nov-2007', 'VOCdevkit', 'VOC2007', 'Annotations')
                mAP = voc_eval(test_root, det_img_name, det_additional, det_boxes, det_scores, det_labels)

                    '''
                        # 2. coco_results
                        pred_boxes[:, 2] -= pred_boxes[:, 0]  # x2 to w
                        pred_boxes[:, 3] -= pred_boxes[:, 1]  # y2 to h

                        image_info = self.test_loader.dataset.coco.loadImgs(ids=img_id)[0]
                        w = image_info['width']
                        h = image_info['height']

                        pred_boxes[:, 0] *= w
                        pred_boxes[:, 2] *= w
                        pred_boxes[:, 1] *= h
                        pred_boxes[:, 3] *= h

                        for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                            if int(pred_label) == self.num_classes - 1:
                                print('background label :', int(pred_label))
                                continue

                            coco_result = {
                                'image_id': img_id,
                                'category_id': self.test_loader.dataset.coco_ids[int(pred_label)],
                                'score': float(pred_score),
                                'bbox': pred_box.tolist(),
                            }
                            results.append(coco_result)

                    elif opts.dataset_type == 'voc':

                        img_names = datas[3][0]
                        additional_info = datas[4][0]

                        det_img_name.append(img_names)  # 4952 len list # [1] - img_ name
                        det_additional.append(additional_info)  # 4952 len list # [2] -  w, h

                        # FIXME ) RetinaNet
                        if opts.model =='retinanet_resnet_50':
                            det_boxes.append(pred_boxes[0].cpu())  # 4952 len list # [obj, 4]
                            det_labels.append(pred_labels[0].cpu())  # 4952 len list # [obj]
                            det_scores.append(pred_scores[0].cpu())  # 4952 len list # [obj]
                        else:
                            det_boxes.append(pred_boxes.cpu())  # 4952 len list # [obj, 4]
                            det_labels.append(pred_labels.cpu())  # 4952 len list # [obj]
                            det_scores.append(pred_scores.cpu())  # 4952 len list # [obj]
                    '''
            # @@@ VISDOM @@@
            if self.vis is not None:
                # loss plot
                self.vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                              Y=torch.Tensor([loss, mAP]).unsqueeze(0).cpu(),
                              win='test_loss',
                              update='append',
                              opts=dict(xlabel='step',
                                        ylabel='test',
                                        title='test loss',
                                        legend=['test Loss', 'mAP']))

    def resume(self):
        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_filename)+'.{}.pth.tar'.format(opts.epochs_start-1))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('\n...Loaded checkpoint from epoch %d...\n' % (int(opts.epochs_start) - 1))