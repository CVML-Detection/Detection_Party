import torch
import glob
import os
from PIL import Image
from torchvision import transforms as tfs
import matplotlib.pyplot as plt
from utils import detect, coco_label_map, coco_color_array, voc_color_array, voc_label_map
from matplotlib.patches import Rectangle
import time
import argparse
from config import device, device_ids
from model import YOLO_VGG_16
from coder import YOLO_Coder


def image_transforms(img):

    transform = tfs.Compose([tfs.Resize((416, 416)),
                             tfs.ToTensor(),
                             tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])])

    tf_img = transform(img)
    tf_img = tf_img.unsqueeze(0)  # make batch
    return tf_img


def visualize_results(images, bbox, cls, scores, label_map, color_array):
    label_array = list(label_map.keys())  # dict
    color_array = color_array

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

        # class and score
        plt.text(x=x1 - 5,
                 y=y1 - 5,
                 s=label_array[int(cls[i])] + ' {:.2f}'.format(scores[i]),
                 fontsize=10,
                 bbox=dict(facecolor=color_array[int(cls[i])],
                           alpha=0.5))

        # bounding box
        plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                      width=x2 - x1,
                                      height=y2 - y1,
                                      linewidth=1,
                                      edgecolor=color_array[int(cls[i])],
                                      facecolor='none'))
    plt.show()


def demo(opts, coder, model, img_type='jpg'):

    img_path_list = glob.glob(os.path.join(opts.img_path, '*' + '.' + img_type))
    total_time = 0
    with torch.no_grad():
        for idx, img_path in enumerate(img_path_list):

            # --------------------- img load ---------------------
            img = Image.open(img_path).convert('RGB')
            img = image_transforms(img).to(device)

            # --------------------- inference time ---------------------
            tic = time.time()
            pred = model(img)
            pred_boxes, pred_labels, pred_scores = detect(pred=pred,
                                                          coder=coder,
                                                          min_score=opts.conf_thres,
                                                          n_classes=opts.num_classes,
                                                          is_demo=True)

            toc = time.time()
            inference_time = toc - tic
            total_time += inference_time

            if idx % 100 == 0 or idx == len(img_path_list) - 1:

                # ------------------- check fps -------------------
                print('Step: [{}/{}]'.format(idx, len(img_path_list)))
                print("fps : {:.4f}".format((idx + 1) / total_time))

            if opts.data_type == 'voc':
                label_map = voc_label_map
                color_array = voc_color_array

            elif opts.data_type == 'coco':
                label_map = coco_label_map
                color_array = coco_color_array

            if opts.visualization:
                visualize_results(img, pred_boxes, pred_labels, pred_scores, label_map, color_array)


if __name__ == '__main__':

    # parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='yolov2_vgg_16_voc')
    parser.add_argument('--conf_thres', type=float, default=0.10)

    parser.add_argument('--img_path', type=str, default='D:\data\\voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages')
    # parser.add_argument('--img_path', type=str, default='D:\data\coco\images\\val2017')
    # parser.add_argument('--img_path', type=str, default='/home/cvmlserver3/Sungmin/data/voc/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages')
    # parser.add_argument('--img_path', type=str, default='/home/cvmlserver3/Sungmin/data/coco/images/val2017')

    parser.add_argument('--visualization', type=bool, default=True)
    parser.add_argument('--data_type', type=str, default='voc', help='choose voc or coco')
    parser.add_argument('--num_classes', type=int, default=20)

    demo_opts = parser.parse_args()
    print(demo_opts)

    # define model
    model = YOLO_VGG_16(num_classes=demo_opts.num_classes).to(device)
    model = torch.nn.DataParallel(module=model, device_ids=device_ids)
    model.eval()

    # load checkpoint
    check_point = torch.load(os.path.join(demo_opts.save_path, demo_opts.save_file_name) + '.{}.pth.tar'.
                             format(demo_opts.epoch),
                             map_location=device)
    model.load_state_dict(check_point['model_state_dict'], strict=True)

    # coder
    coder = YOLO_Coder(demo_opts.data_type)

    # demo
    demo(demo_opts, coder, model, 'jpg')
