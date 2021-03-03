import torch
import glob
import os
from PIL import Image
from torchvision import transforms as tfs
import matplotlib.pyplot as plt
from utils import detect, coco_label_map
from matplotlib.patches import Rectangle
from config import device
import argparse
from model import YOLO_VGG_16


def demo(opts, model=None, img_type='jpg'):
    """

    :param img_path:
    :param model:
    :param img_type:
    :return:
    """
    img_path_list = glob.glob(os.path.join(opts.img_path, '*' + '.' + img_type))
    with torch.no_grad():
        for img_path in img_path_list:
            print(img_path)
            img = Image.open(img_path).convert('RGB')
            img = image_transforms(img).to(device)
            preds = model(img)
            preds = preds.permute(0, 2, 3, 1)  # B, 13, 13, 425
            pred_boxes, pred_labels, pred_scores = detect(pred=preds, min_score=opts.conf_thres,
                                                          center_anchors=model.center_anchors,
                                                          top_k=100, n_classes=model.num_classes)
            visualize_results(img, pred_boxes, pred_labels, pred_scores, coco_label_map)


def image_transforms(img):

    transform = tfs.Compose([tfs.Resize((416, 416)),
                             tfs.ToTensor(),
                             tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])])

    tf_img = transform(img)
    tf_img = tf_img.unsqueeze(0)  # make batch
    return tf_img


def visualize_tensor_img(tensor_img):
    """
    tensor img 가 들어왔을 때, de-normalization 하고, permutation 해서 plt 로 출력하는 부분
    :param tensor_img:
    :return:
    """
    tensor_img = tensor_img.squeeze(0).permute(1, 2, 0)  # B, C, H, W --> H, W, C

    # 1. de-normalization
    tensor_img *= torch.Tensor([0.229, 0.224, 0.225])
    tensor_img += torch.Tensor([0.485, 0.456, 0.406])

    # 2. RGB to BGR
    numpy_img = tensor_img.numpy()

    plt.figure('result')
    plt.imshow(numpy_img)
    plt.show()


def visualize_results(images, bbox, cls, scores, label_dict):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=183)
    parser.add_argument('--img_path', type=str, default="D:\Data\coco\images\\val2014")
    # img_path = "/home/cvmlserver3/Sungmin/data/COCO/images/val2014"
    parser.add_argument('--conf_thres', type=float, default=0.35)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='yolo_v2_vgg_16_coco')
    demo_opts = parser.parse_args()
    epoch = 0

    model = YOLO_VGG_16(num_classes=80).to(device)
    model.eval()
    check_point = torch.load(os.path.join(demo_opts.save_path, demo_opts.save_file_name) + '.{}.pth.tar'.format(demo_opts.epoch))
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    demo(demo_opts, model, 'jpg')
