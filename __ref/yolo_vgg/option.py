import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--conf_thres', type=float, default=0.01)
parser.add_argument('--save_path', type=str, default='./saves')
parser.add_argument('--start_epoch', type=int, default=0)  # to resume
# FIXME
parser.add_argument('--save_file_name', type=str, default='yolo_v2_vgg_16_voc')
parser.add_argument('--num_classes', type=int, default=20, help='20 or 80')
parser.add_argument('--dataset_type', type=str, default='voc', help='which dataset you want to use')
parser.add_argument('--data_root', type=str, default="/home/cvmlserver3/Sungmin/data/VOC_ROOT")

# Paeng의 환경에서 사용 시 --paeng
parser.add_argument('--paeng', dest='paeng', action='store_true')
parser.set_defaults(paeng=False)

opts = parser.parse_args()

# FIXME
opts.save_file_name = 'yolo_v2_vgg_16_coco'
opts.num_classes = 80
opts.dataset_type = 'coco'
# opts.data_root = "D:\Data\VOC_ROOT"
# opts.data_root = "/home/cvmlserver3/Sungmin/data/VOC_ROOT"
if opts.paeng:
    opts.data_root = "/data/coco"
else:
    opts.data_root = "/home/cvmlserver3/Sungmin/data/COCO"

print(opts)