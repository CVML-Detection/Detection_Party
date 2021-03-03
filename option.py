import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='./saves')

# ================================== #
#           HyperParameter           #
# ================================== #
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--epochs_start', type=int, default=0)  # to resume
parser.add_argument('--epochs', type=int, default=300)

parser.add_argument('--conf_thres', type=float, default=0.01, help='min_score')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)

# ================================== #
#               Test                 #
# ================================== #
# --visdom for test mode
# parser.add_argument('--test', dest='test', action='store_true')
# parser.set_defaults(test=False)
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--test_epoch', type=int, default=0)
# Test 할 때 Evaluation 할 것인지.
parser.add_argument('--test_eval', dest='test_eval', action='store_true')
parser.set_defaults(test_eval=True)

# ================================== #
#              Dataset               #
# ================================== #
parser.add_argument('--dataset_type', type=str,
                    default='coco', help='coco or voc')

# ================================== #
#               Model                #
# ================================== #
parser.add_argument('--model', type=str,
                    default='yolov2_vgg_16', help={'ssd_vgg_16', 'yolov2_vgg_16' 'retinanet_resnet_50'})

# ================================== #
#              Visdom                #
# ================================== #
parser.add_argument('--visdom', type=bool, default=True)
parser.add_argument('--visdom_port', type=int, default=8097)

# ================================== #

# Paeng의 환경에서 사용 시 --paeng
parser.add_argument('--user', type=str, default='paeng', help={'paeng', 'sm', 'cvml3'})
opts = parser.parse_args()

# ================================== #
#           각자 설정할 값들             #
# ================================== #

# Paeng의 환경
if opts.user == 'paeng':
    opts.dataset_root = "/data/{}".format(opts.dataset_type)
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SungMin의 환경
elif opts.user == 'sm':
    opts.dataset_root = "D:\data\\{}".format(opts.dataset_type)
    opts.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

elif opts.user == 'cvml3':
    opts.dataset_root = "/home/cvmlserver3/Sungmin/data/{}".format(opts.dataset_type)
    opts.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# ================================== #
#            저장 파일 이름             #
# ================================== #

# ex) yolov2_vgg_16_coco
opts.save_filename = opts.model+'_'+opts.dataset_type
device = opts.device

print('=======================================\n{}\n======================================='.format(opts))