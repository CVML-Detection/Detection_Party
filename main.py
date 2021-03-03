from yolo_party import YOLO_PARTY
from ssd_party import SSD_PARTY
from retinanet_party import RetinaNet_PARTY
from option import opts


def main():
    if opts.model == 'yolov2_vgg_16':
        party_yolo = YOLO_PARTY()
        if opts.test:
            party_yolo.testing(epoch_num=opts.test_epoch)
        else:
            party_yolo.training()
        
    elif opts.model == 'ssd_vgg_16':
        party_ssd = SSD_PARTY()
        if opts.test:
            party_ssd.testing(epoch_num=opts.test_epoch)
        else:
            party_ssd.training()

    elif opts.model == 'retinanet_resnet_50':
        party_retinanet = RetinaNet_PARTY()
        if opts.test:
            party_retinanet.testing(epoch_num=opts.test_epoch)
        else:
            party_retinanet.training()
    else :
        print("option에서 '--model'을 확인해주세요")


if __name__ == '__main__':
    main()
