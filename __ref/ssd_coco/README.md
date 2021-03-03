# SSD coco detection

re-implementation of SSD detection for coco 

### Setting

- Python 3.7
- Numpy
- pytorch >= 1.5.0 

### training

- dataaset

train : trainval35k == train2017

test : minval2014 == val2017
or
test-dev 
```
voc 

batch 32  

60k : 1e-3
20k : 1e-4

32 * 60k = 1920000 / 21503 -> 89.28 epcoh -> 90 epoch
32 * 20k =  640000 / 21503 -> 29.76 epcoh -> 30 epoch

coco 
batch 32  
anchor 변경

160k : 1e-3
 40k : 1e-4
 40k : 1e-5

32 * 160k = 5120000  35158 / 32 -> 146.28 epcoh -> 150 epoch
32 *  40k =  640000 / 21503 ->  36.41 epcoh ->  40 epoch
32 *  40k =  640000 / 21503 ->  36.41 epcoh ->  40 epoch

iter_datasets = len(dataset) // args.batch_size
epoch_size = cfg['max_iter'] // iter_datasets

117266 / 32 = 3664
43 = 160000 / 3664

150, 50, 50
32 * 160k = 5120000 / 117266 / 32

```

learning rate decay 

0 ~ 119 : 1e-3     [120]

120 ~ 199 : 1e-4   [80]

### experiments

```
1. l1 loss + hard negative cls loss and 200 epoch 1e-3 : mAP 나오는 
fps : 40 정도 
```
```
2. rescaling initialization convert to xavier init
```
### Start Guide


