# RETINANET coco detection

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
whole epoch 14
9, 12 때 줄인다. 

총 90K 돈다.
   paper   |    epoch   |   iter   |  accum        |
0 ~ 60K    | (7329 * 8) |  58,632  |  58,632 (59k) | ~ 8 epoch
1e-2
60K ~ 80K  | (7329 * 3) |  21,987  |  80,619 (81K) | ~ 11 epoch
1e-3
80K ~ 90K  | (7329 * 2) |  14,685  |  95,304 (95K) | ~ 13 epoch
1e-4

batch : 16 (논문에서) 
iteration : 90K
batch * iteraion = 16 * 90,000 = 1,440,000 / 117266

117266 / 16 -> 7329 iteration (7K)
1 epoch 에 약 7K
10 epoch 은 약 70K 
20 epoch 은 약 140K

1  ~ 10 epoch : 1e-2 (70K)
11 ~ 15 epoch : 1e-3 (105K)
15 ~ 20 epoch : 1e-4 (140K)
```

### experiments

```
1. 
```
```
2. 
```
### Start Guide


