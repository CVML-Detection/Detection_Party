# SJ Detection Party 디텍션 잔치

## 1) 소문난 잔치 꾸리기 (기존 SSD & YOLO)

### - Training

- [x] SSD With COCO
- [x] SSD With VOC
- [x] YOLO With COCO
- [x] YOLO With VOC

### - Testing

- [ ] SSD With COCO => TODO! (진욱)
- [x] SSD With VOC
- [x] YOLO With COCO
- [x] YOLO With VOC

---

## 2) RetinaNet 파티에 초대

### - Todo (파티 준비물)

- [ ] 논문을 부숴보자
- [x] Training 코드 구현 with VOC
- [x] Testing 코드 구현 with VOC
- [ ] 성능 체크

- [ ] Training 코드 구현 with COCO
- [ ] Testing 코드 구현 with COCO
- [ ] 성능 체크

## 3) 파티 개최! (합치기)

### - TODO

- [x] 파티 Resume 코드
- [x] 이쁜 파티를 위한 tqdm

### 진욱 TODO

- [X] Dataset 코드 파티에 초대
- [X] COCO YOLO일때 80, SSD일때 81 -> 처리

- [ ] 기존 Demo 구현 가져오기
- [ ] Docker를 위한 이미지 저장 방법 구현

### 성민 TODO

- [X] coder를 파티에 초대 (anchor, ...)
- [X] party.py Model과 Loss의 파라미터 및 리턴값 통일
- [ ] utils 내 detect, detect_object, detect_objects_retina 통일
- [X] utils 내 voc_eval들 통일 (VOC Dataset)

## 4) 추가 파티 계획

- 다른 backbone 모델도 적용 되도록
  - ResNet
  - ResNext
- 다른 데이터셋 모델 개발

---

### - Update

- nn 에서 제공하는 Loss 들에 대해서 reduce=false -> reduction='none' (deprecated될 예정)
  https://discuss.pytorch.org/t/userwarning-size-average-and-reduce-args-will-be-deprecated-please-use-reduction-sum-instead/24629/2

- dataset download and visualization update. (데이터 없으면 자동으로 다운로드 되도록)

- requirements.txt update (초기 다운로드 pip install -r requirements.txt 로 진행)

```
convert script
    ./dataset/coco_dataset.py
    ./dataset/voc_dataset.py
    ./dataset/transform.py
    party.py's getDataLoader
```

- coder 와 anchor 를 통한 code update

```
anchor 가 detection 에서 쓰이는 부분은
1. loss 만들 때,
2. test or demo 시 model 의 output 을 실제 bbox 와 label 과 score 로 변경할 때 (decode)
3. coder 라는 class 를 통해서 전체적 framework 변경 필요
```


## 파티 참여 시

```
pip install -r requirements.txt

python3 main.py
```
