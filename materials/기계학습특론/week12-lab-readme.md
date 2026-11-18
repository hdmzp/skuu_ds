# 12주차 실습 문제 — 딥러닝 분할 (건축 CAD 도면 요소 분할)

강의 내용(인코더-디코더, 전치·팽창 합성곱, FCN/U-Net/DeepLab, SegFormer, SAM,
Dice·IoU·Focal 손실)을 `segmentation_example_pytorch/` 코드에 직접 채워 넣는 실습이다.
빈칸은 모두 `# TODO(n)` 과 `raise NotImplementedError("TODO(n)")` 로 표시해 두었다.
**TODO 를 지우고 그 자리에 코드를 쓰면** 아래·위의 코드가 그대로 이어져 돌아간다.

## 1. 환경

```bash
cd segmentation_example_pytorch
pip install -r requirements.txt
```

| 패키지 | 버전 | 확인된 조합 |
|---|---|---|
| torch | ≥ 2.2 | 2.14.0 |
| segmentation-models-pytorch | ≥ 0.4 | 0.5.0 |
| torchmetrics | ≥ 1.3 | 1.9.0 |
| albumentations | ≥ 1.4 | 2.0.8 |
| transformers | ≥ 4.44 | SAM 실습에만 필요 |
| labelme, imgviz | — | `labelme2voc.py` 에만 필요 |

GPU 가 없어도 `--size 128 --batch 2 --epochs 2` 정도로 CPU 에서 동작을 확인할 수 있다
(AMP 는 CUDA 에서만 켜진다).

> albumentations 2.x 는 `ShiftScaleRotate` 에 «Affine 을 쓰라» 는 경고를 띄우지만 동작에는
> 문제가 없다. 경고가 거슬리면 `A.Affine(translate_percent=0.05, scale=(0.85,1.15), rotate=(-10,10), p=0.5)`
> 로 바꿔도 된다.

## 2. 데이터

용량 문제로 이미지·마스크는 포함하지 않았다. **기존 실습 zip 의 `data_annotated2/` 폴더를
`segmentation_example_pytorch/` 아래에 그대로 복사해서 쓴다.**

```
data_annotated2/
  JPEGImages/*.jpg              원본 도면
  SegmentationClassPNG3/*.png   클래스 인덱스 마스크 (0 = _background_, 1..7)
```

라벨 파일은 그대로 들어 있다.

```
labels.txt / class_names.txt : __ignore__, _background_, ow, iw, of, cw, ef, st, oo
```

`__ignore__` 를 뺀 **8클래스**(배경 1 + 도면 기호 7종)가 모델의 출력 채널 수다.
기존 Keras 코드의 `NUM_CLASSES = 7` 은 마지막 클래스 `oo` 가 빠진 값이니 따라 쓰지 말자.

labelme JSON 에서 새로 마스크를 만들려면(변환 스크립트는 완성본 그대로다):

```bash
python labelme2voc.py cad_img_v3 data_annotated2 --labels labels.txt
python -c "from dataset import npy_to_index_png; npy_to_index_png('data_annotated2/SegmentationClass','data_annotated2/SegmentationClassPNG3')"
```

## 3. 실행

```bash
python train_seg.py --data data_annotated2 --arch deeplabv3plus --encoder resnet50 --epochs 30
python train_seg.py --data data_annotated2 --arch segformer     --encoder mit_b2   --lr 6e-5
python train_seg.py --data data_annotated2 --arch unet          --encoder resnet34 --ignore_bg_in_miou
python predict.py --ckpt runs/<run>/best.pt --images data_annotated2/JPEGImages --out preds
python train_sam_pseudolabel.py --json_dir cad_img_v3 --images data_annotated2/JPEGImages \
       --labels labels.txt --out data_annotated2/SegmentationClassSAM
```

## 4. TODO 목록 (총 14개 / 100점)

### `dataset.py` — 3개 (20점)

| # | 무엇을 | 배점 |
|---|---|---|
| TODO(1) | `build_transforms` 학습용 증강 목록 (뒤집기·90° 회전·ShiftScaleRotate·밝기/대비) | 6 |
| TODO(2) | `CadSegDataset.__len__` — 표본 개수 반환 | 4 |
| TODO(3) | `CadSegDataset.__getitem__` — 이미지·마스크를 **같은 변환**으로 처리해 `(float32 (3,S,S), int64 (S,S))` 반환 | 10 |

### `train_seg.py` — 6개 (50점)

| # | 무엇을 | 배점 |
|---|---|---|
| TODO(1) | `build_model` — arch 별 smp 모델 생성 + `mit_*` 인코더는 DeepLabV3+ 에 못 쓴다는 검사 | 10 |
| TODO(2) | `DiceCELoss.__init__` — `nn.CrossEntropyLoss` 와 `smp.losses.DiceLoss(mode="multiclass")` 준비 | 6 |
| TODO(3) | `DiceCELoss.forward` — 두 손실의 합 | 4 |
| TODO(4) | 평가 지표 — `MulticlassJaccardIndex` 로 mIoU(macro, 배경 제외 옵션) 와 클래스별 IoU | 10 |
| TODO(5) | 학습 한 스텝 — AMP autocast → 손실 → `zero_grad` → `backward` → `unscale_`+클리핑 → `step` → `sched.step()` | 12 |
| TODO(6) | 검증 한 스텝 — 손실 누적 + `argmax` 예측으로 두 지표 갱신 | 8 |

### `predict.py` — 2개 (15점)

| # | 무엇을 | 배점 |
|---|---|---|
| TODO(1) | `colorize` — 인덱스 마스크에 `PALETTE` 적용 | 5 |
| TODO(2) | 추론 후처리 — 로짓을 원본 크기로 양선형 보간 후 `argmax` → uint8 인덱스 맵 | 10 |

### `train_sam_pseudolabel.py` — 3개 (15점)

| # | 무엇을 | 배점 |
|---|---|---|
| TODO(1) | `embed` — 무거운 영상 인코더를 이미지당 1회만 실행·캐시 | 5 |
| TODO(2) | `sam_mask` — 상자 프롬프트 → 원본 해상도 이진 마스크 + IoU 점수 | 6 |
| TODO(3) | 의사라벨 합성 루프 — 넓이 내림차순 정렬, 클래스 칠하기, 저품질 마스크는 상자로 대체 | 4 |

> `labelme2voc.py`, `label_convert.py`, `labels.txt`, `class_names.txt` 는 손대지 않는다.

## 5. 자가 점검

```bash
# ① 문법
python -m py_compile dataset.py train_seg.py predict.py train_sam_pseudolabel.py

# ② 데이터셋 (증강·인덱스 보존 확인)
python -c "
from dataset import *
pairs = list_pairs('data_annotated2'); tr, va = split_pairs(pairs)
ds = CadSegDataset(tr, 256, True); x, y = ds[0]
print(len(ds), x.shape, x.dtype, y.shape, y.dtype, int(y.min()), int(y.max()))
assert x.shape == (3,256,256) and y.shape == (256,256) and str(y.dtype)=='torch.int64'
assert 0 <= int(y.min()) and int(y.max()) <= 7   # 보간이 잘못되면 8 이상이나 실수가 나온다
"

# ③ 모델·손실·지표 (데이터 없이 더미 텐서로)
python -c "
import torch
from train_seg import build_model, DiceCELoss
from torchmetrics.classification import MulticlassJaccardIndex
m = build_model('unet','resnet34',8).eval()
o = m(torch.randn(2,3,256,256)); print(o.shape); assert o.shape == (2,8,256,256)
c = DiceCELoss(); y = torch.randint(0,8,(2,256,256))
print(float(c(o,y)))
perfect = torch.nn.functional.one_hot(y,8).permute(0,3,1,2).float()*20
print('완벽 예측 손실 ~0:', float(c(perfect,y)))    # 0 에 가까워야 한다
j = MulticlassJaccardIndex(num_classes=8, average='macro'); j.update(y,y)
print('완벽 예측 mIoU =', j.compute().item())       # 1.0 이어야 한다
try:
    build_model('deeplabv3plus','mit_b2',8); print('TODO(1) 검사 실패')
except ValueError as e: print('OK:', e)
"

# ④ 2 에폭만 돌려 손실이 내려가는지
python train_seg.py --data data_annotated2 --arch unet --encoder resnet34 \
       --size 128 --batch 2 --epochs 2 --workers 0
```

## 6. 제출물

1. 완성한 `dataset.py`, `train_seg.py`, `predict.py`, `train_sam_pseudolabel.py`
2. `runs/<run>/history.json` 과 `curves.png`, 최종 `val mIoU` 와 클래스별 IoU 표
3. `preds/` 의 오버레이 이미지 3장 이상
4. 보고서(2~3쪽)
   - 백본 3종(`resnet50` / `mit_b2` / `efficientnet-b3`) 의 val mIoU·학습 시간 비교표
   - 손실 ablation: CE 만 / Dice 만 / CE+Dice 의 클래스별 IoU (특히 소수 클래스 `ef`, `st`)
   - 사람 라벨 vs SAM 의사라벨 학습 모델의 mIoU·경계 정확도 비교와 실패 사례

예상 소요시간: 코드 채우기 2시간 + 학습·실험 3시간.
