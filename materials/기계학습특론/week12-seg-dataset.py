"""CAD 도면 요소 분할용 PyTorch Dataset.  [실습 문제본]

data_annotated2/
  JPEGImages/*.jpg                 원본 도면
  SegmentationClassPNG3/*.png      클래스 인덱스 마스크 (0=_background_, 1..7)
  class_names.txt                  __ignore__, _background_, ow, iw, of, cw, ef, st, oo
                                   (__ignore__ 를 뺀 8클래스 = 배경 1 + 도면 기호 7종)

labelme JSON -> VOC 형식 변환은 labelme2voc.py 를 그대로 사용한다.
(python labelme2voc.py <json_dir> data_annotated2 --labels labels.txt)
labelme2voc.py 가 만드는 SegmentationClass/*.npy 를 인덱스 PNG 로 바꾸려면 아래
`npy_to_index_png()` 를 쓴다 (기존 label_convert.py 의 컬러 PNG → /255 방식은
클래스 3개 이상에서 인덱스가 깨지므로 대체).

────────────────────────────────────────────────────────────
이 파일의 TODO
  TODO(1) build_transforms  : 학습용 기하·강도 증강 목록을 채운다
  TODO(2) CadSegDataset.__len__     : 표본 개수 반환
  TODO(3) CadSegDataset.__getitem__ : 이미지·마스크를 읽고 같은 증강을 적용해 텐서로 반환
────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_class_names(path: str) -> list[str]:
    names = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    return [n for n in names if n != "__ignore__"]  # _background_ 가 인덱스 0


def npy_to_index_png(npy_dir: str, out_dir: str) -> None:
    """labelme2voc 의 SegmentationClass/*.npy (int32 라벨) -> 인덱스 PNG(uint8)."""
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(glob(os.path.join(npy_dir, "*.npy"))):
        lbl = np.load(f).astype(np.int64)
        lbl[lbl < 0] = 0  # __ignore__(-1) 은 배경 처리 (필요 시 255 로 두고 ignore_index=255)
        Image.fromarray(lbl.astype(np.uint8)).save(os.path.join(out_dir, Path(f).stem + ".png"))


def list_pairs(data_dir: str, img_sub="JPEGImages", mask_sub="SegmentationClassPNG3"):
    imgs = sorted(glob(os.path.join(data_dir, img_sub, "*.jp*g")) + glob(os.path.join(data_dir, img_sub, "*.png")))
    pairs = []
    for im in imgs:
        m = os.path.join(data_dir, mask_sub, Path(im).stem + ".png")
        if os.path.exists(m):
            pairs.append((im, m))
    if not pairs:
        raise FileNotFoundError(f"{data_dir} 에서 이미지-마스크 쌍을 찾지 못했습니다.")
    return pairs


def build_transforms(size: int, train: bool) -> A.Compose:
    """Resize → (학습이면 증강) → Normalize → ToTensorV2 순서의 albumentations 파이프라인.

    반환: A.Compose. `tf(image=img, mask=mask)` 로 호출하면 이미지와 마스크에
          **같은 기하 변환**이 적용되고, 마스크는 최근접 보간이라 클래스 인덱스가 보존된다.
    """
    aug = [A.Resize(size, size)]  # 마스크는 자동으로 최근접 보간 → 클래스 인덱스 보존
    if train:
        # TODO(1): 학습용 증강 목록을 aug 에 추가한다 (좌우 뒤집기, 90도 회전, 평행이동·확대·회전, 밝기·대비).
        #   힌트: A.HorizontalFlip / A.RandomRotate90 / A.ShiftScaleRotate(border_mode=0) / A.RandomBrightnessContrast, 각 p=0.5
        #[설명] albumentations 는 image 와 mask 를 함께 넘기면 기하 변환(뒤집기·회전·이동)을 두 배열에 똑같이 적용하고,
        #[설명] 마스크에는 최근접 보간을 써서 클래스 번호가 섞이지 않게 한다. 밝기·대비는 이미지에만 적용된다.
        #[설명] ShiftScaleRotate 의 border_mode=0 은 회전 후 생기는 빈 영역을 0(검정, 마스크는 배경)으로 채운다는 뜻이다.
        #[목적] 도면은 방향이 바뀌어도 같은 기호이므로 90도 회전·뒤집기로 적은 데이터를 늘리고 과적합을 줄인다.
        aug += [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=10, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ]
    aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)


class CadSegDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], size: int = 512, train: bool = True):
        self.pairs = pairs
        self.tf = build_transforms(size, train)

    def __len__(self) -> int:
        """데이터셋의 표본 개수."""
        # TODO(2): 표본 개수를 반환한다. (빠뜨리면 DataLoader 가 TypeError 를 낸다)
        #[설명] DataLoader 는 len(dataset) 으로 에폭당 배치 수와 셔플 인덱스 범위를 정한다.
        return len(self.pairs)

    def __getitem__(self, i: int):
        """i 번째 (이미지, 마스크) 를 텐서로 반환.

        반환: (image, mask)
              image = float32 (3, size, size)  정규화된 텐서
              mask  = int64   (size, size)     클래스 인덱스 0..7
        """
        img_path, mask_path = self.pairs[i]
        # TODO(3): 이미지를 RGB 로, 마스크를 그대로 읽어 self.tf 로 **한 번에** 변환한 뒤
        #          (image, mask.long()) 를 반환한다.
        #   힌트: np.array(Image.open(p).convert("RGB")), 마스크가 3채널이면 mask[..., 0] 로 줄인다.
        #   힌트: out = self.tf(image=img, mask=mask) → out["image"], out["mask"]
        #[설명] 마스크 PNG 는 화소값 자체가 클래스 번호(0~7)인 인덱스 영상이므로 convert 없이 그대로 읽는다.
        #[설명] RGB 로 바꾸면 팔레트 색으로 풀려 번호가 사라지므로 마스크에는 convert("RGB") 를 쓰면 안 된다.
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        #[목적] 이미지와 마스크를 한 번의 호출로 변환해야 같은 난수(같은 회전·뒤집기)가 적용되어 화소 대응이 유지된다.
        out = self.tf(image=img, mask=mask)
        #[설명] ToTensorV2 는 image 를 (3,H,W) float32 로 바꾸고 mask 는 (H,W) 그대로 두므로,
        #[설명] CrossEntropyLoss 가 요구하는 int64 정답으로 .long() 변환한다.
        #[결과] 자가 점검 ②: x = (3, 256, 256) float32, y = (256, 256) int64, 마스크 값 {0, 1, 2} 로 클래스 번호가 보간 후에도 유지되었다.
        #[결과] 배포된 SegmentationClassPNG3 는 전부 0 이어서 npy_to_index_png 로 .npy 에서 다시 만든 마스크를 사용했다.
        return out["image"], out["mask"].long()


def split_pairs(pairs, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_ratio))
    val = [pairs[i] for i in idx[:n_val]]
    train = [pairs[i] for i in idx[n_val:]]
    return train, val


def denormalize(x: torch.Tensor) -> np.ndarray:
    """(3,H,W) 정규화 텐서 -> uint8 HWC."""
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
