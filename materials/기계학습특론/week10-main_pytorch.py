# -*- coding: utf-8 -*-
"""
기계학습특론 10주차 실습 — 뇌 MRI 4-클래스 종양 분류 (PyTorch 개정판)

기존 main.py (TensorFlow 2.13 + tensorflow-addons) 를 PyTorch 2.x 로 전환하고
다음 문제를 수정했습니다.

  1. 이중 정규화 (X/255 후 모델 내부 Rescaling(1/255)) 제거
  2. EarlyStopping(monitor='val_loss', mode='max') 의 방향 오류 -> 검증 macro-F1 최대화 기준
  3. 의미 없이 3번 호출되던 LabelEncoder.fit 제거 (클래스 사상을 한 번만 정의)
  4. 전체 이미지를 메모리에 적재(약 2.2GB) -> Dataset 지연 로딩 + DataLoader 병렬화
  5. 데이터 증강 추가 (Crop/Flip/Rotation/RandAugment/RandomErasing)
  6. 개발이 종료된 tensorflow-addons 의존 제거 (scikit-learn 지표로 대체)
  7. Training <-> Testing 중복(누수) 검사 추가 — 이 데이터셋에서는 테스트 394장 중
     252장(64%)이 학습 이미지와 픽셀 단위로 동일하며, no_tumor 는 105장 전부가 중복입니다.
     --protocol 로 세 가지 분할 프로토콜을 비교할 수 있습니다.
  8. 확률 캘리브레이션(temperature scaling)·ECE 보고 추가

사용 예)
    python main_pytorch.py --model resnet50 --pretrained --epochs 20
    python main_pytorch.py --model simplecnn --epochs 30 --seed 0
    python main_pytorch.py --model resnet18 --epochs 30 --balanced
    python main_pytorch.py --model resnet50 --pretrained --protocol original   # 누수 포함 비교용

자세한 설명과 시각화(Grad-CAM, 신뢰도 다이어그램)는
10주차_딥러닝분류_실습_PyTorch.ipynb 노트북을 참고하십시오.

────────────────────────────────────────────────────────────────────────────────
[문제본] 아래 10개의 TODO 를 채워야 이 스크립트가 동작한다.
각 TODO 는  # TODO(n): 설명  +  raise NotImplementedError("TODO(n)")  형태로 비어 있다.
주석과 raise 줄을 지우고 그 자리에 코드를 쓰면 된다 (시그니처·docstring 은 그대로 둘 것).

  TODO(1)  conv_block                       — [3x3 Conv → BN → ReLU] × 2 + MaxPool(2)
  TODO(2)  SimpleCNN.__init__               — conv_block 쌓기 + GAP 분류 헤드
  TODO(3)  BasicBlock.forward               — y = ReLU(F(x) + shortcut(x))
  TODO(4)  ResNet18._init_weights           — He 초기화 + zero-init residual
  TODO(5)  make_scheduler                   — 워밍업(선형) → 코사인 감쇠
  TODO(6)  param_groups                     — 백본/헤드 × decay/no-decay 파라미터 그룹
  TODO(7)  train() 의 학습 한 걸음           — 순전파 → 손실 → zero_grad → backward → step
  TODO(8)  fit_temperature 의 closure        — 온도 스케일링(NLL 최소화)
  TODO(9)  expected_calibration_error       — ECE
  TODO(10) main() 의 불균형 처리 가중치       — 역빈도 손실 가중치 + 균형 샘플링 가중치

노트북(10주차_딥러닝분류_실습_PyTorch.ipynb)의 같은 이름 TODO 와 내용이 겹치므로,
노트북을 먼저 푼 뒤 이 스크립트로 옮겨 적어도 좋다.
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

CLASSES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------------------
# 재현성
# --------------------------------------------------------------------------------------
def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------------------
# 데이터
# --------------------------------------------------------------------------------------
def scan_folder(root: Path) -> pd.DataFrame:
    rows = []
    for c in CLASSES:
        folder = root / c
        if not folder.exists():
            raise FileNotFoundError(f'클래스 폴더가 없습니다: {folder}')
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                rows.append({'path': str(p), 'class': c, 'label': CLASS_TO_IDX[c]})
    return pd.DataFrame(rows)


def md5_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def dhash(path: str, size: int = 16) -> bytes:
    """차이 해시: 가로 방향 이웃 픽셀의 밝기 대소 관계 size*size 비트.

    JPEG 재인코딩·리사이즈로 파일 바이트가 달라져도 '같은 사진'이면 같은 해시가 된다.
    8x8 평균 해시(aHash)는 배경이 검은 MRI 에서 서로 다른 이미지도 뭉뚱그리므로 부적합하다.
    """
    im = Image.open(path).convert('L').resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten()).tobytes()


def build_splits(root: Path, protocol: str, val_size: float, seed: int):
    """분할 프로토콜 (PART 5: 평가 프로토콜과 데이터 누수).

    original  : 출처가 제공한 Training/Testing 를 그대로 사용 (중복 방치 — 낙관적 편향)
    drop_test : Testing 에서 Training 과 픽셀이 같은 이미지를 제거
    repool    : Training+Testing 을 합쳐 전역 중복 제거 후 층화 재분할 (권장)
    """
    df_tv = scan_folder(root / 'Training')
    df_te = scan_folder(root / 'Testing')
    for df in (df_tv, df_te):
        df['md5'] = [md5_of(q) for q in df['path']]
        df['dh'] = [dhash(q) for q in df['path']]

    md5_tr, dh_tr = set(df_tv['md5']), set(df_tv['dh'])
    dup_md5 = int(df_te['md5'].isin(md5_tr).sum())
    dup_dh = int(sum(h in dh_tr for h in df_te['dh']))
    print(f'[누수 검사] Training<->Testing 파일 중복(MD5)  : {dup_md5}장 / {len(df_te)}장')
    print(f'[누수 검사] Training<->Testing 픽셀 중복(dHash): {dup_dh}장 / {len(df_te)}장 '
          f'({dup_dh / max(1, len(df_te)):.0%})')

    if protocol == 'original':
        pool_tv, pool_te = df_tv, df_te
    elif protocol == 'drop_test':
        pool_tv = df_tv
        pool_te = df_te[[h not in dh_tr for h in df_te['dh']]].reset_index(drop=True)
        print(f'            테스트 {len(df_te)} -> {len(pool_te)}장으로 정리')
    elif protocol == 'repool':
        pool = pd.concat([df_tv, df_te], ignore_index=True)
        before = len(pool)
        pool = pool.drop_duplicates('dh').reset_index(drop=True)
        print(f'            전체 {before}장 -> 고유 {len(pool)}장 (중복 {before - len(pool)}장 제거)')
        pool_tv, pool_te = train_test_split(pool, test_size=0.15, random_state=seed,
                                            stratify=pool['label'])
    else:
        raise ValueError(f'알 수 없는 프로토콜: {protocol}')

    df_train, df_val = train_test_split(pool_tv, test_size=val_size, random_state=seed,
                                        stratify=pool_tv['label'])
    return (df_train.reset_index(drop=True), df_val.reset_index(drop=True),
            pool_te.reset_index(drop=True))


class MRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df['path'].tolist()
        self.labels = df['label'].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


def build_transforms(img_size: int, augment: bool = True):
    eval_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if not augment:
        return eval_tf, eval_tf
    train_tf = T.Compose([
        T.Resize((img_size + 32, img_size + 32)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.RandAugment(num_ops=2, magnitude=7),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])
    return train_tf, eval_tf


# --------------------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------------------
def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """입력 (N, c_in, H, W) -> 출력 (N, c_out, H/2, W/2) 인 기본 블록."""
    # TODO(1): [3x3 Conv -> BatchNorm2d -> ReLU] 를 두 번 쌓고 마지막에 MaxPool2d(2) 를 둔다.
    #   힌트: padding=1 이면 'same' (크기 유지). 뒤에 BN 이 오므로 Conv2d 는 bias=False.
    raise NotImplementedError("TODO(1)")


class SimpleCNN(nn.Module):
    """강의자료 PART 2 의 기준선 CNN (GAP 헤드)."""

    def __init__(self, num_classes: int = NUM_CLASSES, widths=(32, 64, 128, 256), in_ch: int = 3):
        super().__init__()
        # TODO(2): widths 를 따라 conv_block 을 쌓아 self.features 를,
        #   GAP -> Flatten -> Dropout(0.3) -> Linear 로 self.head 를 만든다.
        #   힌트: self.features = nn.Sequential(*blocks)
        #         self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        #                                   nn.Dropout(0.3), nn.Linear(마지막 채널 수, num_classes))
        raise NotImplementedError("TODO(2)")

    def forward(self, x):
        return self.head(self.features(x))


class BasicBlock(nn.Module):
    """강의자료 PART 4 의 잔차 블록. residual=False 이면 plain 블록."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                          nn.BatchNorm2d(c_out))

    def forward(self, x):
        """반환: (N, c_out, H/stride, W/stride)"""
        # TODO(3): conv1-bn1-ReLU -> conv2-bn2 로 F(x) 를 만들고,
        #   self.residual 이면 F.relu(F(x) + self.shortcut(x)) 를, 아니면 F.relu(F(x)) 를 반환한다.
        #   힌트: 두 번째 ReLU 는 «더한 뒤» 에 적용한다.
        raise NotImplementedError("TODO(3)")


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, widths=(64, 128, 256, 512),
                 blocks=(2, 2, 2, 2), residual: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        layers, c_prev = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for b in range(n):
                layers.append(BasicBlock(c_prev, w, stride=2 if (b == 0 and i > 0) else 1,
                                         residual=residual))
                c_prev = w
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(c_prev, num_classes))
        self._init_weights()

    def _init_weights(self):
        # TODO(4): He(Kaiming) 초기화와 zero-init residual 을 구현한다.
        #   힌트: self.modules() 를 훑으며 nn.Conv2d 는
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'),
        #         nn.BatchNorm2d 는 weight=1, bias=0 으로 둔다. 그 다음 다시 훑으며
        #         residual=True 인 BasicBlock 의 bn2.weight 를 0 으로 두면 학습 초기에
        #         블록이 «항등 사상» 에서 출발한다.
        raise NotImplementedError("TODO(4)")

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def build_model(name: str, pretrained: bool = False):
    """모델과 '헤드 파라미터 이름 접두사'를 함께 반환 (층별 학습률용)."""
    if name == 'simplecnn':
        return SimpleCNN(), ('head',)
    if name == 'resnet18':
        return ResNet18(), ('head',)
    if name == 'resnet50':
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        return m, ('fc',)
    if name == 'convnext_tiny':
        w = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
        return m, ('classifier',)
    raise ValueError(f'알 수 없는 모델: {name}')


# --------------------------------------------------------------------------------------
# 학습 / 평가
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == 'cuda')):
            logits = model(x)
        logits_all.append(logits.float().cpu())
        y_all.append(y)
    logits, y = torch.cat(logits_all), torch.cat(y_all)
    probs = logits.softmax(1)
    return logits, probs, probs.argmax(1), y


def metrics_from(probs, pred, y) -> dict:
    # float() 로 감싼다: sklearn 이 돌려주는 numpy 스칼라를 그대로 체크포인트에 넣으면
    # torch>=2.6 의 torch.load(weights_only=True) 가 UnpicklingError 로 거부한다.
    out = {'acc': float(accuracy_score(y, pred)),
           'macro_f1': float(f1_score(y, pred, average='macro'))}
    try:
        out['auroc'] = float(roc_auc_score(y, probs, multi_class='ovr', average='macro'))
    except ValueError:
        out['auroc'] = float('nan')
    return out


def make_scheduler(optimizer, epochs: int, warmup: int = 3):
    warmup = max(1, min(warmup, epochs - 1)) if epochs > 1 else 1   # 워밍업이 총 에폭을 넘지 않도록
    # TODO(5): «워밍업(선형 상승) -> 코사인 감쇠» 스케줄러를 만들어 반환한다.
    #   힌트: LinearLR(optimizer, start_factor=0.01, total_iters=max(1, warmup)) 과
    #         CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup)) 을
    #         SequentialLR(optimizer, [warm, cos], milestones=[max(1, warmup)]) 로 잇는다.
    raise NotImplementedError("TODO(5)")


def param_groups(model, head_names, lr_backbone: float, lr_head: float, weight_decay: float):
    """가중치 감쇠는 bias·정규화 층에 적용하지 않고, 백본과 헤드에 서로 다른 학습률을 준다."""
    # TODO(6): 학습 대상 파라미터를 «백본/헤드» × «감쇠 적용/미적용» 네 그룹으로 나눈다.
    #   반환: AdamW 에 그대로 넘길 수 있는 딕셔너리 4개의 리스트.
    #         [{'params': [...], 'lr': lr_backbone, 'weight_decay': weight_decay},
    #          {'params': [...], 'lr': lr_backbone, 'weight_decay': 0.0},
    #          {'params': [...], 'lr': lr_head,     'weight_decay': weight_decay},
    #          {'params': [...], 'lr': lr_head,     'weight_decay': 0.0}]
    #   힌트: n.startswith(tuple(head_names)) 이면 헤드다.
    #         p.ndim <= 1 (bias·BatchNorm 의 gamma/beta) 에는 가중치 감쇠를 «주지 않는다».
    raise NotImplementedError("TODO(6)")


def train(model, head_names, train_loader, val_loader, args, device, amp_dtype):
    model = model.to(device, memory_format=torch.channels_last)
    class_weight = args.class_weight.to(device) if args.class_weight is not None else None
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weight)
    optimizer = torch.optim.AdamW(
        param_groups(model, head_names, args.lr_backbone, args.lr_head, args.weight_decay))
    sched = make_scheduler(optimizer, args.epochs, args.warmup)
    scaler = torch.amp.GradScaler(device, enabled=(device == 'cuda' and amp_dtype is torch.float16))

    history, best = [], {'macro_f1': -1.0, 'epoch': 0}
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            # TODO(7): 한 배치의 학습 한 걸음. 끝나면 스칼라 텐서 loss 가 있어야 한다.
            #   힌트: «순전파 -> 손실 -> zero_grad -> backward -> 클리핑 -> step» 순서로,
            #     1) with torch.autocast(device_type=device, dtype=amp_dtype,
            #                            enabled=(device == 'cuda')):  안에서
            #        loss = criterion(model(x), y)
            #     2) optimizer.zero_grad(set_to_none=True)
            #     3) scaler.scale(loss).backward() 후 scaler.unscale_(optimizer)
            #     4) torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #     5) scaler.step(optimizer); scaler.update()
            raise NotImplementedError("TODO(7)")
            run += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()

        _, probs, pred, yv = predict(model, val_loader, device, amp_dtype)
        mtr = metrics_from(probs.numpy(), pred.numpy(), yv.numpy())
        rec = {'epoch': ep, 'train_loss': run / max(1, seen),
               'lr': optimizer.param_groups[0]['lr'], **mtr, 'sec': time.time() - t0}
        history.append(rec)
        # 기존 코드의 mode='max' 버그를 올바르게: macro-F1 '최대'일 때 저장
        if mtr['macro_f1'] > best['macro_f1']:
            best = dict(rec)
            torch.save({'model': model.state_dict(), 'epoch': ep, **mtr}, args.ckpt)
        print(f"ep {ep:>3}/{args.epochs} loss {rec['train_loss']:.4f} "
              f"val_acc {mtr['acc']:.4f} val_F1 {mtr['macro_f1']:.4f} "
              f"AUROC {mtr['auroc']:.4f} lr {rec['lr']:.2e} ({rec['sec']:.1f}s)", flush=True)

    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])
    print(f"[best] epoch {best['epoch']} · val macro-F1 {best['macro_f1']:.4f}")
    return model, pd.DataFrame(history), best


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, max_iter: int = 200) -> float:
    """검증 세트로 온도 T 를 학습 (Guo et al., 2017). 정확도(순위)는 바뀌지 않는다."""
    logits, y = logits.detach().float(), y.detach()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        # TODO(8): LBFGS 가 부르는 closure. NLL 을 최소화하는 온도 T 를 찾는 것이 목표다.
        #   힌트: opt.zero_grad() -> loss = F.cross_entropy(logits / T, y) -> loss.backward()
        #         -> return loss.  T = log_t.exp().clamp(min=1e-2) 로 항상 양수임을 보장한다.
        raise NotImplementedError("TODO(8)")

    opt.step(closure)
    with torch.no_grad():
        t = float(log_t.exp().clamp(0.05, 20.0))
    return t if np.isfinite(t) else 1.0    # 최적화가 발산하면 보정하지 않음(T=1)


def expected_calibration_error(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    """반환: ECE (파이썬 float)"""
    # TODO(9): ECE = Σ_m (|B_m| / N) · |acc(B_m) − conf(B_m)| 를 구현한다.
    #   힌트: conf, pred = probs.max(1) 로 신뢰도와 예측을, acc = (pred == y).float() 로
    #         맞았는지를 구한다. torch.linspace(0, 1, n_bins + 1) 로 경계를 만들고
    #         각 구간 마스크 m = (conf > lo) & (conf <= hi) 가 비어 있지 않을 때만
    #         m.float().mean() * (acc[m].mean() - conf[m].mean()).abs() 를 더한다.
    raise NotImplementedError("TODO(9)")


# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='뇌 MRI 4-클래스 분류 (PyTorch)')
    p.add_argument('--data', default='./data', help='Training/Testing 이 들어 있는 폴더')
    p.add_argument('--model', default='resnet50',
                   choices=['simplecnn', 'resnet18', 'resnet50', 'convnext_tiny'])
    p.add_argument('--pretrained', action='store_true', help='ImageNet 사전학습 가중치 사용')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr-head', type=float, default=1e-3)
    p.add_argument('--lr-backbone', type=float, default=None,
                   help='미지정 시 사전학습이면 lr_head/10, 아니면 lr_head')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--val-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--balanced', action='store_true',
                   help='클래스 불균형 처리(역빈도 손실 가중치 + 균형 샘플링)')
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--protocol', default='repool',
                   choices=['original', 'drop_test', 'repool'],
                   help='분할 프로토콜: original=출처 제공 분할 그대로(중복 방치), '
                        'drop_test=테스트에서 픽셀 중복 제거, repool=전역 중복 제거 후 재분할(권장)')
    p.add_argument('--ckpt', default=None)
    p.add_argument('--out', default='results_summary.csv')
    return p.parse_args()


def main():
    args = parse_args()
    if args.lr_backbone is None:
        args.lr_backbone = args.lr_head / 10 if args.pretrained else args.lr_head
    if args.ckpt is None:
        args.ckpt = f'ckpt_{args.model}_seed{args.seed}.pt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = (torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported()
                 else torch.float16)
    print(f'torch {torch.__version__} | device {device} | amp {amp_dtype}')
    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    root = Path(args.data)
    df_train, df_val, df_test = build_splits(root, args.protocol, args.val_size, args.seed)
    print(f'[{args.protocol}] train {len(df_train)} / val {len(df_val)} / test {len(df_test)}')
    print('  train:', df_train['class'].value_counts().to_dict())
    print('  test :', df_test['class'].value_counts().to_dict())

    # 분할 인덱스를 저장해 재현 가능하게 (PART 5: 재현성 체크리스트)
    for nm, d in (('train', df_train), ('val', df_val), ('test', df_test)):
        d[['path', 'class', 'label']].to_csv(
            f'split_{args.protocol}_{nm}_seed{args.seed}.csv', index=False, encoding='utf-8-sig')

    train_tf, eval_tf = build_transforms(args.img_size, augment=not args.no_augment)

    counts = torch.tensor([(df_train['label'] == i).sum() for i in range(NUM_CLASSES)],
                          dtype=torch.float)
    sampler, args.class_weight = None, None
    if args.balanced:
        # TODO(10): 역빈도 손실 가중치 args.class_weight (shape (K,), 평균 1) 와
        #   균형 샘플링용 표본 가중치 sample_w (shape (N_train,)) 를 만든 뒤 sampler 를 구성한다.
        #   힌트: w = 1.0 / counts 로 빈도에 반비례하게 두고,
        #         args.class_weight = w / w.sum() * NUM_CLASSES 로 평균이 1 이 되게 정규화한다.
        #         sample_w = w[torch.tensor(df_train['label'].values)] 로 표본별 가중치를 고르고
        #         sampler = WeightedRandomSampler(sample_w.double(), len(sample_w), replacement=True)
        raise NotImplementedError("TODO(10)")
        print('불균형 처리 사용 · 손실 가중치:',
              {c: round(v, 3) for c, v in zip(CLASSES, args.class_weight.tolist())})

    def loader(df, tf, shuffle=False, smp=None):
        return DataLoader(MRIDataset(df, tf), batch_size=args.batch_size,
                          shuffle=(shuffle and smp is None), sampler=smp,
                          num_workers=args.workers, pin_memory=(device == 'cuda'),
                          persistent_workers=args.workers > 0,
                          worker_init_fn=seed_worker, generator=g)

    train_loader = loader(df_train, train_tf, shuffle=True, smp=sampler)
    val_loader = loader(df_val, eval_tf)
    test_loader = loader(df_test, eval_tf)

    model, head_names = build_model(args.model, pretrained=args.pretrained)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'모델 {args.model} (pretrained={args.pretrained}) · 파라미터 {n_total:,}')

    model, history, best = train(model, head_names, train_loader, val_loader,
                                 args, device, amp_dtype)
    history.to_csv(f'history_{args.model}_seed{args.seed}.csv', index=False, encoding='utf-8-sig')

    # ---- 테스트 평가 ----
    logits, probs, pred, y = predict(model, test_loader, device, amp_dtype)
    mtr = metrics_from(probs.numpy(), pred.numpy(), y.numpy())
    print('\n===== 테스트 결과 =====')
    print(f"accuracy {mtr['acc']:.4f} | macro-F1 {mtr['macro_f1']:.4f} | AUROC {mtr['auroc']:.4f}")
    print(classification_report(y, pred, target_names=CLASSES, digits=3, zero_division=0))
    print('혼동행렬 (행=실제, 열=예측)')
    print(pd.DataFrame(confusion_matrix(y, pred, labels=range(NUM_CLASSES)),
                       index=CLASSES, columns=CLASSES).to_string())

    # ---- 캘리브레이션 ----
    val_logits, _, _, val_y = predict(model, val_loader, device, amp_dtype)
    temperature = fit_temperature(val_logits, val_y)
    ece_before = expected_calibration_error(probs, y)
    ece_after = expected_calibration_error((logits / temperature).softmax(1), y)
    print(f'\n[캘리브레이션] T = {temperature:.3f} · ECE {ece_before:.4f} -> {ece_after:.4f}')

    row = {'model': args.model, 'pretrained': args.pretrained, 'protocol': args.protocol,
           'balanced': args.balanced, 'n_test': len(df_test),
           'augment': not args.no_augment, 'epochs': args.epochs, 'seed': args.seed,
           'val_macro_f1': round(best['macro_f1'], 4), 'test_acc': round(mtr['acc'], 4),
           'test_macro_f1': round(mtr['macro_f1'], 4), 'test_auroc': round(mtr['auroc'], 4),
           'temperature': round(temperature, 3),
           'ece_before': round(ece_before, 4), 'ece_after': round(ece_after, 4)}
    out = Path(args.out)
    df_row = pd.DataFrame([row])
    if out.exists():
        df_row = pd.concat([pd.read_csv(out), df_row], ignore_index=True)
    df_row.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'\n결과를 {out} 에 추가했습니다. (여러 번 실행하면 ablation 표가 쌓입니다)')

    # ---- 예측 샘플 출력 (기존 main.py 의 마지막 부분에 해당) ----
    print('\n샘플 5장 예측')
    for i in range(min(5, len(y))):
        print(f'  {i + 1}: 실제 {CLASSES[int(y[i])]:>16} | 예측 {CLASSES[int(pred[i])]:>16} '
              f'(p={float(probs[i, pred[i]]):.3f})')


if __name__ == '__main__':
    main()
