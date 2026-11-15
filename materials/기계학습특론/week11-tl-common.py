"""공통 유틸리티: 시드 고정, CIFAR-10 로더, 학습/평가 루프
2026711051 박민지

기계학습특론 11주차 실습 (PyTorch 2.x · timm · HF PEFT · Optuna 5)

이 파일의 TODO
  TODO(1)  set_seed        — python / numpy / torch(+cuDNN) 시드 고정 (재현성)
  TODO(2)  train_one_epoch — 학습 한 스텝의 핵심 (순전파 → 손실 → zero_grad → backward → 클리핑 → step)
  TODO(3)  evaluate        — 정확도와 macro-F1 계산

데이터 로딩·전처리·로그 출력은 완성되어 있다. TODO 줄만 지우고 구현하면 그대로 이어서 돌아간다.
"""
#[설명] common 파일은 실습 1·2·3 이 함께 쓰는 공용 파일로, import의 대상이 될 것.
#시드 고정, 데이터 나누기, 에폭 학습, 평가에 대한 규칙을 똑같이 설정하기 위함임.
#① 재현성(set_seed) ② PyTorch 학습 루프의 표준 순서(train_one_epoch) ③ 공정한 지표 macro-F1(evaluate)
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset

#[설명] PRETRAINED=True 면 timm 이 ImageNet 등으로 미리 학습된 가중치를 내려받는데 이것이 전이학습의 출발점
# LAB_OFFLINE_TEST=1 로 실행하면 사전학습 가중치·CIFAR 다운로드 없이 코드 경로만 검증
OFFLINE_TEST = os.environ.get("LAB_OFFLINE_TEST") == "1"
PRETRAINED = not OFFLINE_TEST

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


#[설명] 시드는 난수의 출발점이다. 시드가 같으면 가중치 초기화 · 데이터 섞기 · 증강이 매번 똑같이 나온다.
#[목적] 다시 돌려도 같은 결과가 나오게해서(재현성), 성능 차이가 설정 때문인지 운 때문인지 가린다.

#[결과] 실습 3: 시드 0 재학습 값 0.7472 가 trial 0 과 소수점까지 같았다(CPU 완전 재현
#[결과] 반대로 시드만 바꿔도 0.62~0.75 로 흔들렸다 → 시드 3개의 평균 ± 표준편차를 보고하는 이유
def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """재현성: python / numpy / torch 시드 고정 (완전한 결정성은 GPU에서 보장되지 않음)."""
    # TODO(1): random · numpy · torch(CPU/모든 GPU) 시드를 seed 로 고정하고,
    # deterministic=True 이면 cuDNN 을 결정적 모드로 둔다 (benchmark 는 끈다).
    # 힌트: random.seed / np.random.seed / torch.manual_seed / torch.cuda.manual_seed_all,
    # torch.backends.cudnn.deterministic · torch.backends.cudnn.benchmark
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


#[설명] GPU(cuda) → 애플 GPU(mps) → CPU 순으로 쓸 수 있는 장치를 고른다. Colab 에서는 cuda, 이 노트북에서는 cpu 가 잡혔다.
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


#[설명] CIFAR-10: 32×32 컬러 사진 6만 장, 10개 클래스. 공식 train 5만 / test 1만 장.
#train 5만 장에서 5,000장을 떼어 '검증'을 만든다.
def build_cifar10_loaders(
    data_dir: str,
    train_tf,
    eval_tf,
    batch_size: int = 64,
    train_size: int | None = None,
    val_size: int = 5000,
    test_size: int | None = None,
    num_workers: int = 2,
    seed: int = 0,
):
    """CIFAR-10을 학습/검증/테스트로 분할.

    - 검증 세트: 공식 train 50,000장에서 val_size장을 떼어냄 (테스트는 튜닝에 절대 사용하지 않음)
    - train_size / test_size: 빠른 실험(--quick)을 위한 부분 샘플링
    - train_tf / eval_tf: timm.data.create_transform 으로 만든 사전학습 정합 전처리
    """
    if OFFLINE_TEST:  # 네트워크 없는 환경에서 코드 경로만 검증 (무작위 이미지)
        full_train = torchvision.datasets.FakeData(512, (3, 32, 32), 10, transform=train_tf, random_offset=1)
        full_train_eval = torchvision.datasets.FakeData(512, (3, 32, 32), 10, transform=eval_tf, random_offset=1)
        test = torchvision.datasets.FakeData(256, (3, 32, 32), 10, transform=eval_tf, random_offset=2)
    else:
        full_train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        full_train_eval = torchvision.datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_tf)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    #시드가 같으면 어느 실습이든 같은 검증 세트를 쓴다.
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train), generator=g).tolist()
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    if train_size is not None:
        train_idx = train_idx[:train_size]
    test_idx = list(range(len(test)))
    if test_size is not None:
        test_idx = test_idx[:test_size]

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_train_eval, val_idx)  # 검증은 증강 없이(eval_tf)
    test_ds = Subset(test, test_idx)

    pin = torch.cuda.is_available()
    #[설명] DataLoader: 데이터를 batch_size 장씩 묶어 공급한다. 학습용은 shuffle=True(매 에폭 섞기), drop_last=True(끝 자투리 배치 버림).
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader

#[설명] 학습되는 파라미터 수와 전체 수를 센다. 프로빙 / LoRA / Full FT 가 '얼마나 학습하나'를 비교하는 핵심 숫자.
def count_params(model: nn.Module) -> tuple[int, int]:
    """(학습 가능 파라미터 수, 전체 파라미터 수)"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


#[설명] 학습 데이터를 한 바퀴(1 에폭) 돌며 가중치를 고친다. 반환값은 평균 학습 손실.
# label_smoothing=0.1: 정답 확률 목표를 1.0 대신 0.91 로 두어 과신을 막는다. 
# RandAugment 도 있어 학습 손실이 높게 보인다.
def train_one_epoch(model, loader, optimizer, device, scheduler=None, criterion=None, log_every: int = 50, max_steps: int | None = None) -> float:
    """한 에폭 학습. 평균 손실 반환. scheduler는 step 단위(cosine/warmup)라고 가정."""
    model.train()
    criterion = criterion or nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = device.type == "cuda"
    total, n = 0.0, 0
    t0 = time.time()
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # TODO(2): 학습 한 스텝을 구현한다.
        #          (a) autocast(bfloat16, enabled=use_amp) 안에서 순전파 → logits, criterion 으로 loss
        #          (b) optimizer.zero_grad(set_to_none=True) → loss.backward()
        #          (c) grad norm 을 1.0 으로 클리핑 → optimizer.step()
        #          (d) scheduler 가 있으면 step 단위로 scheduler.step()
        #   힌트: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp),
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #   주의: loss 변수를 아래 로그 코드가 그대로 쓰므로 이름을 loss 로 둔다.
        
        #[단계] ① 순전파: 예측(logits) ② 손실: 정답과 얼마나 다른지 ③ zero_grad: 지난 기울기 지우기
        # ④ backward: 손실을 줄이는 방향(기울기) 계산 ⑤ 클리핑: 기울기 크기를 1.0 이하로 잘라 폭주 방지 ⑥ step: 가중치 수정 ⑦ 스케줄러: lr 조정
        #[주의] 순서가 중요하다. zero_grad 를 backward 뒤에 두면 방금 구한 기울기가 지워져 학습이 안 된다.
        #[특이사항] 무료 Colab T4 는 bf16 을 하드웨어로 지원하지 않아 ConvNeXt 에서 오류가 났다. 실행할 때만 bf16 을 끄고 fp32 로 돌렸다.
        #[결과] 간단한 분류 문제에서 손실이 0.657 → 0.269 로 줄었다. 학습이 잘되고있다는 것.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                                            # backward 는 autocast 밖에서
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # backward 와 step 사이
        optimizer.step()
        if scheduler is not None:
            scheduler.step()                                       # step 단위 스케줄
        total += loss.item() * x.size(0)
        n += x.size(0)
        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step + 1:4d}  loss {total / n:.4f}  ({time.time() - t0:.0f}s)")
    return total / max(n, 1)


#[설명] 검증/테스트 데이터로 성능을 잰다
# no_grad: 기울기를 계산하지 않아 빠르고 메모리를 덜 쓴다. model.eval(): dropout 등을 끈다.
@torch.no_grad()
def evaluate(model, loader, device, max_steps: int | None = None) -> dict:
    """정확도·macro-F1·평균 손실 계산."""
    from sklearn.metrics import f1_score

    model.eval()
    criterion = nn.CrossEntropyLoss()
    preds, targets, total, n = [], [], 0.0, 0
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    p = torch.cat(preds).numpy()
    t = torch.cat(targets).numpy()
    # TODO(3): 예측 p 와 정답 t 로 정확도와 macro-F1 을 계산해 아래 형태의 dict 로 반환한다.
    #          {"loss": <평균 손실>, "acc": <정확도>, "macro_f1": <macro-F1>}   (값은 모두 float)
    #   힌트: 정확도는 (p == t).mean(), macro-F1 은 sklearn 의 f1_score(t, p, average="macro").
    #         평균 손실은 total / max(n, 1) (0으로 나누기 방지).
    
    #[설명] accuracy = 맞힌 비율. macro-F1 = 클래스마다 F1(정밀도와 재현율의 조화평균)을 구해 단순 평균한 값.
    #클래스가 불균형하면 정확도는 많은 클래스만 잘 맞혀도 높게 나온다. 반면에 macro-F1 은 모든 클래스를 똑같이 취급해 약한 클래스도 드러낸다.
    return {
        "loss": total / max(n, 1),
        "acc": float((p == t).mean()),
        "macro_f1": float(f1_score(t, p, average="macro")),
    }


#[설명] 사전학습 모델은 학습 때 본 입력 형식(224×224 크기, 평균·표준편차 정규화)을 그대로 받는게 가장 좋다.
# timm 이 모델별 설정을 읽어 전처리를 자동으로 맞춘다. CIFAR 32×32 사진을 224 로 키운다.
# 학습용에는 RandAugment(무작위 회전·색 변화 등)를 넣어 과적합을 줄이고, 평가용은 증강 없이 쓴다.
def resolve_transforms(model, auto_augment: str | None = "rand-m9-mstd0.5"):
    """timm 사전학습 설정(mean/std/해상도/crop)에 정합된 전처리 생성."""
    from timm.data import create_transform, resolve_data_config

    cfg = resolve_data_config({}, model=model)
    train_tf = create_transform(**cfg, is_training=True, auto_augment=auto_augment)
    eval_tf = create_transform(**cfg, is_training=False)
    return cfg, train_tf, eval_tf


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
