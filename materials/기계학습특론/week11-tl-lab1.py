"""실습 1. 전이학습 기본 — 동결 선형 프로빙 vs LLRD 전체 파인튜닝 (timm) 
데이터사이언스융합학과 2026711051 박민지

이 파일의 TODO
  TODO(1)  선형 프로빙(--mode probe): 백본 동결 · 헤드만 학습 · 헤드 전용 AdamW
  TODO(2)  전체 파인튜닝(--mode full): 전체 해동 + 층별 학습률 감쇠(LLRD) 파라미터 그룹 구성
  TODO(3)  warmup + cosine 스케줄러(step 단위) 구성
  TODO(4)  검증 macro-F1 최고 시점의 체크포인트 저장 (기존 자료 6단계)

argparse · 데이터 로딩 · 로그 출력 · 결과 저장은 완성되어 있다.

예)
  # (A) 선형 프로빙: 백본 동결, 헤드만 학습
  python lab1_transfer_learning.py --model convnext_tiny.dinov3_lvd1689m --mode probe --epochs 3

  # (B) 전체 파인튜닝 + 층별 학습률 감쇠(LLRD) + warmup/cosine
  python lab1_transfer_learning.py --model convnext_tiny.dinov3_lvd1689m --mode full --lr 5e-5 --layer_decay 0.75 --epochs 5

  # (C) LP-FT: 프로빙으로 헤드를 먼저 맞춘 뒤(체크포인트) 전체 FT
  python lab1_transfer_learning.py --mode full --init_head_from runs/probe/best.pt

  # CPU/노트북에서 빠른 동작 확인
  python lab1_transfer_learning.py --quick

백본 비교 과제: --model 에 resnet50.a1_in1k / vit_small_patch16_224.augreg_in21k_ft_in1k /
convnext_tiny.dinov3_lvd1689m 등을 바꿔 가며 동일 조건으로 실행하고 표로 정리하세요.
(DINOv3 가중치는 Hugging Face 로그인·라이선스 동의가 필요할 수 있습니다: huggingface-cli login)
"""
#[질문] 남이 미리 학습시킨 모델(백본)을 내 데이터(CIFAR-10)에 어떻게 옮겨 쓸까?' 이게 전이학습(transfer learning)이다.
#[세 가지 전략] 
# ① 선형 프로빙: 백본은 얼리고 마지막 분류층(헤드)만 학습 → 싸고 빠름
# ② Full FT(전체 파인튜닝): 백본까지 전부 조금씩 고침 → 성능 최고, 비용 최대
# ③ LP-FT: ①로 헤드를 먼저 맞춘 뒤 ② → 초반에 백본 특징이 망가지는 것을 막음

# 백본(ResNet-50 · ViT-S · ConvNeXt-T DINOv3)과 전략에 따라 성능 · 학습 파라미터 수가 어떻게 달라지는지 비교한다.
from __future__ import annotations

import argparse
import json
import os

import timm
import torch
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler

from common import (PRETRAINED, build_cifar10_loaders, count_params, ensure_dir, evaluate, get_device,
                    resolve_transforms, set_seed, train_one_epoch)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convnext_tiny.dinov3_lvd1689m")
    p.add_argument("--mode", choices=["probe", "full"], default="probe")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help="기본값: probe 1e-3, full 5e-5")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--layer_decay", type=float, default=0.75, help="full 모드의 LLRD 계수 (1.0 = 없음)")
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=float, default=1.0)
    p.add_argument("--init_head_from", default=None, help="LP-FT: 프로빙 체크포인트에서 헤드 초기화")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="소량 데이터·1 에폭으로 동작 확인")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    lr = args.lr or (1e-3 if args.mode == "probe" else 5e-5)
    out_dir = ensure_dir(args.out_dir or f"runs/lab1_{args.mode}_{args.model.split('.')[0]}")
    print(f"device={device}  model={args.model}  mode={args.mode}  lr={lr}")

    #[설명] timm.create_model(..., pretrained=True, num_classes=10): 사전학습 가중치를 불러오고, 마지막 층만 10개 클래스용 새 층(무작위)으로 바꾼다.
    #drop_path(stochastic depth): Full Fine Tuning 때 블록을 무작위로 건너뛰어 과적합을 줄인다. 프로빙은 백본을 안 고치므로 0.
    #LP-FT: 프로빙 best.pt(같은 백본 + 학습된 헤드)를 통째로 불러와, '헤드가 이미 맞춰진 상태'에서 Full FT 를 시작한다.
    
    # 1) 백본 + 새 헤드
    # 사전학습 모델을 불러오고 마지막 층을 10클래스용으로 교체한다. 
    model = timm.create_model(args.model, pretrained=PRETRAINED, num_classes=10,
                              drop_path_rate=args.drop_path if args.mode == "full" else 0.0)
    if args.init_head_from:
        # 프로빙 체크포인트 = 동일 백본(동결) + 학습된 헤드 → 그대로 로드하면 LP-FT 초기화
        sd = torch.load(args.init_head_from, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        print("LP-FT: 프로빙 체크포인트에서 헤드 초기화 완료")

    # 2) 사전학습 정합 전처리
    # 모델에 맞는 전처리를 만들고 CIFAR-10을 학습,검증,테스트로 나눈다
    cfg, train_tf, eval_tf = resolve_transforms(model)
    print("data config:", {k: cfg[k] for k in ("input_size", "mean", "std", "crop_pct")})

    quick = args.quick
    train_loader, val_loader, test_loader = build_cifar10_loaders(
        args.data_dir, train_tf, eval_tf, batch_size=args.batch,
        train_size=256 if quick else None, val_size=128 if quick else 5000, test_size=256 if quick else None,
        num_workers=0 if quick else 2, seed=args.seed)
    epochs = 1 if quick else args.epochs

    # 3) 학습 전략
    # 프로빙 = 백본 동결 + 헤드만 학습, FULL FT = 전체 학습 + LLRD(층별 학습률 감쇠)
    if args.mode == "probe":
        # TODO(1): 선형 프로빙 — 백본 전체를 동결하고 분류 헤드만 학습 대상으로 두고,
        # «헤드 파라미터만» 넘긴 AdamW 를 optimizer 에 만든다 (lr=lr, weight_decay=0.0).
        # 힌트: model.parameters() 의 requires_grad 를 False 로, model.get_classifier().parameters()
        # 만 True 로. 프로빙은 헤드에 weight decay 를 주지 않는 것이 관례다.
        
        #[설명] 선형 프로빙: requires_grad=False 로 백본 전체를 얼리고(기울기 계산 안 함), 헤드만 다시 True 로 푼다.
        #옵티마이저에도 헤드 파라미터만 넘긴다. weight_decay=0: 헤드가 작아 규제가 필요 없다는 관례.
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(model.get_classifier().parameters(), lr=lr, weight_decay=0.0)
    else:
        # TODO(2): 전체 파인튜닝 + 층별 학습률 감쇠(LLRD) — 모든 파라미터를 해동한 뒤
        #          timm 의 create_optimizer_v2 로 깊이별 lr_scale 이 붙은 파라미터 그룹을 만든다.
        #   힌트: create_optimizer_v2(model, opt="adamw", lr=lr, weight_decay=args.weight_decay,
        #                             layer_decay=...)  — layer_decay 가 1.0 이면 LLRD 를 끄려고
        #         None 을 넘겨야 한다(1.0 을 그대로 넘기면 불필요한 그룹이 생긴다).
        #   참고: lr_scale 은 timm 스케줄러의 step_update 가 실제 lr 에 곱해 준다 (TODO(3) 과 한 쌍).
        
        #[설명] Full FT: 모든 파라미터를 풀어 백본까지 학습한다.
        #LLRD(층별 학습률 감쇠): 출력에 가까운 층은 큰 lr, 입력 쪽(얕은 층)일수록 lr × 0.75^깊이 로 작게 준다.
        #얕은 층은 선·질감 같은 일반 특징이라 거의 그대로 두고, 깊은 층만 새 작업에 맞게 많이 고친다 → 사전학습 지식을 덜 잃는다.
        
        for p in model.parameters():
            p.requires_grad = True
        # layer_decay < 1 이면 깊이별로 lr * layer_decay^(depth) 의 lr_scale 이 붙은 그룹이 생긴다
        optimizer = create_optimizer_v2(model, opt="adamw", lr=lr, weight_decay=args.weight_decay,
                                        layer_decay=args.layer_decay if args.layer_decay < 1 else None)
    trainable, total = count_params(model)
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    #[설명] 학습 중 lr 을 바꾼다. warmup = 처음엔 lr 을 아주 작게 시작해 서서히 올린다
    #초반 큰 기울기로 백본이 망가지는 것을 방지하는 것이다.
    steps_per_epoch = len(train_loader)
    # TODO(3): warmup + cosine 스케줄러를 «step 단위»로 만들어 scheduler 에 담는다.
    #   힌트: timm.scheduler.CosineLRScheduler(optimizer, t_initial=<총 step 수>, lr_min=lr*0.01,
    #         warmup_t=<warmup step 수>, warmup_lr_init=lr*0.01, t_in_epochs=False)
    #         총 step 수 = epochs * steps_per_epoch, warmup step 수 = args.warmup_epochs * steps_per_epoch.
    scheduler = CosineLRScheduler(optimizer, t_initial=epochs * steps_per_epoch, lr_min=lr * 0.01,
                                  warmup_t=int(args.warmup_epochs * steps_per_epoch), warmup_lr_init=lr * 0.01,
                                  t_in_epochs=False)
    model.to(device)
    step_sched = _StepWrapper(scheduler)   # step 단위 스케줄러 (에폭 간 누적)

    #[설명] 에폭마다 학습 → 검증. 검증 macro-F1 이 지금까지 최고면 그 순간의 가중치를 best.pt 로 저장한다.
    # 4) 학습 + 검증 지표 기준 최적 체크포인트 저장 (기존 자료 6단계)
    history, best = [], {"macro_f1": -1}
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, scheduler=step_sched,
                               max_steps=4 if quick else None)
        val = evaluate(model, val_loader, device, max_steps=2 if quick else None)
        history.append({"epoch": epoch + 1, "train_loss": loss, **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {epoch + 1}/{epochs}  train_loss {loss:.4f}  val_acc {val['acc']:.4f}  val_f1 {val['macro_f1']:.4f}")
        # TODO(4): 검증 macro-F1 이 지금까지의 최고치를 넘으면 best 를 갱신하고
        #          out_dir/best.pt 에 model.state_dict() 를 저장한다.
        #   힌트: best 는 {**val, "epoch": epoch + 1} 형태로 두면 아래 출력·result.json 이 그대로 돈다.
        #   주의: «검증» 지표로만 고른다. 테스트 세트는 5)에서 마지막 1회만 쓴다.
        if val["macro_f1"] > best["macro_f1"]:
            best = {**val, "epoch": epoch + 1}
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    #고른 best.pt 로 테스트를 딱 한 번 평가한다. 테스트 점수는 '처음 보는 데이터에서의 실력' 추정치다.
   
    # 5) 최적 가중치로 테스트 1회
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    test = evaluate(model, test_loader, device, max_steps=2 if quick else None)
    print(f"[TEST] acc {test['acc']:.4f}  macro_f1 {test['macro_f1']:.4f}  (best epoch {best['epoch']})")
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "lr": lr, "trainable_params": trainable, "total_params": total,
                   "history": history, "best_val": best, "test": test}, f, ensure_ascii=False, indent=2)
    print("saved:", out_dir)


#[설명] train_one_epoch 은 scheduler.step() 을 부르는데, timm 스케줄러는 step_update(번호) 방식이라 둘을 이어 주는 어댑터다.
#step_update(0) 을 먼저 불러야 첫 배치부터 warmup 시작 lr(lr × 0.01)과 LLRD 배율이 적용된다.
class _StepWrapper:
    """timm 스케줄러(step_update)를 train_one_epoch 의 scheduler.step() 인터페이스에 맞춤."""

    def __init__(self, sched):
        self.sched, self.t = sched, 0
        sched.step_update(0)   # 첫 스텝부터 warmup·LLRD(lr_scale)가 반영되도록 초기 lr을 세팅

    def step(self):
        self.t += 1
        self.sched.step_update(self.t)


if __name__ == "__main__":
    main()