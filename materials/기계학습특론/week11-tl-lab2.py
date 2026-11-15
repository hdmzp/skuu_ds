"""실습 2. PEFT — timm ViT/ConvNeXt 백본에 LoRA · DoRA 적용 (Hugging Face PEFT)
데이터사이언스융합학과 2026711051 박민지

이 파일의 TODO
  TODO(1)  guess_targets — 모델 계열별 LoRA 적용 대상(target_modules) 결정
  TODO(2)  LoraConfig 구성(r · α=alpha_ratio·r · dropout · DoRA/rsLoRA · modules_to_save) + get_peft_model
  TODO(3)  merge_and_unload() 로 W0 + (α/r)·BA 병합 후 추론 시간 측정

argparse · 데이터 로딩 · 학습 루프 · 어댑터 저장 · 표 출력은 완성되어 있다.

예)
  python lab2_peft_lora.py --model vit_small_patch16_224.augreg_in21k_ft_in1k --r 16 --epochs 3
  python lab2_peft_lora.py --model vit_small_patch16_224.augreg_in21k_ft_in1k --r 8 --dora --epochs 3
  python lab2_peft_lora.py --model convnext_tiny.dinov3_lvd1689m --r 16 --epochs 3
  python lab2_peft_lora.py --quick

과제: r ∈ {4, 16, 64} × {LoRA, DoRA} 조건으로 실행해 학습 파라미터 수·피크 메모리·정확도·
merge 전/후 추론 시간을 한 표로 정리하세요. 실습 1의 프로빙/Full FT 결과와 같은 표에 두면
전략 스펙트럼(프로빙 → PEFT → Full FT)의 trade-off가 보입니다.

핵심 포인트
  - target_modules: timm ViT는 'qkv','proj','fc1','fc2', ConvNeXt는 'fc1','fc2' (MLP) 등 이름이 모델마다 다름
    → --list_modules 로 확인 후 지정. 'LoRA Without Regret'(2025): 어텐션만이 아니라 MLP까지 전 층 적용.
  - modules_to_save: 새로 초기화된 분류 헤드는 LoRA가 아닌 완전 학습 대상.
  - 학습률: Full FT(≈5e-5)의 약 10배(≈5e-4~1e-3)가 LoRA의 적정 범위.
  - merge_and_unload(): W0 + BA 병합 → 추론 비용 증가 0.
"""
#[질문] Full FT 는 너무 비싼데, 아주 적은 파라미터만 학습해 비슷한 효과를 낼 수 없을까? = PEFT(파라미터 효율적 파인튜닝)
#[설명] LoRA: 원래 가중치 W₀ 는 얼려 두고, 옆에 작은 행렬 두 개(B: d×r, A: r×k)를 붙여 행렬곱 BA 만 학습한다. 실제 가중치 = W₀ + 스케일·BA
#      r(rank)이 작을수록 학습할 숫자가 적다. 
#[설명] DoRA: 가중치를 '크기(magnitude)'와 '방향'으로 나눠 방향은 LoRA 로, 크기는 따로 학습 → Full FT 에 더 가깝게 움직이게 한 변형.
#r(4·16·64)과 LoRA/DoRA 에 따라 학습 파라미터 · 어댑터 크기 · 추론 속도 · 성능이 어떻게 바뀌는지 본다.
from __future__ import annotations

import argparse
import json
import os
import time

import timm
import torch
from peft import LoraConfig, get_peft_model
from timm.scheduler import CosineLRScheduler

from common import (PRETRAINED, build_cifar10_loaders, count_params, ensure_dir, evaluate, get_device,
                    resolve_transforms, set_seed, train_one_epoch)

#[설명] target_modules: LoRA 를 '어느 층에' 붙일까? 모델마다 층 이름이 달라 계열별로 정해 둔다.
# ViT: qkv(어텐션 입력) · proj(어텐션 출력) · fc1·fc2(MLP). 'LoRA Without Regret'(2025): 어텐션만이 아니라 MLP 까지 붙여야 Full FT 에 가깝다.
#정규식을 blocks\..*\. 로 시작하게 해 이미지를 조각내는 patch_embed.proj 는 빼고 블록 안의 층만 고른다.
# 모델 계열별 기본 target_modules (선형층 이름). TODO(1) 에서 채운다.
# 필요 시 --target_modules 로 덮어쓸 수 있다.
DEFAULT_TARGETS: dict = {
    "vit": r"blocks\..*\.(qkv|proj|fc1|fc2)",   # 어텐션 + MLP 전 층 (정규식; patch_embed.proj 제외)
    "convnext": ["fc1", "fc2"],                  # ConvNeXt 블록의 MLP
    "resnet": r"layer\d+\..*\.conv\d+",         # Conv2d 도 LoRA 지원 (정규식; stem conv1 제외)
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_small_patch16_224.augreg_in21k_ft_in1k")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha_ratio", type=float, default=2.0, help="lora_alpha = alpha_ratio * r")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--dora", action="store_true", help="DoRA (weight-decomposed) 사용")
    p.add_argument("--rslora", action=argparse.BooleanOptionalAction, default=True,
                   help="스케일 α/√r (rank-stabilized). 기본 켜짐, --no-rslora 로 끔")
    p.add_argument("--target_modules", nargs="*", default=None)
    p.add_argument("--list_modules", action="store_true", help="LoRA 적용 가능한 모듈 이름 출력 후 종료")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def guess_targets(model_name: str):
    """모델 이름에서 LoRA 를 붙일 대상을 고른다.

    반환: PEFT LoraConfig.target_modules 형식 — 모듈 이름 리스트 또는 정규식 문자열.
    """
    # TODO(1): DEFAULT_TARGETS 를 모델 계열별로 채우고, model_name 에 해당 키가 들어 있으면
    #          그 값을, 어디에도 안 맞으면 ViT 기본값을 반환한다.
    #   힌트: 'LoRA Without Regret'(2025) — 어텐션만이 아니라 MLP까지 전 층에 붙인다.
    #         timm ViT      : blocks.<i>.attn.{qkv,proj}, blocks.<i>.mlp.{fc1,fc2}
    #                         → 정규식 r"blocks\..*\.(qkv|proj|fc1|fc2)" 로 쓰면 patch_embed.proj 가 빠진다.
    #         timm ConvNeXt : 블록 MLP 의 fc1 / fc2 (이름 리스트로 충분)
    #         timm ResNet   : Conv2d 도 LoRA 지원 → r"layer\d+\..*\.conv\d+" (stem conv1 제외)
    #   확인법: python lab2_peft_lora.py --model <백본> --list_modules

    #모델 이름에 'vit' · 'convnext' · 'resnet' 이 들어 있으면 그 계열의 대상을 돌려주고, 모르면 ViT 기본값을 쓴다.
    for key, targets in DEFAULT_TARGETS.items():
        if key in model_name:
            return targets
    return DEFAULT_TARGETS["vit"]


@torch.no_grad()
def time_inference(model, loader, device, n_batches: int = 5) -> float:
    model.eval()
    xs = [x for i, (x, _) in enumerate(loader) if i < n_batches]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for x in xs:
        model(x.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / len(xs)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    tag = f"{'dora' if args.dora else 'lora'}_r{args.r}_{args.model.split('.')[0]}"
    out_dir = ensure_dir(args.out_dir or f"runs/lab2_{tag}")

    base = timm.create_model(args.model, pretrained=PRETRAINED, num_classes=10)
    if args.list_modules:
        names = sorted({n.split(".")[-1] for n, m in base.named_modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))})
        print("Linear/Conv2d 모듈 이름(마지막 토큰):", names)
        return
    cfg, train_tf, eval_tf = resolve_transforms(base)

    #[설명] head_name: 분류 헤드 층 이름(ViT 는 'head'). 헤드는 새로 만든 무작위 층이라 LoRA 가 아니라 통째로 학습해야 한다.
    # LoRA / DoRA 설정
    targets = args.target_modules or guess_targets(args.model)
    head_name = [n for n, _ in base.named_modules() if n in ("head", "head.fc", "fc", "classifier")][-1]
 
    # TODO(2): LoraConfig 를 만들어 lora_cfg 에 담고, get_peft_model(base, lora_cfg) 로 model 을 만든다.
    #          - r = args.r,  lora_alpha = args.alpha_ratio * args.r  (α 를 r 의 배수로 재매개화)
    #          - lora_dropout = args.dropout,  target_modules = targets
    #          - use_dora = args.dora,  use_rslora = args.rslora  (α/√r 스케일)
    #          - modules_to_save = [head_name]   ← 새로 초기화된 분류 헤드는 LoRA 가 아니라 «완전 학습»
    #   힌트: lora_alpha 는 int 여야 한다. modules_to_save 를 빼먹으면 헤드가 동결되어 학습이 전혀 안 된다.
 
    #[설명] r: LoRA 행렬의 폭(rank). lora_alpha = 2·r: 업데이트 크기를 정하는 배율. α 를 r 의 배수로 두면 r 을 바꿔도 α 를 따로 맞출 일이 줄어든다.
    # use_rslora: 스케일을 α/r 대신 α/√r 로 → r 이 커져도 업데이트가 너무 작아지지 않는다(rank-stabilized).
    # lora_dropout: LoRA 입력에 dropout 을 걸어 과적합을 줄인다. modules_to_save=[head]: 헤드는 완전 학습 + 어댑터와 함께 저장.
    # r16 ViT-S: 학습 1,183,498개(5.18%). DoRA 는 크기 벡터 41,472개가 더 붙어 1,224,970개(5.35%).
    # rsLoRA · α=2r 에서는 스케일 2√r 이 r4·16·64 에서 4·8·16 → r64 는 같은 lr 에서 업데이트가 커져 학습 손실이 가장 높았다(2.19).
    lora_cfg = LoraConfig(
        r=args.r, lora_alpha=int(args.alpha_ratio * args.r), lora_dropout=args.dropout,
        target_modules=targets, use_dora=args.dora, use_rslora=args.rslora,
        modules_to_save=[head_name],   # 분류 헤드는 완전 학습
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    trainable, total = count_params(model)
    print(f"targets={targets}  head={head_name}  dora={args.dora}")

    quick = args.quick
    train_loader, val_loader, test_loader = build_cifar10_loaders(
        args.data_dir, train_tf, eval_tf, batch_size=args.batch,
        train_size=256 if quick else None, val_size=128 if quick else 5000, test_size=256 if quick else None,
        num_workers=0 if quick else 2, seed=args.seed)
    epochs = 1 if quick else args.epochs

    #[설명] 학습 대상(LoRA 행렬 + 헤드)만 옵티마이저에 넘긴다. 백본 W₀ 는 얼어 있어 기울기 · 옵티마이저 메모리가 거의 들지 않는다.
    # lr 5e-4: Full FT(5e-5)의 약 10배. LoRA 는 학습하는 숫자가 적고 0 에서 시작하므로 큰 lr 이 알맞다.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = epochs * len(train_loader)
    sched = CosineLRScheduler(optimizer, t_initial=steps, lr_min=args.lr * 0.01,
                              warmup_t=max(1, steps // 20), warmup_lr_init=args.lr * 0.01, t_in_epochs=False)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    class _S:
        t = 0

        def step(self):
            self.t += 1
            sched.step_update(self.t)

    step_sched = _S()
    sched.step_update(0)   # 첫 스텝부터 warmup lr이 반영되도록 초기 lr을 세팅
    history, best = [], {"macro_f1": -1}
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, scheduler=step_sched, max_steps=4 if quick else None)
        val = evaluate(model, val_loader, device, max_steps=2 if quick else None)
        history.append({"epoch": epoch + 1, "train_loss": loss, **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {epoch + 1}/{epochs}  train_loss {loss:.4f}  val_acc {val['acc']:.4f}  val_f1 {val['macro_f1']:.4f}")
        #[설명] save_pretrained 는 백본 전체가 아니라 어댑터(LoRA 행렬 + 헤드)만 저장한다.
        if val["macro_f1"] > best["macro_f1"]:
            best = {**val, "epoch": epoch + 1}
            model.save_pretrained(os.path.join(out_dir, "adapter"))   # 어댑터만 저장 (수 MB)
    #[설명] GPU 에서 학습 중 가장 많이 쓴 메모리. CPU 에서는 잴 수 없어 None 이 된다(이번 실습 2 는 CPU 라 None).
    peak_mem_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None

    # 추론: 병합 전/후 시간 비교
    t_unmerged = time_inference(model, test_loader, device)
    # TODO(3): LoRA 가중치를 백본에 병합해 «일반 timm 모델» merged 를 만들고,
    #          같은 방식으로 추론 시간을 재 t_merged 에 담는다.
    #   힌트: peft 모델의 merge_and_unload() 는 W0 + (α/r)·BA 를 원래 가중치에 더한 모델을 돌려준다
    #         → 어댑터 분기가 사라져 추론 비용 증가가 0 이 된다. time_inference(...) 를 재사용하자.
    
    #[설명] merge_and_unload: W₀ + 스케일·BA 를 계산해 원래 가중치에 더하고, LoRA 분기를 없앤 '보통 timm 모델'을 돌려준다.
    #학습 때는 W₀ 와 BA 를 따로 두어야 하지만, 배포(추론) 때는 하나로 합쳐 추가 계산을 0 으로 만든다. 수학적으로 같은 모델이다.
    merged = model.merge_and_unload()            # W0 + (α/r)·BA 병합 → 일반 timm 모델
    t_merged = time_inference(merged, test_loader, device)
    test = evaluate(merged, test_loader, device, max_steps=2 if quick else None)
    adapter_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(out_dir, "adapter")) for f in fs) / 2**20
    print(f"[TEST] acc {test['acc']:.4f}  macro_f1 {test['macro_f1']:.4f}")
    print(f"trainable {trainable:,}/{total:,} ({100 * trainable / total:.2f}%)  adapter {adapter_mb:.1f} MB  "
          f"peak_mem {peak_mem_mb}  infer/batch unmerged {t_unmerged * 1000:.1f} ms → merged {t_merged * 1000:.1f} ms")
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "targets": targets, "trainable_params": trainable, "total_params": total,
                   "adapter_mb": adapter_mb, "peak_mem_mb": peak_mem_mb, "infer_ms_unmerged": t_unmerged * 1000,
                   "infer_ms_merged": t_merged * 1000, "history": history, "best_val": best, "test": test},
                  f, ensure_ascii=False, indent=2)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
