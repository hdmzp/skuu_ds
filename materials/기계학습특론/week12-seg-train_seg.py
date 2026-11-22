"""DeepLabV3+ / U-Net / SegFormer(MiT) 로 CAD 도면 요소 분할 학습 (PyTorch 2.x).  [실습 문제본]

기존 building_segmentation.py (TensorFlow/Keras, ResNet-50 DeepLabV3+ 직접 구현) 의 PyTorch 판.
동일 구조(ResNet-50 conv4 → ASPP → 저수준 특징 concat)를 segmentation_models_pytorch 로 생성하고,
  - 손실: CrossEntropy + Dice (nnU-Net 기본 조합)
  - 지표: mIoU (배경 제외 옵션), 클래스별 IoU
  - 혼합 정밀도(AMP), OneCycleLR, 최고 mIoU 체크포인트 저장, 학습 곡선 PNG
를 추가한다.

사용 예
  python train_seg.py --data data_annotated2 --arch deeplabv3plus --encoder resnet50 --epochs 30
  python train_seg.py --data data_annotated2 --arch unet --encoder resnet34
  python train_seg.py --data data_annotated2 --arch segformer --encoder mit_b2 --lr 6e-5

────────────────────────────────────────────────────────────
이 파일의 TODO
  TODO(1) build_model          : arch 별 smp 모델 생성 (+ mit_* 인코더 사용 제한 검사)
  TODO(2) DiceCELoss.__init__  : CE 손실과 Dice 손실 객체 준비
  TODO(3) DiceCELoss.forward   : 두 손실의 합 반환
  TODO(4) 평가 지표 생성        : mIoU(매크로 평균) 와 클래스별 IoU
  TODO(5) 학습 한 스텝          : AMP 순전파 → 손실 → zero_grad → backward → 클리핑 → step → 스케줄러
  TODO(6) 검증 한 스텝          : 손실 누적 + argmax 예측으로 지표 갱신
────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex

from dataset import CadSegDataset, list_pairs, read_class_names, split_pairs


def build_model(arch: str, encoder: str, num_classes: int) -> nn.Module:
    """arch/encoder 조합으로 분할 모델을 만든다. num_classes 는 배경 포함 8.

    arch: "deeplabv3plus" | "unet" | "unetplusplus" | "segformer"
    반환: nn.Module — 입력 (B,3,H,W) → 출력 로짓 (B, num_classes, H, W)
    """
    # TODO(1): MiT(SegFormer) 인코더는 팽창 합성곱(dilated / output_stride 8·16) 모드를 지원하지 않으므로
    #          arch 가 deeplabv3plus(또는 deeplabv3) 인데 encoder 가 mit_* 이면 ValueError 를 낸다.
    #          그 다음 arch 에 따라 smp.DeepLabV3Plus / smp.Unet / smp.UnetPlusPlus / smp.Segformer 를 반환한다.
    #   힌트: 공통 인자 kw = dict(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=num_classes)
    #   힌트: mit_b0~mit_b5 백본은 smp.Segformer 로만 쓸 수 있다 (강의 슬라이드 45 참고).
    #[설명] DeepLabV3+ 는 인코더 뒷단을 팽창(dilated) 합성곱으로 바꿔 output_stride 16 을 유지하고 ASPP 를 붙이는 구조다.
    #[설명] MiT 는 트랜스포머(겹치는 패치 임베딩) 인코더라 팽창 모드가 없으므로 조합 자체를 미리 막는다.
    if arch in ("deeplabv3plus", "deeplabv3") and encoder.startswith("mit_"):
        raise ValueError(f"{encoder} 인코더는 dilated 모드를 지원하지 않아 {arch} 에 쓸 수 없습니다. --arch segformer 를 쓰세요.")
    #[설명] encoder_weights="imagenet" 은 인코더만 ImageNet 사전학습 가중치로 시작하고 디코더는 무작위로 초기화한다(전이학습).
    kw = dict(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=num_classes)
    models = {
        "deeplabv3plus": smp.DeepLabV3Plus,
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "segformer": smp.Segformer,
    }
    if arch not in models:
        raise ValueError(f"지원하지 않는 arch: {arch}")
    #[결과] unet/resnet34 출력 (2, 8, 256, 256), deeplabv3plus + mit_b2 는 ValueError. 파라미터 수는 DeepLabV3+ ResNet-50 26.7M,
    #[결과] SegFormer MiT-B2 24.7M, DeepLabV3+ EfficientNet-B3 11.7M 이다.
    return models[arch](**kw)


class DiceCELoss(nn.Module):
    """L = CE + Dice. ignore_index 픽셀은 두 항 모두에서 제외."""

    def __init__(self, ignore_index: int | None = None, class_weights=None):
        super().__init__()
        # TODO(2): self.ce 와 self.dice 를 만든다.
        #   힌트: nn.CrossEntropyLoss(weight=class_weights, ignore_index=...)  — None 이면 -100 을 쓴다.
        #   힌트: smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
        #[설명] CE 는 화소마다 맞는 클래스 확률을 높이는 손실이라 넓은 배경이 손실을 지배하기 쉽다.
        #[설명] Dice 는 클래스마다 예측·정답 영역의 겹침 비율(2|A∩B|/(|A|+|B|))을 보므로 작은 클래스도 같은 비중을 갖는다.
        #[목적] 두 손실을 더해 CE 의 안정적인 기울기와 Dice 의 클래스 불균형 보정을 함께 얻는다.
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100 if ignore_index is None else ignore_index)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)

    def forward(self, logits, target):
        """logits (B,K,H,W), target (B,H,W) int64 → 스칼라 손실."""
        # TODO(3): CE 와 Dice 를 더해 반환한다 (nnU-Net 기본 조합).
        #[설명] 두 손실 모두 로짓(소프트맥스 전 값)을 받는다. CE 는 내부에서 log_softmax, Dice 는 softmax 를 적용한다.
        #[결과] 자가 점검 ③: 무작위 로짓 손실 2.99, 정답을 one-hot×20 로 준 완벽 예측 손실 0.0.
        return self.ce(logits, target) + self.dice(logits, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_annotated2")
    p.add_argument("--classes", default="class_names.txt")
    p.add_argument("--mask_sub", default="SegmentationClassPNG3", help="마스크 하위 폴더 (SAM pseudo-label: SegmentationClassSAM)")
    p.add_argument("--arch", default="deeplabv3plus", choices=["deeplabv3plus", "unet", "unetplusplus", "segformer"])
    p.add_argument("--encoder", default="resnet50")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--ignore_bg_in_miou", action="store_true", help="mIoU 계산 시 배경(0) 제외")
    p.add_argument("--out", default="runs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = read_class_names(args.classes)
    K = len(class_names)
    run_dir = os.path.join(args.out, f"{args.arch}_{args.encoder}_{time.strftime('%m%d_%H%M')}")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "args.json"), "w"), indent=2, ensure_ascii=False)

    pairs = list_pairs(args.data, mask_sub=args.mask_sub)
    tr_pairs, va_pairs = split_pairs(pairs, args.val_ratio, args.seed)
    print(f"device={dev}  classes={K} {class_names}  train={len(tr_pairs)}  val={len(va_pairs)}")
    train_dl = DataLoader(CadSegDataset(tr_pairs, args.size, True), batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(CadSegDataset(va_pairs, args.size, False), batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(args.arch, args.encoder, K).to(dev)
    criterion = DiceCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.1)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # TODO(4): 평가 지표 두 개를 만든다.
    #   miou          : 클래스 평균 IoU. args.ignore_bg_in_miou 가 True 면 배경(0) 을 제외한다.
    #   iou_per_class : 클래스별 IoU (평균하지 않음).
    #   힌트: MulticlassJaccardIndex(num_classes=K, ignore_index=..., average="macro") / average=None
    #[설명] IoU = TP / (TP + FP + FN). macro 는 클래스별 IoU 를 구한 뒤 단순 평균하므로 작은 클래스도 같은 비중이다.
    #[설명] ignore_index=0 이면 정답이 배경인 화소를 계산에서 빼므로, 배경이 대부분인 도면에서 mIoU 가 부풀려지는 것을 막는다.
    #[설명] 지표 객체는 update 로 혼동행렬을 에폭 동안 누적하고 compute 로 한 번에 계산하므로 배치 평균보다 정확하다.
    miou = MulticlassJaccardIndex(num_classes=K, ignore_index=0 if args.ignore_bg_in_miou else None, average="macro")
    iou_per_class = MulticlassJaccardIndex(num_classes=K, average=None)
    #[결과] 완벽 예측 mIoU 1.0. macro 평균은 예측·정답 어디에도 없는 클래스를 빼고 평균하지만, average=None 은 그 클래스를 0 으로 돌려준다.
    #[결과] 그래서 클래스별 표에서는 데이터에 없는 ef, oo 가 0.0 으로 찍히며, 보고서 표에서는 NaN 으로 바꿔 구분했다.

    hist = {"train_loss": [], "val_loss": [], "val_miou": []}
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            # TODO(5): 한 스텝을 완성한다.
            #   ① torch.autocast("cuda", dtype=torch.float16, enabled=use_amp) 안에서 loss = criterion(model(x), y)
            #   ② opt.zero_grad(set_to_none=True)
            #   ③ scaler.scale(loss).backward()  →  scaler.unscale_(opt)  →  nn.utils.clip_grad_norm_(..., 1.0)
            #   ④ scaler.step(opt) → scaler.update() → sched.step()
            #   주의: 기울기 클리핑 전에 반드시 unscale_ 를 호출해야 스케일된 기울기를 자르지 않는다.
            #[설명] autocast 는 합성곱·행렬곱을 float16 으로 계산해 GPU 메모리와 시간을 줄인다(CPU 에서는 enabled=False 라 그대로 float32).
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            #[설명] float16 은 작은 기울기가 0 으로 사라지기 쉬워 GradScaler 가 손실에 큰 수를 곱한 뒤 역전파한다.
            scaler.scale(loss).backward()
            #[목적] 곱해 둔 배율을 먼저 되돌려야(unscale_) 클리핑 기준 1.0 이 실제 기울기 크기에 적용된다.
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #[설명] scaler.step 은 기울기에 inf/NaN 이 있으면 이번 갱신을 건너뛰고, update 가 배율을 조정한다.
            scaler.step(opt)
            scaler.update()
            #[설명] OneCycleLR 은 배치마다 학습률을 바꾸는 스케줄러라 에폭이 아니라 스텝마다 호출한다.
            sched.step()
            tl += loss.item() * x.size(0)
            n += x.size(0)

        model.eval()
        miou.reset(); iou_per_class.reset()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                # TODO(6): logits = model(x) 로 예측하고
                #          ① vl 에 criterion(logits, y).item() * x.size(0) 을 더하고 vn 을 갱신
                #          ② pred = logits.argmax(1) 로 클래스 인덱스를 얻어 (CPU 로 옮겨)
                #             miou 와 iou_per_class 를 update 한다.
                logits = model(x)
                vl += criterion(logits, y).item() * x.size(0)
                vn += x.size(0)
                #[설명] 채널 축(1)에서 가장 큰 로짓의 위치가 그 화소의 예측 클래스다. (B,K,H,W) → (B,H,W)
                #[설명] 지표 객체는 CPU 에 만들었으므로 예측과 정답도 CPU 로 옮겨 update 한다.
                pred = logits.argmax(1).cpu()
                y_cpu = y.cpu()
                miou.update(pred, y_cpu)
                iou_per_class.update(pred, y_cpu)
                #[결과] Colab T4, 512×512 조각(학습 26 / 검증 6), 30에폭: DeepLabV3+ ResNet-50 최고 val mIoU 0.546 (28에폭, 1.1분),
                #[결과] SegFormer MiT-B2 0.495 (20에폭, 1.3분), DeepLabV3+ EfficientNet-B3 0.492 (30에폭, 1.4분).
        m = miou.compute().item()
        hist["train_loss"].append(tl / n); hist["val_loss"].append(vl / vn); hist["val_miou"].append(m)
        per = {c: round(v, 3) for c, v in zip(class_names, iou_per_class.compute().tolist())}
        print(f"[{epoch:03d}/{args.epochs}] train_loss={tl/n:.4f} val_loss={vl/vn:.4f} val_mIoU={m:.4f}  {per}")
        if m > best:
            best = m
            torch.save({"model": model.state_dict(), "args": vars(args), "class_names": class_names}, os.path.join(run_dir, "best.pt"))

    json.dump(hist, open(os.path.join(run_dir, "history.json"), "w"), indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss (CE+Dice)"); ax[0].legend()
    ax[1].plot(hist["val_miou"]); ax[1].set_title("Validation mIoU")
    for a in ax: a.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "curves.png"), dpi=120)
    print(f"best val mIoU = {best:.4f}  ->  {run_dir}/best.pt")


if __name__ == "__main__":
    main()
