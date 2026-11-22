"""학습된 체크포인트로 예측 마스크와 오버레이 시각화 생성.  [실습 문제본]

  python predict.py --ckpt runs/deeplabv3plus_resnet50_xxxx/best.pt --images data_annotated2/JPEGImages --out preds

────────────────────────────────────────────────────────────
이 파일의 TODO
  TODO(1) colorize : 클래스 인덱스 마스크에 팔레트를 적용해 RGB 로 바꾼다
  TODO(2) 추론 후처리 : 로짓을 원본 해상도로 되돌리고 argmax 로 클래스 인덱스 맵을 만든다
────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import build_transforms
from train_seg import build_model

PALETTE = np.array([
    [0, 0, 0], [15, 124, 140], [63, 184, 175], [242, 165, 65], [228, 87, 46], [202, 220, 252], [44, 95, 45],
    [128, 0, 128], [255, 255, 0],
], dtype=np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    """(H,W) 클래스 인덱스 → (H,W,3) uint8 RGB."""
    # TODO(1): PALETTE 를 이용해 인덱스를 색으로 바꾼다.
    #   힌트: 넘파이 팬시 인덱싱 한 줄이면 된다. 팔레트 길이를 넘는 인덱스는 % 로 감싸 안전하게.
    #[설명] PALETTE[(H,W) 정수 배열] 처럼 배열로 인덱싱하면 각 화소 번호가 팔레트의 한 행(RGB)으로 바뀌어 (H,W,3) 이 된다.
    return PALETTE[mask.astype(np.int64) % len(PALETTE)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", required=True, help="이미지 폴더 또는 단일 파일")
    p.add_argument("--out", default="preds")
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev)
    a = ck["args"]
    model = build_model(a["arch"], a["encoder"], len(ck["class_names"])).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    tf = build_transforms(a["size"], train=False)

    paths = [args.images] if os.path.isfile(args.images) else sorted(glob(os.path.join(args.images, "*.*")))
    os.makedirs(args.out, exist_ok=True)
    for pth in paths:
        img = np.array(Image.open(pth).convert("RGB"))
        H, W = img.shape[:2]
        x = tf(image=img)["image"].unsqueeze(0).to(dev)
        with torch.no_grad():
            # TODO(2): 모델 로짓을 얻고 (H, W) 원본 크기로 양선형 보간한 뒤 argmax 로
            #          pred (uint8 numpy, shape (H, W)) 를 만든다.
            #   힌트: torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            #   힌트: 보간은 로짓에 적용한다 — 인덱스 맵을 키우면 계단 현상이 생긴다.
            logits = model(x)  # (1, K, size, size)
            #[목적] 학습 크기(size)로 줄여 예측했으므로 원본 (H, W) 로 되돌린다. 연속값인 로짓을 보간해야 경계가 매끄럽다.
            logits = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            #[결과] 검증 조각 6개에 대해 인덱스 마스크와 오버레이를 저장했다 (preds_human/, preds_sam/).
        stem = Path(pth).stem
        Image.fromarray(pred).save(os.path.join(args.out, f"{stem}_mask.png"))  # 인덱스 PNG (재학습에 사용 가능)
        over = (img * (1 - args.alpha) + colorize(pred) * args.alpha).astype(np.uint8)
        Image.fromarray(over).save(os.path.join(args.out, f"{stem}_overlay.jpg"), quality=90)
        print(stem, {c: int((pred == i).sum()) for i, c in enumerate(ck["class_names"]) if (pred == i).any()})


if __name__ == "__main__":
    main()
