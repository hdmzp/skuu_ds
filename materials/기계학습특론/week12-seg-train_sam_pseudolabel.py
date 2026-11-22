"""SAM(Segment Anything) 으로 pseudo-label 마스크 생성 → train_seg.py 로 재학습.  [실습 문제본]

파이프라인
  1) labelme JSON 의 각 폴리곤(shape) 에서 경계 상자를 얻는다 (또는 labelme 의 rectangle 라벨 사용).
  2) 상자를 SAM 에 프롬프트로 넣어 이진 마스크를 받는다.
  3) 클래스별 마스크를 인덱스 PNG 로 합쳐 SegmentationClassSAM/ 에 저장한다.
  4) python train_seg.py --data data_annotated2 --mask_sub SegmentationClassSAM  (dataset.list_pairs 의 mask_sub 인자)

사람이 폴리곤을 정밀하게 그리지 않고 대략의 상자만 표시해도 학습 마스크를 얻을 수 있어
라벨링 비용을 크게 줄인다(데이터 엔진 개념). 자연영상과 먼 도면 도메인에서는 텍스트 프롬프트(SAM 3)보다
상자 프롬프트가 안정적이다.

  python train_sam_pseudolabel.py --json_dir cad_img_v3 --images data_annotated2/JPEGImages \
         --labels labels.txt --out data_annotated2/SegmentationClassSAM --model facebook/sam-vit-base

SAM 3 를 쓰려면 facebookresearch/sam3 저장소의 API 로 `sam_mask()` 만 교체하면 된다.

────────────────────────────────────────────────────────────
이 파일의 TODO
  TODO(1) SamBoxSegmenter.embed    : 영상 인코더를 이미지당 1회만 실행하고 캐시한다
  TODO(2) SamBoxSegmenter.sam_mask : 상자 프롬프트 → 이진 마스크 + 예측 IoU 점수
  TODO(3) 의사라벨 합성 루프       : 큰 객체부터 칠해 포함 관계를 보존하고, 저품질 마스크는 상자로 대체
────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


def load_labels(path: str) -> dict[str, int]:
    names = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    names = [n for n in names if n != "__ignore__"]
    return {n: i for i, n in enumerate(names)}  # _background_ -> 0


class SamBoxSegmenter:
    def __init__(self, model_id: str = "facebook/sam-vit-base", device: str | None = None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SamModel.from_pretrained(model_id).to(self.dev).eval()
        self.proc = SamProcessor.from_pretrained(model_id)
        self._cache_key = None
        self._emb = None

    @torch.no_grad()
    def embed(self, img: Image.Image, key: str):
        """무거운 영상 인코더는 이미지당 1회만 실행 (SAM 설계 철학)."""
        # TODO(1): key 가 캐시된 것과 다를 때만 이미지를 전처리해 임베딩을 계산하고
        #          self._emb, self._cache_key 를 갱신한다.
        #   힌트: inp = self.proc(img, return_tensors="pt").to(self.dev)
        #   힌트: self.model.get_image_embeddings(inp["pixel_values"])
        #[설명] SAM 은 무거운 ViT 영상 인코더(이미지 1장당 1회)와 가벼운 프롬프트 인코더·마스크 디코더(프롬프트마다)로 나뉜다.
        #[목적] 한 도면에 객체가 수십 개여도 인코더는 한 번만 돌리고 임베딩을 재사용해 시간을 크게 줄인다.
        if key != self._cache_key:
            inp = self.proc(img, return_tensors="pt").to(self.dev)
            self._emb = self.model.get_image_embeddings(inp["pixel_values"])
            self._cache_key = key
        return self._emb

    @torch.no_grad()
    def sam_mask(self, img: Image.Image, box: list[float], key: str) -> tuple[np.ndarray, float]:
        """상자 프롬프트 하나 → (이진 마스크 (H,W) bool, SAM 의 IoU 예측 점수 float)."""
        self.embed(img, key)
        # TODO(2): 상자를 프롬프트로 넣어 마스크를 얻는다.
        #   힌트: inp = self.proc(img, input_boxes=[[box]], return_tensors="pt").to(self.dev)
        #         이미 계산해 둔 임베딩을 쓰므로 inp.pop("pixel_values") 로 화소를 빼고
        #         self.model(image_embeddings=self._emb, multimask_output=False, **inp) 를 호출한다.
        #   힌트: self.proc.image_processor.post_process_masks(
        #             out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
        #         의 [0, 0] 이 원본 해상도 마스크, out.iou_scores[0, 0, 0] 이 품질 점수다.
        #[설명] 프로세서가 상자 좌표를 SAM 입력 크기(긴 변 1024)에 맞게 바꿔 준다. 화소값은 캐시한 임베딩으로 대신하므로 뺀다.
        inp = self.proc(img, input_boxes=[[box]], return_tensors="pt").to(self.dev)
        inp.pop("pixel_values", None)
        #[설명] multimask_output=False 는 후보 3개 대신 마스크 1개만 받는다. 상자 프롬프트는 모호성이 적어 1개로 충분하다.
        out = self.model(image_embeddings=self._emb, multimask_output=False, **inp)
        #[설명] pred_masks 는 256×256 저해상도 로짓이라 post_process_masks 가 원본 크기로 키우고 0 기준으로 이진화한다.
        masks = self.proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
        mask = masks[0, 0].numpy().astype(bool)
        #[설명] iou_scores 는 SAM 이 스스로 예측한 "이 마스크가 정답과 얼마나 겹칠지" 점수로, 품질 필터에 쓴다.
        iou = float(out.iou_scores[0, 0, 0])
        return mask, iou


def shapes_to_boxes(shape: dict) -> list[float]:
    pts = np.array(shape["points"], dtype=float)
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    return [float(x0), float(y0), float(x1), float(y1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", required=True, help="labelme JSON 폴더")
    p.add_argument("--images", required=True, help="원본 이미지 폴더")
    p.add_argument("--labels", default="labels.txt")
    p.add_argument("--out", default="data_annotated2/SegmentationClassSAM")
    p.add_argument("--model", default="facebook/sam-vit-base")
    p.add_argument("--min_iou", type=float, default=0.5, help="SAM 의 IoU 예측이 낮은 마스크는 상자 채움으로 대체")
    args = p.parse_args()

    label2idx = load_labels(args.labels)
    seg = SamBoxSegmenter(args.model)
    os.makedirs(args.out, exist_ok=True)

    for jp in sorted(glob(os.path.join(args.json_dir, "*.json"))):
        meta = json.load(open(jp, encoding="utf-8"))
        stem = Path(jp).stem
        cands = glob(os.path.join(args.images, stem + ".*"))
        if not cands:
            print("이미지 없음:", stem); continue
        img = Image.open(cands[0]).convert("RGB")
        W, H = img.size
        label_map = np.zeros((H, W), dtype=np.uint8)
        # TODO(3): 의사라벨 합성 루프를 완성한다.
        #   ① shapes : meta["shapes"] 를 **넓이 내림차순**으로 정렬한다 (큰 객체 먼저 칠하고
        #      작은 객체를 나중에 덮어써야 포함 관계가 보존된다).
        #      힌트: key=lambda s: -np.prod(np.ptp(np.array(s["points"]), axis=0))
        #   ② 각 shape 에 대해 label2idx 로 클래스 인덱스를 찾고, 없거나 0(배경) 이면 건너뛴다.
        #   ③ shapes_to_boxes(sh) 로 상자를 만들고 seg.sam_mask(img, box, key=stem) 로 (마스크, iou) 를 얻는다.
        #   ④ iou < args.min_iou 면 상자 전체를 채운 마스크로 보수적으로 대체한다.
        #   ⑤ label_map[mask] = 클래스 인덱스
        #[설명] np.ptp 는 최댓값-최솟값이라 점들의 (가로폭, 세로폭)을 주고, 그 곱이 경계 상자 넓이다.
        #[목적] 큰 영역(예: 벽)을 먼저 칠하고 작은 기호(예: 창호)를 나중에 덮어써야 작은 객체가 지워지지 않는다.
        shapes = sorted(meta["shapes"], key=lambda s: -np.prod(np.ptp(np.array(s["points"], dtype=float), axis=0)))
        for sh in shapes:
            cls = label2idx.get(sh["label"])
            if not cls:  # None(목록에 없는 라벨) 또는 0(배경)
                continue
            box = shapes_to_boxes(sh)
            mask, iou = seg.sam_mask(img, box, key=stem)
            if iou < args.min_iou:
                #[설명] SAM 이 자신 없어 하는 마스크는 버리지 않고 상자 전체로 채워, 라벨이 비는 것보다 거칠더라도 위치는 남긴다.
                x0, y0, x1, y1 = (int(round(v)) for v in box)
                mask = np.zeros((H, W), dtype=bool)
                mask[max(y0, 0):min(y1 + 1, H), max(x0, 0):min(x1 + 1, W)] = True
            label_map[mask] = cls
        #[결과] goldensquare_cadf4: 객체 30개 → 클래스 6종. 사람 라벨 대비 IoU 는 iw 0.85, of 0.96, cw 0.90, st 0.94 였지만 ow 는 0.18 이었다.
        #[결과] ow 상자가 건물 전체를 덮어 SAM 이 건물 전체를 마스크로 내고, 그 위에 칠한 방 마스크가 복도와 문 열림 호를 덮지 못해 그 자리가 ow 로 남았다.
        Image.fromarray(label_map).save(os.path.join(args.out, stem + ".png"))
        print(stem, "objects:", len(shapes), "classes:", sorted(set(label_map.flatten().tolist())))


if __name__ == "__main__":
    main()
