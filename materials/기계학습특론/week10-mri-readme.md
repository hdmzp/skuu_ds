# 10주차 실습 — 뇌 MRI 종양 분류 (PyTorch 개정판)

## 프로젝트

* 뇌 MRI 이미지를 4개 클래스로 분류: `glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor`
* 입력: MRI 축상면 이미지 (해상도 제각각, 대부분 512×512)
* 데이터 출처: [Brain Tumor Classification (MRI) — Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data)

## 폴더 구조

```
1.mri/
├── data/
│   ├── Training/   {glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor}   총 2,870장
│   └── Testing/    {동일한 4개 폴더}                                              총   394장
├── 10주차_딥러닝분류_실습_PyTorch.ipynb   ← 강의용 메인 노트북 (권장)
├── main_pytorch.py                        ← 명령줄 실행용 스크립트
├── main.py                                ← 기존 TensorFlow 코드 (참고용, 실행 비권장)
├── requirements.txt                       ← PyTorch 환경
└── requirements_tf_legacy.txt             ← 기존 TensorFlow 환경 (보관용)
```

## 환경 구축

```bash
conda create -n mlspecial python=3.11
conda activate mlspecial

# GPU (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU 전용이면
# pip install torch torchvision

pip install -r requirements.txt
```

설치 확인:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 실행

### 1) 노트북 (권장)

```bash
jupyter lab            # 또는 VS Code 에서 .ipynb 열기
```

`10주차_딥러닝분류_실습_PyTorch.ipynb` 를 **`1.mri/` 폴더 안에서** 실행합니다
(노트북이 `./data` 를 상대경로로 참조합니다).

### 2) 명령줄 스크립트

```bash
# 전이학습 (권장 — 가장 좋은 성능)
python main_pytorch.py --model resnet50 --pretrained --epochs 20

# 기준선 CNN
python main_pytorch.py --model simplecnn --epochs 30

# 직접 구현한 ResNet-18 (사전학습 없이, 불균형 처리)
python main_pytorch.py --model resnet18 --epochs 30 --balanced

# 데이터 누수의 크기 확인: 같은 모델을 두 프로토콜로
python main_pytorch.py --model resnet50 --pretrained --epochs 20 --protocol original
python main_pytorch.py --model resnet50 --pretrained --epochs 20 --protocol repool

# seed 를 바꿔 3회 반복 (보고는 평균 ± 표준편차로)
for s in 0 1 2; do python main_pytorch.py --model resnet50 --pretrained --epochs 20 --seed $s; done
```

실행할 때마다 `results_summary.csv` 에 한 줄씩 누적되므로, 그대로 ablation 표로 쓸 수 있습니다.

주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | `resnet50` | `simplecnn`, `resnet18`, `resnet50`, `convnext_tiny` |
| `--pretrained` | off | ImageNet 사전학습 가중치 사용 (첫 실행 시 다운로드) |
| `--epochs` | 20 | 학습 에폭 |
| `--balanced` | off | 역빈도 손실 가중치 + 균형 샘플링 |
| `--no-augment` | off | 데이터 증강 끄기 (ablation 용) |
| `--protocol` | `repool` | 분할 프로토콜: `original` / `drop_test` / `repool` (아래 참조) |
| `--seed` | 42 | 난수 시드 |

---

## ⚠ 이 데이터셋에서 확인된 데이터 누수

노트북 1단계(또는 스크립트 실행 시 자동 출력)에서 재현할 수 있는 결과입니다.

| 항목 | 결과 |
|---|---|
| 파일이 **바이트 단위로 동일** (MD5) — Training ↔ Testing | 88장 |
| **픽셀이 동일** (dHash 16×16, 재인코딩·리사이즈 포함) — Training ↔ Testing | **252장 / 394장 = 64%** |
| 중복 252장의 클래스 | `no_tumor` **105/105 (전부)**, `meningioma` 107/115, `pituitary` 22/74, `glioma` 18/100 |
| 중복을 뺀 '깨끗한' 테스트 세트 | 142장 — glioma 82, pituitary 52, meningioma 8, **no_tumor 0** |
| Training 폴더 **내부** 중복 (dHash) | 103장 |
| Testing 폴더 **내부** 중복 (dHash) | 82장 |
| 전체 3,264장 중 **고유 이미지** | 2,867장 |

무작위로 뽑은 중복 쌍의 픽셀 평균절대오차(MAE)는 **0.00** 이었습니다 — JPEG 재인코딩으로 파일 바이트만
다를 뿐 **같은 사진**입니다. 즉 기존 `main.py` 가 출력하던 테스트 정확도는 **테스트의 64%가 학습에서 이미
본 이미지**인 상태의 수치이며, 특히 `no_tumor` 는 단 한 장도 새로운 이미지가 아닙니다.

### 분할 프로토콜 세 가지 (`--protocol`)

| 값 | 방식 | 테스트 세트 | 의미 |
|---|---|---|---|
| `original` | 기존 `main.py` 와 동일 (중복 방치) | 394장 | 누수를 포함한 **낙관적** 수치 |
| `drop_test` | Testing 에서 픽셀 중복 제거 | 142장 (`no_tumor` 0장) | 4클래스 평가가 **불가능**해짐을 확인하는 용도 |
| `repool` **(기본·권장)** | Training+Testing 을 합쳐 전역 중복 제거 후 층화 재분할 | 2,867장을 대략 70/15/15 로 | 누수 없는 **정직한** 평가 |

`original` 과 `repool` 로 같은 모델을 돌려 나오는 성능 차이가 **이 데이터셋에서 데이터 누수가 만들어낸
허수**입니다. 어떤 프로토콜을 쓰든 보고서에는 그 선택과 근거를 명시해야 하며,
출처가 제공한 분할을 그대로 쓰는 것이 항상 옳지는 않다는 것이 이 단원의 교훈입니다.
`no_tumor` 표본이 어느 프로토콜에서도 적으므로 **부트스트랩 신뢰구간을 반드시 함께 보고**하십시오.

또한 이 데이터셋에는 **환자 ID가 없어 환자 단위 분할이 불가능**합니다. 같은 환자의 인접 슬라이스가
학습·테스트에 나뉘어 들어갔을 가능성을 배제할 수 없으므로, 여기서 얻은 수치는 **교육용 상한값**으로
해석해야 합니다. 논문에 쓸 결과라면 환자 ID가 있는 데이터셋(예: BraTS)에 `GroupKFold` 를 적용하십시오.

---

## 기존 `main.py` 대비 변경 사항

| # | 기존 코드의 문제 | 영향 | 개정판의 처리 |
|---|---|---|---|
| 1 | `X = X / 255.` 후 모델 안에서 다시 `layers.Rescaling(1.0/255)` | 입력이 0~0.0039 로 **이중 축소** → 학습 불안정 | 정규화를 전처리 한 곳에서만 (ImageNet 평균·표준편차) |
| 2 | `EarlyStopping(monitor='val_loss', mode='max')` | 검증 손실이 **커질 때** 좋다고 판단 → 조기 종료가 반대로 동작 | 검증 macro-F1 **최대화** 기준으로 체크포인트 저장 |
| 3 | `LabelEncoder` 를 train/val/test 에 각각 `fit` | 마지막 `fit` 만 남아 의미 없음 | 클래스→인덱스 사상을 한 번만 정의 |
| 4 | 전체 이미지를 `np.array` 로 메모리 적재 | 2,870장 × 256×256×3 float32 ≈ **2.2 GB** | `Dataset` 지연 로딩 + `DataLoader` 병렬 디코딩 |
| 5 | 데이터 증강 없음 | 소규모 의료 영상에서 과적합 | Crop/Flip/Rotation/RandAugment/RandomErasing |
| 6 | `tensorflow-addons` 의존 (개발 종료) | 재현 불가 | scikit-learn 지표 + 직접 구현으로 대체 |
| 7 | Training↔Testing 중복 미점검 | 위의 **252장(64%) 누수** | dHash 기반 검사 + 세 가지 분할 프로토콜 |
| 8 | 캘리브레이션·해석 없음 | 임상 활용 근거 부족 | Temperature scaling·ECE·Grad-CAM 추가 |
| 9 | 검증 비율 40% (`test_size=1-0.6`) | 학습 데이터 감소 | 20% + 5-겹 교차검증 골격 제공 |

## 노트북 구성과 강의자료 대응

| 노트북 단계 | 10주차 강의자료 |
|---|---|
| 1. 데이터 점검 · 누수 확인 | PART 5 (평가 프로토콜과 데이터 누수) |
| 2. Dataset / DataLoader / 증강 | PART 1·3 |
| 3. 기준선 SimpleCNN | PART 2 (CNN 기본 요소) |
| 4. 학습 루프 (AdamW · 코사인 · AMP) | PART 3 (학습 안정화 기법) |
| 5. 평가 (macro-F1 · 혼동행렬 · AUROC) | PART 2·5 |
| 6. ResNet-18 직접 구현 · 잔차 연결 비교 | PART 4 |
| 7. 전이학습 (ResNet-50 / ConvNeXt-T) | PART 4·5 |
| 8. 클래스 불균형 처리 | PART 5 |
| 9. 캘리브레이션 (Temperature scaling · ECE) | PART 5 |
| 10. Grad-CAM 해석 | PART 5 |
| 11. 결과 정리 · 교차검증 · 과제 | PART 6 |

## 제출물

* 실행 결과가 포함된 `.ipynb`
* `results_summary.csv` (ablation 표)
* 2쪽 보고서: 실험 설계, 결과(평균 ± 표준편차, 95% 신뢰구간), 한계

---

문의: 최승호 · jcn99250@naver.com
