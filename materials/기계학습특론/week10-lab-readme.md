# 10주차 실습 «문제본» — 딥러닝 분류모델 (CNN · PyTorch)

기계학습특론 2026-2학기 · 주제: **뇌 MRI 4-클래스 종양 분류**
(`glioma_tumor` / `meningioma_tumor` / `no_tumor` / `pituitary_tumor`)

| 파일 | 역할 | TODO 개수 |
|---|---|---|
| `10주차_딥러닝분류_실습_PyTorch.ipynb` | 강의용 메인 노트북 (**이것을 제출한다**) | **16** |
| `1.mri/10주차_딥러닝분류_실습_PyTorch.ipynb` | 위와 **완전히 같은 파일** (데이터 옆에서 바로 열 수 있게 둔 사본) | 16 |
| `1.mri/main_pytorch.py` | 명령줄 실행용 스크립트 (선택 과제) | **10** |
| `1.mri/main.py` | 기존 TensorFlow 코드 — **참고용, 손대지 않는다** | – |
| `1.mri/README.md` | 원본 실습 안내 (데이터·실행법) | – |

> 노트북 두 개는 같은 파일이다. **`1.mri/` 안의 것을 열어 풀면** 상대경로 `./data` 가 바로 맞는다.

---

## 1. 실행 방법

### 데이터 준비

용량이 커서 배포본에는 **데이터가 들어 있지 않다.**
기존 실습 zip 의 `data` 폴더를 그대로 쓰거나,
[Kaggle — Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data)
를 내려받아 아래 구조로 둔다.

```
1.mri/
└── data/
    ├── Training/{glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor}   총 2,870장
    └── Testing/ {동일한 4개 폴더}                                              총   394장
```

### 환경

```bash
conda create -n mlspecial python=3.11
conda activate mlspecial

# GPU (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU 전용이면
# pip install torch torchvision

pip install -r 1.mri/requirements.txt
```

패키지 버전은 `1.mri/requirements.txt` 를 따른다
(`torch>=2.2`, `torchvision>=0.17`, `scikit-learn>=1.4`, `pandas>=2.0`,
`matplotlib>=3.8`, `pillow>=10.0`, `timm>=1.0`(선택), `jupyterlab>=4.0`).

### 검증 환경 (정답본이 실제로 돌아간 버전)

Python 3.11.15 · torch 2.14.0 · torchvision 0.29.0 · scikit-learn 1.8.0 ·
pandas 3.0.2 · numpy 2.4.4 · matplotlib 3.10.9

### 실행

```bash
cd 1.mri
jupyter lab            # 또는 VS Code 에서 .ipynb 를 연다
```

GPU 가 없으면 노트북의 `EPOCHS` 를 **3~5** 로 줄인다.
`7. 전이학습` 절은 사전학습 가중치를 인터넷에서 내려받는다(수십~수백 MB).
오프라인이면 그 셀은 `weights=None` 으로 넘어가도록 예외 처리가 되어 있으니 그대로 진행해도 된다.

---

## 2. 푸는 방법

각 `TODO(n)` 은 이런 모양으로 비어 있다.

```python
# TODO(8): 한 배치의 학습 한 걸음을 구현한다. 끝나면 스칼라 텐서 loss 가 있어야 한다.
#   힌트: 순서는 «순전파 -> 손실 -> zero_grad -> backward -> (기울기 클리핑) -> step».
raise NotImplementedError("TODO(8)")
```

- 주석 줄과 `raise` 줄을 지우고 그 자리에 코드를 쓴다.
- **함수 시그니처·docstring·반환 형태 설명은 그대로 둔다.** 무엇을 만들어야 하는지가 거기 적혀 있다.
- TODO 아래 줄부터는 이미 완성된 코드다. **변수 이름과 모양(shape)** 을 설명대로 맞추면 그대로 이어서 돌아간다.
- 번호 순서대로 푸는 것이 가장 쉽다 (뒤 TODO 가 앞 TODO 의 결과를 쓴다).
- 데이터 로딩·증강 파이프라인·시각화·혼동행렬·체크포인트 저장은 **이미 완성되어 있다.** 손대지 않아도 된다.

---

## 3. TODO 목록과 배점 제안

### (1) `10주차_딥러닝분류_실습_PyTorch.ipynb` — 75점

| # | 위치 (절) | 무엇을 구현하는가 | 배점 |
|---|---|---|---|
| 1 | 2-3. Mixup·CutMix | `MixupCutmix.__call__` — λ~Beta(α,α) 로 두 표본과 라벨을 섞고, CutMix 는 **실제로 덮인 면적 비**로 λ 를 다시 계산 | 7 |
| 2 | 3. SimpleCNN | `conv_block` — [3x3 Conv → BN → ReLU] × 2 + `MaxPool2d(2)` | 4 |
| 3 | 3. SimpleCNN | `SimpleCNN.__init__` — conv_block 쌓기(224→14) + GAP 분류 헤드 | 6 |
| 4 | 3-2. 손 계산 | `conv_out_size` — 출력 크기 `⌊(n+2p−d(k−1)−1)/s⌋+1` | 3 |
| 5 | 3-2. 손 계산 | `conv_params` — 파라미터 수 `k·k·C_in·C_out (+C_out)` | 2 |
| 6 | 3-2. 손 계산 | `receptive_field` — 수용영역 재귀 `r_L = r_(L−1) + (k_L−1)·Π s_i` | 5 |
| 7 | 4. 학습 루프 | `make_scheduler` — 워밍업(선형 상승) → 코사인 감쇠 | 5 |
| 8 | 4. 학습 루프 | `train_model` 의 학습 한 걸음 — autocast 순전파·손실·`zero_grad`·`backward`·클리핑·`step` | 8 |
| 9 | 6. ResNet-18 | `BasicBlock.forward` — `y = ReLU(F(x) + shortcut(x))` | 6 |
| 10 | 6. ResNet-18 | `ResNet18._init_weights` — He 초기화 + zero-init residual | 5 |
| 11 | 7. 전이학습 | `param_groups_llrd` — 백본/헤드 층별 학습률 파라미터 그룹 | 5 |
| 12 | 8. 불균형 | 역빈도 손실 가중치 `class_weight` (평균 1) + 균형 샘플링 가중치 `sample_w` | 5 |
| 13 | 8. 불균형 | `FocalLoss.forward` — `(1−p_t)^γ · CE` | 5 |
| 14 | 9. 캘리브레이션 | `fit_temperature` 의 `closure` — NLL 최소화로 온도 T 학습 | 5 |
| 15 | 9. 캘리브레이션 | `expected_calibration_error` — ECE 와 신뢰도 다이어그램용 `stats` | 5 |
| 16 | 10. Grad-CAM | `GradCAM.__call__` — 채널별 기울기 평균 × 활성값 가중합 | 4 |

### (2) `1.mri/main_pytorch.py` — 25점 (선택 과제)

노트북의 TODO 와 내용이 겹친다. **노트북을 먼저 푼 뒤 옮겨 적으면 된다.**

| # | 함수 | 무엇을 구현하는가 | 배점 |
|---|---|---|---|
| 1 | `conv_block` | [3x3 Conv → BN → ReLU] × 2 + MaxPool | 2 |
| 2 | `SimpleCNN.__init__` | conv_block 쌓기 + GAP 헤드 | 3 |
| 3 | `BasicBlock.forward` | 잔차 연결 | 3 |
| 4 | `ResNet18._init_weights` | He 초기화 + zero-init residual | 2 |
| 5 | `make_scheduler` | 워밍업 → 코사인 | 2 |
| 6 | `param_groups` | 백본/헤드 × decay/no-decay **네** 그룹 | 4 |
| 7 | `train()` 의 학습 한 걸음 | 순전파 → 손실 → zero_grad → backward → step | 4 |
| 8 | `fit_temperature` 의 `closure` | 온도 스케일링 | 2 |
| 9 | `expected_calibration_error` | ECE | 2 |
| 10 | `main()` 의 불균형 처리 | 역빈도 가중치 + `WeightedRandomSampler` | 1 |

완성하면 이렇게 돌아간다.

```bash
cd 1.mri
python main_pytorch.py --model simplecnn --epochs 5
python main_pytorch.py --model resnet50 --pretrained --epochs 20
python main_pytorch.py --model resnet18 --epochs 30 --balanced
python main_pytorch.py --model resnet50 --pretrained --epochs 20 --protocol original   # 누수 포함 비교
```

---

## 4. 자가 점검 방법

TODO 를 채운 뒤 **위에서부터 순서대로** 실행하면 다음이 자동으로 걸러진다.

| 확인 지점 | 통과 조건 |
|---|---|
| 2-3절 Mixup 확인 코드 | `소프트 라벨 각 행의 합` 이 모두 1.0, `assert` 두 개 통과 |
| 3절 SimpleCNN | `출력 모양: torch.Size([2, 4])`, 파라미터 **1,174,244** 개, block1~4 가 112→56→28→14 |
| 3-2절 손 계산 | `assert n == 14` 와 `assert hand == torch_total` 통과, 3x3 두 개의 수용영역 = **5** |
| 4절 학습 | 에폭마다 `loss` 가 줄고 `val_F1` 이 오른다. 손실이 `nan` 이면 TODO(8) 의 순서를 의심할 것 |
| 6절 ResNet-18 | 파라미터 **11,178,564** 개 — torchvision `resnet18` 과 **정확히 같아야 한다** |
| 8절 불균형 | `손실 가중치` 의 합이 4(=클래스 수), `균형 샘플링 배치의 클래스 분포` 가 대체로 고르다 |
| 9절 캘리브레이션 | `정확도(불변)` 두 값이 **같다** (온도는 순위를 바꾸지 않는다). 보통 ECE 가 줄어든다 |
| 10절 Grad-CAM | 열지도가 검은 배경이 아니라 뇌 실질 쪽에 뜬다 |

문법만 먼저 확인하고 싶으면:

```bash
python -m py_compile 1.mri/main_pytorch.py
python - <<'EOF'
import json
nb = json.load(open('10주차_딥러닝분류_실습_PyTorch.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), f'<cell {i}>', 'exec')
print('모든 코드 셀 문법 OK')
EOF
```

---

## 5. 제출 형식

1. **`10주차_딥러닝분류_실습_PyTorch.ipynb`** — 모든 TODO 를 채우고 **끝까지 실행한 결과(출력·그림)가 남아 있는** 상태로 제출한다.
   출력을 지운 노트북은 미제출로 간주한다.
2. **`results_summary.csv`** — 11절에서 자동 저장된다.
3. **2쪽 보고서 (PDF)** — 노트북 마지막 절 «제출 과제» 1~5번을 담는다.
   - ablation 표 (A~G), seed 3개의 평균 ± 표준편차
   - 5-겹 교차검증과 부트스트랩 95% 신뢰구간
   - **누수 분석**: `PROTOCOL='original'` vs `'repool'` 의 macro-F1 차이 (가장 중요)
   - 오분류 10장의 Grad-CAM 과 실패 원인 분류
4. (선택) `1.mri/main_pytorch.py`

**파일명**: `10주차_학번_이름.zip`

> 채점 시 **시드 분산보다 작은 차이를 '개선'이라고 주장한 보고서는 감점**한다.
> 수치 하나가 아니라 «평균 ± 표준편차»와 «비교 조건»을 함께 써야 한다.

---

## 6. 예상 소요시간

| 단계 | 시간 |
|---|---|
| 노트북 TODO 16개 구현 | 2.5 ~ 3.5시간 |
| 전체 실행 (GPU) | 30 ~ 60분 |
| 전체 실행 (CPU, `EPOCHS=3`) | 1.5 ~ 3시간 |
| `main_pytorch.py` TODO 10개 (선택) | 40분 (노트북을 먼저 풀었다면) |
| 보고서 작성 | 1.5 ~ 2시간 |
