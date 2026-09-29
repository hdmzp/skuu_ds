# 5주차 실습 문제 — 퍼셉트론: 학습 규칙 · 논리 게이트 · MLP

**기계학습특론 (2026-2학기)** · 최승호 (jcn99250@naver.com)

## 1. 파일
| 파일 | 설명 |
|---|---|
| `5주차_perceptron_pytorch.ipynb` | 실습 문제본. 핵심 로직 15곳이 `TODO(n)` 으로 비어 있다. |

별도의 데이터 파일은 없다. digits 데이터는 `sklearn.datasets.load_digits()` 로 내려받고,
나머지 데이터(선형 분리 가능 2차원 점, 논리 게이트 진리표)는 노트북 안에서 직접 생성한다.

## 2. 실행 방법
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install "numpy>=2.0" "scikit-learn>=1.5" matplotlib "torch>=2.2" jupyter
jupyter lab 5주차_perceptron_pytorch.ipynb
```
- Python ≥ 3.10, CPU 로 충분하다(GPU 불필요). 전체 실행 시간은 1분 내외.
- 검증 환경: numpy 2.4.4 · scikit-learn 1.8.0 · torch 2.14.0 · matplotlib 3.10.9.
- Colab 에서는 한글 폰트를 위해 `!apt-get -qq install fonts-nanum` 후 런타임을 재시작하고 첫 셀을 다시 실행한다.
- 셀은 **맨 위부터 순서대로** 실행한다. 뒤 셀은 앞 셀에서 만든 변수(`perceptron_train`, `X_gate`,
  `X_train`, `single`, `mlp` 등)를 그대로 쓴다.

## 3. TODO 목록
| # | 셀(파트) | 무엇을 구현하는가 | 배점(제안) |
|---|---|---|---|
| 1 | 파트 1 · NumPy | `perceptron_train` 안에서 오분류 판정($y_i\,\mathbf{w}^\top\mathbf{x}_i \le 0$)과 갱신 $\mathbf{w}\leftarrow\mathbf{w}+\eta y_i\mathbf{x}_i$ | 12 |
| 2 | 파트 1 · NumPy | 학습된 $\mathbf{w}$ 에서 결정 경계 직선의 $x_2$ 좌표(`boundary_x2`) 계산 | 5 |
| 3 | 파트 1 · NumPy | 반지름 $R$ 과 Novikoff 상한 $(R/\gamma)^2$ 을 구해 `rows` 에 추가 | 8 |
| 4 | 파트 2 · 논리 게이트 | 진리표 $\{0,1\}$ → 레이블 $\{-1,+1\}$ 변환 후 게이트별 퍼셉트론 학습 | 7 |
| 5 | 파트 2 · 논리 게이트 | 계단 함수로 게이트 출력 `pred` 예측(정확도 검증용) | 5 |
| 6 | 파트 3 · sklearn | `StandardScaler` + `Perceptron(tol=1e-3, random_state=0)` 파이프라인 구성·학습·예측 | 6 |
| 7 | 파트 4 · PyTorch | 학습 한 스텝: 순전파 → `binary_cross_entropy_with_logits` → `zero_grad` → `backward` → `step` | 12 |
| 8 | 파트 4 · PyTorch | XOR 을 푸는 2층 MLP (2→2→1, tanh) 정의 | 7 |
| 9 | 파트 4 · PyTorch | 격자 전체에 대한 예측 확률 `zz` 계산(결정 경계 시각화용) | 6 |
| 10 | 파트 4 · PyTorch | 은닉층 출력 $\mathbf{h}=\tanh(W_1\mathbf{x}+\mathbf{b}_1)$ 계산 | 6 |
| 11 | 파트 4-b · PyTorch | digits 미니배치 학습 루프(교차 엔트로피) | 10 |
| 12 | 파트 4-b · PyTorch | 테스트 정확도 계산 및 반환 | 5 |
| 13 | 파트 4-b · PyTorch | 2층 MLP (64→128→10, GELU + Dropout(0.1)) 정의·학습 | 6 |
| 14 | 파트 5 · 트랜스포머 | `TransformerEncoderLayer` 의 FFN 파라미터 수와 전체 대비 비율 계산 | 8 |
| 15 | 파트 5 · 트랜스포머 | FFN 첫 선형층 + ReLU 통과(`h`)로 뉴런 발화 비율 확인 | 7 |
| | | **합계** | **110** |

> 배점은 제안값이다(100점 만점 + 보너스 10점으로 운영해도 좋다).

## 4. 제출 형식
1. 15개 TODO 를 모두 채우고 `Kernel → Restart & Run All` 로 **처음부터 끝까지 실행한** 노트북
   (`.ipynb`, 그림·출력 포함)을 제출한다.
2. 노트북 맨 끝의 «과제(선택)» 중 최소 1개에 대한 답변을 마크다운 셀로 추가한다.
3. 파일명: `5주차_퍼셉트론_학번_이름.ipynb`

## 5. 자가 점검
- **문법 검사** (제출 전 필수)
  ```bash
  python - <<'PY'
  import json
  nb = json.load(open('5주차_perceptron_pytorch.ipynb', encoding='utf-8'))
  for i, c in enumerate(nb['cells']):
      if c['cell_type'] == 'code':
          compile(''.join(c['source']), f'cell{i}', 'exec')
  print('문법 OK')
  PY
  ```
- **`NotImplementedError` 가 하나도 남아 있지 않아야 한다.** 노트북 전체 검색으로 확인한다.
- **기대 결과** (시드 고정이므로 값이 거의 그대로 나와야 한다)

  | 확인 항목 | 기대값 |
  |---|---|
  | 파트 1, 마진 0.3 | 갱신 4회, 2 에폭 만에 수렴 |
  | 파트 1, 마진 실험 | 실측 갱신 횟수가 항상 $(R/\gamma)^2$ 상한보다 작고, 마진이 줄수록 증가 |
  | 파트 2 | AND 1.00 / OR 1.00 / **XOR 0.50** (XOR 은 못 푸는 게 정답이다) |
  | 파트 3 | digits 테스트 정확도 0.940 |
  | 파트 4 | 단층 loss 0.693·acc ≤ 0.75, MLP loss ~1e-4·acc 1.00 |
  | 파트 4 은닉 출력 | 네 입력의 $\mathbf{h}$ 가 XOR 을 선형 분리 가능하도록 재배치됨 |
  | 파트 4-b | 선형 0.960 / MLP 0.971 내외 |
  | 파트 5 | FFN 파라미터 비율 ≈ 66% |

- 파트 2 에서 XOR 정확도가 1.00 이 나온다면 구현이 틀린 것이다(단층 퍼셉트론은 XOR 을 풀 수 없다).
