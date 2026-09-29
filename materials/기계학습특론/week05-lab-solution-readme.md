# 5주차 실습 정답 — 퍼셉트론: 학습 규칙 · 논리 게이트 · MLP

**기계학습특론 (2026-2학기)** · 채점자용 문서

- 정답 노트북: `5주차_perceptron_pytorch.ipynb` (실행 결과·그림 포함)
- 문제본: `../problem/5주차_perceptron_pytorch.ipynb` (TODO 15개)
- 검증 환경: Python 3.11 · numpy 2.4.4 · scikit-learn 1.8.0 · torch 2.14.0 · matplotlib 3.10.9
  (CPU, 전체 실행 1분 내외). 정답 노트북은 이 환경에서 **오류 없이 끝까지 실행됨을 확인**했다.

---

## TODO별 정답과 채점 포인트

### TODO(1) — Rosenblatt 학습 규칙 (파트 1, NumPy)
```python
if yi * (w @ xi) <= 0:          # 오분류(또는 경계 위)일 때만 갱신
    w += lr * yi * xi; n_updates += 1; errors += 1
```
- **채점 포인트**: ① 조건이 `<= 0`(등호 포함, 경계 위도 오분류로 본다) ② 오분류 샘플에만 갱신
  ③ 부호가 `+ lr*yi*xi` ④ `n_updates`/`errors` 를 함께 증가.
- **흔한 오답**: 모든 샘플에 대해 갱신(→ 퍼셉트론이 아니라 그냥 SGD 가 됨),
  조건을 `< 0` 으로 써서 $\mathbf{w}=\mathbf{0}$ 초기값에서 첫 스텝이 영원히 일어나지 않음,
  `w = w + ...` 로 새 배열을 만들며 바깥 스코프 변수를 잃는 실수, `yi` 를 곱하지 않고 `-=` 사용.

### TODO(2) — 결정 경계 직선 (파트 1)
```python
boundary_x2 = -(w[0] + w[1]*xs)/w[2]
```
- **채점 포인트**: $w_0 + w_1x_1 + w_2x_2 = 0$ 을 $x_2$ 에 대해 푼 식. 바이어스 $w_0$ 를 빠뜨리지 않는다.
- **흔한 오답**: 부호 누락(`(w[0]+w[1]*xs)/w[2]`), $w_2$ 로 나누지 않고 기울기만 그림,
  `w[1]/w[2]` 를 기울기로 쓰면서 부호를 반대로 씀.

### TODO(3) — Novikoff 상한 (파트 1)
```python
R = np.sqrt(3.0)   # |x0|=1, x1,x2 ∈[-1,1] → ||x|| ≤ sqrt(3)
rows.append((g, np.mean(ups), (R/g)**2))
```
- **채점 포인트**: ① 바이어스 성분 $x_0=1$ 을 포함해 $R=\sqrt{3}$ ② 상한이 $(R/\gamma)^2$
  ③ 시드 10개의 **평균** 갱신 횟수를 기록.
- **해석 채점**: 실측값이 항상 상한보다 작고, 마진이 절반이 되면 갱신 횟수가 대략 4배 쪽으로
  증가($1/\gamma^2$ 경향)한다는 설명이 있으면 가점. 상한은 최악의 샘플 순서에 대한 것이라 느슨하다.
- **흔한 오답**: $R=\sqrt{2}$ (바이어스 성분을 빼먹음), 상한을 $R/\gamma$ 로 씀,
  `np.max(ups)` 를 기록.

### TODO(4) — 진리표 → 퍼셉트론 레이블, 게이트 학습 (파트 2)
```python
y_pm = np.where(t==1, 1, -1)
w, n_upd, hist = perceptron_train(Xb_gate, y_pm, max_epochs=100)
```
- **채점 포인트**: 출력 표현 $\{0,1\}$ 과 학습 레이블 $\{-1,+1\}$ 을 구분했는가.
- **흔한 오답**: `t` 를 그대로 넘김(0 레이블은 `yi*(w@xi)` 를 항상 0 으로 만들어 학습이 망가진다).

### TODO(5) — 게이트 출력 예측 (파트 2)
```python
pred = (Xb_gate @ w > 0).astype(int)
```
- **채점 포인트**: 계단 함수(임계값 0) 적용 후 다시 $\{0,1\}$ 로 되돌림. `Xb_gate`(바이어스 열 포함) 사용.
- **기대 결과**: AND 1.00, OR 1.00, **XOR 0.50** — XOR 정확도가 1.00 이면 구현 오류다.
- **흔한 오답**: `X_gate`(바이어스 없음)를 써서 차원 불일치, `>=0` 을 써서 XOR 이 우연히 높게 나옴.

### TODO(6) — sklearn 파이프라인 (파트 3)
```python
clf = make_pipeline(StandardScaler(), Perceptron(tol=1e-3, random_state=0))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```
- **채점 포인트**: ① 표준화를 **파이프라인 안에** 넣어 테스트 누수를 막았는가
  ② `random_state` 고정 ③ 다중 클래스는 sklearn 이 자동 one-vs-rest 로 처리한다는 이해.
- **기대 결과**: 테스트 정확도 **0.940**.
- **흔한 오답**: 전체 `X` 에 `StandardScaler().fit_transform` 을 먼저 적용(데이터 누수),
  스케일링 생략(정확도 하락·불안정), `clf[-1].coef_` 대신 `clf.coef_` 접근(파이프라인에는 없음).

### TODO(7) — PyTorch 학습 루프 (파트 4)
```python
logits = model(X)
loss = F.binary_cross_entropy_with_logits(logits, y)
opt.zero_grad(); loss.backward(); opt.step()
```
- **채점 포인트**: ① 순전파 → 손실 → `zero_grad` → `backward` → `step` 순서
  ② 모델이 **로짓**을 내고 손실이 `*_with_logits` 라 시그모이드를 따로 적용하지 않음.
- **흔한 오답**: `zero_grad()` 누락(기울기 누적), `step()` 을 `backward()` 앞에 둠,
  `sigmoid` 를 적용한 값을 `binary_cross_entropy_with_logits` 에 넣어 이중 적용,
  `loss.item()` 으로 backward 시도.

### TODO(8) — XOR 용 2층 MLP (파트 4)
```python
mlp = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1))
```
- **채점 포인트**: 은닉 뉴런 2개 + **비선형** 활성화 + 출력 1. 활성화가 없으면 층을 쌓아도
  선형 변환의 합성이라 여전히 XOR 을 못 푼다 — 이 설명을 쓰면 가점.
- **기대 결과**: loss ≈ 1e-4, acc 1.00. 단층은 loss 0.693(= ln 2), acc 0.50~0.75 에 머문다.
- **흔한 오답**: `nn.Sequential(nn.Linear(2,2), nn.Linear(2,1))`(활성화 없음),
  출력에 `nn.Sigmoid()` 를 붙이고 `*_with_logits` 손실을 그대로 사용, 은닉 폭을 1로 둠
  (은닉 1개로는 XOR 이 안 된다 — 노트북 말미 «과제(b)» 와 연결).

### TODO(9) — 격자 예측 확률 (파트 4)
```python
with torch.no_grad(): zz = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)
```
- **채점 포인트**: `torch.no_grad()` 사용, 로짓 → 확률 변환, `xx` 모양으로 reshape.
- **흔한 오답**: `no_grad` 없이 `.numpy()` 호출 → *"Can't call numpy() on Tensor that requires grad"*,
  시그모이드를 빼먹어 0.5 등고선이 엉뚱한 위치에 그려짐.

### TODO(10) — 은닉층 출력 (파트 4)
```python
with torch.no_grad(): H = torch.tanh(mlp[0](Xt)).numpy()
```
- **채점 포인트**: 첫 Linear 통과 후 **활성화까지** 적용. `mlp[0]` 이 $W_1\mathbf{x}+\mathbf{b}_1$.
- **해석 채점**: 출력 `H` 에서 XOR 의 두 클래스가 선형 분리 가능하게 재배치되었음을 지적하면 가점
  (은닉층 = 학습된 특징 변환, 출력층 = 그 공간의 퍼셉트론).
- **흔한 오답**: 활성화 누락(선형 변환만으로는 분리 불가), `mlp(Xt)` 로 최종 출력을 찍음.

### TODO(11) — digits 미니배치 학습 루프 (파트 4-b)
```python
loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
opt.zero_grad(); loss.backward(); opt.step()
```
- **채점 포인트**: 다중 클래스이므로 `cross_entropy`(로짓 입력, 정수 레이블),
  배치 인덱싱 `Xtr[idx]`/`ytr[idx]` 일치.
- **흔한 오답**: 레이블을 원-핫으로 바꿔 넣음, `log_softmax` 를 적용한 뒤 다시 `cross_entropy`,
  `zero_grad()` 를 안쪽 루프 밖에 둠.

### TODO(12) — 테스트 정확도 (파트 4-b)
```python
return (model(Xte).argmax(1) == yte).float().mean().item()
```
- **채점 포인트**: `argmax(dim=1)`, 불리언 → float 평균, `.item()` 으로 파이썬 float 반환.
  이미 `with torch.no_grad():` 블록 안이라는 점(평가 시 기울기 불필요).
- **흔한 오답**: `argmax(0)`, `.sum()` 만 하고 개수로 안 나눔, `model.eval()` 없이 Dropout 이 켜진
  상태로 평가(본 실습 규모에서는 영향이 작지만 감점 사유로 언급할 것).

### TODO(13) — digits 용 2층 MLP (파트 4-b)
```python
acc_mlp = fit_classifier(nn.Sequential(nn.Linear(64,128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128,10)))
```
- **채점 포인트**: 입력 64 → 은닉 128 → 출력 10, 활성화 뒤에 Dropout, 마지막 층에 softmax 를
  붙이지 않음(`cross_entropy` 가 로짓을 받는다).
- **기대 결과**: 선형 0.960 / MLP 0.971 내외 — 은닉층 하나로 약 +1%p.
- **흔한 오답**: 출력층 뒤에 Dropout 또는 활성화 추가, 차원 불일치(128→10 이 아닌 64→10).

### TODO(14) — FFN 파라미터 비율 (파트 5)
```python
n_ffn = sum(p.numel() for n,p in layer.named_parameters() if 'linear' in n)
n_all = sum(p.numel() for p in layer.parameters())
```
- **기대 결과**: **66.2%** (`d_model=64`, `dim_feedforward=256`).
- **채점 포인트**: 이름 필터가 `linear1`/`linear2` 만 잡고 어텐션의 `out_proj` 는 제외되는지
  (`'linear' in n` 은 `self_attn.out_proj` 를 잡지 않는다). 가중치와 바이어스를 모두 세는지.
- **개념 채점**: $d_{ff}=4d$ 일 때 FFN 파라미터 $8d^2$ 대 어텐션 $4d^2$ → 약 2/3 라는 손계산을
  덧붙이면 가점. 슬라이드의 "GPT 계열 파라미터의 약 2/3" 와 일치.
- **흔한 오답**: `named_parameters()` 대신 `parameters()` 를 쓰며 이름 필터가 동작하지 않음,
  LayerNorm 을 FFN 에 포함, 바이어스 제외.

### TODO(15) — FFN 뉴런 발화 (파트 5)
```python
h = F.relu(layer.linear1(x))
```
- **채점 포인트**: `linear1` 만 통과 후 ReLU(= 256개의 퍼셉트론). `layer(x)` 전체를 통과시키면 안 된다.
- **기대 결과**: 토큰별 활성 비율이 대략 0.4~0.5 (무작위 초기화이므로 절반 근처).
- **흔한 오답**: `layer(x)` 사용, ReLU 누락, `linear2` 까지 통과.

---

## 총평 채점 기준(제안)
| 구분 | 비중 | 내용 |
|---|---|---|
| 구현 정확성 | 70% | 15개 TODO 가 기대 결과를 재현하는가 |
| 실행 완결성 | 15% | Restart & Run All 로 끝까지 실행되고 그림·출력이 저장되었는가 |
| 해석 | 15% | XOR 실패의 이유(표현력 한계), 은닉층의 역할, 마진-수렴 관계, FFN=MLP 를 설명했는가 |

## 자주 나오는 «개념» 오답
1. "XOR 은 에폭을 더 돌리거나 학습률을 키우면 풀린다" → **아니다.** 단층 퍼셉트론에는 해 자체가 없다
   (Minsky & Papert, 1969). 손실이 볼록해도 최적해의 정확도가 0.75 를 넘지 못한다.
2. "학습률 $\eta$ 를 키우면 더 빨리 수렴한다" → $\mathbf{w}$ 초기값이 0 이면 $\eta$ 는 전체를 스칼라
   배할 뿐이라 예측 부호가 바뀌지 않아 갱신 횟수가 동일하다(노트북 말미 «과제(a)»).
3. "마진이 작을수록 데이터가 어려워서 정확도가 떨어진다" → 정확도가 아니라 **수렴까지의 갱신 횟수**가
   $1/\gamma^2$ 로 늘어나는 것이다. 선형 분리 가능하면 결국 훈련 오류 0 에 도달한다.
4. "트랜스포머는 퍼셉트론과 무관한 새 구조다" → FFN 은 정확히 2층 MLP 이고, 어텐션도 입력에 의존하는
   가중합, 마지막 출력은 softmax 퍼셉트론이다.
