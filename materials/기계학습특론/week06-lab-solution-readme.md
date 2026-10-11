# 6주차 실습 정답 해설 — 다층 퍼셉트론(MLP)

문제본(`../problem/`)의 각 `TODO(n)` 에 대한 정답 요약 · 채점 포인트 · 흔한 오답을 정리한다.
정답 코드 자체는 이 폴더의 두 노트북에 그대로 들어 있다.

---

## 실습 1 — XOR (`6주차_MLP_실습1_XOR_PyTorch.ipynb`)

### TODO(1) — NAND·OR·AND 퍼셉트론의 가중치·바이어스
```python
w_nand, b_nand = np.array([-2, -2]), 3
w_or,   b_or   = np.array([ 2,  2]), -1
w_and,  b_and  = np.array([ 1,  1]), -1
```
- **채점 포인트**: 값이 위와 똑같을 필요는 없다. `z > 0` 판정이 각 게이트의 진리표를 모두 만족하면 정답.
  (예: AND 를 `w=[1,1], b=-1.5` 로 두어도 맞다.)
- **흔한 오답**: AND 의 바이어스를 `-2` 로 두면 `(1,1)` 에서 `z = 0` 이 되어 `z > 0` 이 거짓 → 항상 0 출력.
  계단 함수의 부등호가 `>` 인지 `>=` 인지 반드시 확인해야 한다.

### TODO(2) — XOR 합성
```python
XOR = lambda x1, x2: AND(NAND(x1, x2), OR(x1, x2))
```
- **채점 포인트**: 은닉층 두 개(NAND, OR)의 출력이 출력층(AND)의 «입력»이 된다는 구조를 이해했는지.
  출력이 `0, 1, 1, 0` 이면 정답.
- **흔한 오답**: `AND(OR(...), NAND(...))` 는 AND 가 대칭이라 그대로 맞지만,
  `OR(NAND, AND)` 처럼 층 역할을 섞으면 틀린다.

### TODO(3) — `forward()`
```python
Z1 = X @ W0 + b0;  H1 = sigmoid(Z1)
Z2 = H1 @ W1 + b1; Y_hat = sigmoid(Z2)
return Z1, H1, Z2, Y_hat
```
- **채점 포인트**: 네 값을 모두 반환해야 한다(역전파에서 `Z1`, `H1` 이 필요). shape `(4,2),(4,2),(4,1),(4,1)`.
- **흔한 오답**: `W0 @ X` 로 쓰면 shape 오류. 이 노트북은 슬라이드의 열 벡터 표기를 **행 벡터 규약(`X @ W`)** 으로 옮긴 것이다.

### TODO(4) — `bce()`
```python
return -np.mean(Y * np.log(Y_hat + eps) + (1 - Y) * np.log(1 - Y_hat + eps))
```
- **채점 포인트**: `mean`(= 1/N)을 썼는지, `eps` 로 `log(0)` 을 막았는지.
- **흔한 오답**: `np.sum` 을 써서 손실이 4배가 된다 → `TODO(5)` 의 `/N` 과 어긋나 autograd 검증이 실패한다.

### TODO(5) — `backward()`
```python
d2  = (Y_hat - Y) / N
dW1 = H1.T @ d2;  db1 = d2.sum(0, keepdims=True)
d1  = (d2 @ W1.T) * dsigmoid(Z1)
dW0 = X.T @ d1;   db0 = d1.sum(0, keepdims=True)
return dW0, db0, dW1, db1
```
- **채점 포인트**: 시그모이드+BCE 결합 미분 `∂L/∂z = (ŷ-y)/N` 을 썼는지,
  상위층 오차에 국소 기울기 `σ'(Z1)` 를 곱하는 연쇄법칙 구조를 썼는지, 바이어스는 배치 축으로 합했는지.
- **흔한 오답**
  - `d2 = (Y_hat - Y) * dsigmoid(Z2) / N` — 시그모이드 미분을 **한 번 더** 곱하는 실수(결합 미분에 이미 포함).
  - `dsigmoid(H1)` 처럼 활성화 «출력»을 넣는다 (정답은 «입력» `Z1`). `H1*(1-H1)` 로 쓰면 맞다.
  - 전치 방향을 뒤집어 `d1 @ X.T` 로 쓴다.

### TODO(6) — autograd 검증
```python
H    = torch.sigmoid(Xt @ tW0 + tb0)
Yp   = torch.sigmoid(H @ tW1 + tb1)
loss = F.binary_cross_entropy(Yp, Yt)
loss.backward()
```
- **채점 포인트**: 네 항목 모두 `✔ 일치`(최대 오차 < 1e-5)가 출력되면 정답.
- **흔한 오답**: `loss.backward()` 를 빠뜨려 `tW0.grad` 가 `None` → `AttributeError`.
  `F.binary_cross_entropy_with_logits` 에 `Yp`(확률)를 넣으면 시그모이드가 두 번 적용돼 불일치.

### TODO(7) — NumPy 경사 하강 루프
```python
Z1, H1, Z2, Y_hat = forward(X, W0, b0, W1, b1)
L = bce(Y, Y_hat); hist.append(L)
dW0, db0, dW1, db1 = backward(X, Y, Z1, H1, Z2, Y_hat, W1)
W0 -= lr * dW0; b0 -= lr * db0; W1 -= lr * dW1; b1 -= lr * db1
```
- **채점 포인트**: 네 파라미터를 **모두** 갱신했는지, 부호가 `-=` 인지. 5000 에폭 후 손실 ≈ 0.01 이하.
- **흔한 오답**: 바이어스 갱신 누락, `+=` 로 경사 «상승», 매 스텝 `forward` 를 다시 부르지 않고 첫 결과를 재사용.

### TODO(8) — `nn.Sequential`
```python
model = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid(), nn.Linear(2, 1))
```
- **채점 포인트**: 파라미터 수 **9** (2·2+2 + 2·1+1). 출력층 뒤에 시그모이드가 **없어야** 한다.
- **흔한 오답**: `nn.Sigmoid` (클래스)를 인스턴스화하지 않고 넣는다 / 출력층 뒤에 `nn.Sigmoid()` 를 붙여
  `BCEWithLogitsLoss` 와 이중 적용 → 학습이 매우 느려진다.

### TODO(9) — PyTorch 학습 루프
```python
logits = model(Xt)
loss = criterion(logits, Yt)
optimizer.zero_grad(); loss.backward(); optimizer.step()
```
- **채점 포인트**: 5단계 순서. 예측 클래스가 `[0, 1, 1, 0]`.
- **흔한 오답**: `zero_grad()` 누락 → 기울기 누적으로 발산.
  `loss.backward()` 를 `optimizer.step()` 뒤에 두는 순서 뒤바뀜.

### TODO(10) — 결정 경계 격자 순전파
```python
with torch.no_grad():
    zz = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)
```
- **채점 포인트**: `torch.no_grad()` 사용, 출력이 로짓이므로 시그모이드를 씌워 확률로 바꿨는지,
  `(200,200)` 로 reshape 했는지.
- **흔한 오답**: `reshape` 없이 `contourf` 에 넘겨 shape 오류. `.detach()` 없이 `.numpy()` 호출 → RuntimeError.

---

## 실습 2 — MNIST (`6주차_MLP_실습2_MNIST_PyTorch.ipynb`)

### TODO(1) — `MLP.__init__` 층 구성
```python
for din, dout in [(in_dim, h1), (h1, h2)]:
    layers.append(nn.Linear(din, dout))
    if use_bn: layers.append(nn.BatchNorm1d(dout))
    layers.append(act())
    if dropout > 0: layers.append(nn.Dropout(dropout))
layers.append(nn.Linear(h2, n_classes))
```
- **채점 포인트**: 파라미터 수 **776,510** 출력. 순서가 `Linear → Norm → 활성화 → Dropout`.
  출력층 뒤에 `Softmax` 가 **없어야** 한다(로짓 반환).
- **흔한 오답**
  - 출력층에 `nn.Softmax(dim=1)` 을 붙인다 → `CrossEntropyLoss` 와 이중 소프트맥스, 기울기가 약해져 정확도 정체.
  - `act` 는 «클래스»이므로 `act()` 로 인스턴스를 만들어야 한다 (`act` 를 그대로 append 하면 TypeError).
  - `nn.Flatten()` 을 빼먹어 `(B,1,28,28)` 이 Linear 에 들어가 shape 오류.

### TODO(2) — He 초기화
```python
for m in self.net:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)
```
- **채점 포인트**: `nn.Linear` 만 골라서 적용했는지(BatchNorm 가중치를 건드리면 안 된다), 바이어스 0.
- **흔한 오답**: `self.net.parameters()` 를 돌며 `kaiming_normal_` 을 적용 → 1차원 바이어스 텐서에서 오류.

### TODO(3) — `evaluate()`
```python
logits = model(xb)
loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
correct += (logits.argmax(1) == yb).sum().item(); total += yb.size(0)
```
- **채점 포인트**: `reduction="sum"` 으로 모아 전체 개수로 나눴는지, `argmax(1)` 로 예측 클래스를 얻었는지.
- **흔한 오답**: 배치 «평균» 손실을 그대로 누적 → 마지막 배치 크기가 다를 때 값이 어긋난다.
  `model.eval()`/`torch.no_grad()` 는 이미 주어져 있으므로 지우지 말 것.

### TODO(4) — 손실·최적화기·스케줄러
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))
```
- **채점 포인트**: `steps_per_epoch=len(train_loader)` — OneCycleLR 은 «스텝» 단위라 총 스텝 수가 맞아야 한다.
- **흔한 오답**: `steps_per_epoch` 를 데이터 «개수»로 준다 / `scheduler.step()` 을 에폭 끝에서만 불러
  총 스텝이 모자라 `ValueError: Tried to step ... times` 가 뜬다.

### TODO(5) — 학습 루프
```python
logits = model(xb)
loss = criterion(logits, yb)
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step(); scheduler.step()
```
- **채점 포인트**: 5단계 순서 + 스케줄러 스텝. `clip_grad_norm_` 은 `backward()` 뒤 `step()` 앞.
  5 에폭 후 검증 정확도 0.97 이상.
- **흔한 오답**: `model.train()` 을 부르지 않는다(셀에 이미 있으므로 지우지 말 것),
  `zero_grad()` 를 `backward()` 뒤에 둔다(기울기가 지워져 학습이 안 된다).

### TODO(6) — 혼동행렬
```python
for t, p in zip(trues, preds): cm[t, p] += 1
```
- **채점 포인트**: `cm.sum() == 10000`, 행이 «실제»·열이 «예측». 대각합 / 전체 = 테스트 정확도와 일치.
- **흔한 오답**: `cm[p, t] += 1` 로 행·열을 바꾼다 → 혼동 패턴 해석이 반대가 된다.

### TODO(7) — 클래스별 재현율
```python
per_class = cm.diagonal() / cm.sum(1)
```
- **채점 포인트**: 분모가 «행 합»(실제 개수)이어야 재현율(recall). 열 합으로 나누면 정밀도(precision)다.
- **흔한 오답**: `cm.sum(0)` 으로 나누고 «클래스별 정확도»라 부른다.

### TODO(8) — 구성 비교 실험 A~D
```python
set_seed(1); m = MLP(act=nn.Sigmoid).to(device)
experiments["A: Sigmoid + SGD"] = train_generic(m, torch.optim.SGD(m.parameters(), lr=0.1))
set_seed(1); m = MLP(act=nn.ReLU).to(device)
experiments["B: ReLU + SGD(momentum)"] = train_generic(m, torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9))
set_seed(1); m = MLP(act=nn.ReLU).to(device); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
experiments["C: ReLU + AdamW + OneCycle"] = train_generic(
    m, opt, torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader)))
set_seed(1); m = MLP(act=nn.GELU, use_bn=True, dropout=0.2).to(device); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
experiments["D: GELU + BN + Dropout + AdamW"] = train_generic(
    m, opt, torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader)))
```
- **채점 포인트**: 네 구성 모두 `set_seed(1)` 로 같은 출발점에서 시작했는지,
  결과 해석에서 «활성화(A→B) · 최적화기(B→C) · 정규화(C→D)» 기여를 분리해 설명했는지.
  3 에폭 기준 A 가 가장 낮고 C·D 가 가장 높게 나오면 정상.
- **흔한 오답**: 옵티마이저를 만들 때 이전 모델의 `parameters()` 를 넘긴다(변수 재사용 실수),
  `.to(device)` 누락으로 CPU/GPU 텐서 불일치.

---

## 전체 채점 시 유의 사항
- **재현성**: 시드를 고정해도 CPU/GPU·PyTorch 버전에 따라 소수점 아래 값은 달라질 수 있다.
  정확도 기준은 «이상/이하»로 느슨하게 본다.
- **실습 1 §3(심화 비교) · 실습 2 §7(torch.compile·AMP)** 은 문제로 비우지 않은 참고 셀이다.
  실행만 되면 되고, 환경에 따라 `torch.compile` 이 실패해도 `try/except` 로 넘어가도록 되어 있다.
- **실습 1 은 시드에 민감하다.** `TODO(9)` 를 옳게 짜도 다른 시드에서는 국소 최적에 갇힐 수 있다.
  노트북에 있는 `set_seed(2)` 를 그대로 두었는지 확인한다.
