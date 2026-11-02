# 10주차 «정답본» 채점 안내 — 딥러닝 분류모델 (CNN · PyTorch)

문제본: `../problem/`  (노트북 TODO 16개 · `main_pytorch.py` TODO 10개)
이 문서는 **채점자용**이다. 각 TODO 의 정답 요약, 채점 포인트, 흔한 오답을 적는다.

> 노트북은 루트와 `1.mri/` 두 곳에 **같은 파일**로 둔다 (내용 동일, md5 일치).
> `1.mri/main.py` 는 기존 TensorFlow 코드로 **참고용이며 손대지 않았다.**

---

## 0. 문제본 대비 정답본의 차이

| 항목 | 내용 |
|---|---|
| 코드 | 문제본의 `TODO(n)` 자리에 아래 «정답 요약» 의 코드가 들어 있다. 그 밖은 동일하다. |
| 추가 절 | `2-3. Mixup·CutMix` 와 `3-2. 출력 크기·파라미터 수·수용영역` 두 절은 이번 개정에서 **새로 넣었다** (아래 5장 참조). |
| 버그 수정 | `metrics_from` 이 numpy 스칼라 대신 파이썬 `float` 을 돌려주도록 고쳤다 (아래 5장 참조). |

---

## 1. 노트북 TODO 정답 요약과 채점 포인트

### TODO(1) `MixupCutmix.__call__` — Mixup / CutMix (7점)

```python
y1 = self._onehot(y)
if self.alpha <= 0 or random.random() > self.prob:
    return x, y1
lam = float(np.random.beta(self.alpha, self.alpha))
idx = torch.randperm(x.size(0), device=x.device)
if random.random() < self.switch:                       # CutMix
    H, W = x.shape[-2:]
    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    ya, yb = max(cy - rh // 2, 0), min(cy + rh // 2, H)
    xa, xb = max(cx - rw // 2, 0), min(cx + rw // 2, W)
    x = x.clone()
    x[:, :, ya:yb, xa:xb] = x[idx][:, :, ya:yb, xa:xb]
    lam = 1.0 - (yb - ya) * (xb - xa) / (H * W)         # ★ 면적 비로 λ 재계산
else:                                                   # Mixup
    x = lam * x + (1 - lam) * x[idx]
return x, lam * y1 + (1 - lam) * y1[idx]
```

**채점 포인트**
- 라벨을 이미지와 **같은 λ** 로 섞었는가 (이미지만 섞고 라벨을 그대로 두면 0점에 가깝다).
- CutMix 에서 λ 를 **실제로 덮인 면적 비로 다시 계산**했는가. ★ 여기가 핵심이다.
- 소프트 라벨의 각 행 합이 1 인가 (셀의 `assert` 가 잡아준다).

**흔한 오답**
- 표본마다 서로 다른 λ 를 뽑음 → 배치 하나에 λ 는 **하나**다 (구현이 훨씬 단순하고 원논문도 그렇다).
- `idx` 없이 `x.flip(0)` 로 짝을 만듦 → 동작은 하지만 다양성이 줄어든다. 감점 없음.
- CutMix 에서 λ 재계산을 빠뜨림 → 경계에서 잘린 만큼 라벨이 어긋난다. **−3점**.
- `x = x.clone()` 을 빠뜨리고 제자리 수정 → 입력 배치를 망가뜨린다. **−2점**.

---

### TODO(2) `conv_block` (4점)

```python
return nn.Sequential(
    nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
    nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
    nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
)
```

**채점 포인트**: 순서가 `Conv → BN → ReLU`인가. `padding=1`(same)인가. `bias=False`인가.

**흔한 오답**
- `Conv → ReLU → BN` 순서. 돌아가긴 하지만 표준이 아니다. **−1점**.
- `bias=True` → BN 의 β 와 중복이라 파라미터가 낭비된다. 3-2절의 `assert hand == torch_total` 가 **실패**하므로 학생이 스스로 알아챈다. **−1점**.
- `padding` 생략 → 크기가 계속 줄어 마지막 특징 맵이 14 가 아니게 된다.

---

### TODO(3) `SimpleCNN.__init__` (6점)

```python
blocks, c_prev = [], in_ch
for w in widths:
    blocks.append(conv_block(c_prev, w)); c_prev = w
self.features = nn.Sequential(*blocks)
self.head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Dropout(0.3), nn.Linear(c_prev, num_classes))
```

**채점 포인트**
- 채널 수를 `c_prev` 로 이어받는가 (`widths[i-1] → widths[i]`).
- 헤드가 **GAP → Flatten → Dropout → Linear** 인가.
- 출력이 로짓인가 — **마지막에 `nn.Softmax` 를 붙이면 안 된다.**

**흔한 오답**
- 헤드를 `Flatten → Linear(256*14*14, 4)` 로 만듦 → 파라미터가 200배로 늘고 입력 크기에 고정된다. **−3점**. (정답의 전체 파라미터는 **1,174,244**개다.)
- 마지막에 `nn.Softmax(dim=1)` 추가 → `CrossEntropyLoss` 가 로짓을 기대하므로 **이중 소프트맥스**가 되어 학습이 느려진다. **−3점**. 가장 흔한 오답이다.
- `nn.Sequential(blocks)` (별표 없음) → `TypeError`.

---

### TODO(4) `conv_out_size` — 출력 크기 (3점)

```python
return (n + 2 * p - d * (k - 1) - 1) // s + 1
```

**채점 포인트**: `floor` 를 썼는가(`//`). `-1 ... +1` 항을 빠뜨리지 않았는가.

**흔한 오답**
- `(n + 2p - k) / s + 1` 을 `/` 로 나눔 → float 이 되어 이후 `assert` 와 인덱싱이 깨진다. **−1점**.
- `d` 를 무시 → `d=1` 인 이번 실습에서는 값이 같지만 확장 합성곱에서 틀린다. **−1점**.

### TODO(5) `conv_params` — 파라미터 수 (2점)

```python
return k * k * c_in * c_out + (c_out if bias else 0)
```

**채점 포인트**: `c_in` 을 곱했는가. **위치(H·W)는 곱하지 않는다** — 가중치 공유의 핵심이다.

**흔한 오답**
- `k*k*c_out` (c_in 누락) → 가중치 공유를 오해한 것. **−2점**.
- `k*k*c_in*c_out*H*W` → 완전연결층과 혼동. **−2점**. 3-2절의 `assert hand == torch_total` 가 잡아준다.

---

### TODO(6) `receptive_field` (5점)

```python
r, jump = 1, 1
for k, s in layers:
    r += (k - 1) * jump
    jump *= s
return r
```

**채점 포인트**: `r` 을 갱신한 **뒤에** `jump` 를 곱하는가. 순서를 바꾸면 틀린다.

**검산값** — `[(3,1),(3,1)] → 5`, SimpleCNN 전체 `→ 76`, ResNet-18 stem `[(7,2),(3,2)] → 11`.

**흔한 오답**
- `jump *= s` 를 먼저 실행 → 첫 층의 스트라이드가 그 층 자신에게 적용되어 값이 커진다. **−3점**.
- `r` 을 0 에서 시작 → 결과가 1 작다. **−2점**.

---

### TODO(7) `make_scheduler` (5점)

```python
warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup * steps_per_epoch)
cos  = CosineAnnealingLR(optimizer, T_max=max(1, (epochs - warmup) * steps_per_epoch))
return SequentialLR(optimizer, [warm, cos], milestones=[warmup * steps_per_epoch])
```

**채점 포인트**
- 워밍업 뒤에 코사인이 오는가 (`SequentialLR` 의 `milestones` 가 워밍업 길이와 같은가).
- `T_max` 가 **남은** 에폭 수인가 (`epochs - warmup`). 전체 에폭을 넣으면 학습이 끝날 때 lr 이 0 까지 내려가지 않는다. **−2점**.

**흔한 오답**
- `milestones=[epochs]` → 코사인 구간이 아예 실행되지 않는다. **−3점**.
- `T_max` 에 0 이 들어갈 수 있는데 `max(1, ...)` 를 빼먹음 → `epochs == warmup` 일 때 `ZeroDivisionError`.

---

### TODO(8) `train_model` 의 학습 한 걸음 (8점) ★ 핵심

```python
with torch.autocast(device_type=device, dtype=AMP_DTYPE, enabled=(device == 'cuda')):
    loss = criterion(model(x), y)
optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer); scaler.update()
```

**채점 포인트**
- **순서**: 순전파 → 손실 → `zero_grad` → `backward` → `step`.
- `zero_grad` 가 `backward` **앞**에 있는가. ★ 가장 중요하다.
- `clip_grad_norm_` 앞에 `scaler.unscale_(optimizer)` 가 있는가 — 스케일이 걸린 기울기를 자르면 임계값이 무의미해진다.
- 손실 계산이 `autocast` 블록 **안**에 있는가.

**흔한 오답**
- `zero_grad()` 를 `backward()` 뒤에 둠 → 기울기가 지워져 **학습이 전혀 안 된다**. 손실이 평평하다. **−5점**.
- `zero_grad()` 를 아예 빼먹음 → 기울기가 누적되어 손실이 발산한다. **−5점**.
- `scaler.unscale_` 없이 클리핑 → fp16 에서만 조용히 잘못된다. **−2점**.
- `loss.backward()` (scaler 없이) → fp16 + GradScaler 조합에서 `scaler.step` 이 경고를 내거나 건너뛴다. **−2점**.
- `optimizer.step()` 을 직접 호출 → GradScaler 를 쓰는 의미가 없어진다. **−2점**.

---

### TODO(9) `BasicBlock.forward` (6점)

```python
out = F.relu(self.bn1(self.conv1(x)))
out = self.bn2(self.conv2(out))
if not self.residual:
    return F.relu(out)
return F.relu(out + self.shortcut(x))
```

**채점 포인트**
- **더한 뒤에 ReLU** 를 적용하는가 (`ReLU(F(x)+x)`, `ReLU(F(x))+x` 가 아니다).
- 두 번째 conv 뒤에는 ReLU 를 넣지 않는가.
- `self.shortcut(x)` 를 썼는가 (`x` 를 그대로 더하면 채널·크기가 달라 `RuntimeError`).

**흔한 오답**
- `return F.relu(out) + self.shortcut(x)` → 항등 경로가 비선형 뒤로 가서 잔차의 의미가 사라진다. **−4점**.
- `out = F.relu(self.bn2(self.conv2(out)))` 처럼 ReLU 를 두 번 → 더하기 전에 음수가 잘린다. **−3점**.
- `residual=False` 분기를 빼먹음 → 6-2절의 plain 비교 실험이 돌지 않는다. **−2점**.

---

### TODO(10) `ResNet18._init_weights` (5점)

```python
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
for m in self.modules():
    if isinstance(m, BasicBlock) and m.residual:
        nn.init.zeros_(m.bn2.weight)
```

**채점 포인트**
- ReLU 에 맞는 **He(Kaiming)** 초기화인가 (`nonlinearity='relu'`).
- zero-init residual 을 **두 번째 순회**에서 했는가 — 첫 순회에서 BN weight 를 1 로 두므로, 같은 순회 안에서 0 으로 만들면 뒤에 다시 1 로 덮일 수 있다.

**흔한 오답**
- `xavier_uniform_` 사용 → ReLU 망에서는 분산이 절반으로 줄어드는 것을 보정하지 못한다. **−2점**.
- `mode='fan_in'` → 틀리지는 않으나 원논문·torchvision 은 `fan_out` 이다. 감점 없음.
- zero-init 을 `m.bn1` 에 적용 → 블록 출력이 0 이 되지 않는다. **−2점**.
- 파라미터 수가 **11,178,564** 와 다르면 초기화가 아니라 구조 문제다.

---

### TODO(11) `param_groups_llrd` (5점)

```python
backbone = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith(head_names)]
head     = [p for n, p in model.named_parameters()
            if p.requires_grad and n.startswith(head_names)]
return [{'params': backbone, 'lr': lr_backbone}, {'params': head, 'lr': lr_head}]
```

**채점 포인트**
- `p.requires_grad` 로 걸렀는가 (고정된 백본 파라미터를 옵티마이저에 넣으면 안 된다).
- 두 그룹이 **겹치지도, 빠뜨리지도** 않는가.
- `head_names` 가 튜플임을 알고 `startswith` 에 그대로 넘겼는가.

**흔한 오답**
- 백본과 헤드에 같은 lr → 전이학습의 요점을 놓쳤다. **−3점**.
- `'fc' in n` 처럼 부분 문자열 검사 → `layer4.0.fc_like` 같은 이름을 오인할 수 있다. **−1점**.

---

### TODO(12) 역빈도 가중치 · 균형 샘플링 (5점)

```python
class_weight = (1.0 / counts); class_weight = class_weight / class_weight.sum() * NUM_CLASSES
sample_w = (1.0 / counts)[torch.tensor(df_train['label'].values)]
```

**채점 포인트**
- 빈도에 **반비례**하는가.
- 합이 클래스 수가 되도록(평균 1) 정규화했는가 — 안 하면 손실의 절대 크기가 달라져 lr 을 다시 맞춰야 한다.
- `sample_w` 의 길이가 학습 표본 수(`len(df_train)`)와 같은가.

**흔한 오답**
- 정규화 없이 `1/counts` 를 그대로 사용 → 손실이 0.001 수준으로 작아진다. **−2점**.
- `sample_w` 를 클래스 4개짜리로 만듦 → `WeightedRandomSampler` 가 인덱스 0~3 만 뽑아 4장짜리 학습이 된다. **−3점**.
- 손실 가중치와 균형 샘플링을 **동시에** 적용하고 그 사실을 보고서에 안 씀 → 소수 클래스를 이중으로 강조하게 된다. 보고서에서 감점.

---

### TODO(13) `FocalLoss.forward` (5점)

```python
ce = F.cross_entropy(logits, target, weight=self.weight,
                     label_smoothing=self.ls, reduction='none')
pt = torch.exp(-ce)
return ((1 - pt) ** self.gamma * ce).mean()
```

**채점 포인트**
- `reduction='none'` 으로 **표본별** CE 를 구했는가.
- `p_t = exp(-CE)` 관계를 썼는가 (또는 `softmax` 에서 정답 클래스 확률을 직접 gather).
- 마지막에 `.mean()` 으로 배치 평균을 냈는가.

**흔한 오답**
- `reduction='mean'` 으로 구한 뒤 `(1-pt)**γ` 를 곱함 → 배치 전체에 스칼라 하나를 곱한 것이라 focal 이 아니다. **−4점**.
- `pt = logits.softmax(1).max(1)` → **예측** 클래스 확률이지 정답 클래스 확률이 아니다. **−3점**.
- `label_smoothing` 을 함께 쓰면 `CE ≠ -log p_t` 가 되어 `exp(-ce)` 가 근사가 된다. 기본값 0.0 이므로 문제없지만, 지적한 학생에게는 **가산점**.

---

### TODO(14) `fit_temperature` 의 `closure` (5점)

```python
opt.zero_grad()
loss = F.cross_entropy(logits / logT.exp().clamp(min=1e-2), y)
loss.backward(); return loss
```

**채점 포인트**
- **로짓을 T 로 나눴는가** (곱하면 방향이 반대다).
- `logT.exp()` 로 T > 0 을 보장했는가.
- `closure` 가 손실을 **반환**하는가 — LBFGS 는 반환값을 쓴다.
- **검증 세트** 로짓으로 학습했는가 (테스트로 맞추면 누수다). ★

**흔한 오답**
- `closure` 가 아무것도 반환하지 않음 → LBFGS 가 `TypeError`. **−2점**.
- `opt.zero_grad()` 누락 → LBFGS 가 여러 번 부르므로 기울기가 누적된다. **−2점**.
- 테스트 로짓으로 T 를 학습 → **−5점**. 캘리브레이션 단원의 핵심을 놓친 것이다.
- T 최적화 후 **정확도가 달라졌다면** 어딘가 틀린 것이다. 온도는 순위를 바꾸지 않는다 (셀이 두 정확도를 나란히 출력한다).

---

### TODO(15) `expected_calibration_error` (5점)

```python
conf, pred = probs.max(1)
acc = (pred == y).float()
bins, ece = torch.linspace(0, 1, n_bins + 1), 0.0
stats = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (conf > lo) & (conf <= hi)
    if m.any():
        a, c, w = acc[m].mean(), conf[m].mean(), m.float().mean()
        ece += w * (a - c).abs(); stats.append((float(lo), float(a), float(c), int(m.sum())))
return float(ece), stats
```

**채점 포인트**
- 구간별 **가중 평균**인가 (`|B_m|/N` 을 곱했는가). 단순 평균이면 표본이 1개인 구간이 과대평가된다.
- **절댓값**을 취했는가 — 안 하면 과신과 과소신이 상쇄되어 ECE 가 0 에 가까워진다. ★
- 빈 구간을 건너뛰는가 (`m.any()`) — 안 하면 `nan`.
- `stats` 를 함께 반환하는가 (신뢰도 다이어그램 셀이 쓴다).

**흔한 오답**
- 절댓값 누락 → **−3점**. 결과가 수상하게 작다.
- 가중치 없이 `.mean()` → **−2점**.
- 구간 경계를 `>=`/`<=` 둘 다로 잡아 경계 표본을 두 번 셈 → **−1점**.

---

### TODO(16) `GradCAM.__call__` (4점)

```python
w = self.grads.mean(dim=(2, 3), keepdim=True)
cam = F.relu((w * self.acts).sum(1, keepdim=True))
cam = F.interpolate(cam, x.shape[-2:], mode='bilinear', align_corners=False)
```

**채점 포인트**
- 채널별 중요도 α 가 **기울기의 공간 평균**인가 (`dim=(2,3)`).
- 가중합을 **채널 축**으로 더했는가 (`sum(1)`).
- `ReLU` 로 음의 기여를 잘랐는가 — Grad-CAM 의 정의다.

**흔한 오답**
- `self.acts.mean(dim=(2,3))` 처럼 활성값을 평균 → α 는 **기울기**에서 나온다. **−3점**.
- `sum` 을 공간 축으로 → 모양이 (N, C) 가 되어 `interpolate` 가 실패한다.
- ReLU 생략 → 열지도에 음수가 섞여 정규화 후 배경이 밝게 보인다. **−2점**.
- `.backward()` 전에 `zero_grad()` 를 안 해 이전 기울기가 남음 → 열지도가 흐려진다. **−1점**.

---

## 2. `main_pytorch.py` TODO 정답

노트북과 내용이 겹치므로 위를 그대로 적용한다. 다른 점만 적는다.

| # | 노트북 대응 | 다른 점 |
|---|---|---|
| 1~5 | TODO(2)(3)(8)(9)(6) | 동일 (스케줄러는 `steps_per_epoch` 인자가 없다) |
| 6 | TODO(11) | **네 그룹**이다: 백본/헤드 × decay/no-decay. `p.ndim <= 1` (bias·BN 의 γ,β) 에는 `weight_decay=0.0` 을 준다. 여기를 빠뜨리는 답이 가장 많다 (**−2점**) |
| 7 | TODO(8) | 동일 (`amp_dtype` 가 전역이 아니라 인자다) |
| 8, 9 | TODO(14)(14) | ECE 는 `stats` 없이 float 하나만 반환한다 |
| 10 | TODO(12) | `sampler` 까지 만들어야 한다. `WeightedRandomSampler(sample_w.double(), len(sample_w), replacement=True)` — `replacement=True` 가 빠지면 균형 샘플링이 되지 않는다 (**−1점**) |

---

## 3. 전체 실행 후 기대되는 수치 (`PROTOCOL='repool'`, 20 에폭)

정확한 값은 시드·GPU 에 따라 달라지므로 **순서와 자릿수**만 본다.

| 모델 | 테스트 macro-F1 | 비고 |
|---|---|---|
| SimpleCNN (기준선) | 0.75 ~ 0.88 | 파라미터 1,174,244 |
| ResNet-18 (scratch) | 0.80 ~ 0.90 | 파라미터 11,178,564 (torchvision 과 일치) |
| SimpleCNN + 불균형 처리 | 기준선과 비슷하거나 `no_tumor` F1 이 오름 | macro-F1 만 보면 차이가 작을 수 있다 |
| **ResNet-50 전이학습** | **0.92 ~ 0.97** | 가장 높아야 정상 |

전이학습이 scratch 보다 **낮다면** TODO(11) 의 파라미터 그룹이나 사전학습 가중치 로딩(`PRETRAINED_OK`)을 확인한다.

캘리브레이션: 보통 `T > 1` (과신)이고 ECE 가 줄어든다. `T < 1` 이 나오면 라벨 스무딩을 세게 준 모델일 수 있으니 그 자체가 오답은 아니다.

---

## 4. 보고서 채점 포인트 (과제 1~5)

1. **ablation** — 한 번에 한 변수만 바꿨는가. 같은 예산(에폭·배치)인가.
2. **seed 3개** — 평균 ± 표준편차를 썼는가. **시드 분산보다 작은 차이를 '개선'이라 주장하면 감점.**
3. **교차검증·부트스트랩** — 신뢰구간을 제시했는가.
4. **누수 분석** ★ 가장 중요 — `original` 과 `repool` 의 macro-F1 차이를 표로 보였는가.
   이 데이터셋은 테스트 394장 중 **252장(64%)** 이 학습 이미지와 픽셀 단위로 같고,
   `no_tumor` 는 **105장 전부**가 중복이다. `original` 에서 macro-F1 이 거의 1.0 에 가깝게 나오는 것이
   «허수»임을 지적했는지가 핵심이다. 지적하지 못했다면 이 문항은 만점을 줄 수 없다.
5. **해석** — Grad-CAM 이 배경·두개골 윤곽·문자 주석에 뜨는 사례를 찾아냈는가.
   Grad-CAM 이 **충실성이 보장되지 않는 사후 설명**임을 언급했다면 가산점.

---

## 5. 이번 개정에서 정답본을 손본 내역

### (a) 버그 수정 — 체크포인트를 다시 읽을 수 없던 문제

`metrics_from` 이 sklearn 의 **numpy 스칼라**(`np.float64`)를 그대로 돌려주고,
그것이 `torch.save({'model': ..., **mtr}, ckpt)` 로 체크포인트에 들어갔다.
PyTorch 2.6 부터 `torch.load` 의 `weights_only` 기본값이 `True` 로 바뀌었기 때문에,
학습이 끝나고 최고 성능 가중치를 복원하는 마지막 줄

```python
model.load_state_dict(torch.load(ckpt, map_location=device)['model'])
```

에서 `UnpicklingError` 로 **학습 전체가 무산되는** 버그였다. `metrics_from` 이 파이썬 `float` 을
돌려주도록 고쳤다 (노트북 4절, `main_pytorch.py` 의 `metrics_from`). torch 2.5 이하에서는 증상이 없었다.

### (b) 새로 넣은 두 절

- **`2-3. Mixup · CutMix`** — `train_model` 에 `mixup=` 인자가 있는데 정작 구현이 어디에도 없어
  쓸 수 없는 상태였다. 강의자료 PART 3 의 `v2.MixUp`/`v2.CutMix` 에 해당하는 `MixupCutmix` 를 넣었다.
- **`3-2. 출력 크기·파라미터 수·수용영역`** — 강의자료 PART 2 의 세 공식
  (`⌊(n+2p−d(k−1)−1)/s⌋+1`, `k²·C_in·C_out+C_out`, `r_L = r_(L−1)+(k_L−1)·Π s_i`) 을
  직접 계산해 PyTorch 결과와 `assert` 로 맞춰보는 절이다. 슬라이드 47쪽 실습 안내의
  «파라미터 수 표 작성» 에 대응한다.

### (c) 고치지 않고 남겨둔 것 (수업 중 지적하면 좋은 지점)

6-2절의 출력

```
Plain-34   파라미터: 21,286,724  (shortcut 의 1x1 투영만큼만 차이)
```

에서 두 모델의 파라미터 수는 실제로 **완전히 같다**. `BasicBlock.__init__` 이 `residual` 값과
무관하게 `self.shortcut` 을 만들기 때문에, plain 모델은 1x1 투영 파라미터를 **가지고만 있고 쓰지 않는다**
(그래서 학습되지 않은 채 가중치 감쇠만 받는다). 주석 문구는 이 사실과 맞지 않는다.
"plain 에서는 왜 파라미터가 같은가, 그 파라미터는 어떻게 되는가"는 좋은 토론거리이므로 그대로 두었다.
