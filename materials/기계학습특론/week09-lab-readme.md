# 9주차 실습 «문제본» — 군집화 / 의사결정나무

기계학습특론 2026-2학기 · 9주차는 실습 노트북이 **두 개**다. 두 개 모두 제출한다.

| 파일 | 주제 | TODO 개수 |
|---|---|---|
| `9주차_군집화_2026.ipynb` | k-평균·계층·밀도·확률·그래프 기반 군집화와 평가 | **10** |
| `9주차_의사결정트리_2026.ipynb` | 의사결정나무, 불순도·정보이득, 가지치기, 앙상블, 해석 | **12** |

> 파일명은 기존 실습 자료와의 호환을 위해 `의사결정트리` 를 유지하지만,
> 과목의 표준 용어는 **의사결정나무** 다.

---

## 1. 실행 방법

### Google Colab (권장)
1. 노트북을 Colab 에 업로드한다.
2. 위에서부터 순서대로 실행한다. 필요한 데이터(`clustering_data.csv`,
   `district_dict_list.csv`, `dong_dict_list.csv`)는 **없으면 자동 생성**되므로 별도 내려받기가 필요 없다.
3. 선택 패키지는 해당 셀의 `!pip install` 주석을 해제한다.
   - 의사결정나무 5절 — `xgboost`
   - 의사결정나무 6절 — `shap`
   - 군집화 6-2절 — `sentence-transformers` (선택 과제)

### 로컬
```bash
python -m venv .venv && source .venv/bin/activate
pip install "scikit-learn>=1.4" "numpy>=2.0" pandas matplotlib jupyter
pip install xgboost shap          # 선택
jupyter lab
```

### 검증 환경 (정답본이 실제로 끝까지 실행된 버전)
Python 3.11.15 · numpy 2.4.4 · pandas 3.0.2 · scikit-learn 1.8.0 ·
matplotlib 3.10.9 · xgboost 3.2.0 · shap 0.51.0

---

## 2. 풀이 방법

각 `TODO(n)` 은 이런 모양으로 비어 있다.

```python
# TODO(3): 각 관측치에서 «가장 가까운 기존 중심점»까지의 거리 제곱 D(x)^2 을 구하고, ...
#   힌트: ...
raise NotImplementedError("TODO(3)")
```

- 주석 두 줄과 `raise` 줄을 지우고 그 자리에 코드를 쓴다.
- **함수 시그니처·docstring·반환 형태 설명은 그대로 둔다.** 무엇을 만들어야 하는지가 거기 적혀 있다.
- TODO 바로 아래 줄부터는 이미 완성된 코드다. 변수 **이름과 모양(shape)** 을 설명대로 맞추면 그대로 이어서 돌아간다.
- 번호 순서대로 푸는 것이 가장 쉽다 (뒤 TODO 가 앞 TODO 의 결과를 쓴다).

---

## 3. TODO 목록과 배점 제안

### (1) `9주차_군집화_2026.ipynb` — 50점

| # | 절 | 무엇을 구현하는가 | 배점 |
|---|---|---|---|
| 1 | 2 | `KMeans` 를 `init="k-means++"`, `n_init=10`, 시드 고정으로 구성하고 군집 라벨 산출 | 3 |
| 2 | 2-2 | k = 2..10 각각의 SSE(`inertia_`)와 **평균 실루엣 계수** 계산 | 5 |
| 3 | 3 | **k-means++ 초기화** — D(x)² 에 비례하는 확률로 다음 중심점 표본추출 | 7 |
| 4 | 3 | Lloyd **할당 단계** — 브로드캐스팅 (n,k) 거리 제곱 행렬 → 최근접 중심 할당 → SSE 기록 | 8 |
| 5 | 3 | Lloyd **갱신 단계** — 군집 평균으로 중심점 갱신, 빈 군집 처리, `tol` 수렴 판정 | 7 |
| 6 | 4 | 알고리즘 6종 구성 (k-평균·Ward·DBSCAN·HDBSCAN·GMM·스펙트럴) | 6 |
| 7 | 4-1 | k-거리 그래프에서 DBSCAN 의 `eps` 결정 | 4 |
| 8 | 4-2 | GMM 의 **BIC/AIC** 계산 → BIC 최소 K 선택 | 4 |
| 9 | 5 | 평가 함수 — 실루엣·DB·CH(내부) + **ARI·AMI**(외부), 잡음(-1) 제외 | 3 |
| 10 | 5-1 | 부분표본 반복 군집화 → **안정성 ARI** 분포 | 3 |

### (2) `9주차_의사결정트리_2026.ipynb` — 50점

| # | 절 | 무엇을 구현하는가 | 배점 |
|---|---|---|---|
| 1 | 2 | `DecisionTreeClassifier(criterion="gini", max_depth=3)` 학습·예측 | 3 |
| 2 | 3-1 | **엔트로피** H(D) = −Σ p log₂ p | 5 |
| 3 | 3-1 | **조건부 엔트로피** H_A(D) = Σ (|D_v|/|D|)·H(D_v) | 5 |
| 4 | 3-1 | **SplitInfo** (C4.5 — 속성값 분포 자체의 엔트로피) | 4 |
| 5 | 3-1 | **지니 불순도** 1 − Σ p² | 3 |
| 6 | 3-2 | 속성별 Gain · SplitInfo · **GainRatio** · ΔGini 계산표 | 7 |
| 7 | 3-3 | **최적 분할 속성 탐색** — 정보이득 최대 속성을 루트로 | 4 |
| 8 | 4-1 | **CCP α 경로** 계산 (`cost_complexity_pruning_path`) | 4 |
| 9 | 4-1 | α 별 교차검증 정확도 · 잎 개수 · 테스트 정확도 기록 | 5 |
| 10 | 5 | **RandomForest · HistGradientBoosting · XGBoost** 구성 | 4 |
| 11 | 6 | **순열 중요도** 계산 | 3 |
| 12 | 6-1 | **TreeSHAP** SHAP 값 계산 | 3 |

과제(각 노트북 마지막 절의 «제출 과제» 필수 1·2번)는 별도 20점.

---

## 4. 자가 점검

노트북에 **정답을 맞히면 통과하는 검사**가 이미 들어 있다.

### 군집화
- 3절 `assert np.all(np.diff(sse_hist) <= 1e-9)` — SSE 가 증가하면 TODO(4)·(5) 가 틀렸다.
- 3절 끝 «직접 구현 vs sklearn 의 ARI» 가 **1.0** 이면 TODO(3)~(5) 전체가 맞다.
- TODO(1) 이 맞으면 SSE ≈ **22.297**, 반복 3회, 군집별 100개씩.
- TODO(2) 가 맞으면 «실루엣 최대 k = **5**», TODO(8) 이 맞으면 «BIC 최소 K = **5**».

### 의사결정나무
- TODO(2)·(5) 가 맞으면 `H(D) = 0.940`, `Gini(D) = 0.459`.
- TODO(3)·(6) 이 맞으면 `Gain(D, 나이) = 0.247` 이 최대 → TODO(7) 의 루트는 **나이(age)**.
- 3-1절 `assert abs(g - mi) < 1e-9` — 직접 계산한 정보이득이 `mutual_info_score` 와 일치해야 한다.
- TODO(9) 가 맞으면 선택된 α ≈ **0.0228**, 잎 **4개**.

### 문법만 먼저 확인하고 싶을 때
```bash
python - <<'PY'
import json, glob
for f in glob.glob("*.ipynb"):
    nb = json.load(open(f, encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"{f}:{i}", "exec")
    print("OK", f)
PY
```

---

## 5. 제출 형식

```
학번_이름_9주차/
├── 9주차_군집화_2026.ipynb          # 모든 셀 실행 결과 포함
└── 9주차_의사결정트리_2026.ipynb     # 모든 셀 실행 결과 포함
```

- **출력(그림·표·print)이 저장된 상태**로 제출한다. `raise NotImplementedError` 가 하나라도 남아 있으면 감점.
- 각 노트북 마지막에 «제출 과제» 필수 1·2번 답을 마크다운 셀로 덧붙인다.
- 데이터 파일은 자동 생성되므로 제출물에 포함하지 않아도 된다.

---

## 6. 알아 둘 점

- 군집화 노트북을 **두 번째로 실행**하면 이미 저장된 `clustering_data.csv` 를 읽기 때문에
  참조 라벨 `y_true` 가 `None` 이 되고, 5절 평가표에서 **ARI·AMI 열이 사라진다**.
  외부 지표를 보려면 `clustering_data.csv` 를 지우고 처음부터 다시 실행한다.
- `HDBSCAN` 의 `copy` 관련 `FutureWarning`, 스펙트럴 군집화의
  «Graph is not fully connected» `UserWarning` 은 정상이며 결과에 영향이 없다.
- `xgboost` / `shap` 이 없어도 나머지 셀은 모두 동작하도록 `try/except ImportError` 로 감싸 두었다.
