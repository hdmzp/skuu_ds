# 9주차 실습 «정답 해설» — 군집화 / 의사결정나무

문제본: `../problem/` · 이 문서는 각 TODO 의 정답 요약, 채점 포인트, 흔한 오답을 정리한다.
정답 코드는 같은 폴더의 두 노트북에 그대로 들어 있다.

정답본 실행 확인 환경: Python 3.11.15 · numpy 2.4.4 · pandas 3.0.2 · scikit-learn 1.8.0 ·
matplotlib 3.10.9 · xgboost 3.2.0 · shap 0.51.0 — **두 노트북 모두 끝까지 오류 없이 실행됨.**

---

# A. `9주차_군집화_2026.ipynb` (TODO 10개)

## TODO(1) — sklearn k-평균 구성 (2절)
```python
model = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=RANDOM_STATE)
labels = model.fit_predict(Xs)
```
- **채점 포인트**: ① `n_init` 을 명시했는가(국소 최적 회피) ② `random_state` 로 재현성을 확보했는가
  ③ 원 데이터가 아니라 **표준화된 `Xs`** 를 넣었는가.
- **기대 출력**: SSE ≈ 22.297, 반복 3회, 군집별 100개씩.
- **흔한 오답**: `feature` 를 그대로 넣음 / `fit` 만 하고 `labels` 를 만들지 않음 /
  `n_init` 을 빼서 실행할 때마다 SSE 가 달라짐.

## TODO(2) — SSE·실루엣 곡선 (2-2절)
```python
km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
sse.append(km.inertia_)
sil.append(silhouette_score(Xs, km.labels_))
```
- **채점 포인트**: SSE 는 `inertia_` 로 «모델이 준 값»을 쓰고, 실루엣은 «데이터+라벨»로 계산한다는 구분.
- **기대 출력**: 실루엣 최대 k = **5**. SSE 는 k 에 대해 단조 감소(엘보는 눈으로 판단).
- **흔한 오답**: `silhouette_score(Xs, k)` 처럼 라벨 대신 k 를 넘김 /
  k=1 을 범위에 넣어 `silhouette_score` 가 예외를 던짐.

## TODO(3) — k-means++ 초기화 (3절)
```python
d2 = ((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(-1).min(axis=1)   # D(x)^2
C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
```
- **채점 포인트**: ① `min(axis=1)` — «가장 가까운» 기존 중심점까지의 거리라는 점
  ② 확률이 D(x)**²** 에 비례(거리 1승이 아님) ③ `p` 의 합이 1이 되도록 정규화.
- **흔한 오답**: `argmax(d2)` 로 가장 먼 점을 «확정» 선택 → 이상치에 끌려간다(k-means++ 는 확률적 선택) /
  `sum(-1)` 없이 (n,k,d) 를 그대로 min.

## TODO(4) — Lloyd 할당 단계 (3절)
```python
D2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)      # (n, k)
label = D2.argmin(axis=1)
sse_hist.append(D2[np.arange(len(X)), label].sum())
```
- **채점 포인트**: ① 이중 for 루프가 아닌 **브로드캐스팅** ② `(n,1,d) − (1,k,d) → (n,k,d)` 후
  마지막 축 합 ③ SSE 는 «할당된 중심점까지의 거리 제곱 합» — `D2.min(axis=1).sum()` 도 정답.
- **흔한 오답**: `argmin(axis=0)` (축 혼동, shape 이 k 가 됨) /
  거리 제곱이 아니라 `np.sqrt` 를 씌운 거리로 SSE 를 계산 / SSE 를 갱신 **후**에 기록해 단조성 assert 실패.

## TODO(5) — Lloyd 갱신 단계 (3절)
```python
C_new = np.array([X[label == j].mean(axis=0) if np.any(label == j) else C[j]
                  for j in range(k)])
history.append(C_new.copy())
if np.linalg.norm(C_new - C) < tol:
    C = C_new
    break
C = C_new
```
- **채점 포인트**: ① **빈 군집** 처리(`np.any(label == j)`) ② 종료 조건을 «중심점 이동량»으로
  ③ `history` 에 `.copy()` 를 넣었는가(참조만 넣으면 궤적이 전부 같은 값이 된다).
- **기대 출력**: `assert np.all(np.diff(sse_hist) <= 1e-9)` 통과, sklearn 과의 **ARI = 1.0**.
- **흔한 오답**: 빈 군집에서 `mean` 이 `nan` → 이후 전부 `nan` /
  `history.append(C_new)` (copy 누락) / 종료 조건을 SSE 변화량으로 두어 `tol` 의미가 달라짐.

## TODO(6) — 알고리즘 6종 구성 (4절)
```python
return {"k-means": KMeans(k, n_init=10, random_state=RANDOM_STATE),
        "계층(Ward)": AgglomerativeClustering(n_clusters=k, linkage="ward"),
        "DBSCAN": DBSCAN(eps=0.3, min_samples=5),
        "HDBSCAN": HDBSCAN(min_cluster_size=15),
        "GMM": GaussianMixture(k, covariance_type="full", random_state=RANDOM_STATE),
        "스펙트럴": SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=10,
                                      random_state=RANDOM_STATE)}
```
- **채점 포인트**: ① DBSCAN·HDBSCAN 은 **k 를 받지 않는다**(밀도 기반) ② 스펙트럴은
  `affinity="nearest_neighbors"` 여야 초승달·동심원을 푼다(기본 RBF 로는 잘 안 된다)
  ③ 시드 있는 추정기에 `random_state` 를 넣었는가.
- **기대 결과**: 초승달·동심원에서 k-평균·Ward·GMM 은 실패, DBSCAN·HDBSCAN·스펙트럴은 성공.
- **흔한 오답**: `DBSCAN(n_clusters=k)` (해당 인자 없음) / 딕셔너리 키 순서·이름을 바꿔
  이후 `names`, `models["..."]` 참조가 깨짐.

## TODO(7) — DBSCAN 의 `eps` (4-1절)
```python
kdist = np.sort(dist[:, -1])
eps = float(np.quantile(kdist, 0.95))
```
- **채점 포인트**: ① `dist[:, -1]` — minPts 번째(마지막) 이웃까지의 거리 ② **정렬**해야 곡선이 생긴다
  ③ 꺾이는 지점 ≈ 상위 분위수.
- **기대 출력**: 잡음으로 분류된 관측치 4개 내외.
- **흔한 오답**: `dist[:, 0]` (자기 자신까지의 거리 0) / 정렬을 빠뜨려 그래프가 잡음처럼 보임.

## TODO(8) — GMM 의 BIC/AIC (4-2절)
```python
gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=RANDOM_STATE).fit(Xs)
bics.append(gm.bic(Xs)); aics.append(gm.aic(Xs))
```
- **채점 포인트**: BIC/AIC 는 «**작을수록** 좋다» — 아래 `np.argmin` 과 방향이 맞아야 한다.
- **기대 출력**: BIC 최소 K = **5**, max γ < 0.7 인 불확실 관측치 0개.
- **흔한 오답**: `gm.bic()` 에 데이터를 안 넘김 / `argmax` 로 K 선택 / `covariance_type` 을 바꿔 K 가 흔들림.

## TODO(9) — 평가 지표 (5절)
```python
if n_cl > 1:
    out["silhouette"] = silhouette_score(X[m], lab[m])
    out["davies_bouldin"] = davies_bouldin_score(X[m], lab[m])
    out["calinski_harabasz"] = calinski_harabasz_score(X[m], lab[m])
if y_ref is not None:
    out["ARI"] = adjusted_rand_score(y_ref, lab)
    out["AMI"] = adjusted_mutual_info_score(y_ref, lab)
```
- **채점 포인트**: ① **잡음 라벨 −1 을 내부 지표에서 제외**(`X[m], lab[m]`) ② 군집이 1개면 내부 지표가
  정의되지 않으므로 `n_cl > 1` 가드 ③ 외부 지표는 전체 `lab` 으로 계산(잡음도 하나의 «군집»으로 본다)
  ④ 방향: 실루엣↑, DB↓, CH↑.
- **기대 출력(표준화 블롭)**: k-평균·Ward·GMM 이 실루엣 0.768 로 동일, HDBSCAN 0.781(잡음 16개),
  DBSCAN 은 군집 4개로 과소 병합.
- **흔한 오답**: 잡음 제외 없이 DBSCAN 실루엣을 계산해 값이 낮게 나옴 / `y_ref` 가 `None` 인데 ARI 호출.

## TODO(10) — 안정성 (5-1절)
```python
idx = rng.choice(len(X), int(frac * len(X)), replace=False)
lab = make_model().fit_predict(X[idx])
scores.append(adjusted_rand_score(base[idx], lab))
```
- **채점 포인트**: ① **비복원**(`replace=False`) ② `base` 전체가 아니라 **`base[idx]`** 와 비교(같은 관측치끼리)
  ③ 매번 `make_model()` 로 새 모델을 만들어야 한다(같은 객체 재사용 금지).
- **기대 출력**: k = 3·4·5 는 ARI 1.000, k = 6 에서 0.936 ± 0.053 으로 흔들리기 시작.
- **흔한 오답**: `adjusted_rand_score(base, lab)` — 길이가 달라 예외 / 복원 추출로 중복 관측치 발생.

---

# B. `9주차_의사결정트리_2026.ipynb` (TODO 12개)

## TODO(1) — 의사결정나무 학습 (2절)
```python
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=1,
                             random_state=RANDOM_STATE)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
```
- **채점 포인트**: 문자열 라벨을 `LabelEncoder` 없이 그대로 넣는다(`clf.classes_` 로 순서 확인).
- **기대 출력**: 훈련 정확도 0.917, 테스트 정확도 0.812.
- **흔한 오답**: `max_depth` 를 지정하지 않아 훈련 정확도 1.0 / 테스트 정확도 하락.

## TODO(2) — 엔트로피
```python
p = y.value_counts(normalize=True).to_numpy()
p = p[p > 0]
return float(-(p * np.log2(p)).sum())
```
- **채점 포인트**: ① 밑이 **2** ② `p = 0` 항 제외(0·log0 = 0) ③ 개수가 아니라 **비율**.
- **기대 출력**: `H(D) = 0.940`.
- **흔한 오답**: `np.log`(자연로그) 사용 → 0.652 / `p > 0` 필터 누락 → `nan`.

## TODO(3) — 조건부 엔트로피
```python
return float(sum(len(sub) / n * entropy(sub[TARGET]) for _, sub in data.groupby(attr)))
```
- **채점 포인트**: **부분집합 크기로 가중**해야 한다(단순 평균이 아님).
- **기대 출력**: H_나이(D) = 0.694 → Gain = 0.940 − 0.694 = **0.247**.
- **흔한 오답**: 가중 없이 `np.mean` / `groupby` 대상 컬럼을 `TARGET` 으로 착각.

## TODO(4) — SplitInfo
```python
p = data[attr].value_counts(normalize=True).to_numpy()
return float(-(p * np.log2(p)).sum())
```
- **채점 포인트**: 클래스가 아니라 **속성값**의 분포라는 점. 값이 많은 속성일수록 커져 벌점이 된다.
- **기대 출력**: 나이 1.577, 학생 1.000, 신용 0.985, 수입 1.557.
- **흔한 오답**: `entropy(data[TARGET])` 을 그대로 호출 → 모든 속성에서 같은 값.

## TODO(5) — 지니 불순도
```python
p = y.value_counts(normalize=True).to_numpy()
return float(1 - (p ** 2).sum())
```
- **기대 출력**: `Gini(D) = 0.459`.
- **흔한 오답**: `1 - p.sum() ** 2` (제곱 위치 착오) → 항상 0.

## TODO(6) — 계산표
```python
rows.append({"속성": KOR.get(a, a), "값별 분포": dist,
             "H_A(D)": round(cond_entropy(data, a), 3),
             "Gain": round(g, 3), "SplitInfo": round(si, 3),
             "GainRatio": round(g / si, 3) if si > 0 else np.nan,
             "ΔGini": round(base_G - sum(len(s) / n * gini(s[TARGET])
                                         for _, s in data.groupby(a)), 3)})
```
- **채점 포인트**: ① `SplitInfo == 0` 방어(0 나누기) ② ΔGini 도 «가중» 평균을 뺀다
  ③ 키 이름을 정확히 맞춰야 아래 `sort_values("Gain")`, `t[["속성","Gain","GainRatio"]]` 가 동작.
- **기대 출력**: 나이 Gain 0.247 / GainRatio 0.156, 학생 0.152 / 0.152, 신용 0.048 / 0.049, 수입 0.029 / 0.019.
  → **Gain 기준으로는 나이가 1위지만 GainRatio 로는 나이(0.156)와 학생(0.152)이 거의 붙는다**는 점이 C4.5 의 요지.
- **흔한 오답**: ΔGini 를 `base_G - gini(전체)` 로 계산 → 항상 0.

## TODO(7) — 최적 분할 속성 탐색
```python
root = gain_table(df).iloc[0]["속성"]
root_attr = [a for a in ATTRS if KOR.get(a, a) == root][0]
```
- **채점 포인트**: `gain_table` 이 Gain 내림차순 정렬이므로 `.iloc[0]` 이 argmax 와 같다는 이해.
- **기대 출력**: 첫 분할 = **나이(age)**. 자식에서 senior → **신용**, youth → **학생**,
  middle_aged 는 H = 0 인 **순수 노드**(잎).
- **흔한 오답**: 한글명을 그대로 `df.groupby` 에 넘겨 `KeyError` (역매핑 누락).

## TODO(8) — CCP α 경로
```python
path = DecisionTreeClassifier(random_state=RANDOM_STATE).cost_complexity_pruning_path(X_train, y_train)
alphas = np.unique(path.ccp_alphas)
alphas = alphas[alphas >= 0]
```
- **채점 포인트**: 경로는 «가지치기 전» 완전 성장 트리에서 뽑는다. `np.unique` 로 중복 제거.
- **흔한 오답**: `ccp_alpha` 를 이미 준 트리에서 경로를 뽑음 / `path.impurities` 를 α 로 착각.

## TODO(9) — α 별 성능 기록
```python
m = DecisionTreeClassifier(ccp_alpha=a, random_state=RANDOM_STATE)
cv_scores.append(cross_val_score(m, X_train, y_train, cv=cv).mean())
m.fit(X_train, y_train)
leaves.append(m.get_n_leaves()); test_scores.append(m.score(X_test, y_test))
```
- **채점 포인트**: ① α 선택은 **교차검증(훈련 데이터)** 으로 하고, 테스트 점수는 «확인용»으로만 그린다
  ② `cross_val_score` 는 fit 되지 않은 모델을 받는다 → 이후 별도로 `fit`.
- **기대 출력**: 선택된 α ≈ **0.02275**, 잎 **4개**, 테스트 정확도 0.787.
- **흔한 오답**: 테스트 점수로 α 를 골라 데이터 누수 / `get_n_leaves()` 를 fit 전에 호출해 예외.

## TODO(10) — 앙상블 구성
```python
models["random_forest"] = RandomForestClassifier(500, max_features="sqrt", oob_score=True,
                                                 n_jobs=-1, random_state=RANDOM_STATE)
models["hist_gb"] = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                   early_stopping=True, validation_fraction=0.25,
                                                   random_state=RANDOM_STATE)
try:
    from xgboost import XGBClassifier
    models["xgboost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                      random_state=RANDOM_STATE)
except ImportError:
    print("xgboost 가 설치되어 있지 않습니다 — 해당 모델은 건너뜁니다.")
```
- **채점 포인트**: ① RF 는 **깊은 트리 + 다수 평균**(분산 감소), 부스팅은 **얕은 트리 + 순차 보정**(편향 감소)
  이라는 대비가 하이퍼파라미터에 드러나야 한다 ② `oob_score=True` 를 켜야 아래 `oob_score_` 출력이 동작
  ③ xgboost 는 없을 수 있으므로 `try/except ImportError`.
- **기대 출력**: OOB 0.783 / tree(pruned) 0.787 · random_forest 0.800 · hist_gb 0.819 · xgboost 0.812.
- **흔한 오답**: `oob_score` 누락 → `AttributeError` / 부스팅에 `max_depth=None` 을 줘 과적합.

## TODO(11) — 순열 중요도
```python
pi = permutation_importance(rf, X_te_aug, y_te_i, n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1)
```
- **채점 포인트**: **테스트 데이터**(`X_te_aug`)로 계산해야 의미가 있다. 훈련 데이터로 하면 MDI 와 같은 편향이 생긴다.
- **기대 출력**: longitude MDI 0.444 / 순열 0.375, latitude 0.420 / 0.358,
  **random_num MDI 0.121 → 순열 0.008**, random_bin 0.016 → 0.000.
  즉 순수 잡음이지만 고유값이 많은 연속형 특징이 MDI 에서 과대평가된다.
- **흔한 오답**: `X_tr_aug` 로 계산 / `y_test`(문자열)를 넘겨 정수 라벨 모델과 불일치.

## TODO(12) — TreeSHAP
```python
explainer = shap.TreeExplainer(model_for_shap)
sv = explainer.shap_values(X_test)
sv0 = sv[..., 0] if isinstance(sv, np.ndarray) and sv.ndim == 3 else (sv[0] if isinstance(sv, list) else sv)
```
- **채점 포인트**: `shap_values` 의 반환 형태가 버전·모델에 따라 **(n,d,클래스) 배열 / 클래스별 리스트 / (n,d)**
  로 달라진다는 점을 처리했는가. 아래 `summary_plot` 은 (n, d) 2차원을 요구한다.
- **흔한 오답**: `sv[0]` 만 써서 3차원 배열일 때 «첫 **관측치**»를 꺼냄 → shape 오류 /
  `KernelExplainer` 사용(느리고 근사값).

---

# C. 공통 채점 유의사항

1. **문제본에 남아 있던 `raise NotImplementedError` 가 하나라도 있으면** 해당 항목 0점.
2. 변수명·딕셔너리 키를 바꾸면 아래 셀이 깨진다. 이름은 문제본대로 유지했는지 확인한다.
3. 시드를 고정하지 않아 «실행할 때마다 결과가 다른» 제출물은 재현성 항목에서 감점.
4. 군집화 노트북을 **다시 실행하면** `clustering_data.csv` 를 파일에서 읽어 `y_true` 가 `None` 이 되고
   5절 표에서 ARI·AMI 열이 빠진다. ARI·AMI 가 없는 제출물은 오답이 아니므로 감점하지 않는다.
5. `HDBSCAN` 의 `copy` `FutureWarning`, 스펙트럴의 «Graph is not fully connected» `UserWarning`,
   `tqdm` 의 `IProgress` 경고는 모두 정상이며 감점 대상이 아니다.
