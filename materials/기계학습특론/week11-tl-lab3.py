"""실습 3. Optuna 5.0 으로 LoRA 하이퍼파라미터(lr · r · α/r · dropout · DoRA) 튜닝
데이터사이언스융합학과 2026711051 박민지

이 파일의 TODO
  TODO(1)  build_and_train — 탐색된 HP 로 LoraConfig 구성 (α = alpha_ratio·r 재매개화, rsLoRA)
  TODO(2)  build_and_train — 학습 파라미터 예산 제약을 trial 에 등록 (user_attr + set_constraint)
  TODO(3)  build_and_train — 에폭마다 중간 지표 보고 후 pruning 판정 (report + should_prune)
  TODO(4)  objective       — 탐색 공간 정의 (suggest_float / suggest_categorical)
  TODO(5)  main            — 샘플러(TPE) · 프루너(Median) 설정

스토리지·CSV/그림 내보내기·중요도 분석·다중 시드 재학습은 완성되어 있다.

예)
  python lab3_optuna_hpo.py --n_trials 40 --epochs 3
  python lab3_optuna_hpo.py --quick                       # 동작 확인 (3 trials)
  optuna-dashboard sqlite:///runs/lab3/study.db          # 대시보드 (별도 터미널)

산출물 (runs/lab3/):
  study.db            Optuna 스토리지 (같은 study_name 으로 여러 프로세스에서 optimize() 실행 시 병렬 탐색)
  trials.csv          모든 시행 (논문 부록 표)
  importance.json     PED-ANOVA 하이퍼파라미터 중요도 (5.0 기본 평가기, 상위 20% 국소 중요도)
  *.html              최적화 이력 · 중요도 · 평행좌표 · slice 그림 (plotly)
  best_seed*.json     최적 HP로 시드 3개 재학습 결과 → 평균 ± 표준편차 보고

Optuna 5.0 (2026.9) 포인트
  - TPESampler 기본값이 multivariate=True + constant_liar=True 로 바뀜 (HP 상관 반영·병렬 중복 억제)
  - 제약은 trial.set_constraint(key, g) (g ≤ 0 이면 feasible) 코어 API
  - 중요도 기본 평가기: PED-ANOVA — PedAnovaImportanceEvaluator(target_quantile=0.2)
"""
#[질문] lr · r · α · dropout · DoRA · 배치 · weight decay 7개를 어떤 값으로 해야 가장 좋을까? = 하이퍼파라미터(HP) 튜닝
#[설명] Optuna: HP 조합을 하나 뽑아 학습 → 점수 → 다음 조합 고르기를 자동으로 반복한다.
# TPE: 지난 결과를 보고 '좋았던 시행들의 HP 분포'에서 다음 후보를 더 자주 뽑는 탐색(무작위보다 효율적).
# pruning(가지치기): 중간 점수가 다른 시행들의 중앙값보다 나쁘면 끝까지 돌리지 않고 중단 → 시간 절약.

# 같은 시간 안에 더 좋은 HP 를 찾고, 어떤 HP 가 중요한지, 최적 HP 가 시드에 얼마나 흔들리는지(평균 ± 표준편차)를 본다.
from __future__ import annotations

import argparse
import json
import os
import statistics

import optuna
import timm
import torch
from peft import LoraConfig, get_peft_model

from common import (PRETRAINED, build_cifar10_loaders, count_params, ensure_dir, evaluate, get_device,
                    resolve_transforms, set_seed, train_one_epoch)
from lab2_peft_lora import guess_targets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_small_patch16_224.augreg_in21k_ft_in1k")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default="runs/lab3")
    p.add_argument("--study_name", default="lora_hpo")
    p.add_argument("--n_trials", type=int, default=40)
    p.add_argument("--timeout", type=int, default=None, help="초 단위 예산 상한")
    p.add_argument("--epochs", type=int, default=3, help="시행당 에폭 (pruning 대상)")
    p.add_argument("--max_trainable", type=int, default=3_000_000, help="제약: 학습 파라미터 상한")
    p.add_argument("--final_seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


#[설명] HP 한 세트(params)로 모델을 만들고 학습해 최고 검증 macro-F1 을 돌려준다. 
# 탐색(trial 있음)과 최종 재학습(trial 없음) 모두 이 함수를 쓴다.
def build_and_train(params: dict, args, device, seed: int, trial: optuna.Trial | None = None) -> dict:
    """주어진 HP로 LoRA 모델을 학습하고 검증 지표를 반환. trial 이 있으면 report/prune."""
    set_seed(seed)
    quick = args.quick
    base = timm.create_model(args.model, pretrained=PRETRAINED, num_classes=10)
    cfg, train_tf, eval_tf = resolve_transforms(base)
    head_name = [n for n, _ in base.named_modules() if n in ("head", "head.fc", "fc", "classifier")][-1]
    # TODO(1): params 딕셔너리의 HP 로 LoraConfig 를 만들어 lora_cfg 에 담고,
    #          get_peft_model(base, lora_cfg).to(device) 로 model 을 만든다.
    #          r=params["r"], lora_alpha=int(params["alpha_ratio"] * params["r"]),
    #          lora_dropout=params["lora_dropout"], use_dora=params["dora"], use_rslora=True,
    #          target_modules=guess_targets(args.model), modules_to_save=[head_name]
    #   힌트: α 를 r 의 배수(alpha_ratio)로 두면 r 과 α 의 상호작용이 줄어 탐색이 쉬워진다.
    
    #[설명] 실습 2 와 같은 LoRA 설정이지만, 값을 고정하지 않고 Optuna 가 뽑은 params 에서 가져온다.
    # r 과 α 를 따로 뽑으면 둘이 얽혀 탐색이 어렵다.
    lora_cfg = LoraConfig(r=params["r"], lora_alpha=int(params["alpha_ratio"] * params["r"]),
                          lora_dropout=params["lora_dropout"], use_dora=params["dora"], use_rslora=True,
                          target_modules=guess_targets(args.model), modules_to_save=[head_name])
    model = get_peft_model(base, lora_cfg).to(device)
    n_train, _ = count_params(model)
    if trial is not None:
        # TODO(2): 학습 파라미터 수를 trial 에 기록하고(user_attr "trainable_params"),
        #          «학습 파라미터 ≤ args.max_trainable» 제약을 Optuna 5.0 코어 API 로 등록한다.
        #   힌트: trial.set_user_attr(key, value) / trial.set_constraint(key, g)
        #         set_constraint 는 g ≤ 0 일 때 feasible 이다 → g = n_train - args.max_trainable.
        
        #[설명] user_attr: 시행에 메모를 붙인다(trials.csv 의 user_attrs_trainable_params 열).
        # set_constraint(이름, g): g ≤ 0 이면 '조건 만족'. 여기서는 학습 파라미터가 예산(max_trainable) 이하인지 검사한다.
        # 성능만 보면 무조건 큰 r 을 고를 수 있다. 메모리 · 저장 비용 예산을 걸어 현실적인 해만 최적 후보로 삼는다.
        trial.set_user_attr("trainable_params", n_train)
        trial.set_constraint("param_budget", n_train - args.max_trainable)   # ≤ 0 이면 feasible

    train_loader, val_loader, _ = build_cifar10_loaders(
        args.data_dir, train_tf, eval_tf, batch_size=params["batch"],
        train_size=256 if quick else 10000, val_size=128 if quick else 2000, test_size=256,
        num_workers=0 if quick else 2, seed=seed)
    epochs = 1 if quick else args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=params["lr"],
                            weight_decay=params["weight_decay"])
    
    #OneCycleLR: 처음 10% 동안 lr 을 올렸다가 나머지 동안 내린다(warmup + 감쇠)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=params["lr"], total_steps=epochs * len(train_loader),
                                                pct_start=0.1)
    best = {"macro_f1": -1}
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, opt, device, scheduler=sched, log_every=0, max_steps=4 if quick else None)
        val = evaluate(model, val_loader, device, max_steps=2 if quick else None)
        best = max(best, val, key=lambda d: d["macro_f1"])
        if trial is not None:
            # TODO(3): 이번 에폭의 검증 macro-F1 을 중간 결과로 보고하고,
            #          프루너가 «가망 없음» 으로 판정하면 시행을 중단한다.
            #   힌트: trial.report(value, step=epoch) 로 보고 → trial.should_prune() 이 True 면
            #         raise optuna.TrialPruned() (일반 예외로 죽이면 시행이 FAIL 로 기록된다).
            
            #[설명] report: 에폭마다 중간 점수를 Optuna 에 알린다. should_prune: MedianPruner 가 '같은 에폭의 다른 시행 중앙값보다 나쁜가'를 판단한다.
            # TrialPruned 예외로 끝내야 PRUNED(가지치기)로 기록된다. 다른 예외면 FAIL(실패)로 남는다.
            trial.report(val["macro_f1"], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    best["trainable_params"] = n_train
    return best


#[설명] objective(목적 함수): Optuna 가 시행마다 부르는 함수. HP 를 뽑고 → 학습하고 → 점수(검증 macro-F1)를 돌려준다.
def make_objective(args, device):
    def objective(trial: optuna.Trial) -> float:
        # TODO(4): 탐색 공간을 정의해 params 딕셔너리를 만든다. 아래 7개 키를 모두 채워야
        #          build_and_train 이 그대로 돌아간다.
        #            "lr"           : 1e-4 ~ 1e-2, 로그 스케일   (LoRA 는 Full FT 의 ~10배 영역)
        #            "r"            : {4, 8, 16, 32, 64} 중 하나
        #            "alpha_ratio"  : {1, 2, 4} 중 하나          (α = ratio·r 재매개화)
        #            "lora_dropout" : 0.0 ~ 0.2 (선형)
        #            "dora"         : {False, True}
        #            "batch"        : {16, 32, 64}               (LoRA 는 큰 배치 페널티 → 작은 쪽 포함)
        #            "weight_decay" : 1e-4 ~ 0.1, 로그 스케일
        #   힌트: trial.suggest_float(name, low, high, log=True) / trial.suggest_categorical(name, choices).
        #         suggest_* 를 objective «안에서» 호출해야 define-by-run 조건부 공간이 가능하다.
        #         lr·weight_decay 처럼 자릿수로 움직이는 HP 는 반드시 log=True.
        
        #[설명] suggest_float(log=True): 1e-4~1e-2 를 '자릿수' 기준으로 고르게 뽑는다. lr 처럼 0.0001·0.001·0.01 이 비슷하게 중요한 값에 쓴다.
        # suggest_categorical: 정해진 후보 중 하나(r 은 4·8·16·32·64). suggest_* 를 objective 안에서 불러야 시행마다 새로 뽑힌다(define-by-run).
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "r": trial.suggest_categorical("r", [4, 8, 16, 32, 64]),
            "alpha_ratio": trial.suggest_categorical("alpha_ratio", [1, 2, 4]),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.2),
            "dora": trial.suggest_categorical("dora", [False, True]),
            "batch": trial.suggest_categorical("batch", [16, 32, 64]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 0.1, log=True),
        }
        return build_and_train(params, args, device, seed=args.seed, trial=trial)["macro_f1"]
    return objective


def main():
    args = parse_args()
    device = get_device()
    out = ensure_dir(args.out_dir)
    storage = f"sqlite:///{os.path.join(out, 'study.db')}"
    n_trials = 3 if args.quick else args.n_trials

    # TODO(5): study 를 만든다. 스토리지·study_name·load_if_exists 는 아래 인자를 그대로 쓰고,
    #          «방향 · 샘플러 · 프루너» 를 채운다.
    #            direction : macro-F1 이 목적함수이므로 어느 쪽?
    #            sampler   : TPESampler(seed=args.seed, n_startup_trials=5)
    #                        — Optuna 5.0 은 multivariate=True · constant_liar=True 가 기본이라
    #                          HP 상관을 반영하고 병렬 시행의 중복 탐색을 억제한다.
    #            pruner    : MedianPruner(n_startup_trials=5, n_warmup_steps=1)  ← 실습 코드 ⑦과 동일
    #   힌트: optuna.create_study(study_name=..., storage=storage, load_if_exists=True,
    #                             direction=..., sampler=..., pruner=...)
    #         같은 study_name·storage 로 다른 프로세스에서 다시 돌리면 그대로 병렬 탐색이 된다.

    #[설명] TPESampler(n_startup_trials=5): 처음 5시행은 무작위로 뽑아 자료를 모은 뒤, 6번째부터 TPE 로 뽑는다.
    # MedianPruner(n_startup_trials=5, n_warmup_steps=1): 완료 시행이 5개 쌓이고 첫 에폭(step 0)이 지난 뒤부터 가지치기를 판단.
    # 축소 실행은 15분 안에 4시행만 끝나 TPE · pruning 이 시작되기 전(5시행 미만) 중단
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(make_objective(args, device), n_trials=n_trials, timeout=args.timeout, gc_after_trial=True)

    print("\n=== best trial ===")
    print(study.best_trial.number, study.best_value, study.best_params)

    # 결과 내보내기
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs", "duration"))
    df.to_csv(os.path.join(out, "trials.csv"), index=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) >= 3:
        try:
            # 5.0 기본 평가기 = PED-ANOVA. target_quantile=0.2 → 상위 20% 영역의 «국소» 중요도
            evaluator = optuna.importance.PedAnovaImportanceEvaluator(target_quantile=0.2)
            imp = optuna.importance.get_param_importances(study, evaluator=evaluator)
            with open(os.path.join(out, "importance.json"), "w") as f:
                json.dump(imp, f, indent=2)
            print("importance:", imp)
            from optuna.visualization import (plot_contour, plot_optimization_history, plot_parallel_coordinate,
                                              plot_param_importances, plot_slice)
            plot_optimization_history(study).write_html(os.path.join(out, "history.html"))
            plot_param_importances(study, evaluator=evaluator).write_html(os.path.join(out, "importance.html"))
            plot_parallel_coordinate(study, params=["lr", "r", "lora_dropout"]).write_html(os.path.join(out, "parallel.html"))
            plot_slice(study, params=["lr", "r"]).write_html(os.path.join(out, "slice.html"))
            plot_contour(study, params=["lr", "weight_decay"]).write_html(os.path.join(out, "contour.html"))
        except Exception as e:  # plotly 미설치 등
            print("시각화 건너뜀:", e)

    #[목적] 최적 HP 가 운 좋은 시드 하나에서만 좋았던 건 아닌지 확인한다. 시드 0·1·2 로 다시 학습해 평균 ± 표준편차를 보고한다.
    # 최적 HP로 다중 시드 재학습 → 평균 ± SD
    seeds = [0] if args.quick else list(range(args.final_seeds))
    scores = []
    for s in seeds:
        res = build_and_train(study.best_params, args, device, seed=s)
        scores.append(res["macro_f1"])
        with open(os.path.join(out, f"best_seed{s}.json"), "w") as f:
            json.dump({"seed": s, "params": study.best_params, **res}, f, indent=2)
        print(f"seed {s}: val macro_f1 {res['macro_f1']:.4f}")
    mean = statistics.mean(scores)
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    print(f"\n최적 HP 재학습 macro-F1: {mean:.4f} ± {sd:.4f} (n={len(scores)})")
    print("saved:", out, "| dashboard:", f"optuna-dashboard {storage}")


if __name__ == "__main__":
    main()
