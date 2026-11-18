"""Tune the PathSnap MLP using grouped training CV, then evaluate once."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from .mlp import build_mlp

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

SEED = 42
KEYS = ["trip_id", "obs_idx"]

# Replace with the exact input columns used by the winning baseline.
# Never include label, ref_arc, ref_dist or ref_in_candidates.
FEATURES = [
    "perp_dist", "proj_frac", "dist_to_start", "dist_to_end",
    "seg_length", "heading_cos", "n_candidates", "dist_rank",
    "dist_minus_min", "from_in_degree", "from_out_degree",
    "to_in_degree", "to_out_degree", "has_reverse_arc", "is_dead_end",
]

# Example only: replace with the settings from your team's winning run.
BASELINE = {}


def make_model(params):
    model = build_mlp(random_state=SEED)
    return model.set_params(**{f"classifier__{name}": value for name, value in params.items()})


def load_split(root, prefix):
    cand = pd.read_parquet(root / f"{prefix}_candidates.parquet")
    gps = pd.read_parquet(root / f"{prefix}_gps.parquet").copy()

    # The earlier runner preserves GPS row order within each trip but
    # may not explicitly save obs_idx in the GPS parquet.
    if "obs_idx" not in gps:
        gps["obs_idx"] = gps.groupby("trip_id", sort=False).cumcount()

    obs = gps[KEYS]
    if obs.empty or obs.duplicated().any():
        raise ValueError(f"{prefix}: expected unique, nonempty GPS observation keys.")
    if cand.empty or not cand["label"].isin([0, 1]).all():
        raise ValueError(f"{prefix}: expected nonempty candidates with binary labels.")
    if cand.duplicated(KEYS + ["arc_id"]).any():
        raise ValueError(f"{prefix}: duplicate candidate arcs for an observation.")

    missing = set(FEATURES) - set(cand.columns)
    if missing:
        raise ValueError(f"{prefix}: missing model features: {sorted(missing)}")
    if not pd.MultiIndex.from_frame(cand[KEYS]).isin(
        pd.MultiIndex.from_frame(obs)
    ).all():
        raise ValueError(f"{prefix}: candidate keys don't match GPS keys.")

    return cand.reset_index(drop=True), obs


def model_inputs(cand):
    return cand[FEATURES].replace([np.inf, -np.inf], np.nan)


def top1_counts(cand, scores, n_obs):
    ranked = cand[KEYS + ["arc_id", "label"]].assign(score=np.asarray(scores))
    winners = (
        ranked.sort_values(
            ["score", "arc_id"], ascending=[False, True]
        ).drop_duplicates(KEYS)
    )
    # Equal scores are broken by lower arc ID, not by the reference label.
    return int(winners["label"].sum()), n_obs


def candidate_miss_rate(cand, n_obs):
    covered = cand.loc[cand["label"].eq(1), KEYS].drop_duplicates()
    return 1.0 - len(covered) / n_obs


def grouped_folds(obs, n_splits):
    # Split the GPS table, not the candidate table, so trips with zero
    # candidates still contribute to the validation denominator.
    return GroupKFold(n_splits=n_splits).split(obs, groups=obs["trip_id"])


def cv_top1(params, cand, obs, folds):
    correct = total = 0
    for fit_idx, val_idx in folds:
        fit_trips = obs.iloc[fit_idx]["trip_id"]
        val_trips = obs.iloc[val_idx]["trip_id"]
        train = cand[cand["trip_id"].isin(fit_trips)]
        valid = cand[cand["trip_id"].isin(val_trips)]

        if train["label"].nunique() != 2:
            raise ValueError("A training fold lacks both classes. Check your labels.")

        model = make_model(params)
        model.fit(model_inputs(train), train["label"])
        scores = (
            model.predict_proba(model_inputs(valid))[:, 1]
            if not valid.empty else np.array([])
        )
        hits, n = top1_counts(valid, scores, len(val_idx))
        correct += hits
        total += n

    # Pool observations across folds, rather than weighting every fold equally.
    return correct / total


def trial_params(trial):
    layouts = {
        "64": (64,),
        "128": (128,),
        "64_32": (64, 32),
        "128_64": (128, 64),
    }
    return {
        "hidden_layer_sizes": layouts[
            trial.suggest_categorical("architecture", list(layouts))
        ],
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init", 1e-4, 1e-2, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }


def evaluate(model, cand, obs):
    start = perf_counter()
    scores = model.predict_proba(model_inputs(cand))[:, 1]
    predict_seconds = perf_counter() - start
    hits, n_obs = top1_counts(cand, scores, len(obs))
    return {
        "top1_accuracy": hits / n_obs,
        "candidate_log_loss": float(log_loss(cand["label"], scores, labels=[0, 1])),
        "candidate_f1_at_0_5": float(
            f1_score(cand["label"], scores >= 0.5, zero_division=0)
        ),
        "candidate_miss_rate": candidate_miss_rate(cand, n_obs),
        "predict_seconds": predict_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=Path("features"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/mlp"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--cv-only", action="store_true",
        help="Run the search without reading student-testing data.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train, train_obs = load_split(args.features_dir, "train")
    folds = list(grouped_folds(train_obs, args.folds))
    baseline_cv = cv_top1(BASELINE, train, train_obs, folds)

    start = perf_counter()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
    )
    study.optimize(
        lambda trial: cv_top1(trial_params(trial), train, train_obs, folds),
        n_trials=args.trials,
        n_jobs=1,
    )
    search_seconds = perf_counter() - start
    best = trial_params(optuna.trial.FixedTrial(study.best_params))
    study.trials_dataframe().to_csv(args.out / "trials.csv", index=False)

    report = {
        "seed": SEED,
        "folds": args.folds,
        "features": FEATURES,
        "train_trips": int(train_obs["trip_id"].nunique()),
        "train_observations": len(train_obs),
        "train_candidate_miss_rate": candidate_miss_rate(train, len(train_obs)),
        "baseline_params": BASELINE,
        "tuned_params": best,
        "baseline_cv_top1": baseline_cv,
        "tuned_cv_top1": study.best_value,
        "search_seconds": search_seconds,
    }

    if not args.cv_only:
        test, test_obs = load_split(args.features_dir, "student_test")
        overlap = set(train_obs["trip_id"]) & set(test_obs["trip_id"])
        if overlap:
            raise ValueError(f"Trip IDs overlap across splits: {sorted(overlap)}")

        for name, params in [("baseline", BASELINE), ("tuned", best)]:
            model = make_model(params)
            start = perf_counter()
            model.fit(model_inputs(train), train["label"])
            fit_seconds = perf_counter() - start
            report[f"{name}_student_test"] = {
                **evaluate(model, test, test_obs),
                "fit_seconds": fit_seconds,
            }
            if name == "tuned":
                joblib.dump(
                    {"pipeline": model, "features": FEATURES},
                    args.out / "tuned_mlp.joblib",
                )

    text = json.dumps(report, indent=2)
    (args.out / "report.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()