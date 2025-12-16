#!/usr/bin/env python3
"""
Learning curve by training-set size (row count), keeping fixed val/test splits.

Usage example (build subsets and evaluate):
  python scripts/run_learning_curve_sizes.py \
      --train data/grandprix_features_train.csv \
      --val data/grandprix_features_val.csv \
      --test data/grandprix_features_test.csv \
      --train-sizes 500 1000 2000 3740 \
      --output data/learning_curve_sizes.csv \
      --subset-dir data/train_subsets
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learning curve over different train set sizes.")
    p.add_argument("--train", type=Path, default=Path("data/grandprix_features_train.csv"))
    p.add_argument("--val", type=Path, default=Path("data/grandprix_features_val.csv"))
    p.add_argument("--test", type=Path, default=Path("data/grandprix_features_test.csv"))
    p.add_argument("--train-sizes", type=int, nargs="+", required=True, help="Row counts to sample for training")
    p.add_argument("--target", type=str, default="points_scored")
    p.add_argument(
        "--cat-cols",
        nargs="+",
        default=["driver_id", "constructor_id", "circuit_id"],
        help="Categorical columns (one-hot)",
    )
    p.add_argument(
        "--num-cols",
        nargs="+",
        default=[
            "grid_position",
            "quali_delta",
            "quali_tm_delta",
            "season_pts_driver",
            "season_pts_team",
            "last_3_avg",
            "is_street_circuit",
            "is_wet",
        ],
        help="Numeric columns",
    )
    p.add_argument("--learning-rate", type=float, default=0.05, help="LightGBM learning rate")
    p.add_argument("--max-estimators", type=int, default=2000, help="LightGBM n_estimators")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("learning_curve_sizes.csv"))
    p.add_argument(
        "--subset-dir",
        type=Path,
        default=Path("data/train_subsets"),
        help="Directory to save sampled train subsets as CSV (one file per size).",
    )
    return p.parse_args()


def stratified_sample(df: pd.DataFrame, n: int, target: str, random_state: int) -> pd.DataFrame:
    if n >= len(df):
        return df.sample(frac=1, random_state=random_state)  # just shuffle
    grouped = []
    for _, g in df.groupby(target):
        k = max(1, int(round(len(g) * n / len(df))))
        grouped.append(g.sample(n=min(k, len(g)), random_state=random_state))
    sampled = pd.concat(grouped).sample(n=n, random_state=random_state, replace=False)
    return sampled


def build_preprocess(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ]
    )


def fit_lightgbm(X_train, y_train, X_val, y_val, learning_rate: float, max_estimators: int):
    import lightgbm as lgb  # type: ignore

    params = {
        "objective": "binary",
        "learning_rate": learning_rate,
        "num_leaves": 63,
        "max_depth": -1,
        "n_estimators": max_estimators,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model


def evaluate(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
    }


def main() -> None:
    args = parse_args()
    try:
        import lightgbm  # noqa: F401
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit("Please install lightgbm (pip install lightgbm)") from exc

    df_train = pd.read_csv(args.train)
    df_val = pd.read_csv(args.val)
    df_test = pd.read_csv(args.test)

    for df, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        missing = set(args.cat_cols + args.num_cols + [args.target]) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    features = args.cat_cols + args.num_cols
    preprocess = build_preprocess(args.cat_cols, args.num_cols)
    rng_seed = args.random_state

    results = []
    args.subset_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted(set(args.train_sizes))
    for size in sizes:
        subset = stratified_sample(df_train, size, args.target, rng_seed)
        subset_path = args.subset_dir / f"train_subset_{len(subset)}.csv"
        subset.to_csv(subset_path, index=False)

        X_tr, y_tr = subset[features], subset[args.target]
        X_val, y_val = df_val[features], df_val[args.target]
        X_test, y_test = df_test[features], df_test[args.target]

        pre = preprocess.fit(X_tr)
        X_tr_enc = pre.transform(X_tr)
        X_val_enc = pre.transform(X_val)
        X_test_enc = pre.transform(X_test)

        model = fit_lightgbm(X_tr_enc, y_tr, X_val_enc, y_val, args.learning_rate, args.max_estimators)
        prob_val = model.predict_proba(X_val_enc)[:, 1]
        prob_test = model.predict_proba(X_test_enc)[:, 1]

        # Threshold tuning on val F1
        thresholds = np.linspace(0.05, 0.95, 19)
        best_thr, best_f1 = 0.5, -np.inf
        for t in thresholds:
            f1 = f1_score(y_val, (prob_val >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, t

        val_metrics = evaluate(y_val.to_numpy(), prob_val, best_thr)
        test_metrics = evaluate(y_test.to_numpy(), prob_test, best_thr)

        results.append(
            {
                "train_size": len(subset),
                "subset_path": str(subset_path),
                "threshold": best_thr,
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_bal_acc": val_metrics["balanced_acc"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_bal_acc": test_metrics["balanced_acc"],
            }
        )
        print(
            f"Train size {len(subset)} -> thr {best_thr:.2f} | "
            f"Val F1 {val_metrics['f1']:.3f} BalAcc {val_metrics['balanced_acc']:.3f} | "
            f"Test F1 {test_metrics['f1']:.3f} BalAcc {test_metrics['balanced_acc']:.3f}"
        )

    out_df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote results to {args.output}")
    if not out_df.empty:
        print(out_df)


if __name__ == "__main__":
    main()
