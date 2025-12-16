#!/usr/bin/env python3
"""
Run a simple learning curve by increasing the number of training seasons.

Assumes per-season CSVs in data/years (or configurable), with a 'year' column.
Keeps a fixed val/test split, and for multiple training-year subsets trains a
LightGBM model (with OHE + imputed/scaled numeric features) and reports metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learning curve over training years.")
    p.add_argument("--data-dir", type=Path, default=Path("data/years"), help="Dir with per-year CSVs")
    p.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for per-year files")
    p.add_argument(
        "--train-sets",
        nargs="+",
        required=True,
        help="Comma-separated year lists, e.g. '2022,2023' '2020,2021,2022,2023'",
    )
    p.add_argument("--val-years", nargs="+", required=True, type=int, help="Validation years")
    p.add_argument("--test-years", nargs="+", required=True, type=int, help="Test years")
    p.add_argument(
        "--target", type=str, default="points_scored", help="Target column for classification"
    )
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
    p.add_argument("--max-estimators", type=int, default=2000, help="LightGBM n_estimators")
    p.add_argument("--learning-rate", type=float, default=0.05, help="LightGBM learning rate")
    p.add_argument("--output", type=Path, default=Path("learning_curve_results.csv"))
    return p.parse_args()


def load_seasons(data_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "year" not in df.columns:
            raise ValueError(f"'year' column missing in {f}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def subset_by_year(df: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    return df[df["year"].isin(years)].copy()


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


def evaluate(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
    }


def main() -> None:
    args = parse_args()
    df_all = load_seasons(args.data_dir, args.pattern)
    df_all = df_all.dropna(subset=[args.target, "year"])
    df_all["year"] = df_all["year"].astype(int)

    val_df = subset_by_year(df_all, [int(y) for y in args.val_years])
    test_df = subset_by_year(df_all, [int(y) for y in args.test_years])
    features = args.cat_cols + args.num_cols

    preprocess = build_preprocess(args.cat_cols, args.num_cols)
    # Fit preprocess on train subset each time for fair comparison

    results = []
    for spec in args.train_sets:
        train_years = [int(y) for y in spec.split(",")]
        train_df = subset_by_year(df_all, train_years)
        if train_df.empty:
            print(f"[skip] train years {train_years} produced empty dataset")
            continue

        X_train, y_train = train_df[features], train_df[args.target]
        X_val, y_val = val_df[features], val_df[args.target]
        X_test, y_test = test_df[features], test_df[args.target]

        pre = preprocess.fit(X_train)
        X_train_enc = pre.transform(X_train)
        X_val_enc = pre.transform(X_val)
        X_test_enc = pre.transform(X_test)

        model = fit_lightgbm(X_train_enc, y_train, X_val_enc, y_val, args.learning_rate, args.max_estimators)
        prob_val = model.predict_proba(X_val_enc)[:, 1]
        prob_test = model.predict_proba(X_test_enc)[:, 1]

        # simple threshold tuning on val F1
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
                "train_years": ",".join(map(str, train_years)),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
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
            f"Train years {train_years} -> thr {best_thr:.2f} | "
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
