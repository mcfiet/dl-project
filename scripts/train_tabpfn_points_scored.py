from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.points_scored_model import (
    CAT_COLS,
    NUM_COLS,
    TARGET_COL,
    MASK_TOKENS,
    DROPOUT_TOKEN,
    ModelBundle,
    apply_dropout_token,
    build_category_mappings,
    mask_unseen_categories,
    encode_categories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TabPFN model for points_scored classification and save bundle."
    )
    parser.add_argument(
        "--train",
        default="data/points_scored/dataset_years_train.csv",
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--val",
        default="data/points_scored/dataset_years_val.csv",
        help="Path to validation CSV.",
    )
    parser.add_argument(
        "--test",
        default="data/points_scored/dataset_years_test.csv",
        help="Path to test CSV (used for mapping only).",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.5,
        help="Apply categorical dropout during training (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for TabPFN (default: cpu).",
    )
    parser.add_argument(
        "--output",
        default="models/tabpfn_points_scored.pkl",
        help="Where to save the model bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    test = pd.read_csv(args.test)

    known_categories = {col: set(train[col].unique()) for col in MASK_TOKENS}

    train = mask_unseen_categories(train, known_categories, MASK_TOKENS)
    val = mask_unseen_categories(val, known_categories, MASK_TOKENS)
    test = mask_unseen_categories(test, known_categories, MASK_TOKENS)

    train = apply_dropout_token(train, MASK_TOKENS.keys(), args.dropout_rate, DROPOUT_TOKEN)

    mappings = build_category_mappings(
        train=train,
        val=val,
        test=test,
        cat_cols=CAT_COLS,
        mask_tokens=MASK_TOKENS,
        dropout_token=DROPOUT_TOKEN,
    )

    train = encode_categories(train, mappings, MASK_TOKENS, CAT_COLS)
    val = encode_categories(val, mappings, MASK_TOKENS, CAT_COLS)

    X_train, y_train = train[CAT_COLS + NUM_COLS], train[TARGET_COL]
    X_val, y_val = val[CAT_COLS + NUM_COLS], val[TARGET_COL]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    cat_idx = [X_train.columns.get_loc(c) for c in CAT_COLS]

    clf = TabPFNClassifier(device=device, categorical_features_indices=cat_idx)
    clf.fit(X_train, y_train)

    val_acc = (clf.predict(X_val) == y_val).mean()
    print(f"Validation accuracy: {val_acc:.4f}")

    bundle = ModelBundle(
        model=clf,
        cat_cols=CAT_COLS,
        num_cols=NUM_COLS,
        mappings=mappings,
        mask_tokens=MASK_TOKENS,
        dropout_token=DROPOUT_TOKEN,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved model bundle to {output_path}")


if __name__ == "__main__":
    main()
