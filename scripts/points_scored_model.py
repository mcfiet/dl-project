from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

MASK_TOKENS = {
    "driver_id": "new_driver",
    "constructor_id": "new_constructor",
    "circuit_id": "new_circuit",
}
DROPOUT_TOKEN = "masked_dropout"

CAT_COLS = [
    "driver_id",
    "constructor_id",
    "circuit_id",
    "year",
]
NUM_COLS = [
    "grid_position",
    "quali_delta",
    "quali_tm_delta",
    "season_pts_driver",
    "season_pts_team",
    "last_3_avg",
    "is_street_circuit",
    "is_wet",
]
TARGET_COL = "points_scored"


@dataclass
class ModelBundle:
    model: Any
    cat_cols: List[str]
    num_cols: List[str]
    mappings: Dict[str, Dict[Any, int]]
    mask_tokens: Dict[str, str]
    dropout_token: str = DROPOUT_TOKEN


def mask_unseen_categories(
    df: pd.DataFrame,
    known_categories: Mapping[str, Iterable[Any]],
    mask_tokens: Mapping[str, str],
) -> pd.DataFrame:
    df = df.copy()
    for col, token in mask_tokens.items():
        if col not in df.columns:
            continue
        known = set(known_categories[col])
        df.loc[~df[col].isin(known), col] = token
    return df


def apply_dropout_token(
    df: pd.DataFrame,
    cols: Iterable[str],
    p: float,
    token: str = DROPOUT_TOKEN,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if p <= 0:
        return df
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    for col in cols:
        if col not in df.columns:
            continue
        mask = rng.random(len(df)) < p
        df.loc[mask, col] = token
    return df


def build_category_mappings(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: Iterable[str],
    mask_tokens: Mapping[str, str],
    dropout_token: str = DROPOUT_TOKEN,
) -> Dict[str, Dict[Any, int]]:
    mappings: Dict[str, Dict[Any, int]] = {}
    for col in cat_cols:
        all_vals = pd.concat([train[col], val[col], test[col]], axis=0)
        categories = all_vals.astype("category").cat.categories
        if col in mask_tokens:
            categories = categories.union([mask_tokens[col], dropout_token])
        mapping = {v: i for i, v in enumerate(categories)}
        mappings[col] = mapping
    return mappings


def encode_categories(
    df: pd.DataFrame,
    mappings: Mapping[str, Mapping[Any, int]],
    mask_tokens: Mapping[str, str],
    cat_cols: Iterable[str],
) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        mapping = mappings[col]
        token = mask_tokens.get(col)
        if token is not None:
            df.loc[~df[col].isin(mapping), col] = token
        mapped = df[col].map(mapping)
        if mapped.isna().any():
            missing = sorted({str(v) for v in df.loc[mapped.isna(), col].unique()})
            raise ValueError(f"Unknown category in '{col}': {missing}")
        df[col] = mapped.astype("int64")
    return df


def prepare_features(
    df: pd.DataFrame,
    bundle: ModelBundle,
) -> pd.DataFrame:
    missing = [c for c in bundle.cat_cols + bundle.num_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    for col in bundle.num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[bundle.num_cols].isna().any().any():
        bad = df[bundle.num_cols].isna().any()
        missing_nums = [col for col, is_bad in bad.items() if is_bad]
        raise ValueError(f"Invalid numeric values in: {missing_nums}")
    df = encode_categories(df, bundle.mappings, bundle.mask_tokens, bundle.cat_cols)
    return df[bundle.cat_cols + bundle.num_cols]
