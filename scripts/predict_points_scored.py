from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.points_scored_model import ModelBundle, prepare_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict points_scored from JSON input.")
    parser.add_argument(
        "--model",
        default="../models/tabpfn_points_scored.pkl",
        help="Path to saved model bundle.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSON object or path to JSON file with feature values.",
    )
    return parser.parse_args()


def load_input(payload: str) -> dict:
    path = Path(payload)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(payload)


def main() -> None:
    args = parse_args()

    with Path(args.model).open("rb") as f:
        bundle: ModelBundle = pickle.load(f)

    payload = load_input(args.input)
    df = pd.DataFrame([payload])
    X = prepare_features(df, bundle)

    proba = bundle.model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)

    print(json.dumps({"prediction": pred, "probability": float(proba)}, indent=2))


if __name__ == "__main__":
    main()
