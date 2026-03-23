#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import Counter
import pandas as pd


BASE_COLUMNS = [
    "id",
    "Tokens",
    "Subject_Entity",
    "Object_Entity",
    "Subject_Type",
    "Object_Type",
    "Subject_Start",
    "Subject_End",
    "Object_Start",
    "Object_End",
    "True_Labels",
]

PREDICTION_COL_CANDIDATES = [
    "Prediction",
    "LLM_Prediction",
]


def detect_prediction_column(df: pd.DataFrame, path: Path):
    for col in PREDICTION_COL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(f"No prediction column found in {path}")


def load_voter_file(voters_root: Path, voter_name: str, dataset: str, split: str) -> pd.DataFrame:
    csv_path = voters_root / voter_name / dataset / f"{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing voter file: {csv_path}")

    df = pd.read_csv(csv_path)

    pred_col = detect_prediction_column(df, csv_path)

    required_cols = set(BASE_COLUMNS + [pred_col])
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df[BASE_COLUMNS + [pred_col]].copy()
    df = df.rename(columns={pred_col: f"Prediction_{voter_name}"})

    return df


def vote_row(predictions, k: int, fallback: str = "no_relation") -> str:
    valid_preds = [p for p in predictions if pd.notna(p)]

    if not valid_preds:
        return fallback

    counts = Counter(valid_preds)
    pred, count = counts.most_common(1)[0]

    if count >= k:
        return pred

    return fallback


def main():
    parser = argparse.ArgumentParser(description="VoteRE voting script")

    parser.add_argument("--voters_root", type=str, default="Data/3_Voters_Predictions")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["TACRED", "TACREV", "ReTACRED"])
    parser.add_argument("--split", type=str, required=True,
                        choices=["train", "dev", "test", "example"])
    parser.add_argument("--voters", type=str, nargs="+", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--fallback", type=str, default="no_relation")

    args = parser.parse_args()

    if args.k > len(args.voters):
        raise ValueError("k cannot be greater than number of voters")

    voters_root = Path(args.voters_root)

    merged_df = None
    prediction_cols = []

    print(f"Voting on dataset={args.dataset}, split={args.split}")
    print(f"Voters: {args.voters}")
    print(f"k = {args.k}")

    for voter in args.voters:
        voter_df = load_voter_file(voters_root, voter, args.dataset, args.split)

        pred_col = f"Prediction_{voter}"
        prediction_cols.append(pred_col)

        if merged_df is None:
            merged_df = voter_df
        else:
            merged_df = merged_df.merge(
                voter_df[["id", pred_col]],
                on="id",
                how="inner"
            )

    if merged_df is None or merged_df.empty:
        raise ValueError("No data after merging voters")

    merged_df["Final_Prediction"] = merged_df[prediction_cols].apply(
        lambda row: vote_row(row.tolist(), args.k, args.fallback),
        axis=1
    )

    output_cols = BASE_COLUMNS + prediction_cols + ["Final_Prediction"]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_df[output_cols].to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Rows: {len(merged_df)}")


if __name__ == "__main__":
    main()