from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from prompt_utils import project_root, ensure_parent_dir


def sample_balanced(
    df: pd.DataFrame,
    n_per_label: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[int, int]]:
    if n_per_label <= 0:
        raise ValueError("n_per_label must be positive")
    samples: list[pd.DataFrame] = []
    counts: dict[int, int] = {}
    for label in [0, 1]:
        subset = df[df["label"] == label]
        if len(subset) < n_per_label:
            raise ValueError(f"FEVEROUS label {label} has {len(subset)} rows; need {n_per_label}")
        take = n_per_label
        counts[label] = take
        if take > 0:
            samples.append(subset.sample(n=take, random_state=seed))
    if not samples:
        return pd.DataFrame(columns=df.columns), counts
    return pd.concat(samples, ignore_index=True), counts


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default=None,
        help=(
            "Path to combined_datasets_with_split.csv "
            "(defaults to data/processed/combined_datasets_with_split.csv)"
        ),
    )
    parser.add_argument(
        "--output-pqt",
        default=None,
        help=(
            "Output parquet path "
            "(defaults to data/processed/experiment_subset_seed42.pqt)"
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Optional CSV output path "
            "(defaults to data/processed/experiment_subset_seed42.csv)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for FEVEROUS balanced subsampling (default: 42)",
    )
    parser.add_argument(
        "--feverous-per-label",
        type=int,
        default=2000,
        help="Rows per FEVEROUS label (0/1) to sample (default: 2000)",
    )
    parser.add_argument(
        "--hover-split",
        default="dev",
        help="Hover split value to keep (default: dev)",
    )
    args = parser.parse_args()

    root = project_root()

    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else root / "data" / "processed" / "combined_datasets_with_split.csv"
    )
    output_pqt = (
        Path(args.output_pqt)
        if args.output_pqt
        else root / "data" / "processed" / "experiment_subset_seed42.pqt"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else root / "data" / "processed" / "experiment_subset_seed42.csv"
    )

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = {"dataset", "split", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    dataset_norm = df["dataset"].fillna("").astype(str).str.strip().str.lower()
    split_norm = df["split"].fillna("").astype(str).str.strip().str.lower()

    # Keep Hover dev only.
    hover_df = df[(dataset_norm == "hover") & (split_norm == args.hover_split.lower())].copy()

    # Keep all ClimateFever rows.
    climate_df = df[dataset_norm == "climatefever"].copy()

    # Balanced FEVEROUS sample, same idea as in 5_results_analysis.py.
    feverous_df = df[dataset_norm == "feverous"].copy()
    feverous_df = feverous_df[feverous_df["label"].isin([0, 1])]
    feverous_sampled, fever_counts = sample_balanced(
        feverous_df,
        n_per_label=args.feverous_per_label,
        seed=args.seed,
    )

    subset_df = pd.concat([hover_df, climate_df, feverous_sampled], ignore_index=True)

    ensure_parent_dir(output_pqt)
    subset_df.to_parquet(output_pqt, index=False)

    if output_csv:
        ensure_parent_dir(output_csv)
        subset_df.to_csv(output_csv, index=False)

    print(f"Saved parquet: {output_pqt}")
    if output_csv:
        print(f"Saved csv: {output_csv}")
    print(
        "Counts | "
        f"Hover({args.hover_split}): {len(hover_df)} | "
        f"ClimateFever: {len(climate_df)} | "
        f"FEVEROUS label0: {fever_counts.get(0, 0)} | "
        f"FEVEROUS label1: {fever_counts.get(1, 0)} | "
        f"Total: {len(subset_df)}"
    )


if __name__ == "__main__":
    main()
