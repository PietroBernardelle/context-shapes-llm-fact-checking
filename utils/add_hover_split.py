"""Add a split column to combined_datasets.csv based on Hover dev claims."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_hover_claims(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items: Iterable[object]
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if isinstance(data.get("data"), list):
            items = data["data"]
        elif isinstance(data.get("claims"), list):
            items = data["claims"]
        else:
            items = []
    else:
        items = []

    claims: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            claim = item.get("claim")
            if isinstance(claim, str):
                claims.add(claim.strip())
    return claims


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--hover-json",
        default=None,
        help="Path to hover_dev_release_v1.1.json (defaults to data/raw/hover_dev_release_v1.1.json)",
    )
    parser.add_argument(
        "--combined-csv",
        default=None,
        help="Path to combined_datasets.csv (defaults to data/processed/combined_datasets.csv)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (defaults to data/processed/combined_datasets_with_splits.csv)",
    )
    args = parser.parse_args()

    hover_path = Path(args.hover_json) if args.hover_json else PROJECT_ROOT / "data" / "raw" / "hover_dev_release_v1.1.json"
    combined_path = (
        Path(args.combined_csv)
        if args.combined_csv
        else PROJECT_ROOT / "data" / "processed" / "combined_datasets.csv"
    )
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else combined_path.with_name("combined_datasets_with_split.csv")
    )

    if not hover_path.exists():
        raise SystemExit(f"Hover JSON not found: {hover_path}")
    if not combined_path.exists():
        raise SystemExit(f"Combined CSV not found: {combined_path}")

    hover_claims = load_hover_claims(hover_path)
    if not hover_claims:
        raise SystemExit(f"No Hover claims found in: {hover_path}")

    df = pd.read_csv(combined_path)
    if "claim" not in df.columns:
        raise SystemExit("combined_datasets.csv is missing a 'claim' column.")
    if "dataset" not in df.columns:
        raise SystemExit("combined_datasets.csv is missing a 'dataset' column.")

    df["split"] = "dev"

    dataset_series = df["dataset"].fillna("").astype(str).str.strip().str.lower()
    claim_series = df["claim"].fillna("").astype(str).str.strip()

    hover_mask = dataset_series == "hover"
    in_dev_mask = claim_series.isin(hover_claims)

    df.loc[hover_mask & ~in_dev_mask, "split"] = "train"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    hover_rows = int(hover_mask.sum())
    train_rows = int((df["split"] == "train").sum())
    dev_rows = int((df["split"] == "dev").sum())
    print(f"Saved: {output_path}")
    print(f"Hover rows: {hover_rows}")
    print(f"Split counts: dev={dev_rows}, train={train_rows}")


if __name__ == "__main__":
    main()
