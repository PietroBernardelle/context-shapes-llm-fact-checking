from __future__ import annotations

from pathlib import Path

import pandas as pd

from prompt_utils import load_prompt_config


def macro_f1_from_counts(counts: dict[str, int]) -> float:
    if counts["valid"] == 0:
        return float("nan")
    errors = counts["fp"] + counts["fn"]
    true_refutes = counts["tp"]
    true_supports = counts["correct"] - true_refutes
    scores = [2 * n / (2 * n + errors) if 2 * n + errors else 0.0
              for n in (true_supports, true_refutes)]
    return sum(scores) / 2


def read_subset_keys(path: Path) -> set[tuple[str, str]]:
    frame = pd.read_parquet(path) if path.suffix in {".pqt", ".parquet"} else pd.read_csv(path)
    keys = result_keys(frame)
    if not keys:
        raise ValueError(f"Empty experiment subset: {path}")
    return set(keys)


def result_keys(frame: pd.DataFrame) -> list[tuple[str, str]]:
    missing = {"dataset", "claim"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing identity columns: {sorted(missing)}")
    if frame[["dataset", "claim"]].isna().any().any():
        raise ValueError("Dataset and claim identifiers must not be null")
    datasets = frame["dataset"].astype(str).str.strip().str.lower()
    claims = frame["claim"].astype(str).str.strip()
    return list(zip(datasets, claims))


def filter_subset(frame: pd.DataFrame, keys: set[tuple[str, str]]) -> pd.DataFrame:
    return frame.loc[[key in keys for key in result_keys(frame)]].copy()


def filter_configured_metrics(
    frame: pd.DataFrame, config_path: Path, all_positions: bool = False,
) -> pd.DataFrame:
    """Select the paper grid; all_positions is for historical Figure 1 inspection."""
    if frame.empty:
        return frame.copy()
    config = load_prompt_config(config_path)
    positions = config["settings"]["evidence_positions"]
    lengths = config["settings"]["filler_lengths"]
    keep = []
    for row in frame.itertuples():
        position_matches = pd.notna(row.evidence_position) and (all_positions or any(
            abs(float(row.evidence_position) - float(p)) < 1e-6 for p in positions
        ))
        keep.append(
            row.model in config["models"]
            and row.filler_length in lengths.get(row.model, [])
            and getattr(row, "evidence_order", "multi") in config["settings"]["evidence_orders"]
            and position_matches
        )
    return frame.loc[keep].copy()
