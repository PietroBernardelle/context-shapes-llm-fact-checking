from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from evaluation_utils import filter_subset, result_keys
from prompt_utils import format_position, load_prompt_config, project_root


@dataclass
class Coverage:
    rows: int
    missing_rows: int
    extra_rows: int
    invalid_predictions: int
    label_mismatches: int

    @property
    def complete(self) -> bool:
        return not (self.missing_rows or self.extra_rows or
                    self.invalid_predictions or self.label_mismatches)


def check_coverage(
    frame: pd.DataFrame, subset: pd.DataFrame, allow_outside_subset: bool = False,
) -> Coverage:
    """Compare row multiplicities as well as identities; duplicates are significant."""
    expected_keys = result_keys(subset)
    if allow_outside_subset:
        frame = filter_subset(frame, set(expected_keys))
    observed_keys = result_keys(frame)
    expected = Counter(expected_keys)
    observed = Counter(observed_keys)
    if not {"label", "pred_label"}.issubset(frame.columns):
        raise ValueError("Result file requires label and pred_label columns")
    valid = frame["label"].isin([0, 1]) & frame["pred_label"].isin([0, 1])
    expected_labels = Counter(zip(expected_keys, subset["label"].tolist()))
    observed_labels = Counter(zip(observed_keys, frame["label"].tolist()))
    mismatches = sum((observed_labels - expected_labels).values())
    return Coverage(len(frame), sum((expected - observed).values()),
                    sum((observed - expected).values()), int((~valid).sum()), mismatches)


def expected_files(config: dict, results_dir: Path) -> list[Path]:
    files = []
    for model in config["models"]:
        for condition in ["claim_only", "multi"]:
            files.append(results_dir / model / f"res_{condition}_no_filler.pqt")
        for length in config["settings"]["filler_lengths"][model]:
            for position in config["settings"]["evidence_positions"]:
                token = format_position(position)
                files.append(results_dir / "subset_seed42" / model /
                             f"res_multi_len{length}_pos{token}_subset_seed42.pqt")
    return files


def main() -> None:
    root = project_root()
    parser = ArgumentParser(description="Check paper run coverage before producing figures and tables.")
    parser.add_argument("--config", type=Path, default=root / "config/paper_config.json")
    parser.add_argument("--results-dir", type=Path, default=root / "out/results")
    parser.add_argument("--subset-file", type=Path, default=root / "data/processed/experiment_subset_seed42.pqt")
    args = parser.parse_args()
    config = load_prompt_config(args.config)
    subset = (pd.read_csv(args.subset_file) if args.subset_file.suffix == ".csv"
              else pd.read_parquet(args.subset_file))
    if subset.empty or not subset["label"].isin([0, 1]).all():
        raise ValueError("The subset must contain binary labelled rows")
    counts = subset["dataset"].astype(str).str.strip().str.lower().value_counts().to_dict()
    expected_counts = {key.lower(): value for key, value in config["expected_dataset_counts"].items()}
    if counts != expected_counts:
        raise ValueError(f"Subset sizes differ from configuration: {counts}; expected {expected_counts}")
    files = expected_files(config, args.results_dir)
    problems = 0
    for path in files:
        relative = path.relative_to(args.results_dir)
        if not path.is_file():
            print(f"MISSING {relative}")
            problems += 1
            continue
        try:
            frame = pd.read_parquet(path, columns=["dataset", "claim", "label", "pred_label"])
            coverage = check_coverage(frame, subset, allow_outside_subset=path.name.endswith("_no_filler.pqt"))
        except Exception as error:
            print(f"UNREADABLE {relative}: {type(error).__name__}")
            problems += 1
            continue
        if not coverage.complete:
            print(f"INCOMPLETE {relative}: {coverage}")
            problems += 1
    print(f"Checked {len(files)} expected files; {problems} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
