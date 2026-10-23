from __future__ import annotations

import math
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from prompt_utils import project_root


def empty_counts() -> dict[str, int]:
    return {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "correct": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


def compute_counts(df: pd.DataFrame) -> dict[str, int]:
    labels = df["label"]
    preds = df["pred_label"]
    valid_mask = labels.isin([0, 1]) & preds.isin([0, 1])
    valid = df.loc[valid_mask, ["label", "pred_label"]]

    tp = int(((valid["label"] == 1) & (valid["pred_label"] == 1)).sum())
    fp = int(((valid["label"] == 0) & (valid["pred_label"] == 1)).sum())
    fn = int(((valid["label"] == 1) & (valid["pred_label"] == 0)).sum())
    correct = int((valid["label"] == valid["pred_label"]).sum())
    total = int(len(df))
    valid_count = int(len(valid))

    return {
        "total": total,
        "valid": valid_count,
        "invalid": total - valid_count,
        "correct": correct,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def init_label_counts() -> dict[int, dict[str, int]]:
    return {
        0: {"actual": 0, "predicted": 0, "correct": 0},
        1: {"actual": 0, "predicted": 0, "correct": 0},
    }


def compute_label_counts(df: pd.DataFrame) -> dict[int, dict[str, int]]:
    labels = df["label"]
    preds = df["pred_label"]
    valid_mask = labels.isin([0, 1]) & preds.isin([0, 1])
    valid = df.loc[valid_mask, ["label", "pred_label"]]

    counts = init_label_counts()
    for label in [0, 1]:
        actual = int((valid["label"] == label).sum())
        predicted = int((valid["pred_label"] == label).sum())
        correct = int(((valid["label"] == label) & (valid["pred_label"] == label)).sum())
        counts[label]["actual"] = actual
        counts[label]["predicted"] = predicted
        counts[label]["correct"] = correct
    return counts


def add_counts(target: dict[str, int], counts: dict[str, int]) -> None:
    for key, value in counts.items():
        target[key] += value


def add_label_counts(target: dict[int, dict[str, int]], counts: dict[int, dict[str, int]]) -> None:
    for label in [0, 1]:
        for key in ["actual", "predicted", "correct"]:
            target[label][key] += counts[label][key]


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def format_metric(value: float) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def format_metric_diff(value: float, baseline: float | None = None) -> str:
    if value is None or math.isnan(value):
        return "NA"
    if baseline is None or math.isnan(baseline):
        return f"{value:.4f}"
    diff = value - baseline
    return f"{value:.4f} ({diff:+.4f})"


def read_subset_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pqt", ".parquet"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_subset_metadata(path: Path) -> tuple[set[tuple[str, str]], dict[str, str], list[str]]:
    df = read_subset_file(path)
    required = {"claim", "dataset"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in subset file {path}: {sorted(missing)}")

    dataset_raw = df["dataset"].fillna("").astype(str).str.strip()
    dataset_norm = dataset_raw.str.lower()
    claim_norm = df["claim"].fillna("").astype(str).str.strip()

    keys = set(zip(dataset_norm, claim_norm))
    if not keys:
        raise SystemExit(f"No (dataset, claim) keys found in subset file: {path}")

    display_map: dict[str, str] = {}
    dataset_order: list[str] = []
    seen: set[str] = set()
    for raw, norm in zip(dataset_raw.tolist(), dataset_norm.tolist()):
        if norm and norm not in seen:
            seen.add(norm)
            dataset_order.append(norm)
        if norm and norm not in display_map:
            display_map[norm] = raw if raw else norm

    return keys, display_map, dataset_order


def summarize_files(
    files: list[Path],
    subset_keys: set[tuple[str, str]],
) -> tuple[
    dict[str, int],
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
    dict[str, set[str]],
    dict[str, dict[int, dict[str, int]]],
    set[str],
    list[Path],
    list[Path],
    list[Path],
]:
    totals = empty_counts()
    dataset_totals: dict[str, dict[str, int]] = {}
    dataset_totals_norm: dict[str, dict[str, int]] = {}
    dataset_files: dict[str, set[str]] = {}
    label_counts_norm: dict[str, dict[int, dict[str, int]]] = {}
    used_files: set[str] = set()

    skipped: list[Path] = []
    missing_dataset: list[Path] = []
    missing_claim: list[Path] = []

    for file_path in sorted(files):
        df = pd.read_parquet(file_path)
        if "label" not in df.columns or "pred_label" not in df.columns:
            skipped.append(file_path)
            continue
        if "dataset" not in df.columns:
            missing_dataset.append(file_path)
            continue
        if "claim" not in df.columns:
            missing_claim.append(file_path)
            continue

        dataset_norm = df["dataset"].fillna("").astype(str).str.strip().str.lower()
        claim_norm = df["claim"].fillna("").astype(str).str.strip()
        subset_mask = [(d, c) in subset_keys for d, c in zip(dataset_norm, claim_norm)]
        filtered_df = df.loc[subset_mask]
        if filtered_df.empty:
            continue

        used_files.add(file_path.name)
        add_counts(totals, compute_counts(filtered_df))

        for dataset_name, group in filtered_df.groupby("dataset", dropna=False):
            dataset_key = "NA" if pd.isna(dataset_name) else str(dataset_name)
            normalized_key = dataset_key.strip().lower()

            if dataset_key not in dataset_totals:
                dataset_totals[dataset_key] = empty_counts()
                dataset_files[dataset_key] = set()
            if normalized_key not in dataset_totals_norm:
                dataset_totals_norm[normalized_key] = empty_counts()
            if normalized_key not in label_counts_norm:
                label_counts_norm[normalized_key] = init_label_counts()

            dataset_counts = compute_counts(group)
            add_counts(dataset_totals[dataset_key], dataset_counts)
            add_counts(dataset_totals_norm[normalized_key], dataset_counts)
            dataset_files[dataset_key].add(file_path.name)
            add_label_counts(label_counts_norm[normalized_key], compute_label_counts(group))

    return (
        totals,
        dataset_totals,
        dataset_totals_norm,
        dataset_files,
        label_counts_norm,
        used_files,
        skipped,
        missing_dataset,
        missing_claim,
    )


def ordered_dataset_keys(available: dict[str, object], preferred_order: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for key in preferred_order:
        if key in available and key not in seen:
            out.append(key)
            seen.add(key)
    for key in sorted(available.keys()):
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def print_table(title: str, rows: list[dict[str, str]], headers: list[str]) -> None:
    if not rows:
        return
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))

    print(title)
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    separator_line = "  ".join("-" * widths[h] for h in headers)
    print(header_line)
    print(separator_line)
    for row in rows:
        print("  ".join(row[h].ljust(widths[h]) for h in headers))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Path to results directory (defaults to out/results)",
    )
    parser.add_argument(
        "--subset-file",
        default=None,
        help=(
            "Subset file generated for experiments (.pqt/.parquet/.csv). "
            "Defaults to data/processed/experiment_subset_seed42.pqt"
        ),
    )
    args = parser.parse_args()

    root = project_root()
    results_dir = Path(args.results_dir) if args.results_dir else root / "out" / "results"
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    subset_file = (
        Path(args.subset_file)
        if args.subset_file
        else root / "data" / "processed" / "experiment_subset_seed42.pqt"
    )
    if not subset_file.exists():
        raise SystemExit(f"Subset file not found: {subset_file}")

    subset_keys, dataset_display_map, dataset_order = load_subset_metadata(subset_file)
    print(f"Using subset filter: {subset_file}")

    overall_rows: list[dict[str, str]] = []
    dataset_rows: list[dict[str, str]] = []
    final_rows: list[dict[str, str]] = []
    final_label_rows: list[dict[str, str]] = []
    multi_final_rows: list[dict[str, str]] = []
    multi_label_rows: list[dict[str, str]] = []

    skipped_files: list[Path] = []
    missing_dataset_files: list[Path] = []
    missing_claim_files: list[Path] = []

    claim_metrics: dict[tuple[str, str], dict[str, float]] = {}
    claim_label_metrics: dict[tuple[str, str, int], dict[str, float]] = {}

    for model_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        all_pqts = sorted(model_dir.glob("*.pqt"))
        claim_files = [f for f in all_pqts if f.name != "res_multi_no_filler.pqt"]
        multi_files = [f for f in all_pqts if f.name == "res_multi_no_filler.pqt"]

        if claim_files:
            (
                totals,
                dataset_totals,
                dataset_totals_norm,
                dataset_files,
                label_counts_norm,
                used_files,
                skipped,
                missing_dataset,
                missing_claim,
            ) = summarize_files(claim_files, subset_keys)

            skipped_files.extend(skipped)
            missing_dataset_files.extend(missing_dataset)
            missing_claim_files.extend(missing_claim)

            if totals["total"] > 0:
                accuracy = safe_ratio(totals["correct"], totals["valid"])
                f1 = safe_ratio(2 * totals["tp"], (2 * totals["tp"] + totals["fp"] + totals["fn"]))
                overall_rows.append(
                    {
                        "Model": model_dir.name,
                        "Files": str(len(used_files)),
                        "Rows": str(totals["total"]),
                        "Valid": str(totals["valid"]),
                        "Invalid": str(totals["invalid"]),
                        "Accuracy": format_metric(accuracy),
                        "F1": format_metric(f1),
                    }
                )

                for dataset_key in sorted(dataset_totals.keys()):
                    ds_counts = dataset_totals[dataset_key]
                    ds_accuracy = safe_ratio(ds_counts["correct"], ds_counts["valid"])
                    ds_f1 = safe_ratio(
                        2 * ds_counts["tp"],
                        (2 * ds_counts["tp"] + ds_counts["fp"] + ds_counts["fn"]),
                    )
                    dataset_rows.append(
                        {
                            "Model": model_dir.name,
                            "Dataset": dataset_key,
                            "Files": str(len(dataset_files.get(dataset_key, set()))),
                            "Rows": str(ds_counts["total"]),
                            "Valid": str(ds_counts["valid"]),
                            "Invalid": str(ds_counts["invalid"]),
                            "Accuracy": format_metric(ds_accuracy),
                            "F1": format_metric(ds_f1),
                        }
                    )

                for dataset_key in ordered_dataset_keys(dataset_totals_norm, dataset_order):
                    ds_counts = dataset_totals_norm[dataset_key]
                    ds_accuracy = safe_ratio(ds_counts["correct"], ds_counts["valid"])
                    ds_f1 = safe_ratio(
                        2 * ds_counts["tp"],
                        (2 * ds_counts["tp"] + ds_counts["fp"] + ds_counts["fn"]),
                    )
                    ds_name = dataset_display_map.get(dataset_key, dataset_key)
                    claim_metrics[(model_dir.name, ds_name)] = {"accuracy": ds_accuracy, "f1": ds_f1}

                    final_rows.append(
                        {
                            "Model": model_dir.name,
                            "Dataset": ds_name,
                            "Rows": str(ds_counts["total"]),
                            "Valid": str(ds_counts["valid"]),
                            "Invalid": str(ds_counts["invalid"]),
                            "Accuracy": format_metric(ds_accuracy),
                            "F1": format_metric(ds_f1),
                        }
                    )

                    label_counts = label_counts_norm.get(dataset_key)
                    if label_counts:
                        for label in [0, 1]:
                            precision = safe_ratio(label_counts[label]["correct"], label_counts[label]["predicted"])
                            recall = safe_ratio(label_counts[label]["correct"], label_counts[label]["actual"])
                            claim_label_metrics[(model_dir.name, ds_name, label)] = {
                                "precision": precision,
                                "recall": recall,
                            }
                            final_label_rows.append(
                                {
                                    "Model": model_dir.name,
                                    "Dataset": ds_name,
                                    "Label": str(label),
                                    "Precision": format_metric(precision),
                                    "Recall": format_metric(recall),
                                }
                            )

        if multi_files:
            (
                _,
                _,
                dataset_totals_norm,
                _,
                label_counts_norm,
                _,
                _,
                _,
                _,
            ) = summarize_files(multi_files, subset_keys)

            for dataset_key in ordered_dataset_keys(dataset_totals_norm, dataset_order):
                ds_counts = dataset_totals_norm[dataset_key]
                ds_accuracy = safe_ratio(ds_counts["correct"], ds_counts["valid"])
                ds_f1 = safe_ratio(
                    2 * ds_counts["tp"],
                    (2 * ds_counts["tp"] + ds_counts["fp"] + ds_counts["fn"]),
                )
                ds_name = dataset_display_map.get(dataset_key, dataset_key)
                base_metrics = claim_metrics.get((model_dir.name, ds_name), {})
                multi_final_rows.append(
                    {
                        "Model": model_dir.name,
                        "Dataset": ds_name,
                        "Rows": str(ds_counts["total"]),
                        "Valid": str(ds_counts["valid"]),
                        "Accuracy": format_metric_diff(ds_accuracy, base_metrics.get("accuracy")),
                        "F1": format_metric_diff(ds_f1, base_metrics.get("f1")),
                    }
                )

                label_counts = label_counts_norm.get(dataset_key)
                if label_counts:
                    for label in [0, 1]:
                        precision = safe_ratio(label_counts[label]["correct"], label_counts[label]["predicted"])
                        recall = safe_ratio(label_counts[label]["correct"], label_counts[label]["actual"])
                        base_l_metrics = claim_label_metrics.get((model_dir.name, ds_name, label), {})
                        multi_label_rows.append(
                            {
                                "Model": model_dir.name,
                                "Dataset": ds_name,
                                "Label": str(label),
                                "Precision": format_metric_diff(precision, base_l_metrics.get("precision")),
                                "Recall": format_metric_diff(recall, base_l_metrics.get("recall")),
                            }
                        )

    if not overall_rows:
        print(f"No subset-matching result rows found under: {results_dir}")
        return

    print_table(
        "Overall by model (subset-filtered)",
        overall_rows,
        ["Model", "Files", "Rows", "Valid", "Invalid", "Accuracy", "F1"],
    )

    if dataset_rows:
        print()
        print_table(
            "By model and dataset (subset-filtered)",
            dataset_rows,
            ["Model", "Dataset", "Files", "Rows", "Valid", "Invalid", "Accuracy", "F1"],
        )
    else:
        print("\nNo dataset breakdown available (missing dataset columns).")

    if final_rows:
        print()
        print_table(
            "Subset summary (per dataset)",
            final_rows,
            ["Model", "Dataset", "Rows", "Valid", "Invalid", "Accuracy", "F1"],
        )
    else:
        print("\nNo subset summary rows generated.")

    if final_label_rows:
        print()
        print_table(
            "Label precision/recall (subset-filtered)",
            final_label_rows,
            ["Model", "Dataset", "Label", "Precision", "Recall"],
        )
    else:
        print("\nNo label precision/recall rows generated.")

    if skipped_files:
        print("\nSkipped files (missing label/pred_label columns):")
        for path in skipped_files:
            print(f"- {path}")

    if missing_dataset_files:
        print("\nFiles missing dataset column (excluded):")
        for path in missing_dataset_files:
            print(f"- {path}")

    if missing_claim_files:
        print("\nFiles missing claim column (excluded):")
        for path in missing_claim_files:
            print(f"- {path}")

    print("\n" + "-" * 120 + "\n")
    print("Multi-step (no filler) Analysis vs Claim Only (subset-filtered)")

    if multi_final_rows:
        print()
        print_table(
            "Subset summary (Value and Diff from Claim Only)",
            multi_final_rows,
            ["Model", "Dataset", "Rows", "Valid", "Accuracy", "F1"],
        )
    else:
        print("\nNo multi result files found.")

    if multi_label_rows:
        print()
        print("Label precision/recall columns format: Value (Diff from Claim Only)")
        print_table(
            "",
            multi_label_rows,
            ["Model", "Dataset", "Label", "Precision", "Recall"],
        )


if __name__ == "__main__":
    main()
