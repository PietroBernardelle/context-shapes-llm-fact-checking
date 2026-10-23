from __future__ import annotations

import math
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation_utils import macro_f1_from_counts, filter_configured_metrics, read_subset_keys, filter_subset
from prompt_utils import project_root, ensure_parent_dir


def get_short_model_name(name: str) -> str:
    return name.replace("-Instruct", "")


def parse_position_token(token: str) -> float | None:
    token = token.strip()
    if token.upper() == "NA" or token == "":
        return None
    try:
        return float(token.replace("p", "."))
    except Exception:
        return None


def extract_file_metadata(file_name: str) -> dict[str, Any] | None:
    stem = Path(file_name).stem
    if not stem.startswith("res_"):
        return None

    core = stem[len("res_") :]
    out: dict[str, Any] = {
        "file_stem": stem,
        "file_name": file_name,
        "evidence_order": None,
        "filler_length": None,
        "evidence_position": None,
        "tag": "",
        "is_filler": False,
    }

    if "_len" in core and "_pos" in core:
        try:
            order_part, rest = core.split("_len", 1)
            len_part, pos_and_tag = rest.split("_pos", 1)
            if "_" in pos_and_tag:
                pos_part, tag = pos_and_tag.split("_", 1)
            else:
                pos_part, tag = pos_and_tag, ""

            filler_length = int(len_part)
            # Map 29k/32k to 32768 to combine them in plots
            if 28000 <= filler_length <= 33000:
                filler_length = 32768
            evidence_position = parse_position_token(pos_part)

            out["evidence_order"] = order_part
            out["filler_length"] = filler_length
            out["evidence_position"] = evidence_position
            out["tag"] = tag
            out["is_filler"] = filler_length > 0 and evidence_position is not None
            return out
        except Exception:
            return None

    # Handles no-filler files (kept for optional summaries).
    if core.endswith("_no_filler"):
        order_part = core[: -len("_no_filler")]
        out["evidence_order"] = order_part
        out["filler_length"] = 0
        out["evidence_position"] = None
        out["is_filler"] = False
        return out

    match = re.match(r"^(?P<order>.+?)_no_filler_(?P<tag>.+)$", core)
    if match:
        out["evidence_order"] = match.group("order")
        out["tag"] = match.group("tag")
        out["filler_length"] = 0
        out["evidence_position"] = None
        out["is_filler"] = False
        return out

    return None


def compute_counts(df: pd.DataFrame) -> dict[str, int]:
    labels = df["label"]
    preds = df["pred_label"]
    valid_mask = labels.isin([0, 1]) & preds.isin([0, 1])
    valid_df = df.loc[valid_mask, ["label", "pred_label"]]

    tp = int(((valid_df["label"] == 1) & (valid_df["pred_label"] == 1)).sum())
    fp = int(((valid_df["label"] == 0) & (valid_df["pred_label"] == 1)).sum())
    fn = int(((valid_df["label"] == 1) & (valid_df["pred_label"] == 0)).sum())
    correct = int((valid_df["label"] == valid_df["pred_label"]).sum())
    total = int(len(df))
    valid = int(len(valid_df))

    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "correct": correct,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def safe_ratio(num: int, den: int) -> float:
    if den == 0:
        return float("nan")
    return num / den


def accuracy_f1(df: pd.DataFrame) -> tuple[float, float, int, int]:
    c = compute_counts(df)
    acc = safe_ratio(c["correct"], c["valid"])
    f1 = macro_f1_from_counts(c)
    return acc, f1, c["valid"], c["total"]


def format_metric(v: float) -> str:
    if v is None or math.isnan(v):
        return "NA"
    return f"{v:.4f}"


def read_results_file(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["dataset", "label", "pred_label"])


def build_metrics_dataframe(
    results_dir: Path,
    tag_filter: str | None = None,
    min_valid: int = 1,
) -> tuple[pd.DataFrame, list[Path], list[Path]]:
    records: list[dict[str, Any]] = []
    skipped_files: list[Path] = []
    bad_schema_files: list[Path] = []

    files = sorted(results_dir.rglob("*.pqt"))
    for file_path in files:
        meta = extract_file_metadata(file_path.name)
        if meta is None:
            skipped_files.append(file_path)
            continue
        if tag_filter and tag_filter not in file_path.name:
            continue

        rel = file_path.relative_to(results_dir)
        model = rel.parts[0] if len(rel.parts) > 1 else "UNKNOWN_MODEL"

        try:
            df = read_results_file(file_path)
        except Exception:
            skipped_files.append(file_path)
            continue

        required = {"label", "pred_label", "dataset"}
        if not required.issubset(df.columns):
            bad_schema_files.append(file_path)
            continue

        # Overall (combined datasets).
        acc, f1, valid, total = accuracy_f1(df)
        if valid >= min_valid:
            records.append(
                {
                    "model": model,
                    "dataset": "ALL",
                    "dataset_norm": "all",
                    "accuracy": acc,
                    "f1": f1,
                    "n_valid": valid,
                    "n_total": total,
                    "source_file": str(rel),
                    **meta,
                }
            )

        # Per-dataset.
        for dataset_name, group in df.groupby("dataset", dropna=False):
            ds = "NA" if pd.isna(dataset_name) else str(dataset_name).strip()
            ds_norm = ds.lower()
            g_acc, g_f1, g_valid, g_total = accuracy_f1(group)
            if g_valid < min_valid:
                continue
            records.append(
                {
                    "model": model,
                    "dataset": ds,
                    "dataset_norm": ds_norm,
                    "accuracy": g_acc,
                    "f1": g_f1,
                    "n_valid": g_valid,
                    "n_total": g_total,
                    "source_file": str(rel),
                    **meta,
                }
            )

    metrics_df = pd.DataFrame(records)
    if metrics_df.empty:
        return metrics_df, skipped_files, bad_schema_files

    metrics_df["filler_length"] = metrics_df["filler_length"].astype("Int64")
    return metrics_df, skipped_files, bad_schema_files


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, output_dir: Path, base_name: str, formats: list[str], dpi: int) -> None:
    for fmt in formats:
        out_path = output_dir / f"{base_name}.{fmt}"
        ensure_parent_dir(out_path)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_single(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    cmap: str,
    output_dir: Path,
    file_stem: str,
    formats: list[str],
    dpi: int,
    num_ticks: int = 15,
    fixed_ticks: np.ndarray | None = None,
) -> None:
    # Ensure folder existence
    ensure_parent_dir(output_dir / "placeholder")
    
    # Save source data
    save_table(data, output_dir / f"{file_stem}.csv")

    pivot = (
        data.groupby(["filler_length", "evidence_position"], as_index=False)[value_col]
        .mean()
        .pivot(index="filler_length", columns="evidence_position", values=value_col)
        .sort_index()
    )
    if pivot.empty:
        return

    # Scale to percentage
    pivot = pivot * 100
    vmin, vmax = 0.0, 100.0

    fig, ax = plt.subplots(figsize=(8, 8))
    
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        linecolor="white",
    )
    # ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Evidence depth (%)")
    ax.set_ylabel("Filler length (tokens)")

    # Handle x-ticks for heatmap
    # Heatmap x-axis uses indices 0.5, 1.5, ... corresponding to columns
    cols = pivot.columns
    n_cols = len(cols)
    
    # Try to convert columns to floats for smart tick location
    try:
        col_floats = np.array([float(x) for x in cols])
        is_numeric = True
    except ValueError:
        is_numeric = False

    if fixed_ticks is not None and is_numeric:
        # Force specific ticks
        tick_locs = []
        tick_labels = []
        for val in fixed_ticks:
            # Find closest column
            idx = (np.abs(col_floats - val)).argmin()
            # Check if it's reasonably close (within 0.01) to avoid misleading labels
            if abs(col_floats[idx] - val) < 0.02:
                tick_locs.append(idx + 0.5)
                tick_labels.append(f"{int(round(col_floats[idx] * 100))}")
        
        # Remove duplicates if any (e.g. if resolution is low)
        unique_ticks = {}
        for l, t in zip(tick_locs, tick_labels):
            unique_ticks[l] = t
        
        ax.set_xticks(list(unique_ticks.keys()))
        ax.set_xticklabels(list(unique_ticks.values()), rotation=45)

    elif is_numeric:
        # Use MaxNLocator to find "nice" values covering the range
        locator = MaxNLocator(nbins=num_ticks)
        desired_ticks = locator.tick_values(col_floats.min(), col_floats.max())
        
        # Filter desired_ticks to only those within actual range
        desired_ticks = [t for t in desired_ticks if col_floats.min() <= t <= col_floats.max()]

        tick_locs = []
        tick_labels = []
        for val in desired_ticks:
            # Find closest column
            idx = (np.abs(col_floats - val)).argmin()
            # Only add if getting closer to this val than any other existing tick
            # (Simple heuristic: if closest match is close enough)
            if abs(col_floats[idx] - val) < 0.025: 
                tick_locs.append(idx + 0.5)
                tick_labels.append(f"{int(round(col_floats[idx] * 100))}")

        # Remove duplicates
        unique_locs = sorted(list(set(tick_locs)))
        unique_labels = [f"{int(round(col_floats[int(l-0.5)] * 100))}" for l in unique_locs]

        if len(unique_locs) > 0:
            ax.set_xticks(unique_locs)
            ax.set_xticklabels(unique_labels, rotation=45)
        else:
            # Fallback for numeric but weird distribution
            target_ticks = num_ticks
            if n_cols > target_ticks:
                step = int(np.ceil(n_cols / target_ticks))
                valid_indices = np.arange(0, n_cols, step)
                ax.set_xticks(valid_indices + 0.5)
                ax.set_xticklabels(cols[valid_indices], rotation=45)

    else:
        # Non-numeric, use simple stepping
        target_ticks = num_ticks
        if n_cols > target_ticks:
            step = int(np.ceil(n_cols / target_ticks))
            valid_indices = np.arange(0, n_cols, step)
            ax.set_xticks(valid_indices + 0.5)
            ax.set_xticklabels(cols[valid_indices], rotation=45)
    
    fig.tight_layout()
    save_figure(fig, output_dir, file_stem, formats, dpi)


def get_baselines(model_name: str, results_root: Path, valid_claims: set[tuple[str, str]] | None) -> dict[str, float]:
    baselines = {}
    # Files are expected in results_root / model_name / ...
    # Mapping: label -> filename
    files = {
        "claim_only": "res_claim_only_no_filler.pqt",
        "multi": "res_multi_no_filler.pqt"
    }
    
    model_dir = results_root / model_name
    if not model_dir.exists():
        return baselines

    for key, fname in files.items():
        fpath = model_dir / fname
        if not fpath.exists():
            continue
        
        try:
            df = pd.read_parquet(fpath, columns=["dataset", "claim", "label", "pred_label"])
            if valid_claims is not None and "claim" in df.columns:
                df = filter_subset(df, valid_claims)
            
            acc, f1, valid, _ = accuracy_f1(df)
            if valid > 0:
                print(f"[{model_name}] Baseline {key}: n={valid}, acc={acc:.4f}, f1={f1:.4f}")
                baselines[f"{key}_accuracy"] = acc
                baselines[f"{key}_f1"] = f1
        except Exception as e:
            print(f"Warning: Failed to process baseline {fpath}: {e}")
            
    return baselines


def plot_line_single(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    output_dir: Path,
    file_stem: str,
    formats: list[str],
    dpi: int,
    model_name: str,
    num_ticks: int = 15,
    fixed_ticks: np.ndarray | None = None,
    baselines: dict[str, float] | None = None,
    xlabel_y: float | None = None,
    xlabel_size: int | None = None,
    title_y: float | None = None,
) -> None:
    ensure_parent_dir(output_dir / "placeholder")
    save_table(data, output_dir / f"{file_stem}.csv")

    # Aggregate if multiple rows per position/length exist
    mdf = (
        data.groupby(["evidence_position", "filler_length"], as_index=False)[value_col]
        .mean()
        .sort_values(["filler_length", "evidence_position"])
    )
    if mdf.empty:
        return

    # Check for values below 0.4
    min_val = mdf[value_col].min()
    if min_val < 0.4:
        print(f"ERROR: Found {value_col} values below 0.4 for model {model_name} (min={min_val:.4f}).")

    # Scale to percentage
    mdf[value_col] = mdf[value_col] * 100

    fig, ax = plt.subplots(figsize=(8, 8))
    
    if value_col == "accuracy":
        palette = sns.color_palette("Set1", n_colors=max(3, mdf["filler_length"].nunique()))
    else:
        palette = sns.color_palette("viridis", n_colors=max(3, mdf["filler_length"].nunique()))
    
    sns.lineplot(
        data=mdf,
        x="evidence_position",
        y=value_col,
        hue="filler_length",
        marker="o",
        linewidth=4.0,
        markersize=15.0,
        palette=palette,
        ax=ax,
        legend=False if value_col == "accuracy" else "auto",
    )
    
    # Plot Baselines
    if baselines:
        if f"claim_only_{value_col}" in baselines:
            val = baselines[f"claim_only_{value_col}"]
            ax.axhline(val * 100, color="lightgrey", linestyle="--", linewidth=3.5, alpha=0.9, label=f"Claim Only ({val * 100:.1f})")
        
        if f"multi_{value_col}" in baselines:
            val = baselines[f"multi_{value_col}"]
            ax.axhline(val * 100, color="grey", linestyle="--", linewidth=3.5, alpha=0.9, label=f"Claim + Evidence ({val * 100:.1f})")

    # ax.set_title(title, fontsize=14, weight="bold")
    if value_col == "accuracy":
        title_kwargs = {}
        if title_y is not None:
            title_kwargs["y"] = title_y
        ax.set_title(get_short_model_name(model_name), fontsize=25, weight="bold", **title_kwargs)

    label_kwargs = {}
    if xlabel_size is not None:
        label_kwargs["fontsize"] = xlabel_size

    ax.set_xlabel("Evidence depth (%)", **label_kwargs)
    if xlabel_y is not None:
        ax.xaxis.set_label_coords(0.5, xlabel_y)

    if value_col == "accuracy":
        ax.set_ylabel("")
    else:
        ax.set_ylabel(value_col.upper())
    ax.set_ylim(40, 80)
    
    if fixed_ticks is not None:
        ax.set_xticks(fixed_ticks)
        ax.set_xticklabels([f"{int(round(t * 100))}" for t in fixed_ticks])
        ax.set_xlim(0.0, 1.0)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x * 100))}"))

    ax.grid(alpha=0.25)
    # ax.legend(title="Legend", fontsize=20, title_fontsize=11)
    
    fig.tight_layout()
    save_figure(fig, output_dir, file_stem, formats, dpi)


def plot_box_single(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    output_dir: Path,
    file_stem: str,
    formats: list[str],
    dpi: int,
    model_name: str,
) -> None:
    ensure_parent_dir(output_dir / "placeholder")
    save_table(data, output_dir / f"{file_stem}.csv")

    if data.empty:
        return
        
    # Check min value on raw rows for boxplot? Or aggregated? 
    # Usually boxplot shows raw distribution. We check min value of data points.
    min_val = data[value_col].min()
    if min_val < 0.4:
        print(f"ERROR: Found {value_col} values below 0.4 in boxplot data for {model_name} (min={min_val:.4f}).")

    # Local copy and scale
    plot_data = data.copy()
    plot_data[value_col] = plot_data[value_col] * 100

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.boxplot(
        data=plot_data,
        x="filler_length",
        y=value_col,
        color="#6EA8FE",
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x="filler_length",
        y=value_col,
        color="#1F4E79",
        size=4.0,
        alpha=0.35,
        ax=ax,
    )
    
    # ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Filler length (tokens)")
    ax.set_ylabel(value_col.upper())
    ax.set_ylim(40, 80)
    ax.grid(axis="y", alpha=0.22)
    
    fig.tight_layout()
    save_figure(fig, output_dir, file_stem, formats, dpi)


def plot_combined_accuracy_row(
    models_data: list[tuple[str, pd.DataFrame, dict[str, float]]],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    num_ticks: int = 15,
    fixed_ticks: np.ndarray | None = None,
    xlabel_y: float | None = None,
    xlabel_size: int | None = None,
    title_y: float | None = None,
    legend_y: float | None = None,
    wspace: float | None = None,
    ylabel_x: float | None = None,
) -> None:
    """
    Plots a 1xN figure of accuracy line plots for the specified models.
    models_data: list of (model_name, dataframe, baselines_dict)
    """
    n_models = len(models_data)
    if n_models == 0:
        print("No models to plot in combined plot.")
        return

    ensure_parent_dir(output_dir / "placeholder")
    
    # Setup global color palette based on all filler lengths found
    all_lengths = set()
    for _, df, _ in models_data:
        all_lengths.update(df["filler_length"].dropna().unique())
    sorted_lengths = sorted(list(all_lengths))
    
    # Use Set1 as in single plots, or viridis if many
    palette_colors = sns.color_palette("Set1", n_colors=max(3, len(sorted_lengths)))
    color_map = {l: c for l, c in zip(sorted_lengths, palette_colors)}

    # Adjust figsize based on number of models (8 inches per model)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8), sharey=True)
    if n_models == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    
    # Common settings
    line_width = 4.0
    marker_size = 15.0
    
    for idx, (model_name, df, baselines) in enumerate(models_data):
        ax = axes[idx]
        
        # Aggregate
        mdf = (
            df.groupby(["evidence_position", "filler_length"], as_index=False)["accuracy"]
            .mean()
            .sort_values(["filler_length", "evidence_position"])
        )
        mdf["accuracy"] = mdf["accuracy"] * 100
        
        # Plot lines
        for length in sorted_lengths:
            sub = mdf[mdf["filler_length"] == length]
            if sub.empty:
                continue
            ax.plot(
                sub["evidence_position"], 
                sub["accuracy"], 
                marker="o", 
                linewidth=line_width, 
                markersize=marker_size, 
                color=color_map[length],
                label=f"{length} tokens"
            )

        # Plot Baselines
        if baselines:
            if "claim_only_accuracy" in baselines:
                val = baselines["claim_only_accuracy"]
                ax.axhline(val * 100, color="lightgrey", linestyle="--", linewidth=3.5, alpha=0.9)
            
            if "multi_accuracy" in baselines:
                val = baselines["multi_accuracy"]
                ax.axhline(val * 100, color="grey", linestyle="--", linewidth=3.5, alpha=0.9)

        # Labels and Titles
        title_kwargs = {}
        if title_y is not None:
            title_kwargs["y"] = title_y
        ax.set_title(get_short_model_name(model_name), fontsize=25, weight="bold", **title_kwargs)
        
        label_kwargs = {}
        if xlabel_size is not None:
            label_kwargs["fontsize"] = xlabel_size
        ax.set_xlabel("Evidence depth (%)", **label_kwargs)
        if xlabel_y is not None:
            ax.xaxis.set_label_coords(0.5, xlabel_y)
        
        if idx == 0:
            ax.set_ylabel("Accuracy", **label_kwargs)
            if ylabel_x is not None:
                ax.yaxis.set_label_coords(ylabel_x, 0.5)
        else:
            ax.set_ylabel("")

        ax.set_ylim(40, 80)
        ax.grid(alpha=0.25)
        
        # Ticks
        if fixed_ticks is not None:
            ax.set_xticks(fixed_ticks)
            ax.set_xticklabels([f"{int(round(t * 100))}" for t in fixed_ticks])
            ax.set_xlim(0.0, 1.0)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x * 100))}"))

    # Unified Legend
    legend_elements = []
    # Lengths
    for l in sorted_lengths:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=f"{l} Context",
                   markerfacecolor=color_map[l], markersize=15)
        )
    # Baselines
    legend_elements.append(Line2D([0], [0], color='lightgrey', lw=3.5, linestyle='--', label='Claim Only'))
    legend_elements.append(Line2D([0], [0], color='grey', lw=3.5, linestyle='--', label='Claim + Evidence (No Context)'))

    legend_y_val = legend_y if legend_y is not None else -0.15
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, legend_y_val), ncol=len(legend_elements), frameon=False, fontsize=20)
    
    plt.tight_layout() # Uses default padding
    
    adjust_args = {"bottom": 0.2}
    if wspace is not None:
        adjust_args["wspace"] = wspace
    
    plt.subplots_adjust(**adjust_args)
    
    save_figure(fig, output_dir, f"combined_accuracy_{n_models}models", formats, dpi)


def plot_bar_comparison_single(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    output_dir: Path,
    file_stem: str,
    formats: list[str],
    dpi: int,
    model_name: str,
) -> None:
    # Requires multiple datasets in 'data'
    ensure_parent_dir(output_dir / "placeholder")
    save_table(data, output_dir / f"{file_stem}.csv")

    mdf = (
        data.groupby(["dataset", "filler_length"], as_index=False)[value_col]
        .mean()
    )
    if mdf.empty:
        return

    min_val = mdf[value_col].min()
    if min_val < 0.4:
        print(f"ERROR: Found {value_col} values below 0.4 for model {model_name} in bar comparison (min={min_val:.4f}).")

    mdf[value_col] = mdf[value_col] * 100

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.barplot(
        data=mdf,
        x="filler_length",
        y=value_col,
        hue="dataset",
        errorbar=None,
        ax=ax,
    )
    
    # ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Filler length (tokens)")
    ax.set_ylabel(f"Mean {value_col.upper()} across depths")
    ax.set_ylim(40, 80)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Dataset", fontsize=10, title_fontsize=11)

    fig.tight_layout()
    save_figure(fig, output_dir, file_stem, formats, dpi)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results root directory (defaults to out/results/subset_seed42)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for tables and figures (defaults to images)",
    )
    parser.add_argument(
        "--tag-filter",
        default=None,
        help="Optional substring filter on result file names (example: subset_seed42)",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated figure formats (default: png,pdf)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="Figure DPI (default: 320)",
    )
    parser.add_argument(
        "--min-valid",
        type=int,
        default=1,
        help="Minimum valid rows required to keep a metric row (default: 1)",
    )
    parser.add_argument(
        "--num-ticks",
        type=int,
        default=15,
        help="Number of ticks on x-axis (default: 15)",
    )
    parser.add_argument(
        "--ticks-01-spaced",
        action="store_true",
        help="Force 11 ticks from 0.0 to 1.0 spaced by 0.1",
    )
    parser.add_argument(
        "--ticks-02-spaced",
        action="store_true",
        help="Force 6 ticks from 0.0 to 1.0 spaced by 0.2",
    )
    parser.add_argument(
        "--evidence-depth-xlabel-y",
        type=float,
        default=-0.11,
        help="Custom y-coordinate for the Evidence depth x-axis label on line plots (axes coords, e.g. -0.15)",
    )
    parser.add_argument(
        "--evidence-depth-xlabel-size",
        type=int,
        default=22,
        help="Custom font size for the Evidence depth x-axis label on line plots",
    )
    parser.add_argument(        
        "--title-y",
        type=float,
        default=1.02,
        help="Custom y-coordinate for the title on line plots (axes coords, e.g. 1.05)",
    )
    parser.add_argument(
        "--combined-legend-y",
        type=float,
        default=-0.05,
        help="Custom y-coordinate (bbox_to_anchor y) for the legend in the combined 1x4 plot (default: -0.15)",
    )
    parser.add_argument(
        "--combined-wspace",
        type=float,
        default=0.2,
        help="Custom width spacing (wspace) between subplots in the combined 1x4 plot",
    )
    parser.add_argument(
        "--accuracy-ylabel-x",
        type=float,
        default=-0.11,
        help="Custom x-coordinate for the Accuracy y-axis label on combined plot (axes coords, e.g. -0.15)",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="If set, only generate the combined 1x4 plot and skip individual model plots",
    )
    parser.add_argument(
        "--use-existing-csv",
        action="store_true",
        help="If set, load metrics from the existing table_filler_metrics_per_file_dataset.csv in the output directory instead of parsing raw results",
    )
    parser.add_argument(        
        "--debug",
        action="store_true",
        help="Debug mode: only plot accuracy line plots for Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--exclude-32k",
        action="store_true",
        help="Exclude data with context length ~32k (29k-33k) from analysis",
    )
    parser.add_argument("--config", type=Path, default=project_root() / "config/paper_config.json")
    parser.add_argument("--subset-file", type=Path, default=project_root() / "data/processed/experiment_subset_seed42.csv")
    args = parser.parse_args()

    root = project_root()
    results_dir = Path(args.results_dir) if args.results_dir else root / "out" / "results" / "subset_seed42"
    
    base_output_dir = Path(args.output_dir) if args.output_dir else root / "out" / "figures"
    if args.debug:
        base_output_dir = base_output_dir / "debug"

    if args.ticks_01_spaced:
        output_dir = base_output_dir / "spaced_by_0.1"
    elif args.ticks_02_spaced:
        output_dir = base_output_dir / "spaced_by_0.2"
    else:
        output_dir = base_output_dir / f"ticks_{args.num_ticks}"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    if not formats:
        formats = ["png"]

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "grid.color": "#DDE3EA",
            "font.size": 11,
            "axes.titleweight": "regular",
            "axes.labelweight": "regular",
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
        }
    )

    metrics_df = pd.DataFrame()
    skipped_files = []
    bad_schema_files = []
    
    csv_path = output_dir / "table_filler_metrics_per_file_dataset.csv"
    
    if args.use_existing_csv and csv_path.exists():
        print(f"Loading metrics from existing CSV: {csv_path}")
        filler_df = pd.read_csv(csv_path)
    else:
        metrics_df, skipped_files, bad_schema_files = build_metrics_dataframe(
            results_dir=results_dir,
            tag_filter=args.tag_filter,
            min_valid=args.min_valid,
        )

        if metrics_df.empty:
            raise SystemExit("No matching metrics were extracted from result files.")

        if args.exclude_32k:
            print("Excluding 32k context data...")
            # We already mapped 29k to 32768 in the extraction step, so we just filter 32768
            metrics_df = metrics_df[metrics_df["filler_length"] != 32768]
            if metrics_df.empty:
                raise SystemExit("No metrics remaining after excluding 32k context.")

        # Persist raw/processed tables for reproducibility.
        save_table(metrics_df, output_dir / "table_raw_metrics_per_file_dataset.csv")

        filler_df = metrics_df[metrics_df["is_filler"] == True].copy()
        filler_df = filler_df[filler_df["evidence_position"].notna()].copy()

        # Filter data if restricted ticks requested
        allowed_positions = None
        if args.ticks_01_spaced:
            allowed_positions = np.linspace(0.0, 1.0, 11)
        elif args.ticks_02_spaced:
            allowed_positions = np.linspace(0.0, 1.0, 6)
        
        if allowed_positions is not None:
            # Filter rows where evidence_position is close to one of the allowed
            # Use a tolerance (e.g. 0.01)
            mask = filler_df["evidence_position"].apply(
                lambda x: any(np.abs(allowed_positions - x) < 0.001)
            )
            filler_df = filler_df[mask].copy()

        filler_df = filter_configured_metrics(filler_df, args.config)
        filler_df = filler_df.sort_values(["model", "dataset", "filler_length", "evidence_position"])

        if filler_df.empty:
            raise SystemExit("No filler-positioning rows found (len/pos files).")

        save_table(filler_df, output_dir / "table_filler_metrics_per_file_dataset.csv")

    filler_df = filter_configured_metrics(filler_df, args.config)
    if filler_df.empty:
        raise ValueError("No metrics match the selected experiment configuration")

    # Paper-facing summaries.
    model_summary = (
        filler_df[filler_df["dataset_norm"] == "all"]
        .groupby("model", as_index=False)
        .agg(
            n_conditions=("accuracy", "count"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            best_accuracy=("accuracy", "max"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            best_f1=("f1", "max"),
        )
        .sort_values("mean_accuracy", ascending=False)
    )
    save_table(model_summary, output_dir / "table_summary_overall_by_model.csv")

    model_dataset_summary = (
        filler_df[filler_df["dataset_norm"] != "all"]
        .groupby(["model", "dataset"], as_index=False)
        .agg(
            n_conditions=("accuracy", "count"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            best_accuracy=("accuracy", "max"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            best_f1=("f1", "max"),
        )
        .sort_values(["model", "dataset"])
    )
    save_table(model_dataset_summary, output_dir / "table_summary_by_model_and_dataset.csv")

    combined_dataset_summary = (
        filler_df[filler_df["dataset_norm"] != "all"]
        .groupby("dataset", as_index=False)
        .agg(
            n_conditions=("accuracy", "count"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
        )
        .sort_values("mean_accuracy", ascending=False)
    )
    save_table(combined_dataset_summary, output_dir / "table_summary_combined_across_models_by_dataset.csv")

    # Console summaries.
    print("\nOverall Accuracy/F1 by model (combined datasets):")
    for _, row in model_summary.iterrows():
        print(
            f"- {row['model']}: "
            f"ACC={format_metric(row['mean_accuracy'])} "
            f"(best {format_metric(row['best_accuracy'])}), "
            f"F1={format_metric(row['mean_f1'])} "
            f"(best {format_metric(row['best_f1'])}), "
            f"conditions={int(row['n_conditions'])}"
        )

    print("\nOverall Accuracy/F1 by model and dataset:")
    for _, row in model_dataset_summary.iterrows():
        print(
            f"- {row['model']} | {row['dataset']}: "
            f"ACC={format_metric(row['mean_accuracy'])}, "
            f"F1={format_metric(row['mean_f1'])}, "
            f"conditions={int(row['n_conditions'])}"
        )

    print("\nCombined across models (by dataset):")
    for _, row in combined_dataset_summary.iterrows():
        print(
            f"- {row['dataset']}: "
            f"ACC={format_metric(row['mean_accuracy'])}, "
            f"F1={format_metric(row['mean_f1'])}, "
            f"conditions={int(row['n_conditions'])}"
        )

    
    # Load experiment subset for baseline filtering
    valid_claims = read_subset_keys(args.subset_file)

    # Identical setup for ticks calculation for passing to plot functions
    fixed_ticks = None
    if args.ticks_01_spaced:
        fixed_ticks = np.linspace(0.0, 1.0, 11)
    elif args.ticks_02_spaced:
        fixed_ticks = np.linspace(0.0, 1.0, 6)

    # ----------------------------------------------------
    # Collect data for Combined Plot
    # ----------------------------------------------------
    combined_targets = [
        "Llama-3.1-8B", 
        "Llama-3.1-70B",
        "Qwen2.5-7B", 
        "Qwen3-8B", # Searching for flexible match
        "Qwen3-32B"
    ]
    # Map friendly keys to actual model names found
    found_models_map = {} 
    all_models = filler_df["model"].unique().tolist()
    
    # Simple heuristic to find specific models
    for m in all_models:
        m_upper = m.upper()
        if "LLAMA-3.1-8B" in m_upper:
            found_models_map["llama"] = m
        elif "LLAMA-3.1-70B" in m_upper:
            found_models_map["llama_70b"] = m
        elif "QWEN2.5-7B" in m_upper:
            found_models_map["qwen2.5"] = m
        # For Qwen3, assume user meant Qwen2.5 variants or literal Qwen3 if they exist
        elif "QWEN3-8B" in m_upper:
            found_models_map["qwen3_8b"] = m
        elif "QWEN3-32B" in m_upper:
            found_models_map["qwen3_32b"] = m
        # Fallback if user meant Qwen2.5-14B/32B for "Qwen3"
        elif "QWEN2.5-14B" in m_upper:
             # Just in case this maps to "qwen3 8B" based on user context? 
             # Let's stick to literal search primarily.
             pass
        elif "QWEN2.5-32B" in m_upper:
             # If Qwen3-32B not found, maybe this is it?
             if "qwen3_32b" not in found_models_map:
                 found_models_map["qwen3_32b"] = m

    # Construct the ordered list of data
    combined_plot_data = []
    # Order: llama, qwen2.5, qwen3 8B, qwen3 32B, llama_70b
    target_keys = ["llama", "qwen2.5", "qwen3_8b", "qwen3_32b", "llama_70b"]
    
    # If explicit Qwen3 keys missing, try to fill "qwen3_8b" with Qwen2.5-14B or similar if needed?
    # Or just warn.
    
    for k in target_keys:
        if k in found_models_map:
            m_name = found_models_map[k]
            m_df = filler_df[(filler_df["model"] == m_name) & (filler_df["dataset_norm"] == "all")].copy()
            if not m_df.empty:
                m_base = get_baselines(m_name, results_dir.parent, valid_claims)
                combined_plot_data.append((m_name, m_df, m_base))
            else:
                print(f"Warning: No data found for model {m_name} (key={k})")
        else:
            print(f"Warning: Could not find model for key '{k}' in {all_models}")

    if len(combined_plot_data) > 0:
        print(f"Generating combined 1x{len(combined_plot_data)} accuracy plot...")
        # Save in output_dir (which is e.g. spaced_by_0.1)
        # combined function logic handles saving
        plot_combined_accuracy_row(
            combined_plot_data, 
            output_dir, 
            ["png", "pdf"], # Forced jpg/pdf as requested
            args.dpi,
            num_ticks=args.num_ticks,
            fixed_ticks=fixed_ticks,
            xlabel_y=args.evidence_depth_xlabel_y,
            xlabel_size=args.evidence_depth_xlabel_size,
            title_y=args.title_y,
            legend_y=args.combined_legend_y,
            wspace=args.combined_wspace,
            ylabel_x=args.accuracy_ylabel_x
        )
    else:
        print(f"Skipping combined plot: found {len(combined_plot_data)} required models.")

    if args.combined_only:
        print("Skipping individual plots (--combined-only set).")
        return

    # ----------------------------------------------------
    # New single-model plot loop
    # ----------------------------------------------------
    models = sorted(filler_df["model"].unique().tolist())
    
    # Identify results root (parent of the specific experiment folder)
    results_root = results_dir.parent

    for model in models:
        if args.debug and model != "Qwen2.5-7B-Instruct":
            continue

        model_df = filler_df[filler_df["model"] == model].copy()
        if model_df.empty:
            continue

        baselines = get_baselines(model, results_root, valid_claims)
            
        # (A) Aggregated across all datasets (dataset_norm == 'all')
        agg_df = model_df[model_df["dataset_norm"] == "all"]
        if not agg_df.empty:
            # Heatmaps
            if not args.debug:
                plot_heatmap_single(agg_df, "accuracy", f"{model} Accuracy (Aggregated)", "YlGnBu", 
                                    output_dir / "01_heatmaps_accuracy", f"{model}_ALL", formats, args.dpi,
                                    num_ticks=args.num_ticks, fixed_ticks=fixed_ticks)
                plot_heatmap_single(agg_df, "f1", f"{model} F1 (Aggregated)", "YlOrRd", 
                                    output_dir / "02_heatmaps_f1", f"{model}_ALL", formats, args.dpi,
                                    num_ticks=args.num_ticks, fixed_ticks=fixed_ticks)
            # Line Plots
            plot_line_single(agg_df, "accuracy", f"{model} Accuracy (Aggregated)", 
                             output_dir / "03_lineplots_accuracy", f"{model}_ALL", formats, args.dpi, model,
                             num_ticks=args.num_ticks, fixed_ticks=fixed_ticks, baselines=baselines,
                             xlabel_y=args.evidence_depth_xlabel_y, xlabel_size=args.evidence_depth_xlabel_size,
                             title_y=args.title_y)
            
            if not args.debug:
                plot_line_single(agg_df, "f1", f"{model} F1 (Aggregated)", 
                                 output_dir / "04_lineplots_f1", f"{model}_ALL", formats, args.dpi, model,
                                 num_ticks=args.num_ticks, fixed_ticks=fixed_ticks, baselines=baselines,
                                 xlabel_y=args.evidence_depth_xlabel_y, xlabel_size=args.evidence_depth_xlabel_size,
                                 title_y=args.title_y)
                # Box Plots
                plot_box_single(agg_df, "accuracy", f"{model} Accuracy Distribution (Aggregated)", 
                                output_dir / "05_boxplots_accuracy", f"{model}_ALL", formats, args.dpi, model)
                plot_box_single(agg_df, "f1", f"{model} F1 Distribution (Aggregated)", 
                                output_dir / "06_boxplots_f1", f"{model}_ALL", formats, args.dpi, model)

        # (B) Per dataset plots (dataset_norm != 'all')
        per_ds_df = model_df[model_df["dataset_norm"] != "all"]
        if not per_ds_df.empty:
            # Dataset Comparison (Bar Chart)
            if not args.debug:
                plot_bar_comparison_single(per_ds_df, "accuracy", f"{model} Accuracy by Dataset", 
                                           output_dir / "07_bars_accuracy_comparison", f"{model}_Overview", formats, args.dpi, model)
                plot_bar_comparison_single(per_ds_df, "f1", f"{model} F1 by Dataset", 
                                           output_dir / "08_bars_f1_comparison", f"{model}_Overview", formats, args.dpi, model)
            
            # Individual plots per dataset
            for ds_norm in per_ds_df["dataset_norm"].unique():
                single_ds = per_ds_df[per_ds_df["dataset_norm"] == ds_norm]
                if single_ds.empty:
                    continue
                # Use dataset name from data for title/file
                ds_name_str = str(single_ds["dataset"].iloc[0]).strip()
                safe_ds = re.sub(r"[^a-zA-Z0-9_\-]", "_", ds_name_str)
                
                # Heatmaps
                if not args.debug:
                    plot_heatmap_single(single_ds, "accuracy", f"{model} Accuracy ({ds_name_str})", "YlGnBu", 
                                        output_dir / "01_heatmaps_accuracy", f"{model}_{safe_ds}", formats, args.dpi,
                                        num_ticks=args.num_ticks, fixed_ticks=fixed_ticks)
                    plot_heatmap_single(single_ds, "f1", f"{model} F1 ({ds_name_str})", "YlOrRd", 
                                        output_dir / "02_heatmaps_f1", f"{model}_{safe_ds}", formats, args.dpi,
                                        num_ticks=args.num_ticks, fixed_ticks=fixed_ticks)
                
                # Line Plots
                plot_line_single(single_ds, "accuracy", f"{model} Accuracy ({ds_name_str})", 
                                 output_dir / "03_lineplots_accuracy", f"{model}_{safe_ds}", formats, args.dpi, model,
                                 num_ticks=args.num_ticks, fixed_ticks=fixed_ticks, baselines=baselines,
                                 xlabel_y=args.evidence_depth_xlabel_y, xlabel_size=args.evidence_depth_xlabel_size,
                                 title_y=args.title_y)
                
                if not args.debug:
                    plot_line_single(single_ds, "f1", f"{model} F1 ({ds_name_str})", 
                                     output_dir / "04_lineplots_f1", f"{model}_{safe_ds}", formats, args.dpi, model,
                                     num_ticks=args.num_ticks, fixed_ticks=fixed_ticks, baselines=baselines,
                                     xlabel_y=args.evidence_depth_xlabel_y, xlabel_size=args.evidence_depth_xlabel_size,
                                     title_y=args.title_y)
                    
                    # Box Plots
                    plot_box_single(single_ds, "accuracy", f"{model} Accuracy Distribution ({ds_name_str})", 
                                    output_dir / "05_boxplots_accuracy", f"{model}_{safe_ds}", formats, args.dpi, model)
                    plot_box_single(single_ds, "f1", f"{model} F1 Distribution ({ds_name_str})", 
                                    output_dir / "06_boxplots_f1", f"{model}_{safe_ds}", formats, args.dpi, model)

    # Diagnostics logs.
    if skipped_files:
        skipped_df = pd.DataFrame({"file": [str(p) for p in skipped_files]})
        save_table(skipped_df, output_dir / "table_skipped_files_unparsed_or_unreadable.csv")
    if bad_schema_files:
        bad_df = pd.DataFrame({"file": [str(p) for p in bad_schema_files]})
        save_table(bad_df, output_dir / "table_skipped_files_missing_required_columns.csv")

    print(f"\nSaved analysis tables and figures to: {output_dir}")


if __name__ == "__main__":
    main()
