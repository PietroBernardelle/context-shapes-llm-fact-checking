from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import seaborn as sns

from evaluation_utils import macro_f1_from_counts, filter_configured_metrics
from prompt_utils import project_root, ensure_parent_dir

# -----------------------------------------------------------------------------
# Reused extraction logic from 7_filler_positioning_analysis.py
# -----------------------------------------------------------------------------

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
            # Map 29k/32k to 32768 to combine them
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

    # Handles no-filler files
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
        "correct": correct, "tp": tp, "fp": fp, "fn": fn, "valid": valid, "total": total
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

def read_results_file(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["dataset", "label", "pred_label"])

def build_metrics_dataframe(results_dir: Path, min_valid: int = 1) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    
    files = sorted(results_dir.rglob("*.pqt"))
    for file_path in files:
        meta = extract_file_metadata(file_path.name)
        if meta is None:
            continue
            
        rel = file_path.relative_to(results_dir)
        # Assuming folder structure is model/filename.pqt or just filename.pqt
        model = rel.parts[0] if len(rel.parts) > 1 else "UNKNOWN_MODEL"
        if model == "UNKNOWN_MODEL":
             # Try to guess from parent folder if it's directly in results_dir check
             # But here we stick to the extraction logic of 7_...
             pass

        try:
            df = read_results_file(file_path)
        except Exception:
            continue

        required = {"label", "pred_label", "dataset"}
        if not required.issubset(df.columns):
            continue

        # Overall (combined datasets)
        acc, f1, valid, total = accuracy_f1(df)
        if valid >= min_valid:
            records.append({
                "model": model,
                "dataset": "ALL",
                "accuracy": acc,
                "f1": f1,
                **meta
            })

        # Per-dataset
        for dataset_name, group in df.groupby("dataset", dropna=False):
            ds = "NA" if pd.isna(dataset_name) else str(dataset_name).strip()
            g_acc, g_f1, g_valid, g_total = accuracy_f1(group)
            if g_valid < min_valid:
                continue
            records.append({
                "model": model,
                "dataset": ds,
                "accuracy": g_acc,
                "f1": g_f1,
                **meta
            })
            
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# New Plotting Logic
# -----------------------------------------------------------------------------


def get_model_order(model_name: str) -> int:
    # Desired order: Llama-8B, Qwen2.5, Qwen3-8B, Qwen3-32B, Llama-70B
    order = [
        "Llama-3.1-8B",
        "Qwen2.5-7B",
        "Qwen3-8B",
        "Qwen3-32B",
        "Llama-3.1-70B"
    ]
    for i, m in enumerate(order):
        if m in model_name:
            return i
    return 999

def plot_context_analysis(
    df: pd.DataFrame, 
    output_dir: Path, 
    only_avg: bool = False,
    xlabel_y: float | None = None,
    ylabel_x: float | None = None
):
    # Style settings similar to script 7 (6_build_experiment_subset ref in prompt)
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D0D7DE",
        "grid.color": "#DDE3EA",
        "font.size": 11,
        "axes.titleweight": "regular", # Regular in main theme, bold in individual titles
        "axes.labelweight": "regular",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })
    
    # Ensure numerical sorting and consistency
    df["filler_length"] = df["filler_length"].astype(int)
    
    # We want to identify the depths available
    depths = sorted([d for d in df["evidence_position"].unique() if d is not None and not math.isnan(d)])
    
    # 0 length context rows (baseline)
    baseline_df = df[df["filler_length"] == 0].copy()
    
    # Non-zero length context rows
    context_df = df[df["filler_length"] > 0].copy()
    
    # Iterate over each dataset to create separate plots
    for dataset in df["dataset"].unique():
        ds_subset = df[df["dataset"] == dataset].copy()
        if ds_subset.empty:
            continue
            
        print(f"Generating plots for Dataset: {dataset}")
        dataset_slug = str(dataset).replace(" ", "_").lower()

        # Compute mapping for equidistant x-axis
        unique_lengths = sorted(ds_subset["filler_length"].unique())
        length_map = {l: i for i, l in enumerate(unique_lengths)}
        ds_subset["x_pos"] = ds_subset["filler_length"].map(length_map)
        
        xticks = sorted(length_map.values())
        xticklabels = []
        for i in xticks:
            l = unique_lengths[i]
            if l == 0:
                xticklabels.append("0")
            elif l >= 1000:
                xticklabels.append(f"{int(l/1000)}k")
            else:
                xticklabels.append(str(l))
        
        # ---------------------------------------------------------------------
        # 1. Average Accuracy vs Context (averaged across depths)
        # ---------------------------------------------------------------------
        # Group by model and length, calculate mean accuracy
        # Note: We include length 0 here as well. Length 0 usually has evidence_position=None.
        # When we group by filler_length, we want the mean over evidence_positions for that length.
        # For length 0, there is only one "position" (None), so mean is just the value.
        # For length > 0, there are multiple positions.
        
        agg_df = ds_subset.groupby(["model", "filler_length", "x_pos"])["accuracy"].mean().reset_index()
        
        # Scale to percentage for consistency with script 7 style
        agg_df["accuracy"] = agg_df["accuracy"] * 100
        
        plt.figure(figsize=(16, 8))
        
        # Use a consistent palette
        models = sorted(agg_df["model"].unique(), key=get_model_order)
        palette = sns.color_palette("Set1", n_colors=max(3, len(models)))
        
        ax = sns.lineplot(
            data=agg_df, 
            x="x_pos", 
            y="accuracy", 
            hue="model", 
            marker="o",
            linewidth=4.0,
            markersize=15.0,
            palette=palette,
            hue_order=models
        )
        
        # ax.set_title(f"Accuracy vs Context Length (Avg) - {dataset}", fontsize=25, weight="bold", y=1.02)
        ax.set_xlabel("Context Length (Tokens)", fontsize=22)
        if xlabel_y is not None:
            ax.xaxis.set_label_coords(0.5, xlabel_y)

        ax.set_ylabel("Average Accuracy (%)", fontsize=22)
        if ylabel_x is not None:
            ax.yaxis.set_label_coords(ylabel_x, 0.5)

        ax.grid(alpha=0.25)
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.set_ylim(50, 75)
        
        # Consistent legend styling
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15, title_fontsize=16)
        plt.tight_layout()
        
        out_path = output_dir / f"avg_accuracy_vs_context_{dataset_slug}.png"
        plt.savefig(out_path, dpi=320)
        plt.close()
        print(f"  Saved: {out_path}")

        if only_avg:
            continue

        # ---------------------------------------------------------------------
        # 2. Accuracy vs Context per Depth
        # ---------------------------------------------------------------------
        # For these plots, we want to show lines for each model as context grows,
        # but specific to a fixed depth (e.g. 10%, 20%).
        # We also want to include Length 0 as the starting point (x=0) for all of them.
        
        # Get baseline (len=0) for this dataset
        ds_baseline = ds_subset[ds_subset["filler_length"] == 0]
        
        for depth in depths:
            # Filter for this depth
            depth_data = ds_subset[ds_subset["evidence_position"] == depth]
            
            # Combine with baseline
            # We want the baseline rows to appear in this plot's dataframe
            # The baseline rows have evidence_position = None, so we temporarily assign them the current depth
            # so they plot correctly as the X=0 point for that series.
            
            current_baseline = ds_baseline.copy()
            current_baseline["evidence_position"] = depth
            
            plot_data = pd.concat([current_baseline, depth_data], ignore_index=True)
            plot_data.sort_values(by="filler_length", inplace=True)
            
            # Scale to percentage
            plot_data["accuracy"] = plot_data["accuracy"] * 100
            
            plt.figure(figsize=(10, 8))
            ax = sns.lineplot(
                data=plot_data, 
                x="x_pos", 
                y="accuracy", 
                hue="model", 
                marker="o",
                linewidth=4.0,
                markersize=15.0,
                palette=palette,
                hue_order=models
            )
            
            ax.set_title(f"Accuracy vs Context Length (Depth {depth}) - {dataset}", fontsize=25, weight="bold", y=1.02)
            ax.set_xlabel("Context Length (Tokens)", fontsize=22)
            if xlabel_y is not None:
                ax.xaxis.set_label_coords(0.5, xlabel_y)

            ax.set_ylabel("Accuracy (%)", fontsize=22)
            if ylabel_x is not None:
                ax.yaxis.set_label_coords(ylabel_x, 0.5)

            ax.grid(alpha=0.25)
            
            # Set specific x-ticks and y-ticks
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.set_ylim(50, 75)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15, title_fontsize=16)
            plt.tight_layout()
            
            # Format depth for filename (replace point with p to avoid double extensions)
            depth_str = str(depth).replace(".", "p")
            out_path = output_dir / f"accuracy_vs_context_depth_{depth_str}_{dataset_slug}.png"
            plt.savefig(out_path, dpi=320)
            plt.close()
            print(f"  Saved: {out_path}")


def main():
    root = project_root()
    import argparse
    parser = argparse.ArgumentParser(description="Analyze results by context length.")
    parser.add_argument("--results-dir", type=Path, default=root / "out" / "results" / "subset_seed42", help="Path to results directory")
    parser.add_argument("--output-dir", type=Path, default=root / "out" / "figures" / "context_length_analysis", help="Path to save plots")
    parser.add_argument("--only-avg", action="store_true", help="Only compute and save the average plot")
    parser.add_argument("--xlabel-y", type=float, default=-0.1, help="Custom y-coordinate for the x-axis label")
    parser.add_argument("--ylabel-x", type=float, default=-0.07, help="Custom x-coordinate for the y-axis label")
    parser.add_argument("--exclude-32k", action="store_true", help="Exclude data with context length ~32k (29k-33k) from analysis")
    
    parser.add_argument("--config", type=Path, default=root / "config/paper_config.json")
    parser.add_argument("--all-positions", action="store_true", help="Historical diagnostic: average all available evidence positions")
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        # Fallback check for common patterns if default failed
        print("Please specify --results-dir if it is different.")
        return

    print(f"Reading results from: {args.results_dir}")
    df = build_metrics_dataframe(args.results_dir)
    
    df = filter_configured_metrics(df, args.config, all_positions=args.all_positions)
    if df.empty:
        raise ValueError("No results match the selected experiment configuration")

    if args.exclude_32k:
        print("Excluding 32k context data...")
         # We already mapped 29k to 32768 in the extraction step, so we just filter 32768
        df = df[df["filler_length"] != 32768]
    
    if df.empty:
        print("No valid results found.")
        return
        
    print(f"Found {len(df)} records.")
    
    ensure_parent_dir(args.output_dir / "placeholder")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_context_analysis(
        df, 
        args.output_dir, 
        only_avg=args.only_avg,
        xlabel_y=args.xlabel_y,
        ylabel_x=args.ylabel_x
    )
    print(f"All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
