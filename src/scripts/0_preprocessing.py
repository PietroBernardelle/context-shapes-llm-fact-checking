"""
Data preprocessing script.

Steps:
1) (Optional) Process raw datasets and write interim files.
2) (Optional) Standardize datasets and write processed files.
3) (Optional) Analyze processed datasets with summary stats and plots.

Select phases with --steps raw standardize combine (and optionally analyze).
"""

# Suppress noisy warnings to keep output readable.
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from prompt_utils import project_root, ensure_parent_dir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = project_root()

# Define base data folders.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Reproducible sampling.
RANDOM_STATE = 42



# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def split_evidence_text(evidence_text: str) -> list[str]:
    if pd.isna(evidence_text):
        return []
    return str(evidence_text).split("++")


def compute_len_evidence(evidences: list[str]) -> int:
    return sum(len(text) for text in evidences)


def process_climatefever_evidences(row: pd.Series) -> dict:
    clean_evidence = [ev["evidence"] for ev in row["evidences"]]
    return {
        "clean_evidence": clean_evidence,
        "number_of_evidences": len(clean_evidence),
        "len_evidences": sum(len(ev) for ev in clean_evidence),
    }


# ---------------------------------------------------------------------------
# Phase 1: Raw dataset processing
# ---------------------------------------------------------------------------

def process_raw_datasets(raw_dir: Path, interim_dir: Path, climatefever_file: Path | None = None) -> None:
    # Define input paths.
    feverous_path = raw_dir / "feverous.pqt"
    hover_path = raw_dir / "hover.pqt"
    climatefever_path = "tdiggelm/climate_fever"
    for path in [feverous_path, hover_path]:
        if not path.is_file():
            raise ValueError(f"Missing prepared input: {path}. See docs/data.md.")

    # Process Feverous dataset.
    df_feverous = pd.read_parquet(feverous_path)
    df_feverous = df_feverous[df_feverous["label"] != "NOT ENOUGH INFO"]

    # Build evidence statistics.
    df_feverous["clean_evidence"] = df_feverous["evidence"].apply(split_evidence_text)
    df_feverous["number_of_evidences"] = df_feverous["clean_evidence"].apply(len)
    df_feverous["len_evidences"] = df_feverous["clean_evidence"].apply(compute_len_evidence)

    # Save full cleaned Feverous.
    feverous_out = interim_dir / "cleaned_feverous.csv"
    ensure_parent_dir(feverous_out)
    df_feverous.to_csv(feverous_out, index=False)
    print("Feverous processed")

    # Process Hover dataset.
    df_hover = pd.read_parquet(hover_path)
    df_hover = df_hover[df_hover["label"].isin(["SUPPORTED", "NOT_SUPPORTED"])]

    # Build evidence statistics.
    df_hover["clean_evidence"] = df_hover["evidence"].apply(split_evidence_text)
    df_hover["number_of_evidences"] = df_hover["clean_evidence"].apply(len)
    df_hover["len_evidences"] = df_hover["clean_evidence"].apply(compute_len_evidence)

    # Save full cleaned Hover.
    hover_out = interim_dir / "cleaned_hover.csv"
    ensure_parent_dir(hover_out)
    df_hover.to_csv(hover_out, index=False)
    print("Hover processed")

    # Process ClimateFever dataset from HuggingFace.
    if climatefever_file is not None:
        df_climatefever = pd.read_parquet(climatefever_file)
    else:
        from datasets import load_dataset
        df_climatefever = pd.DataFrame(load_dataset(climatefever_path, split="test"))

    # Remove NOT ENOUGH INFO claims and DISPUTED claims.
    df_climatefever = df_climatefever[(df_climatefever["claim_label"] != 2) & (df_climatefever["claim_label"] != 3)]
    df_climatefever = df_climatefever.reset_index(drop=True)

    # Extract evidence information.
    evidence_data = df_climatefever.apply(process_climatefever_evidences, axis=1)
    evidence_df = pd.DataFrame(evidence_data.tolist(), index=df_climatefever.index)
    df_climatefever = pd.concat([df_climatefever, evidence_df], axis=1)

    # Save full cleaned ClimateFever.
    climatefever_out = interim_dir / "cleaned_climatefever.csv"
    ensure_parent_dir(climatefever_out)
    df_climatefever.to_csv(climatefever_out, index=False)
    print("ClimateFever processed")

    print("\nAll datasets processed and saved")


# ---------------------------------------------------------------------------
# Phase 2: Dataset standardization
# ---------------------------------------------------------------------------

def standardize_datasets(interim_dir: Path, processed_dir: Path) -> None:
    # Load interim cleaned datasets.
    df_hover = pd.read_csv(interim_dir / "cleaned_hover.csv")
    df_feverous = pd.read_csv(interim_dir / "cleaned_feverous.csv")
    df_climatefever = pd.read_csv(interim_dir / "cleaned_climatefever.csv")

    # Add dataset names for later analysis.
    df_hover["dataset"] = "Hover"
    df_feverous["dataset"] = "Feverous"
    df_climatefever["dataset"] = "ClimateFever"

    print(f"Hover shape: {df_hover.shape}")
    print(f"Feverous shape: {df_feverous.shape}")
    print(f"ClimateFever shape: {df_climatefever.shape}")

    # Align columns across datasets.
    df_hover = df_hover[["claim", "label", "clean_evidence", "number_of_evidences", "len_evidences", "dataset"]]
    df_feverous = df_feverous[["claim", "label", "clean_evidence", "number_of_evidences", "len_evidences", "dataset"]]
    df_climatefever = df_climatefever[["claim", "claim_label", "clean_evidence", "number_of_evidences", "len_evidences", "dataset"]]
    df_climatefever = df_climatefever.rename(columns={"claim_label": "label"})

    print("\n----------------------------------------------------")
    print("Number of SUPPORTED and NOT_SUPPORTED labels in Hover dataset:")
    print(df_hover["label"].value_counts())
    print("\nNumber of SUPPORTED and NOT_SUPPORTED labels in Feverous dataset:")
    print(df_feverous["label"].value_counts())
    print("\nNumber of SUPPORTED and NOT_SUPPORTED labels in ClimateFever dataset:")
    print(df_climatefever["label"].value_counts())

    # Standardize labels to binary (SUPPORTED -> 0, NOT_SUPPORTED/REFUTES -> 1).
    df_hover["label"] = df_hover["label"].apply(lambda x: 0 if x == "SUPPORTED" else 1)
    df_feverous["label"] = df_feverous["label"].apply(lambda x: 0 if x == "SUPPORTS" else 1)

    # Print label distributions after standardization.
    print("\n----------------------------------------------------")
    print("Number of SUPPORTED and NOT_SUPPORTED labels in Hover dataset:")
    print(df_hover["label"].value_counts())
    print("\nNumber of SUPPORTED and NOT_SUPPORTED labels in Feverous dataset:")
    print(df_feverous["label"].value_counts())
    print("\nNumber of SUPPORTED and NOT_SUPPORTED labels in ClimateFever dataset:")
    print(df_climatefever["label"].value_counts())

    # Save standardized datasets.
    hover_out = processed_dir / "hover.csv"
    feverous_out = processed_dir / "feverous.csv"
    climatefever_out = processed_dir / "climatefever.csv"

    ensure_parent_dir(hover_out)
    ensure_parent_dir(feverous_out)
    ensure_parent_dir(climatefever_out)

    df_hover.to_csv(hover_out, index=False)
    df_feverous.to_csv(feverous_out, index=False)
    df_climatefever.to_csv(climatefever_out, index=False)


# ---------------------------------------------------------------------------
# Phase 3: Dataset analysis
# ---------------------------------------------------------------------------

def plot_distribution(data: pd.DataFrame, column: str, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=data,
        x=column,
        hue="dataset",
        alpha=0.6,
        legend=True,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def print_stats(data: pd.DataFrame, column: str) -> None:
    stats_by_dataset = data.groupby("dataset")[column].agg(["mean", "median", "min", "max"])
    print(f"\n--------------------------\nStatistics for {column}:")
    print(stats_by_dataset)
    print("\nOverall statistics:")
    print(data[column].agg(["mean", "median", "min", "max"]))


def compute_correlation(data: pd.DataFrame) -> None:
    corr, p_value = stats.pearsonr(data["number_of_evidences"], data["len_evidences"])
    print(f"\nCorrelation between number of evidences and evidence length: {corr:.4f}")
    print(f"P-value: {p_value}")


def analyze_processed_datasets(processed_dir: Path) -> None:
    # Load processed datasets.
    df_hover = pd.read_csv(processed_dir / "hover.csv")
    df_feverous = pd.read_csv(processed_dir / "feverous.csv")
    df_climatefever = pd.read_csv(processed_dir / "climatefever.csv")

    # Print shapes to confirm sizes.
    print(f"Hover: {df_hover.shape}")
    print(f"Feverous: {df_feverous.shape}")
    print(f"ClimateFever: {df_climatefever.shape}")

    # Print columns and head for each dataset.
    print("\nHover dataset:")
    print(df_hover.columns)
    print(df_hover.head())

    print("\nFeverous dataset:")
    print(df_feverous.columns)
    print(df_feverous.head())

    print("\nClimateFever dataset:")
    print(df_climatefever.columns)
    print(df_climatefever.head())

    # Combine datasets for joint analysis.
    combined_df = pd.concat([df_climatefever, df_feverous, df_hover], ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")

    # save combined_df to processed directory
    combined_out = processed_dir / "combined_datasets.csv"
    ensure_parent_dir(combined_out)
    combined_df.to_csv(combined_out, index=False)

    # Plot distribution of number of evidences.
    plot_distribution(
        combined_df,
        "number_of_evidences",
        "Distribution of Number of Evidences",
        "Number of Evidences",
        "Density",
    )

    # Plot distribution of evidence lengths.
    plot_distribution(
        combined_df,
        "len_evidences",
        "Distribution of Evidence Lengths",
        "Evidence Length",
        "Density",
    )

    # Print summary statistics for both key columns.
    print_stats(combined_df, "number_of_evidences")
    print_stats(combined_df, "len_evidences")

    # Scatter plot: number of evidences vs evidence length.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=combined_df,
        x="number_of_evidences",
        y="len_evidences",
        hue="dataset",
        alpha=0.6,
        legend=True,
    )
    plt.title("Number of Evidences vs Evidence Length")
    plt.xlabel("Number of Evidences")
    plt.ylabel("Evidence Length")
    plt.legend(title="dataset")
    plt.show()

    # Compute and print overall correlation.
    print("\n--------------------------\nOverall correlation:")
    compute_correlation(combined_df)

    # Box plots by dataset.
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_df, x="dataset", y="number_of_evidences")
    plt.title("Number of Evidences by Dataset")
    plt.ylabel("Number of Evidences")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_df, x="dataset", y="len_evidences")
    plt.title("Evidence Length by Dataset")
    plt.ylabel("Evidence Length")
    plt.show()

    # Print counts for claims with exactly 5 evidences.
    print("\nNumber of entries in each dataset if we keep only the claims with exactly 5 evidences:")
    print("ClimateFever:", len(df_climatefever[df_climatefever["number_of_evidences"] == 5]))
    print("Feverous:", len(df_feverous[df_feverous["number_of_evidences"] == 5]))
    print("Hover:", len(df_hover[df_hover["number_of_evidences"] == 5]))


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = ArgumentParser(description="Prepare the fact-checking experiment inputs.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--steps", nargs="+", choices=["raw", "standardize", "combine", "analyze"],
                        default=["raw", "standardize", "combine"])
    parser.add_argument("--climatefever-file", type=Path,
                        help="Optional local raw ClimateFEVER Parquet; otherwise load the official HF test split")
    args = parser.parse_args()
    raw_dir = args.data_dir / "raw"
    interim_dir = args.data_dir / "interim"
    processed_dir = args.data_dir / "processed"
    if "raw" in args.steps:
        process_raw_datasets(raw_dir, interim_dir, args.climatefever_file)
    if "standardize" in args.steps:
        standardize_datasets(interim_dir, processed_dir)
    if "combine" in args.steps:
        frames = [pd.read_csv(processed_dir / f"{name}.csv")
                  for name in ["climatefever", "feverous", "hover"]]
        output = processed_dir / "combined_datasets.csv"
        pd.concat(frames, ignore_index=True).to_csv(output, index=False)
        print(f"Saved combined dataset: {output}")
    if "analyze" in args.steps:
        analyze_processed_datasets(processed_dir)


if __name__ == "__main__":
    main()
