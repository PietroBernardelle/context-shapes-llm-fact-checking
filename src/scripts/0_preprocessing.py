"""
Data preprocessing script.

Steps:
1) (Optional) Process raw datasets and write interim files.
2) (Optional) Standardize datasets and write processed files.
3) (Optional) Analyze processed datasets with summary stats and plots.

Toggle the `RUN_*` flags in the Configuration section to control what runs.
"""

# Suppress noisy warnings to keep output readable.
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
from datasets import load_dataset
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

# Toggle each phase of the workflow.
RUN_RAW_PROCESSING = True
RUN_STANDARDIZATION = False
RUN_ANALYSIS = False


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

def process_raw_datasets() -> None:
    # Define input paths.
    feverous_path = RAW_DIR / "feverous.pqt"
    hover_path = RAW_DIR / "hover.pqt"
    climatefever_path = "climate_fever"

    # Process Feverous dataset.
    df_feverous = pd.read_parquet(feverous_path)
    df_feverous = df_feverous[df_feverous["label"] != "NOT ENOUGH INFO"]

    # Build evidence statistics.
    df_feverous["clean_evidence"] = df_feverous["evidence"].apply(split_evidence_text)
    df_feverous["number_of_evidences"] = df_feverous["clean_evidence"].apply(len)
    df_feverous["len_evidences"] = df_feverous["clean_evidence"].apply(compute_len_evidence)

    # Save full cleaned Feverous.
    feverous_out = INTERIM_DIR / "cleaned_feverous.csv"
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
    hover_out = INTERIM_DIR / "cleaned_hover.csv"
    ensure_parent_dir(hover_out)
    df_hover.to_csv(hover_out, index=False)
    print("Hover processed")

    # Process ClimateFever dataset from HuggingFace.
    climatefever = load_dataset(climatefever_path, split="test")
    df_climatefever = pd.DataFrame(climatefever)

    # Remove NOT ENOUGH INFO claims and DISPUTED claims.
    df_climatefever = df_climatefever[(df_climatefever["claim_label"] != 2) & (df_climatefever["claim_label"] != 3)]
    df_climatefever = df_climatefever.reset_index(drop=True)

    # Extract evidence information.
    evidence_data = df_climatefever.apply(process_climatefever_evidences, axis=1)
    evidence_df = pd.DataFrame(evidence_data.tolist(), index=df_climatefever.index)
    df_climatefever = pd.concat([df_climatefever, evidence_df], axis=1)

    # Save full cleaned ClimateFever.
    climatefever_out = INTERIM_DIR / "cleaned_climatefever.csv"
    ensure_parent_dir(climatefever_out)
    df_climatefever.to_csv(climatefever_out, index=False)
    print("ClimateFever processed")

    print("\nAll datasets processed and saved")


# ---------------------------------------------------------------------------
# Phase 2: Dataset standardization
# ---------------------------------------------------------------------------

def standardize_datasets() -> None:
    # Load interim cleaned datasets.
    df_hover = pd.read_csv(INTERIM_DIR / "cleaned_hover.csv")
    df_feverous = pd.read_csv(INTERIM_DIR / "cleaned_feverous.csv")
    df_climatefever = pd.read_csv(INTERIM_DIR / "cleaned_climatefever.csv")

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
    hover_out = PROCESSED_DIR / "hover.csv"
    feverous_out = PROCESSED_DIR / "feverous.csv"
    climatefever_out = PROCESSED_DIR / "climatefever.csv"

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


def analyze_processed_datasets() -> None:
    # Load processed datasets.
    df_hover = pd.read_csv(PROCESSED_DIR / "hover.csv")
    df_feverous = pd.read_csv(PROCESSED_DIR / "feverous.csv")
    df_climatefever = pd.read_csv(PROCESSED_DIR / "climatefever.csv")

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
    combined_out = PROCESSED_DIR / "combined_datasets.csv"
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
    """Run selected preprocessing steps based on flags."""
    if RUN_RAW_PROCESSING:
        process_raw_datasets()

    if RUN_STANDARDIZATION:
        standardize_datasets()

    if RUN_ANALYSIS:
        analyze_processed_datasets()

    if not (RUN_RAW_PROCESSING or RUN_STANDARDIZATION or RUN_ANALYSIS):
        print("No steps selected. Set a RUN_* flag to True.")


if __name__ == "__main__":
    main()
