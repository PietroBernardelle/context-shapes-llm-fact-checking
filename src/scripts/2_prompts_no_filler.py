from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from prompt_utils import project_root, load_prompt_config, load_combined_df, ensure_parent_dir, build_context


DEFAULT_CLAIM_ONLY_TEMPLATE = "CLAIM:\n{claim}\n\nAnswer:"


def build_prompts(df: pd.DataFrame, user_template: str, evidence_order: str) -> pd.DataFrame:
    df = df.copy()

    # Build the context field by joining evidence lines.
    df["context"] = df["clean_evidence"].apply(lambda evidences: build_context(evidences, evidence_order))

    # Build the prompt using the configured template.
    df["prompt"] = df.apply(
        lambda row: user_template.format(claim=row["claim"], context=row["context"]),
        axis=1,
    )

    # Add metadata columns for tracking.
    df["evidence_order"] = evidence_order
    df["filler_length"] = 0
    df["evidence_position"] = None

    return df


def build_claim_only_prompts(df: pd.DataFrame, claim_only_template: str) -> pd.DataFrame:
    df = df.copy()

    # No evidence/context in this setting.
    df["context"] = ""
    df["prompt"] = df.apply(
        lambda row: claim_only_template.format(claim=row["claim"]),
        axis=1,
    )

    # Add metadata columns for tracking.
    df["evidence_order"] = "claim_only"
    df["filler_length"] = 0
    df["evidence_position"] = None

    return df


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration (default: config/paper_config.json)",
    )
    parser.add_argument(
        "--evidence-order",
        default="all",
        choices=["multi", "multi_reverse", "claim_only", "all"],
        help="Evidence ordering to generate prompts for",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for the number of rows to process",
    )
    args = parser.parse_args()

    root = project_root()
    config_path = Path(args.config) if args.config else root / "config" / "paper_config.json"
    config = load_prompt_config(config_path)

    # Load the combined dataset and normalize evidence lists.
    combined_path = root / config["paths"]["combined_df"]
    df = load_combined_df(combined_path, max_rows=args.max_rows)

    # Decide which evidence orders to run.
    if args.evidence_order == "all":
        evidence_orders = list(config["settings"]["evidence_orders"])
        evidence_orders.append("claim_only")
    else:
        evidence_orders = [args.evidence_order]

    # Generate prompts for each order and save them.
    for evidence_order in evidence_orders:
        # Build the output path for this setting.
        prompts_dir = root / config["paths"]["prompts_dir"] / "no_filler" / evidence_order
        out_path = prompts_dir / "prompts.pqt"
        ensure_parent_dir(out_path)

        if out_path.exists():
            print(f"File already exists: {out_path}")
            continue

        if evidence_order == "claim_only":
            claim_only_template = config["prompt"].get("claim_only_template", DEFAULT_CLAIM_ONLY_TEMPLATE)
            prompts_df = build_claim_only_prompts(df, claim_only_template)
        else:
            prompts_df = build_prompts(df, config["prompt"]["user_template"], evidence_order)

        # Save prompts to Parquet for easy reuse.
        prompts_df.to_parquet(out_path, index=False)
        print(f"Saved prompts to: {out_path}")


if __name__ == "__main__":
    main()
