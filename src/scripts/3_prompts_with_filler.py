from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from prompt_utils import (
    project_root,
    load_prompt_config,
    load_combined_df,
    ensure_parent_dir,
    build_context,
    format_position,
)


def prepare_filler_tokens(filler_path: Path, tokenizer, max_length: int) -> list[str]:
    filler_text = filler_path.read_text(encoding="utf-8")
    filler_tokens = tokenizer.tokenize(filler_text)
    
    if not filler_tokens:
        return []
        
    current_tokens = list(filler_tokens)
    while len(current_tokens) < max_length:
        current_tokens.extend(filler_tokens)
        
    return current_tokens[:max_length]


def insert_evidence_into_tokens(filler_tokens: list[str], statement: str, positions: list[float], tokenizer) -> str:
    statement_tokens = tokenizer.tokenize(statement)
    len_text_tokens = len(filler_tokens)
    len_statement_tokens = len(statement_tokens)

    # Token list to detect sentence boundaries.
    period_tokens = set(tokenizer.tokenize(".") + ["."])
    space_tokens = tokenizer.tokenize(" ")

    # Convert fractional positions into token indexes.
    token_positions = [int(len_text_tokens * pos) for pos in positions]

    # We assume valid positions list has 1 item as per current usage, but loop supports logic
    # Note: This logic assumes we return after the first valid insertion as typical for this script
    for pos in token_positions:
        end_pos = pos + len_statement_tokens

        # If insertion would overflow, insert at the end instead.
        if end_pos > len_text_tokens or pos == len_text_tokens:
            # Remove the last statement-length tokens to make space.
            # Using slicing to create new list
            cutoff = len(filler_tokens) - len_statement_tokens
            if cutoff < 0:
                cutoff = 0
            text_tokens = filler_tokens[:cutoff]

            # Trim to the previous sentence boundary.
            while text_tokens and text_tokens[-1] not in period_tokens:
                text_tokens.pop()

            # Append the statement tokens.
            return tokenizer.convert_tokens_to_string(text_tokens + statement_tokens)
        else:
            # Split the text in two parts around the insertion span.
            first_part = filler_tokens[:pos] # copy
            last_part = filler_tokens[end_pos:] # copy

            # Trim to sentence boundaries on both sides.
            while first_part and first_part[-1] not in period_tokens:
                first_part.pop()
            
            first_part = first_part + space_tokens

            while last_part and last_part[0] not in period_tokens:
                last_part.pop(0)
            
            if last_part:
                # Based on prompt_utils, we skip the boundary token (period)
                last_part.pop(0)

            # Insert the statement tokens in between.
            return tokenizer.convert_tokens_to_string(first_part + statement_tokens + last_part)

    return tokenizer.convert_tokens_to_string(filler_tokens)


def build_prompts_with_filler(
    df: pd.DataFrame,
    user_template: str,
    evidence_order: str,
    filler_tokens: list[str],
    evidence_position: float,
    tokenizer,
) -> pd.DataFrame:
    df = df.copy()

    # Build the evidence string (multi or reverse order).
    df["evidence_text"] = df["clean_evidence"].apply(lambda evidences: build_context(evidences, evidence_order))

    # Insert the evidence into the filler text at the chosen position.
    if filler_tokens:
        df["context"] = df["evidence_text"].apply(
            lambda evidence: insert_evidence_into_tokens(
                filler_tokens=filler_tokens,
                statement=evidence,
                positions=[evidence_position],
                tokenizer=tokenizer,
            )
        )
    else:
        # If filler length is zero, the context is just the evidence.
        df["context"] = df["evidence_text"]

    # Build the prompt using the configured template.
    df["prompt"] = df.apply(
        lambda row: user_template.format(claim=row["claim"], context=row["context"]),
        axis=1,
    )

    # Add metadata columns for tracking.
    df["evidence_order"] = evidence_order
    df["evidence_position"] = evidence_position

    return df


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Path to prompt_config.json",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name as defined in the prompt configuration",
    )
    parser.add_argument(
        "--evidence-order",
        default="all",
        choices=["multi", "multi_reverse", "all"],
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

    # Validate model and load its tokenizer.
    if args.model not in config["models"]:
        raise ValueError(f"Model '{args.model}' not found in prompt_config.json")

    model_id = config["models"][args.model]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=config["models"][args.model].get("revision"))

    # Load dataset and filler text path.
    combined_path = root / config["paths"]["combined_df"]
    filler_path = root / config["paths"]["filler_text"]
    df = load_combined_df(combined_path, max_rows=args.max_rows)

    # Decide which evidence orders to run.
    if args.evidence_order == "all":
        evidence_orders = config["settings"]["evidence_orders"]
    else:
        evidence_orders = [args.evidence_order]

    # Look up the filler lengths and evidence positions for this model.
    filler_lengths = config["settings"]["filler_lengths"].get(args.model, [0])
    evidence_positions = config["settings"]["evidence_positions"]

    # Cache filler tokens per length (tokenizer-dependent).
    filler_tokens_by_length = {
        filler_length: (prepare_filler_tokens(filler_path, tokenizer, max_length=filler_length) if filler_length else [])
        for filler_length in filler_lengths
    }

    # Generate prompts across lengths and positions.
    for evidence_order in evidence_orders:
        prompts_dir = (
            root
            / config["paths"]["prompts_dir"]
            / "with_filler"
            / args.model
            / evidence_order
        )
        for filler_length in filler_lengths:
            filler_tokens = filler_tokens_by_length[filler_length]

            for evidence_position in evidence_positions:
                pos_tag = format_position(float(evidence_position))
                out_path = prompts_dir / f"prompts_len{filler_length}_pos{pos_tag}.pqt"

                if out_path.exists():
                    print(f"Skipping existing prompts: {out_path}")
                    continue

                prompts_df = build_prompts_with_filler(
                    df,
                    config["prompt"]["user_template"],
                    evidence_order,
                    filler_tokens,
                    evidence_position,
                    tokenizer,
                )

                # Add metadata for this run.
                prompts_df["filler_length"] = filler_length

                ensure_parent_dir(out_path)

                # Save prompts to Parquet for easy reuse.
                prompts_df.to_parquet(out_path, index=False)
                print(f"Saved prompts to: {out_path}")


if __name__ == "__main__":
    main()
