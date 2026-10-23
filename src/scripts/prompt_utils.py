from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_prompt_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_list(value: Any) -> list:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return [value]
    if value is None:
        return []
    return value


def load_combined_df(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)
    df["clean_evidence"] = df["clean_evidence"].apply(ensure_list)
    return df


def build_context(evidences: list[str], order: str) -> str:
    if order == "multi_reverse":
        evidences = list(reversed(evidences))
    evidences = [e.lstrip() for e in evidences]
    return " ".join(evidences).lstrip()


def read_filler_text(filler_path: Path, tokenizer, max_length: int) -> str:
    filler_text = filler_path.read_text(encoding="utf-8")
    filler_tokens = tokenizer.tokenize(filler_text)
    filler_truncated_tokens = filler_tokens[:max_length]
    return tokenizer.convert_tokens_to_string(filler_truncated_tokens)


def insert_statement(text: str, statement: str, positions: list[float], tokenizer) -> str:
    # Tokenize the text and the statement.
    text_tokens = tokenizer.tokenize(text)
    statement_tokens = tokenizer.tokenize(statement)
    len_text_tokens = len(text_tokens)
    len_statement_tokens = len(statement_tokens)

    # Token list to detect sentence boundaries.
    period_tokens = tokenizer.tokenize(".") + ["."]

    # Convert fractional positions into token indexes.
    token_positions = [int(len_text_tokens * pos) for pos in positions]

    for pos in token_positions:
        end_pos = pos + len_statement_tokens

        # If insertion would overflow, insert at the end instead.
        if end_pos > len_text_tokens or pos == len_text_tokens:
            # Remove the last statement-length tokens to make space.
            text_tokens = text_tokens[:-len_statement_tokens]

            # Trim to the previous sentence boundary.
            while text_tokens and text_tokens[-1] not in period_tokens:
                text_tokens = text_tokens[:-1]

            # Append the statement tokens.
            text_tokens = text_tokens + statement_tokens
        else:
            # Split the text in two parts around the insertion span.
            first_part = text_tokens[:pos]
            last_part = text_tokens[end_pos:]

            # Trim to sentence boundaries on both sides.
            while first_part and first_part[-1] not in period_tokens:
                first_part = first_part[:-1]
            first_part += tokenizer.tokenize(" ")

            while last_part and last_part[0] not in period_tokens:
                last_part = last_part[1:]
            last_part = last_part[1:]

            # Insert the statement tokens in between.
            text_tokens = first_part + statement_tokens + last_part

    # Convert tokens back to a string.
    return tokenizer.convert_tokens_to_string(text_tokens)


def format_position(pos: float) -> str:
    if isinstance(pos, int) or (isinstance(pos, float) and pos.is_integer()):
        return str(int(pos))
    return str(pos).replace(".", "p")
