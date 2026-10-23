from __future__ import annotations

import gc
import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from vllm import LLM, SamplingParams

try:
    # Newer vLLM versions
    from vllm.sampling_params import GuidedDecodingParams
except Exception:  # pragma: no cover
    from vllm import GuidedDecodingParams

from prompt_utils import project_root, load_prompt_config, ensure_parent_dir, format_position


def parse_json_label(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'"label"\s*:\s*([01])', text)
        if match:
            return {"label": int(match.group(1))}
    return {"label": None}


def build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def read_subset_df(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pqt", ".parquet"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_subset_keys(path: Path) -> set[tuple[str, str]]:
    df = read_subset_df(path)
    required = {"dataset", "claim"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in subset file {path}: {sorted(missing)}")

    dataset_norm = df["dataset"].fillna("").astype(str).str.strip().str.lower()
    claim_norm = df["claim"].fillna("").astype(str).str.strip()
    keys = set(zip(dataset_norm, claim_norm))
    if not keys:
        raise SystemExit(f"No (dataset, claim) keys in subset file: {path}")
    return keys


def infer_nominal_length(prompt_file: Path) -> int | None:
    match = re.search(r"len(\d+)", prompt_file.name)
    if match:
        val = int(match.group(1))
        # Group 29k and 32k together as ~32k
        if 28000 <= val <= 33000:
            return 32000
        return val

    parts_lower = [part.lower() for part in prompt_file.parts]
    if "no_filler" in parts_lower:
        return 0
    return None


def parse_model_size_b(model_name: str) -> float | None:
    # Picks the last "...B" number, e.g. "Llama-3.1-70B-Instruct" -> 70
    matches = re.findall(r"(\d+(?:\.\d+)?)B", model_name, flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def base_runtime_plan(nominal_len: int | None, model_name: str = "") -> dict[str, int]:
    # Tuned for fast/robust long-context inference on H100-class GPUs.
    if nominal_len is None:
        return {"max_model_len": 9216, "batch_size": 24, "max_num_seqs": 24}
    if nominal_len <= 0:
        return {"max_model_len": 4096, "batch_size": 64, "max_num_seqs": 64}
    if nominal_len <= 2048:
        return {"max_model_len": 3072, "batch_size": 64, "max_num_seqs": 64}
    if nominal_len <= 4096:
        return {"max_model_len": 5120, "batch_size": 48, "max_num_seqs": 48}
    if nominal_len <= 8192:
        return {"max_model_len": 9216, "batch_size": 24, "max_num_seqs": 24}
    if nominal_len <= 16384:
        return {"max_model_len": 17408, "batch_size": 12, "max_num_seqs": 12}
    if nominal_len <= 32768:
        # 34k for others, 31792 for qwen
        maxlen = 31792 if "qwen" in model_name.lower() else 34000
        return {"max_model_len": maxlen, "batch_size": 4, "max_num_seqs": 4}
    return {
        "max_model_len": int(nominal_len + 1024),
        "batch_size": 2,
        "max_num_seqs": 2,
    }


def scale_plan_for_model(plan: dict[str, int], model_name: str) -> dict[str, int]:
    size_b = parse_model_size_b(model_name)
    if size_b is None:
        return plan

    divisor = 1
    if size_b >= 60:
        divisor = 4
    elif size_b >= 30:
        divisor = 2

    if divisor == 1:
        return plan

    return {
        "max_model_len": plan["max_model_len"],
        "batch_size": max(1, plan["batch_size"] // divisor),
        "max_num_seqs": max(1, plan["max_num_seqs"] // divisor),
    }


def planned_runtime(
    nominal_len: int | None,
    model_name: str,
    batch_size_override: int | None,
    max_num_seqs_override: int | None,
    max_model_len_override: int | None,
) -> dict[str, int]:
    plan = base_runtime_plan(nominal_len, model_name=model_name)
    plan = scale_plan_for_model(plan, model_name)

    if batch_size_override is not None:
        plan["batch_size"] = batch_size_override
    if max_num_seqs_override is not None:
        plan["max_num_seqs"] = max_num_seqs_override
    if max_model_len_override is not None:
        plan["max_model_len"] = max_model_len_override

    return plan


def output_filename_for_df(df: pd.DataFrame, prompt_file: Path, output_tag: str | None) -> str:
    try:
        if "evidence_order" in df.columns and "filler_length" in df.columns:
            evidence_order = str(df["evidence_order"].iloc[0])
            filler_len = int(df["filler_length"].iloc[0])

            if filler_len == 0:
                base = f"res_{evidence_order}_no_filler"
            else:
                if "evidence_position" in df.columns:
                    ev_pos_val = df["evidence_position"].iloc[0]
                    ev_pos_str = format_position(float(ev_pos_val))
                else:
                    ev_pos_str = "NA"
                base = f"res_{evidence_order}_len{filler_len}_pos{ev_pos_str}"
        else:
            base = f"{prompt_file.stem}_results"
    except Exception:
        base = f"{prompt_file.stem}_results"

    suffix = f"_{output_tag}" if output_tag else ""
    return f"{base}{suffix}.pqt"


def run_inference_for_file(
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer,
    system_prompt: str,
    prompt_file: Path,
    out_dir: Path,
    batch_size: int,
    max_rows: int | None,
    subset_keys: set[tuple[str, str]],
    output_tag: str | None,
    overwrite: bool,
) -> None:
    df = pd.read_parquet(prompt_file)

    if max_rows:
        df = df.head(max_rows)

    if "dataset" not in df.columns or "claim" not in df.columns:
        print(f"Skipping {prompt_file.name}: missing dataset/claim columns")
        return

    dataset_norm = df["dataset"].fillna("").astype(str).str.strip().str.lower()
    claim_norm = df["claim"].fillna("").astype(str).str.strip()
    subset_mask = [(d, c) in subset_keys for d, c in zip(dataset_norm, claim_norm)]
    df = df.loc[subset_mask].copy()

    if df.empty:
        print(f"Skipping {prompt_file.name}: no rows match subset")
        return

    fname = output_filename_for_df(df, prompt_file, output_tag)
    out_path = out_dir / fname
    ensure_parent_dir(out_path)

    if out_path.exists() and not overwrite:
        print(f"Skipping existing results: {out_path}")
        return

    prompts = [
        build_prompt(tokenizer, system_prompt, prompt)
        for prompt in df["prompt"].tolist()
    ]

    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        batch_outputs = llm.generate(batch, sampling_params)
        outputs.extend([item.outputs[0].text for item in batch_outputs])

    parsed = [parse_json_label(text) for text in outputs]
    df["pred_json"] = parsed
    df["pred_label"] = [row.get("label") for row in parsed]

    df.to_parquet(out_path, index=False)
    print(f"Saved results to: {out_path}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Path to prompt_config.json (defaults to config/prompt_config.json)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of model names, or 'all' to run every model",
    )
    parser.add_argument(
        "--prompts-dir",
        required=True,
        help="Directory with prompt files (.pqt). You can use '{model}' as a placeholder.",
    )
    parser.add_argument(
        "--subset-file",
        default=None,
        help=(
            "Subset file (.pqt/.parquet/.csv) with claim+dataset columns "
            "(defaults to data/processed/experiment_subset_seed42.pqt)"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Output results root. Defaults to out/results/subset_seed42 "
            "(model subfolders are created under this path)."
        ),
    )
    parser.add_argument(
        "--output-tag",
        default="subset_seed42",
        help="Tag appended to output result filenames (default: subset_seed42)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for batch size (otherwise auto-tuned by context length/model size)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Optional override for vLLM max_num_seqs (otherwise auto-tuned)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional override for vLLM max_model_len (otherwise auto-tuned by context length)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization (default: 0.9)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for number of rows read from each prompt file before subset filtering",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files (default: skip existing files)",
    )
    args = parser.parse_args()

    root = project_root()
    config_path = Path(args.config) if args.config else root / "config" / "prompt_config.json"
    config = load_prompt_config(config_path)

    subset_file = (
        Path(args.subset_file)
        if args.subset_file
        else root / "data" / "processed" / "experiment_subset_seed42.pqt"
    )
    if not subset_file.exists():
        raise SystemExit(f"Subset file not found: {subset_file}")
    subset_keys = load_subset_keys(subset_file)
    print(f"Loaded subset keys: {len(subset_keys)} from {subset_file}")

    if args.models is None or args.models.lower() == "all":
        model_names = list(config["models"].keys())
    else:
        model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    # Strict JSON output schema for {"label": 0/1}
    json_schema = {
        "type": "object",
        "properties": {"label": {"type": "integer", "enum": [0, 1]}},
        "required": ["label"],
        "additionalProperties": False,
    }
    guided_decoding = GuidedDecodingParams(json=json_schema)

    base_results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else (root / config["paths"]["results_dir"] / "subset_seed42")
    )

    for model_name in model_names:
        if model_name not in config["models"]:
            raise ValueError(f"Model '{model_name}' not found in prompt_config.json")

        model_id = config["models"][model_name]["model_id"]

        def resolve_path(p: Path) -> Path:
            if p.exists():
                return p
            if not p.is_absolute():
                p_root = root / p
                if p_root.exists():
                    return p_root
            return p

        prompts_dir = resolve_path(Path(args.prompts_dir.format(model=model_name)))
        prompt_files = sorted(prompts_dir.rglob("*.pqt"))

        if not prompt_files and "{model}" in args.prompts_dir:
            fallback_dir = resolve_path(Path(args.prompts_dir.format(model="")))
            fallback_files = sorted(fallback_dir.rglob("*.pqt"))
            if fallback_files:
                print(f"No prompts in {prompts_dir}, falling back to {fallback_dir}")
                prompts_dir = fallback_dir
                prompt_files = fallback_files

        if not prompt_files:
            print(f"No prompt files found in: {prompts_dir}")
            continue

        # Order by nominal context length: 2k, 4k, 8k, 16k, 32k.
        files_by_len: dict[int | None, list[Path]] = {}
        for prompt_file in prompt_files:
            nominal_len = infer_nominal_length(prompt_file)
            files_by_len.setdefault(nominal_len, []).append(prompt_file)

        def length_order_key(v: int | None) -> tuple[int, int]:
            # Run positive filler lengths first: 2k -> 4k -> 8k -> ...
            if isinstance(v, int) and v > 0:
                return (0, v)
            # Then no-filler / unknown lengths.
            if isinstance(v, int) and v <= 0:
                return (1, v)
            return (2, 10**9)

        ordered_lengths = sorted(files_by_len.keys(), key=length_order_key)

        results_dir = base_results_dir / model_name

        for nominal_len in ordered_lengths:
            plan = planned_runtime(
                nominal_len,
                model_name,
                batch_size_override=args.batch_size,
                max_num_seqs_override=args.max_num_seqs,
                max_model_len_override=args.max_model_len,
            )

            group_files = sorted(files_by_len[nominal_len])
            print(
                f"Initializing {model_name} for nominal_len={nominal_len} "
                f"with max_model_len={plan['max_model_len']}, "
                f"batch_size={plan['batch_size']}, max_num_seqs={plan['max_num_seqs']} "
                f"for {len(group_files)} files."
            )

            llm_kwargs = {
                "model": model_id,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "tensor_parallel_size": args.tensor_parallel_size,
                "enable_prefix_caching": True,
                "dtype": "auto",
                "max_num_seqs": plan["max_num_seqs"],
                "disable_log_stats": True,
                "seed": 22,
                "max_model_len": plan["max_model_len"],
            }
            llm = LLM(**llm_kwargs)
            tokenizer = llm.get_tokenizer()

            sampling_params = SamplingParams(
                max_tokens=config["inference"]["max_new_tokens"],
                temperature=config["inference"]["temperature"],
                top_p=config["inference"]["top_p"],
                guided_decoding=guided_decoding,
            )

            for prompt_file in group_files:
                print(f"Processing file: {prompt_file.name}")
                try:
                    rel_path = prompt_file.relative_to(prompts_dir)
                    target_dir = results_dir / rel_path.parent
                except ValueError:
                    target_dir = results_dir

                run_inference_for_file(
                    llm=llm,
                    sampling_params=sampling_params,
                    tokenizer=tokenizer,
                    system_prompt=config["prompt"]["system_prompt"],
                    prompt_file=prompt_file,
                    out_dir=target_dir,
                    batch_size=plan["batch_size"],
                    max_rows=args.max_rows,
                    subset_keys=subset_keys,
                    output_tag=args.output_tag,
                    overwrite=args.overwrite,
                )

            del llm
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Completed model: {model_name}")


if __name__ == "__main__":
    main()
