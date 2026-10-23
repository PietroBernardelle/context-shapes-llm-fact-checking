from __future__ import annotations

import json
import re
import gc
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

import pandas as pd
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


def run_inference_for_file(
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer,
    system_prompt: str,
    prompt_file: Path,
    out_dir: Path,
    batch_size: int,
    max_rows: int | None,
) -> None:
    df = pd.read_parquet(prompt_file)
    if max_rows:
        df = df.head(max_rows)

    # Build chat-templated prompts.
    prompts = [
        build_prompt(tokenizer, system_prompt, prompt)
        for prompt in df["prompt"].tolist()
    ]

    # Run generation in batches to keep memory stable.
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        batch_outputs = llm.generate(batch, sampling_params)
        outputs.extend([item.outputs[0].text for item in batch_outputs])

    # Parse JSON labels and attach to the dataframe.
    parsed = [parse_json_label(text) for text in outputs]
    df["pred_json"] = parsed
    df["pred_label"] = [row.get("label") for row in parsed]

    # Determine output filename based on metadata columns if available.
    try:
        if "evidence_order" in df.columns and "filler_length" in df.columns:
            evidence_order = str(df["evidence_order"].iloc[0])
            filler_len = int(df["filler_length"].iloc[0])
            
            if filler_len == 0:
                # No filler case: e.g. res_claim_only_no_filler.pqt
                fname = f"res_{evidence_order}_no_filler.pqt"
            else:
                # Filler case: e.g. res_multi_len100_pos0p1.pqt
                if "evidence_position" in df.columns:
                    ev_pos_val = df["evidence_position"].iloc[0]
                    ev_pos_str = format_position(float(ev_pos_val))
                else:
                    ev_pos_str = "NA"
                fname = f"res_{evidence_order}_len{filler_len}_pos{ev_pos_str}.pqt"
            
            out_path = out_dir / fname
        else:
            # Fallback for generic CSVs
            out_path = out_dir / f"{prompt_file.stem}_results.pqt"
    except Exception:
        # Fallback on error
        out_path = out_dir / f"{prompt_file.stem}_results.pqt"

    ensure_parent_dir(out_path)
    
    if out_path.exists():
        print(f"Skipping existing results: {out_path}")
        return

    df.to_parquet(out_path, index=False)
    print(f"Saved results to: {out_path}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration (default: config/paper_config.json)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of model names, or 'all' to run every model",
    )
    parser.add_argument(
        "--prompts-dir",
        required=True,
        help="Directory with prompt CSV files. You can use '{model}' as a placeholder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for vLLM generation",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for the number of rows to process",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size",
    )
    args = parser.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    try:
        from vllm.sampling_params import GuidedDecodingParams
    except ImportError:
        from vllm import GuidedDecodingParams

    root = project_root()
    config_path = Path(args.config) if args.config else root / "config" / "paper_config.json"
    config = load_prompt_config(config_path)

    # Decide which models to run.
    if args.models is None:
        model_names = list(config["models"].keys())
    elif args.models.lower() == "all":
        model_names = list(config["models"].keys())
    else:
        model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    # Build the structured output schema for guided decoding.
    json_schema = {
        "type": "object",
        "properties": {"label": {"type": "integer", "enum": [0, 1]}},
        "required": ["label"],
        "additionalProperties": False,
    }

    guided_decoding = GuidedDecodingParams(json=json_schema)

    # Run inference for each selected model.
    for model_name in model_names:
        if model_name not in config["models"]:
            raise ValueError(f"Model '{model_name}' not found in prompt_config.json")

        model_id = config["models"][model_name]["model_id"]

        # Helper to resolve paths relative to CWD or Root
        def resolve_path(p: Path) -> Path:
            if p.exists():
                return p
            # If path is relative and doesn't exist locally, try resolving against project root
            if not p.is_absolute():
                p_root = root / p
                if p_root.exists():
                    return p_root
            return p

        # Resolve the prompts directory (replace {model} if present).
        prompts_dir = resolve_path(Path(args.prompts_dir.format(model=model_name)))
        prompt_files = sorted(prompts_dir.rglob("*.pqt"))

        # Fallback: if {model} yielded no files, try generic path (empty model string).
        # This allows using a single config path like "prompts/{model}" even if
        # some subsets (like no_filler) don't use model subfolders.
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

        # Output directory is grouped by model name.
        results_dir = root / config["paths"]["results_dir"] / model_name

        # Group files by max_model_len requirements
        file_groups = {}
        for p in prompt_files:
            length_val = None
            if "2048" in p.name:
                length_val = 2500
            elif "4096" in p.name:
                length_val = 4500
            elif "8192" in p.name:
                length_val = 8600
            elif "16384" in p.name:
                length_val = 17000
            elif "32768" in p.name:
                length_val = 33400
            
            if length_val not in file_groups:
                file_groups[length_val] = []
            file_groups[length_val].append(p)

        # Process each group
        for length_val, group_files in file_groups.items():
            print(f"Initializing vLLM for model {model_name} with max_model_len={length_val} for {len(group_files)} files.")

            # Initialize vLLM for this model.
            llm_kwargs = {
                "model": model_id, 
                "gpu_memory_utilization": 0.9,
                "tensor_parallel_size": args.tensor_parallel_size,
                "enable_prefix_caching": True,
                "dtype": "auto",  
                "max_num_seqs": 200,
                "disable_log_stats": True,
                "seed": config["inference"].get("seed", 22),
            }
            if length_val is not None:
                llm_kwargs["max_model_len"] = length_val

            if config["models"][model_name].get("revision"):
                llm_kwargs["revision"] = config["models"][model_name]["revision"]
                llm_kwargs["tokenizer_revision"] = config["models"][model_name]["revision"]
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

                # Preserve subdirectory structure relative to prompts_dir
                # e.g. if we found "claim_only/prompts.csv", we want output to be in ".../claim_only/"
                try:
                    rel_path = prompt_file.relative_to(prompts_dir)
                    target_dir = results_dir / rel_path.parent
                except ValueError:
                    target_dir = results_dir

                run_inference_for_file(
                    llm,
                    sampling_params,
                    tokenizer,
                    config["prompt"]["system_prompt"],
                    prompt_file,
                    target_dir,
                    batch_size=args.batch_size,
                    max_rows=args.max_rows,
                )

            # Free up GPU memory before loading the next model
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            
        print(f"Freed GPU memory for model {model_name}")


if __name__ == "__main__":
    main()
