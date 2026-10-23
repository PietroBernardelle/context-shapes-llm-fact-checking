# Context Shapes LLMs Retrieval-Augmented Fact-Checking Effectiveness

Initial code and figure snapshot for the paper by Pietro Bernardelle, Stefano Civelli, Kevin Roitero, and Gianluca Demartini.

The Python files and configuration are preserved from the `Needle_new` experiment project. This repository contains the core experiment pipeline and the two figures used in the paper. Data, filler text, generated prompts, model outputs, metrics tables, logs, and model weights are excluded. Their working directories contain only empty `.gitkeep` placeholders.

## Code

| File | Purpose |
| --- | --- |
| `src/scripts/0_preprocessing.py` | Dataset preprocessing and standardization |
| `utils/add_hover_split.py` | Recover the HOVER development split |
| `src/scripts/6_build_experiment_subset.py` | Construct the experiment subset |
| `config/prompt_config.json` | Saved prompt templates, model identifiers, and inference settings |
| `src/scripts/2_prompts_no_filler.py` | Create claim-only and evidence-only prompts |
| `src/scripts/3_prompts_with_filler.py` | Create prompts with controlled context length and evidence position |
| `src/scripts/4_inference_vllm.py` | Baseline inference |
| `src/scripts/4_inference_vllm_subset_bunya.py` | Subset inference with context-dependent runtime settings |
| `src/scripts/5_results_analysis.py` | Baseline evaluation |
| `src/scripts/7_filler_positioning_analysis.py` | Evidence-position plots, including Figure 2 |
| `src/scripts/9_context_length_analysis.py` | Context-length plots, including Figure 1 |
| `src/scripts/10_generate_latex_table.py` | LaTeX table generation |
| `src/scripts/prompt_utils.py` | Shared path, configuration, and prompt helpers |

## Paper figures

Both PNGs were verified against the images embedded in the paper: their dimensions and every RGB pixel match after removing the saved PNG alpha channel.

**Figure 1: accuracy versus context length**

![Figure 1: accuracy versus context length](images/context_length_analysis/avg_accuracy_vs_context_all.png)

**Figure 2: accuracy versus evidence depth**

![Figure 2: accuracy versus evidence depth](images/spaced_by_0.1/combined_accuracy_5models.png)

The existing [PDF version of Figure 2](images/spaced_by_0.1/combined_accuracy_5models.pdf) is also included.

## Snapshot status

This is an archival snapshot for private review, not yet a self-contained reproduction package. The figure matches establish the project and plotting-code association; the exact historical revision and command for every script were not archived or independently verified.

The original experiment files have not been rewritten. In particular:

- The saved configuration includes additional models and context lengths beyond the paper. A paper-specific run should select the five reported models and the four reported context lengths.
- The preprocessing phases are controlled by constants inside the preprocessing script.
- The table generator still uses the original machine-specific paths.
- Figure 1's averaging inputs and the completeness of some inference conditions need review before a full reproduction release.
- Required packages include pandas, a Parquet backend such as pyarrow, NumPy, SciPy, matplotlib, seaborn, scikit-learn, datasets, transformers, PyTorch, and vLLM. A fully verified dependency lockfile is not available in this snapshot.

The old configuration generator, stale launch scripts, unrelated analyses, and notebooks are omitted. Data must be supplied locally before execution. The `.gitignore` explicitly allows only the files in this snapshot and empty directory placeholders, so generated artifacts remain untracked by default.
