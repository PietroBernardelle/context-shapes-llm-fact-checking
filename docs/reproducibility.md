# Reproducibility and manuscript reconciliation

The initial snapshot is commit `1e149ae04f7b7bff2dc45c2d2ebfda5e78e69f8d`. Subsequent corrections improve execution and evaluation; they do not constitute a full experimental rerun.

## Corrected code

1. The original table generator treated `1` as SUPPORTS and `0` as REFUTES, opposite to preprocessing and prompts. It now uses `0=SUPPORTS`, `1=REFUTES`. Recompute and compare the manuscript table; the code finding alone does not prove which historical revision produced that table.
2. Original baseline and plotting summaries used label-1 F1, whereas the manuscript describes macro-F1. Summaries now average the two class F1 scores. This correction does not affect accuracy plots.
3. The LaTeX generator now applies the experiment subset to baseline results. Plot baselines match both dataset and normalized claim text.
4. New context-length analyses default to the eleven specified positions and four lengths. `--all-positions` retains a historical diagnostic mode.

Invalid predictions are excluded from metric denominators as in the historical plotting code. The coverage audit flags them, preventing such a run from passing as a complete validated evaluation.

## Findings requiring review

- Saved metrics report 4,096 rows for Llama-3.1-70B-Instruct at 16,384 tokens and evidence positions 30%, 40%, 50%, 60%, and 70%, versus the expected 8,907. Check the underlying outputs and rerun if necessary. This preparation does not fabricate or regenerate missing predictions.
- Saved raw metrics include additional 5%, 15%, 85%, and 95% positions for some models/lengths. Their all-position means differ from the eleven-position means. The original Figure 1 corresponds to the all-position series and remains unchanged. Decide whether to update the figure or clarify the protocol after validating complete results.
- The manuscript calls ClimateFEVER's available split the training set. The archived code and author-maintained dataset call it `test`; reconcile this naming difference.
- Exact prepared-data provenance, filler article revision, and model revision hashes were not recovered. Model repository names are not immutable historical checkpoints.

## Environment and validation boundaries

Original logs identify vLLM 0.8.5.post1, seed 22, and BF16 loading. A launcher requested CUDA 12.6 and an H100 allocation; that is not proof of the hardware used for every model. The current cluster environment has changed and is not treated as a historical lockfile.

The Python 3.11 analysis environment is pinned and tested separately. Regression tests cover label mapping, macro-F1, invalid predictions, dataset-aware matching, duplicate rows, grid selection, and partial-output detection. CLI and synthetic-input checks do not establish equivalence to all manuscript results. GPU inference is not tested during this preparation.

Original figures are preserved. New plots go to ignored `out/figures/` so corrected analyses cannot overwrite the manuscript images accidentally.

Validation performed for this candidate: eight regression tests, twelve CLI help checks, and a synthetic workflow covering raw preprocessing, split recovery, balanced sampling, no-filler prompts, result auditing, both table analyses, and both figure generators. The analysis environment passed its package consistency check. A plot regenerated locally from the saved all-position metrics followed the original Figure 1 curves, but was not pixel-identical because layout/rendering differed.

Attempts to read the original large files on the Bunya archive stalled and were stopped. The partial-run finding therefore remains based on the saved metrics rather than a new audit of all raw predictions. No GPU jobs were launched.
