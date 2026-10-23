# Local data requirements

Prepared datasets are not redistributed because redistribution permissions have not been confirmed. Users must obtain source datasets separately under their applicable terms. Dataset directories are placeholders; the filler passage is included separately.

## Sources

- HOVER: [official repository](https://github.com/hover-nlp/hover), including development annotations and the source evidence corpus.
- FEVEROUS: [official dataset page](https://fever.ai/dataset/feverous.html), including development annotations and corresponding Wikipedia evidence resources.
- ClimateFEVER: [author-maintained dataset](https://huggingface.co/datasets/tdiggelm/climate_fever). Its Hugging Face split is called `test`; the archived code uses that split although the supplied manuscript calls it the training set.
- Filler source: the [Wikipedia Llama article](https://en.wikipedia.org/wiki/Llama). The original local text is included in `data/raw/filler/llama.txt`; its exact article revision is unknown. See the [attribution, license, and checksum](../data/raw/filler/README.md).

## Required prepared inputs

| Local file | Expected content |
| --- | --- |
| `data/raw/hover.pqt` | Parquet with `claim`, `label`, `evidence`; labels `SUPPORTED` / `NOT_SUPPORTED`; evidence pieces joined with `++` |
| `data/raw/feverous.pqt` | Parquet with `claim`, `label`, `evidence`; labels `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`; evidence pieces joined with `++` |
| `data/raw/hover_dev_release_v1.1.json` | HOVER development annotations containing claim strings |
| `data/raw/filler/llama.txt` | Included UTF-8 filler passage; preserve its literal escapes and surrounding quotes |

The creation scripts and source revisions for the prepared Parquet files were not recovered. Original annotations alone do not recreate them: evidence references must be resolved against their source corpus. This release does not claim an exact raw-data-to-paper reconstruction. Preserve row order, evidence order, and preprocessing choices; a fixed sampling seed alone does not reproduce a sample when input order changes.

Raw preprocessing loads `tdiggelm/climate_fever`, split `test`. For offline use, supply a raw Parquet export with `--climatefever-file /local/path/climate_fever.pqt`. It must contain `claim`, `claim_label`, and an `evidences` list with `evidence` text. The script excludes NOT_ENOUGH_INFO and DISPUTED and preserves evidence-list order.

## Generated local inputs

Preprocessing writes `claim`, `label`, `clean_evidence`, `number_of_evidences`, `len_evidences`, and `dataset`. CSV `clean_evidence` values serialize lists of strings. Labels are `0=SUPPORTS`, `1=REFUTES`.

`utils/add_hover_split.py` writes `data/processed/combined_datasets_with_split.csv`, identifying HOVER development rows by claim-text membership. Its default `dev` assignment for other datasets does not independently verify their provenance.

`6_build_experiment_subset.py` retains HOVER development rows and all binary ClimateFEVER rows, and samples 2,000 FEVEROUS rows per label with seed 42. Expected counts are HOVER 4,000, FEVEROUS 4,000, ClimateFEVER 907. Preserve repeated rows: duplicate claim strings can occur in the original subset.

Alternatively supply an existing local subset as `data/processed/experiment_subset_seed42.csv` and `.pqt` with the standardized columns above. Its datasets and claim strings must match the local prediction files. No dataset records, identifiers, or generated prompts are included in the repository. The reusable [prompt templates](../prompts/README.md) and the filler passage are included.
