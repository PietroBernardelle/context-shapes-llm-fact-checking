# Fact-checking prompt templates

These UTF-8 files contain the exact template strings in the `prompt` section of
[`config/paper_config.json`](../config/paper_config.json), without added trailing
newlines. The generation and inference scripts read that configuration; these
copies make the templates accessible without extracting JSON. Keep both in sync
when changing the prompting protocol.

| File | Role |
| --- | --- |
| [system.txt](system.txt) | System instructions and the `0=SUPPORTS`, `1=REFUTES` output convention |
| [user.txt](user.txt) | User message with `{claim}` and `{context}` placeholders |
| [claim_only.txt](claim_only.txt) | User message with only the `{claim}` placeholder |

For evidence-only prompts, the context contains the ordered evidence pieces.
For filler conditions, it contains evidence placed within the supplied
[filler passage](../data/raw/filler/README.md). The generation code tokenizes the
filler with the selected model's tokenizer, repeats and truncates it to the
configured length, then places evidence at the selected depth with sentence-boundary
adjustments. The configured length describes the filler token budget before
evidence placement, rather than the total chat-formatted input length.

Inference supplies the system and user messages to the model tokenizer's
`apply_chat_template(..., tokenize=False, add_generation_prompt=True)`. These text
files therefore omit model-specific chat delimiters. Model/tokenizer revision
differences can affect the final input.

Use `2_prompts_no_filler.py` and `3_prompts_with_filler.py` as described in the
[project README](../README.md#prepare-inputs-and-prompts) to instantiate the
templates from locally supplied datasets. Generated prompts contain dataset text
and are excluded from version control.

The templates are covered by the repository's [MIT license](../LICENSE).
