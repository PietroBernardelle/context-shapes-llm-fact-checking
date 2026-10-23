# Public paper release

The repository remains private during review. Prepared datasets and generated records stay excluded.

Before publishing a stable version:

1. Reconcile `docs/reproducibility.md` with the manuscript and original outputs.
2. Pass the coverage audit; resolve incomplete runs and invalid predictions.
3. Validate GPU inference and record model/tokenizer revisions.
4. Recompute tables and figures; review numerical changes alongside the manuscript.
5. Add final paper metadata, release version, and date to `CITATION.cff`.
6. Confirm the repository may become public, then tag the reviewed commit, for example `v1.0.0`.
7. Connect the repository to Zenodo and archive the release, or upload its reviewed source archive. Add the assigned version DOI to the paper and citation metadata. No DOI is reserved or invented here.

See [GitHub citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files), [software archiving guidance](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content), and [Zenodo integration](https://help.zenodo.org/docs/github/enable-repository/).

Suggested wording after publication:

> Code, experiment configurations, prompt templates, filler input, and figures are available at https://github.com/PietroBernardelle/context-shapes-llm-fact-checking. Prepared datasets are not redistributed; the repository documents required local inputs and original sources. The accompanying software version is archived at [version DOI].

Omit the last sentence until a DOI exists. Do not claim a complete reproduction package while the documented preparation and validation gaps remain.
