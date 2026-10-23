# Llama filler passage

[`llama.txt`](llama.txt) is the original filler input recovered from the experiment
archive. It is a plain-text adaptation of **Llama**, by Wikipedia contributors:

- Article: https://en.wikipedia.org/wiki/Llama
- Contributor history: https://en.wikipedia.org/w/index.php?title=Llama&action=history
- License: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/legalcode)

This third-party text is distributed under **CC BY-SA 4.0**, separately from the
repository's MIT-licensed code and prompt templates. Retain attribution, the
license notice, and a description of changes when redistributing an adaptation.
See [Wikimedia's reuse terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content).

The archived adaptation removes web formatting and includes escaped characters
and surrounding single quotes. Its original extraction procedure, article revision,
and extraction date were not recovered. The links above identify the source and
contributors; they do not identify an exact historical revision.

The file is preserved byte-for-byte from the archive: 22,847 bytes, SHA-256
`12010eac9b233a3572db477e7a5498c38d6bdebaec4307ee7cc6cbf7832b014e`.
It contains literal `\n` sequences rather than paragraph line breaks. The prompt
generator reads the file as UTF-8 text without interpreting those escapes. Do not
unescape, reformat, or strip the surrounding quotes when reproducing this input;
doing so changes tokenization.
