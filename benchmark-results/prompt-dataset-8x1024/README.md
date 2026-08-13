# Natural Long-Context Prompt Dataset

This dataset contains eight independent, naturally written technical
contexts for MoE cache experiments. The topics, vocabulary, document
structure, and evidence patterns are intentionally different; no prompt is
made by repeating a short paragraph template.

Each file is deliberately longer than 1024 model tokens. The benchmark is
run with `--pp 1024`, so the measured prefill input is the first 1024 tokens
after the model tokenizer processes that file. The exact token count must be
recorded by the benchmark summary for every run; byte count alone is not a
token count.

The complete paired matrix runs all eight prompt ids under every mode and
hardware cell. If only three observations per cell are affordable, use three
different prompt ids but label them as prompt samples rather than process
repeats. The full design, balanced order, and prompt-level statistics are in
`TEST-PROTOCOL.md`.

The `manifest.tsv` file records the intended topic and SHA256. A run must
copy the prompt path, SHA256, byte size, and measured `n_prompt` into its
metadata. Results from the old repeated-template prompt must not be pooled
with this dataset.
