# llama.cpp MoE offload experiment review

- Deck: `llama_cpp_moe_offload_experiment_review_zh.pptx`
- Generator: `build_offload_review.py`
- Scope: branches `0714_ubatch_new`, `0716_eamc`, `0719_speculative_decoding`, `0720_mtp`, and `0724_transfer`
- Evidence date: 2026-07-27

Regenerate with:

```bash
python3 -m venv /tmp/llama-offload-ppt-venv
/tmp/llama-offload-ppt-venv/bin/pip install python-pptx
/tmp/llama-offload-ppt-venv/bin/python ppt/build_offload_review.py
```

The deck uses editable PowerPoint shapes for diagrams and charts. Every slide
has a footer pointing to the relevant branch, commit, or benchmark artifact.
