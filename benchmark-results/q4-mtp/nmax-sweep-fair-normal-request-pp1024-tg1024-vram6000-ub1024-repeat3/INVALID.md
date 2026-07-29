# INVALID FOR PERFORMANCE COMPARISON

Do not use this sweep for performance or acceptance-rate conclusions.

1. The old prompt tokenizer allocated only `pp + 128` token slots. The source
   prompt is 10200 tokens, so tokenization returned a buffer-size error and the
   benchmark silently substituted synthetic sequential token IDs.
2. External GPU jobs entered physical GPU 0 between cases. The starting
   external allocations ranged from 3 MiB to about 18.9 GiB.

The files are retained only as debugging artifacts.
