# INVALID REVERSE PASS

Do not merge this pass into pass-A statistics.

After the reverse pass started, unrelated users launched many concurrent
`cicc`/`ptxas` compilation processes under
`/tmp/llama-dram-cache-baseline-*`, plus download/evaluation work. Physical
GPU 1 remained exclusive, but shared CPU, memory, and transfer resources were
not stable. The completed `n_max=2` TPOT drifted from 53.89 ms in pass-A to
64.84 ms here with identical token output, acceptance, hit rate, and byte
counts. The running `n_max=3` case was terminated by us, and later cases were
not started.

The partial files are retained as an environment-contamination audit.
