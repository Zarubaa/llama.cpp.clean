# Q8 MTP MoE-offload fair A/B

Model: `/workspace/models/Qwen3.6-35BA3B-MTP.moe.gguf`

Common params: pp=1024, tg=1024, repeat=3, ubatch=1024, moe_cache_vram_mb=6000, predictor=lru, CUDA_VISIBLE_DEVICES=0.

Main result uses the explicit prompt saved as `prompt.txt` and truncated by `llama-moe-bench` to 1024 prompt tokens. The default-prompt result is included only as a secondary check.

## normal-request

| run | prefill_target ms | mtp_process ms | prefill_total ms | decode_total ms | TPOT ms | total ms | decode hit | SSD decode GB | VRAM peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-MTP | 5592.9 | 0.0 | 5596.5 | 77380.6 | 75.57 | 82977.1 | 84.0% | 488.55 | 10.08 |
| MTP n_max=1 | 5715.6 | 221.1 | 5936.7 | 10692.0 | 10.44 | 16628.7 | 93.5% | 8.38 | 11.62 |

decode speedup=7.24x; TPOT speedup=7.24x; end-to-end speedup=4.99x; MTP prefill_total/no-MTP prefill_total=1.061x.

MTP acceptance: draft_tokens=1652, draft_accepted=1420, accept_rate=86.0%. Trace check draft_tokens=1652, draft_accepted=1420.

Token trace emitted check: no-MTP {0: 1024, 1: 1024, 2: 1024}; MTP {0: 1024, 1: 1024, 2: 1024}.

MTP decode split rows: target=480, draft=1652, process=1652. Requests: target=1652, draft=1652, process=1652.

## default-prompt

| run | prefill_target ms | mtp_process ms | prefill_total ms | decode_total ms | TPOT ms | total ms | decode hit | SSD decode GB | VRAM peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-MTP | 6258.6 | 0.0 | 6260.8 | 82330.3 | 80.40 | 88591.1 | 84.0% | 488.55 | 9.62 |
| MTP n_max=1 | 7636.7 | 460.9 | 8097.6 | 11274.3 | 11.01 | 19371.9 | 93.7% | 9.12 | 11.63 |

decode speedup=7.30x; TPOT speedup=7.30x; end-to-end speedup=4.57x; MTP prefill_total/no-MTP prefill_total=1.293x.

MTP acceptance: draft_tokens=1767, draft_accepted=1303, accept_rate=73.7%. Trace check draft_tokens=1767, draft_accepted=1303.

Token trace emitted check: no-MTP {0: 1024, 1: 1024, 2: 1024}; MTP {0: 1024, 1: 1024, 2: 1024}.

MTP decode split rows: target=560, draft=1767, process=1769. Requests: target=1769, draft=1767, process=1769.

