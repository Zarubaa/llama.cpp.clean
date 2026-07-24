ggml_cuda_init: found 1 CUDA devices (Total VRAM: 45457 MiB):
  Device 0: NVIDIA L20, compute capability 8.9, VMM: yes, VRAM: 45457 MiB
| model                          |       size |     params | backend    | ngl |     sm | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | ---: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | CUDA       |  99 |   none |    0 |          pp1024 |      5102.37 ± 54.58 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | CUDA       |  99 |   none |    0 |          tg1024 |        158.95 ± 0.67 |

build: 3821d9835 (10001)
