# 4 GiB Host DRAM Cache Comparison

本目录比较三种专家权重传输路径：

- `control`：冻结的 `9f0199dbb` baseline，Linux page cache 热，应用 Host expert cache 关闭。
- `host4gb-cold`：4 GiB pinned Host expert cache 初始为空，运行中按 aged-LFU 准入和淘汰。
- `host4gb-hot`：4 GiB pinned Host expert cache 预加载由同一 prompt calibration 得到、且与初始 GPU per-layer hot experts 互补的 2240 个专家。

固定条件：NVIDIA L20 GPU1，正常中英混合 prompt，`pp=1024`、`tg=1024`、greedy、LRU predictor、`VRAM={6000,8000} MiB`、`ubatch={512,1024}`。每个 cell 使用 3 个全新进程，报告算术平均、样本标准差、最小值和最大值。

每次运行前执行 `--page-cache-policy hot` 并要求模型文件驻留率至少 99%。因此报告中的 `Source bytes`/`SSD reads` 是应用层逻辑源读取，不能解释成物理 SSD I/O；进程 `/proc/self/io read_bytes` 和块设备扇区增量作为物理读取辅助证据。

## 目录

```text
calibration/                 observed-routing 热度校准
hotsets/                     每个 VRAM/ubatch cell 的 2240 专家 Host hotset
control/                     baseline 运行
host4gb-cold/                4 GiB Host cache 空缓存运行
host4gb-hot/                 4 GiB Host cache 预加载运行
scripts/run-experiments.sh   可恢复实验 runner
scripts/validate-run.py      单次运行验收
scripts/summarize.py         逐次、聚合和报告生成
scripts/compare-logits.py    control 与 4 GiB hot logits 数值对比
summary-table.tsv            36 次逐次结果
aggregate-table.tsv          12 个 cell 的三次聚合结果
final-report.md              同时展示延迟和速度的结论
```

每个正式运行保存 `summary.txt`、`profile.csv`、`tokens.csv`、stdout/stderr、metadata、1 秒 Host memory/NUMA 采样、page-cache/process-I/O 摘要、块设备计数和 validation 结果。runner 只会跳过已完整通过验证的 canonical run；发现残缺输出会停止，防止把部分运行混入统计。

## 执行

```bash
GPU_INDEX=1 PREFLIGHT_ONLY=1 ./benchmark-results/0805-4GB-DRAM-cache-comparison/scripts/run-experiments.sh
GPU_INDEX=1 ./benchmark-results/0805-4GB-DRAM-cache-comparison/scripts/run-experiments.sh
```

正式顺序按三轮 Latin-square 交替，减小温度和时钟漂移造成的模式偏差。control 使用冻结 baseline binary；两个 4 GiB 模式使用本分支 experiment binary。
