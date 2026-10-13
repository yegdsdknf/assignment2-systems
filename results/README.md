# 实验结果索引

- [AutoDL RTX 6000D 原始实验归档](autodl/20260928/README.md)：2026-09-28 的两卡通信、DDP、优化器分片、FSDP profiler、单卡计时、显存及 checkpointing。文件校验与采集提交见同目录 `manifest.json`。
- `ddp/`：历史 CPU/Gloo 正确性与重复基准比较；各实现使用不同文件名，部分运行保留代码快照。
- `flash_attention/`：RTX 5060 完整扫描与 smoke 记录。
- `attention/`、`torch_compile/`：PyTorch attention 与 torch.compile 比较。
- `batch/`、`mode/`、`seq/`、`mixed_precision/`：历史单卡规模、运行模式和精度实验。
- `checkpointing/`、`memory_profiling/`：历史显存分析与 checkpointing 实验。
- `nsight/`：历史单卡 Nsight 采集记录。

比较前先检查文件内的设备、实现、模型、batch、context、精度及预热/测量次数。正式基准、probe 和 smoke 不能混为同一组样本。

原 AutoDL `cs336-results/` 的 GitHub 文件路径现迁移为 `results/autodl/20260928/`；原始文件内容保持不变。
