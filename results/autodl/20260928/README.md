# AutoDL 实验归档：2026-09-28

本目录保存 `/root/autodl-tmp/cs336-results` 的原始实验结果与配套脚本；采集所用代码提交记录在 `manifest.json` 中。所有原始文件逐字节复制，并通过 SHA-256 校验。运行环境为两张 RTX 6000D，PyTorch 2.11.0+cu130；GPU 互联拓扑见 `topology.txt`，硬件信息见 `gpus.csv`。

## 文件用途

- `allreduce_2gpu_*.json`：两卡 NCCL all-reduce，覆盖 1/10/100/1000 MB，MB 按十进制字节计。
- `ddp_naive_ddp_xl_*.json`、`ddp_flat_gradient_ddp_xl_*.json` 和 `optimizer_replicated_xl_bf16_512_formal.json`：三种 DDP 的完整训练步比较；最后一项使用 overlapping DDP。
- `ddp_sync_*.json`：反向传播之后的同步区间；计入打包、通信及写回，不能直接当作纯 NCCL 时间。
- `optimizer_*_formal.json`：完整 AdamW 与优化器状态分片的正式比较。
- `single_gpu_*.json`：不同模型规模、运行模式及精度的单卡计时。
- `xl_memory_*.json`、`xl2048_checkpoint_*.json`：XL 显存测量与 checkpointing；OOM 也是实验结果。
- `*.nsys-rep`、`fsdp_xl_nsys.sqlite`：Nsight 原始采集报告与导出数据。
- `*.pickle`：PyTorch 显存快照，可在 https://pytorch.org/memory_viz 查看。
- `*.py`：采集脚本；FSDP profiler 使用的损失与优化器基准不同，显存数据不能作为受控性能比较直接相减。

## 比较口径

以每个 JSON 内的模型、batch、context、precision、warmup 和 measurement 设置为准。正式 DDP/优化器比较使用 `formal` 或对应 5 步预热、20 步测量的文件；`probe`、`smoke`、Nsight 单步计时用于验证和定位。BF16 autocast 不意味着 FP32 主参数、梯度和优化器状态全部转为 BF16。

本次 AutoDL 只测量了两卡配置；没有 4/6 卡结果。RTX 5060 的 FlashAttention 完整扫描位于仓库原有 `results/flash_attention/`；本目录仅含 RTX 6000D 的 smoke 检查。

虚拟环境、依赖缓存与 Jupyter 自动备份不包含在此归档中。原始 AutoDL 文件仍保留在来源目录。
