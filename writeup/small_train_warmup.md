# Small 模型基准测试报告

## 一、实验目标

验证完整训练步骤的基准测试在不同 warmup 次数下何时进入稳定状态，并据此确定后续性能实验的默认 warmup 配置；随后比较 forward、forward + backward 与完整训练步骤，并考察 batch size 和序列长度对性能与显存的影响。

## 二、实验配置

| 项目 | 配置 |
| --- | --- |
| 模型 | Small |
| 测量对象 | 完整训练步骤（forward + backward + optimizer step） |
| 设备 | CUDA |
| batch size | 4 |
| context length | 512 |
| vocab size | 10,000 |
| 参数量 | 128,625,408 |
| 每组正式测量次数 | 30 |
| warmup 配置 | 0、1、2、5、10、20 |

每组实验均在 warmup 完成后开始正式计时和峰值显存统计。GPU 计时边界前后均同步，避免 CUDA 异步执行导致 CPU 侧计时偏小。

原始数据：[`results/small_train_warmup.csv`](../results/small_train_warmup.csv)。

## 三、实验结果

| Warmup 次数 | 平均耗时 (ms) | 标准差 (ms) | 峰值已分配显存 (MiB) | 峰值缓存显存 (MiB) | 状态 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 323.054 | 88.619 | 5159.751 | 5528.000 | success |
| 1 | 295.786 | 1.572 | 5157.548 | 5508.000 | success |
| 2 | 295.724 | 0.451 | 5157.548 | 5508.000 | success |
| 5 | 296.257 | 0.329 | 5157.548 | 5508.000 | success |
| 10 | 296.438 | 0.306 | 5157.548 | 5508.000 | success |
| 20 | 296.636 | 0.343 | 5157.548 | 5508.000 | success |

## 四、分析与结论

未执行 warmup 时，平均耗时为 323.054 ms，标准差为 88.619 ms；相较 warmup=2 的稳定耗时，高约 9.2%，且方差明显更大。这表明首次或早期执行仍包含 CUDA 运行时初始化、内核选择、内存缓存建立等一次性成本，不应纳入正式性能统计。

从 warmup=2 开始，平均耗时处于 295.724–296.636 ms 区间，最大差异为 0.912 ms（约 0.31%）；标准差也降至 0.306–0.451 ms。因此，对当前 Small 模型训练步骤而言，**2 次 warmup 已足以达到稳定状态**。

各稳定配置下的峰值已分配显存均为 5157.548 MiB、峰值缓存显存均为 5508 MiB，未出现随 warmup 次数增长的显存上升趋势，说明本轮实验没有观察到明显的显存泄漏。

> 后续常规 benchmark 采用 **5 次 warmup**。虽然 2 次已是当前配置的最小稳定值，但 5 次的额外开销很小，并可为后续引入 `torch.compile`、Triton kernel 或不同模型规模时可能出现的一次性开销保留余量。

## 五、后续实验约定

后续性能实验固定使用 `warmup_steps=5`，并保持“warmup 与正式测量分离、GPU 同步、正式测量次数固定”的方法。若变更模型规模、精度、`torch.compile` 状态或核心 kernel，则应重新运行一次 warmup sweep，确认该配置的稳定区间。


## 六、实验环境与统计口径

| 项目 | 配置 |
| --- | --- |
| 运行环境 | WSL Ubuntu |
| GPU | NVIDIA GeForce RTX 5060（8151 MiB） |
| GPU 驱动 | 596.21 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| 默认模型 | Small（128,625,408 参数） |
| 常规测量 | 5 次 warmup，30 次正式测量 |

除 warmup sweep 外，模式对比均采用 batch size=4、context length=512。每个模式独立重复两次，主表时间为两次运行均值；显存峰值在重复中一致。

## 七、训练步骤分解

| 模式 | 两次运行均值 (ms) | 两次运行间标准差 (ms) | 峰值已分配显存 (MiB) | 峰值缓存显存 (MiB) |
| --- | ---: | ---: | ---: | ---: |
| forward | 90.046 | 0.062 | 728.702 | 894.000 |
| forward + backward | 266.571 | 0.045 | 4157.589 | 4336.000 |
| forward + backward + optimizer step | 303.938 | 0.697 | 5159.751 | 5528.000 |

相较仅前向传播，反向传播增加 **176.526 ms（196.0%）** 和 **3428.887 MiB** 峰值已分配显存。加入 optimizer step 后，再增加 **37.367 ms（14.0%）** 和 **1002.162 MiB**。完整训练步骤约为仅前向传播时间的 **3.38 倍**、峰值已分配显存的 **7.08 倍**；额外显存与 AdamW 在 warmup 后保留的一阶、二阶优化器状态一致。

主统计只使用 results/mode/ 中带 1、2 后缀的两次独立重复；未编号 JSON 保留为单次历史运行，不混入重复统计。

## 八、Batch size 扫描

固定 Small 模型和 context length=512，吞吐量按 batch size × context length / mean_ms 计算。

| Batch size | 时间 (ms) | 峰值已分配显存 (MiB) | 吞吐量 (tokens/s) | 解释 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 103.119 | 2488.686 | 4965 | GPU 利用率仍偏低 |
| 2 | 168.373 | 3383.665 | 6082 | 吞吐量提升 |
| 4 | 297.429 | 5159.751 | 6886 | 本机的最大有效配置 |
| 8 | 10621.625 | 8835.829 | 386 | 超出显存容量后的性能断崖 |

从 batch size 1 提升到 4，吞吐量提高约 **38.7%**。batch size 8 虽未抛出 PyTorch OOM，但峰值已分配显存超过设备的 8151 MiB 物理显存，吞吐量相较 batch size 4 下降约 **94.4%**，因此不应作为正常吞吐量趋势的一部分。

## 九、序列长度扫描与容量边界

固定 Small 模型和 batch size=4：

| 序列长度 | 时间 (ms) | 峰值已分配显存 (MiB) | 吞吐量 (tokens/s) | 解释 |
| ---: | ---: | ---: | ---: | --- |
| 128 | 93.686 | 2247.840 | 5465 | 正常运行 |
| 256 | 144.313 | 3104.103 | 7096 | 正常运行 |
| 512 | 296.221 | 5159.751 | 6914 | 正常运行 |
| 768 | 5625.513 | 7836.774 | 546 | 接近显存边界，发生严重退化 |
| 1024 | 22143.063 | 11167.570 | 185 | 超出设备容量，不可作常规性能比较 |

从 sequence length 512 增至 768 时，长度仅增加 1.5 倍，单步时间却增加约 **19 倍**；1024 时单步已超过 **22 秒**。768 的峰值缓存显存为 8248 MiB，1024 达到 11968 MiB；结合突增的时间，这高度表明运行已进入显存超售或 WSL/CUDA 回退路径，而非正常计算量增长区间。

因此，768 和 1024 应作为本机容量边界的证据保留，但不能与 128–512 的正常点共同拟合吞吐量趋势。PyTorch 显存统计不包含全部 CUDA 上下文、驱动与系统开销，因此即使已分配显存略低于 8151 MiB，也可能已接近实际可用上限。

## 十、当前结论与后续工作

本机上，Small 模型完整训练步骤的推荐基线为 batch size=4、context length=512、5 次 warmup 和 30 次正式测量。当前报告已覆盖 warmup、三种执行模式、batch size、序列长度、mixed precision、memory profiling 与 Nsight Systems 单步性能分析；后续需补充不同模型规格和上下文长度的 Nsight 对照，以及 activation checkpointing、torch.compile 与分布式训练实验。


## 十一、Mixed Precision 数值正确性

### 11.1 实验方法

数值验证固定随机种子为 0，在 RTX 5060 上完成。首先对 262,144 个位于 $[10^{-4}, 10^{-2}]$ 的正数求和，分别比较 FP32 输入/FP32 累积、FP16 输入/FP16 累积与 FP16 输入/FP32 累积。正数输入避免参考和接近零时相对误差失去解释力。

随后以相同的 Small 模型、同一组 FP32 参数和固定输入比较 FP32 与 BF16 autocast 的模型输出。BF16 实验只在 autocast 上下文内执行模型算子；模型主参数仍是 FP32，且自定义 cross-entropy 的 logits 在 softmax/reduction 前显式转换回 FP32。

原始数据：[`results/mixed_precision/numerical_validation.json`](../results/mixed_precision/numerical_validation.json)。

### 11.2 累积精度

| 输入精度 | 累积精度 | 求和值 | 绝对误差 | 相对误差 |
| --- | --- | ---: | ---: | ---: |
| FP32 | FP32 | 1323.827148 | 0 | 0 |
| FP16 | FP16 | 1324.000000 | 0.172852 | $1.306 \times 10^{-4}$ |
| FP16 | FP32 | 1323.827148 | 0 | 0 |

FP16 累积将求和值量化到更粗的间隔，产生了可观测误差。将 FP16 输入提升到 FP32 再累积后，本次确定性输入上的结果与参考值在记录精度内相同，表明主要误差来自低精度 reduction。该结果不意味着 FP16 输入量化永远没有误差；它只说明在当前输入分布和长度下，输入量化误差相互抵消或低于所记录的 FP32 精度。

### 11.3 模型输出正确性

| 指标 | FP32 | BF16 autocast | 差异/结论 |
| --- | ---: | ---: | --- |
| logits 是否有限 | True | True | 未出现 NaN 或 Inf |
| loss 是否有限 | True | True | 未出现 NaN 或 Inf |
| loss | 9.280806 | 9.280767 | 绝对误差 $3.815 \times 10^{-5}$ |
| logits 最大绝对误差 | — | — | 0.015712 |
| logits 平均绝对误差 | — | — | 0.002106 |
| logits 相对 L2 误差 | — | — | 0.007100 |

BF16 输出与 FP32 存在预期中的舍入差异，但 loss 差异很小，且全部 logits 与 loss 均为有限值。因此，该 mixed-precision 路径在当前配置下具有可接受的数值稳定性。

## 十二、Mixed Precision 性能与显存

固定 Small 模型、batch size=4、context length=512、5 次 warmup 和 30 次正式测量，比较完整训练步骤。原始数据：[`small_train_fp32.json`](../results/mixed_precision/small_train_fp32.json) 与 [`small_train_bf16.json`](../results/mixed_precision/small_train_bf16.json)。

| 精度 | 平均时间 (ms) | 标准差 (ms) | 峰值已分配显存 (MiB) | 峰值缓存显存 (MiB) | 吞吐量 (tokens/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 304.203 | 0.674 | 5159.751 | 5528.000 | 6732 |
| BF16 autocast | 189.108 | 1.995 | 4360.321 | 4580.000 | 10829 |

相较 FP32，BF16 autocast 将训练步骤时间降低 **37.8%**，获得 **1.61 倍**加速，并将吞吐量提高约 **60.9%**。峰值已分配显存减少 **799.430 MiB（15.5%）**，峰值缓存显存减少 **948 MiB（17.2%）**。

显存降幅明显小于 50%，因为参数、梯度以及 AdamW 的一阶和二阶状态仍长期保留为 FP32；BF16 主要减少了矩阵乘和部分中间激活的存储/计算开销。这种设计在保持优化器稳定性的同时，利用 RTX 5060 上 BF16 矩阵计算的更高吞吐量，是本实验中速度与数值稳定性的合理折中。

## 十三、更新后的后续工作

已完成 memory profiling，确认训练峰值的主要单步来源是 backward 保存的激活，而 AdamW 状态主要构成稳态常驻显存。下一阶段进入 activation checkpointing，并继续使用本报告已确定的默认测量口径，比较不同 checkpoint 粒度对峰值显存、训练时间和可运行上下文长度的影响。

## 十四、Memory Profiling 与 Autograd Saved Tensors

### 14.1 方法与测量口径

本节固定 Small 模型、batch size=4、context length=512 与 vocab size=10000。在每种模式正式采样前先执行 5 个 warmup step；训练模式的 warmup 会先物化 AdamW 的一阶、二阶状态，因此随后记录的 baseline 表示稳态训练，而非首次更新的偶然低值。

脚本通过 torch.autograd.graph.saved_tensors_hooks 统计 autograd 为 backward 保存的 tensor 记录数、记录字节数及按底层 storage 去重后的字节数；同时使用 CUDA allocator 的 memory_allocated、max_memory_allocated 和 max_memory_reserved 记录显存。原始结果位于 [results/memory_profiling](../results/memory_profiling/)。

### 14.2 稳态显存分解

| 模式/精度 | 参数 (MiB) | AdamW 状态 (MiB) | 稳态基线 (MiB) | 峰值已分配 (MiB) | 峰值缓存 (MiB) | 单步峰值增量 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forward / FP32 | 490.667 | 0.000 | 500.354 | 728.702 | 894.000 | 228.348 |
| forward + backward / FP32 | 490.667 | 0.000 | 508.479 | 4157.589 | 4336.000 | 3649.109 |
| train / FP32 | 490.667 | 981.334 | 1498.829 | 5149.142 | 5480.000 | 3650.312 |
| train / BF16 autocast | 490.667 | 981.334 | 1504.376 | 4342.946 | 4580.000 | 2838.570 |

相较 forward+backward，完整训练的稳态基线增加 990.350 MiB，接近测得的 AdamW 状态 981.334 MiB；但两者的单步峰值增量只相差 1.203 MiB。因此，优化器状态主要是常驻显存，而反向传播的激活及其梯度才是单步峰值的主要来源。

### 14.3 Saved tensors 与 BF16 影响

| 模式/精度 | saved tensor 记录数 | 记录字节数 (MiB) | 去重 storage 字节数 (MiB) |
| --- | ---: | ---: | ---: |
| forward / FP32 | 0 | 0.000 | 0.000 |
| forward + backward / FP32 | 590 | 5190.440 | 3954.370 |
| train / FP32 | 590 | 5190.440 | 3954.370 |
| train / BF16 autocast | 590 | 3480.792 | 2604.722 |

记录字节数大于去重后的 storage 字节数，是因为 autograd 会保存视图或共享底层 storage 的 tensor；两者都不能与 CUDA allocator 峰值简单一一对应。FP32 的完整训练与单纯 forward+backward 的 saved-tensor 统计完全相同，说明 optimizer step 不会额外保存反向中间值。

BF16 autocast 没有改变计算图的保存记录数（仍为 590），但将去重后的 saved storage 从 3954.370 MiB 降至 2604.722 MiB，减少 1349.648 MiB（34.1%）。完整训练峰值相应降低 806.195 MiB（15.7%），单步峰值增量降低 811.742 MiB（22.2%）。峰值降幅低于 saved-storage 降幅，是因为参数、梯度与 AdamW 状态仍以 FP32 常驻，且 CUDA allocator 还会保留缓存。

四次运行各自随机初始化模型和输入，因此表中的 loss 仅用于确认其为有限值，不用于比较不同模式或精度的数值差异。

### 14.4 结论与下一步

> 当前 Small 配置中，反向传播的激活保存是单步峰值的主要可优化部分；AdamW 状态则决定了训练的常驻显存下限。

下一阶段将实现 activation checkpointing：按若干 Transformer block 为一段重算前向激活，比较不同分段粒度下的峰值显存、训练时间以及可运行的最大 context length，并首先验证 checkpoint 前后的 logits、loss 和参数梯度一致性。


## 十五、Nsight Systems Profiling

### 15.1 实验方法与采集边界

本节首先使用 Nsight Systems 2026.4.1 分析 Small 模型在 FP32 下的 inference forward、forward + backward 与完整训练步骤。三种基线模式均采用 batch size=4、context length=512、vocab size=10000 和 5 次 warmup。随后扩展到 Small 模型的 context length 256、512、1024，以及 Medium/256；扩展实验固定 batch size=1。每份 Nsight trace 只采集 warmup 后的一个 measurement step，另用 10 个未插桩 measurement steps 统计稳定端到端时间。

| 项目 | 配置 |
| --- | --- |
| Trace API | CUDA、cuBLAS、OS runtime、NVTX |
| PyTorch 标注 | `functions-trace-shapes`、`autograd-nvtx` |
| Capture range | `profiled_measurement` |
| 阶段标注 | forward、loss、backward、optimizer step |
| 注意力标注 | attention scores、mask、softmax、final value matmul |
| CPU sampling / context switch | 关闭，降低 profiler 额外开销 |

`torch.cuda.nvtx.range_push()` 发出的名称属于动态字符串，而当前 Nsight 默认只使用 registered string 触发 NVTX capture。因此采集命令设置 `NSYS_NVTX_PROFILER_REGISTER_ONLY=0`，启用完整字符串匹配。计时起止点位于外层 NVTX push/pop 内部：进入 capture range 后才开始计时，在 `torch.cuda.synchronize()` 完成后立即停止计时，从而排除 profiler 启动、停止、flush 和报告生成开销。

保留的原始 trace：

- [`small_forward_fp32_nvtx.nsys-rep`](../results/nsight/small_forward_fp32_nvtx.nsys-rep)
- [`small_forward_backward_fp32_nvtx.nsys-rep`](../results/nsight/small_forward_backward_fp32_nvtx.nsys-rep)
- [`small_train_fp32_nvtx.nsys-rep`](../results/nsight/small_train_fp32_nvtx.nsys-rep)

SQLite 是 `nsys stats` 从 `.nsys-rep` 导出的可再生产物，因此未作为最终结果保留；需要重新分析时使用 `--force-export=true` 生成即可。

### 15.2 端到端时间与训练阶段分解

| 模式 | 常规 benchmark 参考值 (ms) | Nsight 运行中的 Python 计时 (ms) | Nsight GPU/同步时间 (ms) |
| --- | ---: | ---: | ---: |
| inference forward | 91.064 | 94.639 | 94.141 |
| forward + backward | 267.479 | 270.274 | 270.274 |
| forward + backward + optimizer | 297.429 | 294.548 | 294.698 |

Inference 中 Python 计时与 GPU projection 只相差 0.498 ms；完整训练中二者相差 0.150 ms。Nsight 计时与常规 benchmark 的差异处于约 0.1%–3.9% 范围，说明 capture 边界能够代表单步端到端时间。

NVTX push/pop range 是线程局部的。PyTorch autograd engine 会在工作线程中发起大部分 backward CUDA kernel，因此主线程 `backward` range 的 GPU projection 只关联到一个约 608 ns 的小操作，不能视为反向传播时间。完整训练通过 loss 结束与 optimizer 首个 GPU 操作开始的时间线边界推导 backward；forward + backward 模式则用同步后的完整 Python 时间减去 forward 与 loss 得到 backward interval。

完整训练步骤的阶段分解：

| 阶段 | 时间 (ms) | 占完整训练步骤 |
| --- | ---: | ---: |
| forward | 88.796 | 30.13% |
| loss | 2.036 | 0.69% |
| backward（时间线边界推导） | 169.851 | 57.64% |
| optimizer step | 34.013 | 11.54% |
| 合计 | 294.698 | 100% |

Forward + backward 模式的阶段分解：

| 阶段 | 时间 (ms) | 占 forward + backward |
| --- | ---: | ---: |
| forward | 92.444 | 34.20% |
| loss | 2.103 | 0.78% |
| backward（同步总时间余量） | 175.727 | 65.02% |
| 合计 | 270.274 | 100% |

两种模式下 backward 均约为 forward 的 1.9 倍。完整训练还需约 34.0 ms 执行 AdamW 更新。

### 15.3 CUDA Kernel 构成

| 模式 | Kernel 总时间 (ms) | GEMM 时间 (ms) | GEMM 占比 | 最耗时的具体 kernel |
| --- | ---: | ---: | ---: | --- |
| inference forward | 91.178 | 61.162 | 67.1% | `magma_sgemmEx_kernel`：58.771 ms，97 次 |
| forward + backward | 263.038 | 155.285 | 59.0% | `magma_sgemmEx_kernel`：61.245 ms，109 次 |
| 完整训练 | 282.636 | 149.834 | 53.0% | `magma_sgemmEx_kernel`：58.855 ms，109 次 |

按完整 demangled 名称统计，三种模式累计时间最多的具体 kernel 都是 `magma_sgemmEx_kernel`。`cuda_gpu_kern_sum:base` 会把多个不同形状和转置方向的 CUTLASS SGEMM 变体合并为 `Kernel2`；这个合并类别适合计算全部矩阵乘法占比，但不能当作一个具体 kernel。Forward 中 MAGMA GEMM 调用 97 次，加入 backward 后增至 109 次。

矩阵乘法的绝对时间从 inference 的 61.162 ms 增至 forward + backward 的 155.285 ms，但其占全部 kernel 时间的比例从 67.1% 降至 59.0%；加入 AdamW 后进一步降至 53.0%。这表明 backward 和 optimizer 增加了大量非矩阵 kernel。完整训练中主要的非矩阵操作如下：

| Kernel 类别 | 累计时间 (ms) | 实例数 | 占完整训练 kernel 时间 |
| --- | ---: | ---: | ---: |
| vectorized elementwise | 87.318 | 2487 | 30.9% |
| elementwise | 36.781 | 634 | 13.0% |
| reduction | 7.345 | 128 | 2.6% |

其中 optimizer step 单独包含 1776 个 vectorized elementwise kernel，累计约 28.256 ms。当前 AdamW 按参数执行许多细粒度逐元素更新，因而 kernel launch 数量很高；optimizer 没有增加 GEMM 调用。

### 15.4 Self-Attention 内部分析

Inference forward 的 12 层 self-attention GPU projection 如下：

| 注意力阶段 | 12 层合计 (ms) | 每层平均 (ms) | 占完整 forward |
| --- | ---: | ---: | ---: |
| attention scores | 6.345 | 0.529 | 6.74% |
| causal mask | 3.442 | 0.287 | 3.66% |
| softmax | 12.460 | 1.038 | 13.23% |
| final value matmul | 2.795 | 0.233 | 2.97% |
| 整体 scaled dot-product attention | 25.344 | 2.112 | 26.92% |

Softmax 是注意力内部最耗时的子阶段。按 NVTX 操作范围计算，softmax 的 12.460 ms 是 attention scores 与 final value matmul 合计 9.140 ms 的约 1.36 倍。进一步按范围内的具体 kernel 分类，softmax 相关 elementwise/reduction kernel 合计约 12.258 ms，而两个纯 GEMM kernel 合计约 5.178 ms，前者约为后者的 2.37 倍。

这一结果与 FLOPs 数量不成比例：矩阵乘法 FLOPs 更多，但 GPU GEMM kernel 能充分利用计算单元；自定义 softmax 由多次 elementwise 与 reduction kernel 组成，还需要反复读写中间张量，因此更容易受显存带宽和 kernel launch 开销限制。除矩阵乘法外，forward 中累计时间较明显的操作还包括 vectorized elementwise、普通 elementwise、reduction 与张量拼接复制。

### 15.5 作业问题对应结论

1. Small/512 inference forward 的 Nsight GPU 时间为 94.141 ms，与 Python profiling 计时 94.639 ms、常规 benchmark 91.064 ms 接近。
2. Forward 中累计时间最多的具体 kernel 是 `magma_sgemmEx_kernel`，调用 97 次；forward + backward 中仍是同一具体 kernel，共调用 109 次。
3. Forward 中除矩阵乘法外，vectorized elementwise、elementwise、reduction 和 batched copy 也占据非平凡的 GPU 时间。
4. GEMM 占 kernel 时间的比例从 inference 的 67.1% 降至 forward + backward 的 59.0%，加入 AdamW 后进一步降至 53.0%；optimizer 主要增加大量逐元素更新 kernel。
5. Softmax 虽然 FLOPs 远少于两个 attention GEMM，但其 NVTX 范围时间约为二者合计的 1.36 倍，纯 kernel 时间约为二者合计的 2.37 倍，说明当前实现主要受多 kernel 启动、中间张量访问和 reduction 效率限制。


### 15.6 模型规模与 Context Length 扩展实验

#### 15.6.1 实验矩阵与稳定时间

在 Small/512、batch size=4 的详细基线之外，扩展实验固定 FP32、batch size=1、vocab size=10000。Small 模型覆盖 256、512、1024 三个 context length；Medium 模型在显存允许的 context length=256 下分别采集 forward、forward + backward 和完整训练 trace。

| 模型 | Context length | 10 步 mean (ms) | Std (ms) | 变异系数 | Peak allocated (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small | 256 | 74.497 | 9.482 | 12.73% | 2162.27 |
| Small | 512 | 102.241 | 1.086 | 1.06% | 2488.69 |
| Small | 1024 | 206.517 | 1.049 | 0.51% | 3987.00 |
| Medium | 256 | 228.337 | 3.822 | 1.67% | 6690.85 |

Small 的稳定训练时间随 context length 单调增长：256 到 512 增加约 37.2%，512 到 1024 增加约 102.0%。峰值显存从 2162.27 MiB 增至 3987.00 MiB，体现了注意力中间张量随序列长度快速增长的影响。Medium 参数量为 423.18M，约为 Small 的 3.29 倍；在相同 batch size=1、context length=256 下，训练时间约为 Small 的 3.06 倍，峰值显存约为 3.09 倍。

#### 15.6.2 Nsight 插桩开销

| 模型 / Context | 稳定 benchmark (ms) | Nsight 内 Python 计时 (ms) | 相对增幅 |
| --- | ---: | ---: | ---: |
| Small / 256 | 74.497 | 141.435 | 89.9% |
| Small / 512 | 102.241 | 123.689 | 21.0% |
| Small / 1024 | 206.517 | 216.682 | 4.9% |
| Medium / 256 | 228.337 | 246.079 | 7.8% |

Profiler 的固定开销对短任务影响最大，任务变长后相对增幅明显下降。因此 Nsight 单步 trace 用于分析阶段归属、kernel 构成和调用次数，性能趋势采用多步未插桩 benchmark。Small/256 的单份 trace 是明显慢样本，不能据此得出 context 256 比 512 更慢的结论。

Medium/256 完整训练的 Nsight GPU 时间为 244.972 ms，其中 forward 为 44.341 ms、loss 为 0.142 ms、optimizer step 为 127.232 ms；按总时间余量推算 backward 约为 73.258 ms。Optimizer 占完整步骤约 51.9%，说明在小 batch、短 context 下，较大模型的参数更新成本已经超过 forward 和 backward 中任一阶段。

#### 15.6.3 Context Length 对 Kernel 构成的影响

Small、batch size=1 的完整训练 kernel 构成如下；括号中为占该次 kernel 总时间的比例。

| Context | Kernel 总时间约值 (ms) | Vectorized elementwise | `Kernel2` | `magma_sgemmEx_kernel` | Elementwise | Reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 61.8 | 31.587 (51.1%) | 18.802 (30.4%) | 4.456 (7.2%) | 6.119 (9.9%) | 0.538 (0.9%) |
| 512 | 95.2 | 39.226 (41.2%) | 34.824 (36.6%) | 8.982 (9.4%) | 10.342 (10.9%) | 1.341 (1.4%) |
| 1024 | 203.0 | 71.848 (35.4%) | 60.374 (29.8%) | 28.740 (14.2%) | 33.864 (16.7%) | 7.119 (3.5%) |

Context 从 256 增至 1024 时，vectorized elementwise 的占比从 51.1% 降至 35.4%，而普通 elementwise、reduction 与 MAGMA GEMM 的占比均上升。`Kernel2` 与 `magma_sgemmEx_kernel` 两类矩阵乘法合计时间从 23.258 ms 增至 89.114 ms；reduction 从 0.538 ms 增至 7.119 ms。这说明短序列时参数更新等近似固定成本更突出，长序列时 attention、激活处理和 reduction 逐渐主导增量成本。Context 1024 下 `Kernel2` 实例数由 278 降至 230，而 MAGMA 实例数由 49 增至 97，表明矩阵形状改变后底层库选择了不同的 kernel 组合。

Medium/256 forward-only 中，`Kernel2` 累计 13.192 ms（44.4%，144 次），`magma_sgemmEx_kernel` 累计 13.020 ms（43.8%，73 次），两类 GEMM 合计占 kernel 时间约 88.2%。加入 backward 后，累计时间最多的 base-name 类别仍为 `Kernel2`，达到 53.199 ms（56.5%，554 次）；完整训练加入 AdamW 后，vectorized elementwise 上升至 122.162 ms（58.4%，4899 次），成为最大类别。

#### 15.6.4 扩展实验结论

1. 已完成 Small 与 Medium 两种模型规模，以及 256、512、1024 三个大于 128 的 2 的幂次 context length。
2. Small 的时间和显存随 context length 单调增长；context 1024 的训练时间约为 context 256 的 2.77 倍，峰值显存约为 1.84 倍。
3. Medium/256 forward-only 与 forward + backward 中，`Kernel2` base-name 类别均为累计时间最高者，调用次数由 144 增至 554。
4. 加入 AdamW 后，Medium 的 vectorized elementwise kernel 取代 GEMM 成为完整训练的最大类别，反映出大参数量、小 batch/context 配置中的优化器瓶颈。
5. 单步 Nsight trace 存在固定插桩开销，尤其会扭曲短任务的时间；稳定性能结论必须以多步普通 benchmark 为准。

扩展实验保留的 trace：

- [`small_train_fp32_b1_ctx256_nvtx.nsys-rep`](../results/nsight/small_train_fp32_b1_ctx256_nvtx.nsys-rep)
- [`small_train_fp32_b1_ctx512_nvtx.nsys-rep`](../results/nsight/small_train_fp32_b1_ctx512_nvtx.nsys-rep)
- [`small_train_fp32_b1_ctx1024_nvtx.nsys-rep`](../results/nsight/small_train_fp32_b1_ctx1024_nvtx.nsys-rep)
- [`medium_forward_fp32_b1_ctx256_nvtx.nsys-rep`](../results/nsight/medium_forward_fp32_b1_ctx256_nvtx.nsys-rep)
- [`medium_forward_backward_fp32_b1_ctx256_nvtx.nsys-rep`](../results/nsight/medium_forward_backward_fp32_b1_ctx256_nvtx.nsys-rep)
- [`medium_train_fp32_b1_ctx256_nvtx.nsys-rep`](../results/nsight/medium_train_fp32_b1_ctx256_nvtx.nsys-rep)

### 15.7 局限与下一步

本节已经覆盖 Small 与 Medium 两种模型规模，以及 256、512、1024 三种大于 128 的 2 的幂次 context length。受 8 GiB GPU 容量限制，Medium 只在 batch size=1、context length=256 下完成 forward、forward + backward 和完整训练 trace；完整训练的 peak reserved memory 已达到 7128 MiB，再增大 context 存在显存超售或 OOM 风险。

每份 trace 只采集一个已 warmup 的 measurement step，因此适合分析 kernel 构成和调用次数；端到端趋势以 10 次普通 benchmark 的均值与标准差为准。NVTX push/pop range 具有线程局部语义，autograd 工作线程发起的 backward kernel 无法正确投影到主线程 `backward` range；forward + backward 的完整时间采用同步后的 CPU range/Python 计时，训练阶段的 backward 则通过 forward、loss 和 optimizer 边界之间的余量推导。下一节将进入 activation checkpointing，比较重计算带来的显存节省与运行时间开销。

## 十六、Activation Checkpointing

### 16.1 实现与正确性验证

本节实现 [`CheckpointedTransformerLM`](../cs336_systems/checkpointing.py)，将连续 Transformer block 按 `checkpoint_every` 分段，并使用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` 包裹每个分段。Embedding、final RMSNorm 和 LM head 不进入 checkpoint 区间；在 `torch.inference_mode()` 等禁用梯度的环境中直接执行原始 forward，避免无意义的重计算机制。

使用一个 5 层微型 Transformer 验证 baseline 与每 2 层 checkpoint 的实现。测试比较 logits、cross-entropy loss、全部参数梯度以及 inference 输出，均在 `rtol=1e-5`、`atol=1e-6` 下通过。由此确认 checkpoint 只改变 activation 的保存与重计算策略，不改变模型的数学结果；5 层按 2 层分段也覆盖了最后一个不完整分段的边界情况。

### 16.2 Memory-Optimal Recursive Checkpointing

设模型包含 $N$ 个顺序执行的 Transformer block。若不使用 checkpoint，所有 block 的 residual 会同时保留到 backward，峰值 activation memory 为 $\Theta(N)$。

对于不嵌套的单层分段，设每个 checkpoint 区间包含 $k$ 个 block。Forward 需要长期保存约 $N/k$ 个分段边界；backward 重计算一个分段时需要同时物化约 $k$ 个 block 的 residual，因此忽略常数后的峰值为

$$
M(k)=\Theta\left(\frac{N}{k}+k\right).
$$

当 checkpoint 边界和单个 block residual 的大小处于同一量级时，$k=\Theta(\sqrt{N})$ 给出 $\Theta(\sqrt{N})$ 的单层最优峰值。实际最优点还取决于二者的常数比例，因此必须通过 profiling 确认。

若忽略计算成本，可以继续对重计算区间进行平衡递归 checkpoint。递归深度为 $\Theta(\log N)$，任意时刻只需保留每层递归的边界 activation 以及当前 block 的 residual，因此

$$
M(N)=M(N/2)+\Theta(1)=\Theta(\log N).
$$

每一层递归会对全部 $N$ 个 block 产生一次重计算，共有 $\Theta(\log N)$ 层，所以

$$
T(N)=2T(N/2)+\Theta(N)=\Theta(N\log N).
$$

普通 backward 的 $\Theta(N)$ 是低阶项，不改变最终计算复杂度。

### 16.3 单层分段实验

作业要求在 XL、batch size=4、context length=2048 下验证只有一轮重计算的最佳分段策略。由于本机 RTX 5060 只有 8 GiB 显存，无法容纳官方 XL 配置，因此先使用 Small、batch size=4、context length=512、FP32 作为代理实验，扫描 `checkpoint_every` 为 1、2、3、4、6、12 的配置。所有实验使用 5 次 warmup，并记录完整训练步骤的 CUDA peak allocated memory。

| 每段 block 数 | Peak allocated (MiB) | Step peak delta (MiB) | Peak reserved (MiB) | Saved tensor count | Saved tensor (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 无 checkpoint | 5149.14 | 3650.31 | 5480 | 590 | 5190.44 |
| 1 | 2369.39 | 870.99 | 2880 | 38 | 281.62 |
| 2 | 2616.53 | 1112.82 | 3008 | 26 | 245.62 |
| 3 | 2860.49 | 1360.16 | 3158 | 22 | 233.62 |
| 4 | 3100.66 | 1598.66 | 3440 | 20 | 227.62 |
| 6 | 3579.36 | 2078.49 | 3960 | 18 | 221.62 |
| 12 | 5008.83 | 3509.73 | 5440 | 16 | 215.62 |

每个 block 单独 checkpoint（$k=1$）取得最低峰值：相比无 checkpoint，peak allocated memory 减少 2779.75 MiB，即 53.98%；去除参数与已物化 optimizer state 后的 step peak delta 减少 2779.33 MiB，即 76.14%。从 $k=1$ 增至 $k=2$ 后峰值增加 247.13 MiB，并随分段继续增大而单调上升。$k=12$ 需要在 backward 中重新物化几乎完整的 12 层 residual，峰值 5008.83 MiB 已接近 baseline。最优点位于合法搜索边界，不存在更小的 block size，因此使用相邻的 $k=2$ 验证增大分段会提高峰值。

`saved_tensor_mib` 随 $k$ 增大反而下降，不能用于判断训练峰值：该 hook 统计更接近 checkpoint 边界的保存事件，并不直接表示 backward 重计算期间同时物化的 residual。分段策略的结论以 `peak_allocated_mib` 和 `step_peak_delta_mib` 为准。

实验结果：

- [`small_b4_ctx512_baseline.json`](../results/checkpointing/small_b4_ctx512_baseline.json)
- [`small_b4_ctx512_k1.json`](../results/checkpointing/small_b4_ctx512_k1.json)
- [`small_b4_ctx512_k2.json`](../results/checkpointing/small_b4_ctx512_k2.json)
- [`small_b4_ctx512_k3.json`](../results/checkpointing/small_b4_ctx512_k3.json)
- [`small_b4_ctx512_k4.json`](../results/checkpointing/small_b4_ctx512_k4.json)
- [`small_b4_ctx512_k6.json`](../results/checkpointing/small_b4_ctx512_k6.json)
- [`small_b4_ctx512_k12.json`](../results/checkpointing/small_b4_ctx512_k12.json)

### 16.4 XL 配置的容量限制与理论预测

参考 XL 配置约含 3.41B 参数，仅 FP32 参数就需要约 12.69 GiB，已经超过本机 8 GiB 物理显存；完整训练还需要梯度、AdamW 状态和 activation，因此无法在本机获得作业要求的 XL 实测峰值。上述 Small 实验用于验证实现和趋势，不能替代官方 XL 测量。

根据题目示例，XL 的单个编译后 Transformer block residual 约为 3651.31 MiB。一个 residual-stream checkpoint 边界为

$$
4\times2048\times2560\times4\text{ bytes}=80\text{ MiB}.
$$

对于 $N=32$ 层的单层分段，activation peak 可近似写为

$$
M(k)\approx\frac{32}{k}\times80+k\times3651.31\text{ MiB}.
$$

该估算给出 $M(1)\approx6211.31$ MiB、$M(2)\approx8582.62$ MiB，因此理论上同样预测每个 block 单独 checkpoint 最优。这与 Small 代理实验的单调趋势一致，但最终提交若要求 XL 实测值，仍需在高显存 GPU 上重新运行。

### 16.5 本节结论

1. 平衡递归 checkpoint 可将峰值 activation memory 从 $\Theta(N)$ 降至 $\Theta(\log N)$，代价是 $\Theta(N\log N)$ 计算量。
2. 在只允许一轮重计算的非嵌套策略中，Small 代理实验的最佳分段为每个 block 单独 checkpoint。
3. $k=1$ 将完整训练 peak allocated memory 降低 53.98%，并将 step transient peak 降低 76.14%。
4. 分段过大时，backward 重计算会同时物化更多 residual；$k=12$ 的峰值已接近无 checkpoint baseline。
5. 本机无法运行官方 XL 配置，报告明确区分了代理实验、理论预测与尚未取得的官方实测值。


## 十七、PyTorch Attention Benchmarking

### 17.1 实验设置

按照作业要求，固定 batch size 为 8，不使用 multi-head 的额外 head 维度，直接对形状为 $(8, N, d_{model})$ 的 $Q$、$K$、$V$ 进行 attention。扫描

$$
d_{model}\in\{16,32,64,128\},\qquad
N\in\{256,1024,4096,8192,16384\}.
$$

每组实验使用 5 次 warmup、100 次 forward 和 100 次 backward，并在每次操作后调用 torch.cuda.synchronize()。forward 计时中每轮都会释放输出张量；backward 计时在计时区间外重新构造 forward graph，只对 output.backward(grad_output) 计时，避免重复使用计算图或把 loss reduction 混入 attention backward。

本机 GPU 为 NVIDIA GeForce RTX 5060，显存约 8 GiB。由于 WSL/WDDM 在显存不足时可能发生长时间换页，实验通过 torch.cuda.set_per_process_memory_fraction(0.90) 将 PyTorch allocator 上限设为可见显存的 90%，使超限配置及时报告 OOM。该上限不影响成功配置：其最大 peak allocated 约 3.2 GiB，明显低于 7.16 GiB 的 allocator 上限。

### 17.2 完整结果

下表中的 forward/backward 为单次平均时间，显存增量为第一次 forward 输出图相对于 forward 前基线的 allocated 增量。OOM 配置在 warmup 阶段失败，因此没有正式计时。

| $d_{model}$ | $N$ | Forward (ms) | Backward (ms) | Forward 显存增量 (MiB) | Peak allocated (MiB) | 结果 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 256 | 0.215 | 0.479 | 4.148 | 29.023 | 成功 |
| 16 | 1024 | 1.135 | 2.818 | 64.594 | 211.344 | 成功 |
| 16 | 4096 | 17.364 | 42.534 | 1026.375 | 3100.625 | 成功 |
| 16 | 8192 | — | — | — | 6177.000 | OOM（warmup） |
| 16 | 16384 | — | — | — | 6177.000 | OOM（warmup） |
| 32 | 256 | 0.136 | 0.432 | 4.273 | 29.773 | 成功 |
| 32 | 1024 | 1.147 | 2.831 | 65.094 | 214.344 | 成功 |
| 32 | 4096 | 17.507 | 42.754 | 1028.375 | 3112.625 | 成功 |
| 32 | 8192 | — | — | — | 6193.000 | OOM（warmup） |
| 32 | 16384 | — | — | — | 6193.000 | OOM（warmup） |
| 64 | 256 | 0.141 | 0.415 | 4.523 | 31.273 | 成功 |
| 64 | 1024 | 1.200 | 2.951 | 66.094 | 220.344 | 成功 |
| 64 | 4096 | 18.374 | 44.392 | 1032.375 | 3136.625 | 成功 |
| 64 | 8192 | — | — | — | 6225.000 | OOM（warmup） |
| 64 | 16384 | — | — | — | 6225.000 | OOM（warmup） |
| 128 | 256 | 0.161 | 0.423 | 5.023 | 34.273 | 成功 |
| 128 | 1024 | 1.580 | 3.617 | 68.094 | 232.344 | 成功 |
| 128 | 4096 | 24.460 | 54.815 | 1040.375 | 3184.625 | 成功 |
| 128 | 8192 | — | — | — | 6289.000 | OOM（warmup） |
| 128 | 16384 | — | — | — | 6289.000 | OOM（warmup） |

总计 20 组实验中有 12 组成功、8 组 OOM。所有 $N\leq4096$ 的配置成功，所有 $N\geq8192$ 的配置均在 warmup backward 阶段 OOM，因此本机的 attention 长度边界稳定在 4096 与 8192 之间，且主要由序列长度决定，$d_{model}$ 只带来较小的附加开销。

### 17.3 最小 OOM 配置的显存 accounting

最小 OOM 配置为 $B=8$、$d_{model}=16$、$N=8192$。FP32 下，$Q$、$K$、$V$ 的输入显存为

$$
3\times B\times N\times d_{model}\times4
=3\times8\times8192\times16\times4
\approx12\text{ MiB}.
$$

标准 attention 的 score 矩阵形状为 $(B,N,N)$，单个矩阵需要

$$
B\times N^2\times4
=8\times8192^2\times4
=2\text{ GiB}.
$$

forward 至少需要保存 score 以及 softmax 后的 probability 两个 $N\times N$ 矩阵，合计约 4 GiB；再加上输入、输出、梯度和 backward 临时 workspace，warmup backward 过程中已经分配约 6.02 GiB，随后尝试申请额外 2.00 GiB 而失败，记录的 peak allocated 为 6177 MiB。

作为对照，$N=4096$ 时单个 attention 矩阵为 512 MiB，两个矩阵约 1 GiB；$d_{model}=16$ 的实测 forward allocation delta 为 1026.375 MiB，与理论值高度一致。

### 17.4 序列长度缩放与消除显存代价

以 $d_{model}=16$ 的成功配置为例，forward allocation delta 从 $N=256$ 的 4.148 MiB 增长到 $N=1024$ 的 64.594 MiB，再增长到 $N=4096$ 的 1026.375 MiB。序列长度每扩大 4 倍，显存约扩大 16 倍，验证了 attention 中间矩阵的 $O(N^2)$ 缩放；$d_{model}$ 增大只使 $Q/K/V$、输出和相应 GEMM workspace 线性增加，因此在相同 $N$ 下影响明显较小。

要消除这部分显存代价，应使用 FlashAttention-2 一类的 tiled、fused attention kernel：将 $QK^T$ 和 softmax 分块计算，只保留必要的行级统计量（例如 log-sum-exp），不显式物化完整的 $(B,N,N)$ score/probability 矩阵，并在 backward 中按 tile 重计算所需中间量。这样可将 attention activation memory 从 $O(N^2)$ 降至近似 $O(Nd_{model})$，代价是额外重计算。

实验原始结果：pytorch_attention.json（见 results/attention/）。


## 十八、整体 Transformer 的 torch.compile 对比

### 18.1 实验设置

在端到端 benchmark 中增加 compile-model 开关。开启后先构造完整的 BasicsTransformerLM，再调用 torch.compile(model, dynamic=False)，随后使用编译后模型的 parameters 创建 AdamW optimizer。每组实验使用相同的随机种子、Small 模型、batch size=1、context length=256、vocab size=10000、FP32、5 次 warmup 和 10 次正式测量。

torch.compile 的图捕获、算子融合和 Triton kernel 编译发生在首次 warmup 调用中；正式 mean/std 只统计编译完成后的稳态 step，不把一次性编译成本计入 forward、forward + backward 或 train 的平均时间。

### 18.2 Vanilla 与 compiled 结果

| 模式 | Vanilla mean (ms) | Compiled mean (ms) | 加速 | Vanilla std (ms) | Compiled std (ms) | Vanilla peak (MiB) | Compiled peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | 12.080 | 9.539 | 21.0% | 0.131 | 0.067 | 516.84 | 510.78 |
| forward + backward | 39.006 | 29.885 | 23.4% | 2.174 | 0.586 | 1016.11 | 1062.73 |
| train + optimizer | 79.413 | 63.555 | 20.0% | 8.044 | 0.286 | 2162.27 | 2176.14 |

对应的 peak reserved memory 分别为：

| 模式 | Vanilla reserved (MiB) | Compiled reserved (MiB) | 变化 |
| --- | ---: | ---: | ---: |
| forward | 588 | 570 | -3.1% |
| forward + backward | 1276 | 1324 | +3.8% |
| train + optimizer | 2546 | 2528 | -0.7% |

### 18.3 结果分析

torch.compile 在三种模式下都带来约 20%–23% 的稳态加速，其中 forward + backward 的收益最大，为 23.4%。compiled 版本的标准差也明显更小：完整 train 的标准差从 8.044 ms 降到 0.286 ms，说明编译后的 kernel 调度和执行过程更加稳定。

显存收益并不一致。forward 的 peak allocated 仅降低 1.2%；forward + backward 反而增加 4.6%，说明编译后的 backward 可能使用额外的临时 workspace；完整 train 的 peak allocated 只增加 0.6%，而 peak reserved 下降 0.7%。因此 torch.compile 的主要收益是计算融合和 kernel 调度优化，而不是系统性地降低训练峰值显存。

本实验严格保持 FP32 和相同随机种子。Inductor 输出的 TF32 未启用和 max-autotune SM 数不足警告不改变正确性；前者保持了 vanilla/compiled 的精度口径一致，后者只是让编译器回退到可用的 kernel 策略。

### 18.4 本节结论

1. 对 Small/256、batch size=1 的完整 Transformer，torch.compile 将 forward、forward + backward 和 train step 分别加速 21.0%、23.4% 和 20.0%。
2. 编译收益主要来自算子融合与 kernel 调度，不能简单归因于显存优化。
3. Peak allocated 在不同模式下变化方向不一致，因此不能用 torch.compile 替代 activation checkpointing 来解决显存容量问题。
4. 报告中的 mean/std 均为 warmup 后稳态执行时间；首次编译延迟属于一次性成本，未计入正式测量。

实验结果文件：

- results/torch_compile/small_b1_ctx256_forward_vanilla.json
- results/torch_compile/small_b1_ctx256_forward_compiled.json
- results/torch_compile/small_b1_ctx256_forward_backward_vanilla.json
- results/torch_compile/small_b1_ctx256_forward_backward_compiled.json
- results/torch_compile/small_b1_ctx256_train_vanilla.json
- results/torch_compile/small_b1_ctx256_train_compiled.json


## 十九、FlashAttention-2 Benchmarking

### 19.1 实验设置与计时口径

本节使用 [flash_attention_benchmark.py](../cs336_systems/flash_attention_benchmark.py)，比较普通 PyTorch attention 与自定义 FlashAttention-2 的前向、反向及前后向组合性能。实验在本地 RTX 5060 上运行，属于资源受限的代理实验；作业原文指定单张 B200，因此本节结果不能视为 B200 硬件上的复现。

| 项目 | 配置与说明 |
| --- | --- |
| 环境 / GPU | WSL Ubuntu / NVIDIA GeForce RTX 5060，约 8 GiB 显存 |
| 输入形状 | $Q,K,V\in\mathbb{R}^{1\times N\times d}$，batch size=1 |
| Mask | 全部启用 causal masking |
| 序列长度 $N$ | 128、256、512、1024、2048、4096、8192、16384、32768、65536 |
| Embedding dimension $d$ | 16、32、64、128 |
| 输入 dtype | torch.bfloat16、torch.float32 |
| 普通 PyTorch 基线 | 显式计算 score、causal mask、softmax 和 value matmul，再使用 autograd 反向；不是融合 SDPA 基线 |
| 自定义实现 | Triton 分块前向 + torch.compile 编译的 PyTorch 反向 |
| 前向 tile | Q_TILE_SIZE=16，K_TILE_SIZE=16；本轮未按输入形状调优 |
| 计时方法 | triton.testing.do_bench，return_mode="median" |
| 本轮运行方案参数 | warmup=25 ms，rep=100 ms，memory-fraction=0.90 |
| 测量对象 | F、B、F+B；不包含 Transformer 其他模块、loss 或 optimizer step |
| 原始数据 | [rtx5060_full.json](../results/flash_attention/rtx5060_full.json) |

本节的 warmup/rep 采用毫秒预算，不是第五节常规 Transformer benchmark 的固定迭代次数。输入 $Q,K,V$ 与输出梯度在计时前随机生成；每个 case 独立生成输入，当前脚本未固定随机种子，也未在两种实现间复用同一组输入。因此两种实现按相同 shape、dtype 和 mask 配对，但不是逐元素相同输入的配对实验。

三种计时模式分别为：

| 模式 | 计时范围 |
| --- | --- |
| F | 在启用 autograd、输入 requires_grad=True 的条件下执行一次 forward；不是 inference_mode benchmark |
| B | 计时前构造 forward graph，计时中调用 autograd.grad，并以 retain_graph=True 复用该图 |
| F+B | 每次操作重新执行 forward 并计算对 Q、K、V 的梯度 |

每个 latency 是 do_bench 返回的单 case 中位数，而不是均值。F+B 单独测量，不能用 F 与 B 的中位数相加代替；编译、准备和正式测量的总墙钟时间也不能等同于表中 latency。本轮原始 JSON 不记录 warmup/rep、软件版本或随机种子等运行元数据，上表时间预算依据本轮执行方案及脚本默认参数说明，后续跨机器复测应独立保存运行命令与环境信息。

自定义反向函数 [_flash_attention_backward](../cs336_systems/flash_attention.py) 将 Q、K、V、O、dO 等转为 FP32 后重算中间量，最后把梯度转换回原输入 dtype。因此“BF16”列表示输入 dtype，不表示自定义前后向中的所有运算都以 BF16 执行。

### 19.2 完整结果与数据核验

完整矩阵包含 $10\times4\times2=80$ 个输入配置，每个配置测量两种实现、三种模式，共 $80\times2\times3=480$ 条记录。按 (dtype, N, d, implementation, mode) 建立唯一键后，480 个键均唯一，且无缺漏或额外配置；全部记录的 batch size 为 1、is_causal 为 True。400 条成功记录的 latency 均为有限正数，另有 80 条 OOM、0 条其他 error。

| 模式 | PyTorch 成功 | PyTorch OOM | 自定义成功 | 自定义 OOM |
| --- | ---: | ---: | ---: | ---: |
| F | 64 | 16 | 80 | 0 |
| B | 64 | 16 | 64 | 16 |
| F+B | 64 | 16 | 64 | 16 |
| 合计 | 192 | 48 | 208 | 32 |

本轮结果核验已有测试记录为：FlashAttention 的 6 项前向/反向测试与 checkpointing 的 1 项等价性测试全部通过，共 7 passed。单元测试通过与 benchmark 的 status="ok" 是不同证据：后者只代表该 case 成功执行并完成计时，不表示 480 个 case 都做了逐项数值对照。本节整理未重新运行完整 sweep。

下列两张表各包含 40 个输入配置，所有耗时单位均为 ms，保留四位小数；原始精度保留在 JSON 中。F、B、F+B 分别表示 forward、backward、forward_backward；“自定义”始终指 Triton 前向 + 编译 PyTorch 反向。OOM 表示该模式未得到有效 latency，不是 0 ms；原始异常文本保留在 JSON 中。

#### 19.2.1 BF16 输入

| $N$ | $d$ | PyTorch F | PyTorch B | PyTorch F+B | 自定义 F | 自定义 B | 自定义 F+B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 16 | 0.0615 | 0.0303 | 0.0930 | 0.0084 | 0.0848 | 0.0868 |
| 128 | 32 | 0.0560 | 0.0298 | 0.0868 | 0.0084 | 0.0883 | 0.0858 |
| 128 | 64 | 0.0541 | 0.0297 | 0.0831 | 0.0085 | 0.0816 | 0.0899 |
| 128 | 128 | 0.0543 | 0.0300 | 0.0858 | 0.0105 | 0.0909 | 0.0998 |
| 256 | 16 | 0.0552 | 0.0387 | 0.0878 | 0.0105 | 0.0884 | 0.0903 |
| 256 | 32 | 0.0558 | 0.0365 | 0.0917 | 0.0125 | 0.0888 | 0.0960 |
| 256 | 64 | 0.0549 | 0.0366 | 0.0919 | 0.0125 | 0.0880 | 0.0997 |
| 256 | 128 | 0.0593 | 0.0402 | 0.0950 | 0.0146 | 0.1093 | 0.1158 |
| 512 | 16 | 0.0646 | 0.0469 | 0.1036 | 0.0146 | 0.0960 | 0.1152 |
| 512 | 32 | 0.0644 | 0.0465 | 0.1073 | 0.0187 | 0.1018 | 0.1124 |
| 512 | 64 | 0.0663 | 0.0535 | 0.1114 | 0.0206 | 0.1096 | 0.1228 |
| 512 | 128 | 0.0553 | 0.0579 | 0.1197 | 0.0247 | 0.1283 | 0.1472 |
| 1024 | 16 | 0.0753 | 0.0803 | 0.1504 | 0.0248 | 0.1636 | 0.1861 |
| 1024 | 32 | 0.0704 | 0.0838 | 0.1521 | 0.0350 | 0.1662 | 0.2008 |
| 1024 | 64 | 0.0743 | 0.0773 | 0.1498 | 0.0390 | 0.2107 | 0.2476 |
| 1024 | 128 | 0.0828 | 0.0989 | 0.1828 | 0.0534 | 0.2717 | 0.3249 |
| 2048 | 16 | 0.1947 | 0.2120 | 0.3898 | 0.0616 | 0.4849 | 0.5417 |
| 2048 | 32 | 0.1956 | 0.2050 | 0.3895 | 0.0984 | 0.5129 | 0.6036 |
| 2048 | 64 | 0.2084 | 0.2169 | 0.4181 | 0.1107 | 0.5901 | 0.6987 |
| 2048 | 128 | 0.2376 | 0.2683 | 0.4997 | 0.1742 | 0.8066 | 0.9760 |
| 4096 | 16 | 1.2186 | 1.4512 | 2.6481 | 0.1947 | 1.8359 | 2.0252 |
| 4096 | 32 | 1.2318 | 1.4479 | 2.6552 | 0.3340 | 1.8753 | 2.2052 |
| 4096 | 64 | 1.2349 | 1.4520 | 2.6557 | 0.3770 | 2.1988 | 2.5697 |
| 4096 | 128 | 1.3005 | 1.5780 | 2.8813 | 0.5572 | 2.8421 | 3.3940 |
| 8192 | 16 | 4.8860 | 5.9108 | 10.7771 | 0.7231 | 6.9842 | 7.7007 |
| 8192 | 32 | 4.9060 | 5.8563 | 10.7878 | 1.2842 | 7.0968 | 8.3819 |
| 8192 | 64 | 4.8438 | 5.7815 | 10.5364 | 1.4172 | 8.5662 | 9.9835 |
| 8192 | 128 | 5.1067 | 6.2593 | 11.3828 | 2.0686 | 11.0739 | 13.1284 |
| 16384 | 16 | 18.4205 | 23.2269 | 42.3516 | 2.7465 | 26.8037 | 29.5009 |
| 16384 | 32 | 18.4934 | 23.3047 | 42.4562 | 4.9133 | 27.0335 | 31.9112 |
| 16384 | 64 | 18.0722 | 22.9879 | 42.1188 | 5.4120 | 37.4972 | 43.9680 |
| 16384 | 128 | 19.2997 | 24.9531 | 46.2325 | 7.9136 | 50.7947 | 59.0234 |
| 32768 | 16 | OOM | OOM | OOM | 10.7239 | OOM | OOM |
| 32768 | 32 | OOM | OOM | OOM | 19.2480 | OOM | OOM |
| 32768 | 64 | OOM | OOM | OOM | 21.2454 | OOM | OOM |
| 32768 | 128 | OOM | OOM | OOM | 31.0908 | OOM | OOM |
| 65536 | 16 | OOM | OOM | OOM | 45.5301 | OOM | OOM |
| 65536 | 32 | OOM | OOM | OOM | 81.2124 | OOM | OOM |
| 65536 | 64 | OOM | OOM | OOM | 88.6569 | OOM | OOM |
| 65536 | 128 | OOM | OOM | OOM | 130.4136 | OOM | OOM |

#### 19.2.2 FP32 输入

| $N$ | $d$ | PyTorch F | PyTorch B | PyTorch F+B | 自定义 F | 自定义 B | 自定义 F+B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 16 | 0.0645 | 0.0797 | 0.1360 | 0.0084 | 0.0715 | 0.0756 |
| 128 | 32 | 0.0628 | 0.0694 | 0.1289 | 0.0084 | 0.0715 | 0.0786 |
| 128 | 64 | 0.0637 | 0.0694 | 0.1272 | 0.0105 | 0.0746 | 0.0823 |
| 128 | 128 | 0.0657 | 0.0704 | 0.1360 | 0.0106 | 0.0799 | 0.0868 |
| 256 | 16 | 0.0568 | 0.0796 | 0.1288 | 0.0105 | 0.0712 | 0.0786 |
| 256 | 32 | 0.0584 | 0.0651 | 0.1209 | 0.0125 | 0.0762 | 0.0848 |
| 256 | 64 | 0.0577 | 0.0741 | 0.1198 | 0.0144 | 0.0739 | 0.0868 |
| 256 | 128 | 0.0689 | 0.0775 | 0.1401 | 0.0166 | 0.0978 | 0.1006 |
| 512 | 16 | 0.0703 | 0.0820 | 0.1580 | 0.0166 | 0.0829 | 0.0960 |
| 512 | 32 | 0.0742 | 0.0875 | 0.1434 | 0.0206 | 0.0832 | 0.0981 |
| 512 | 64 | 0.0689 | 0.0876 | 0.1493 | 0.0248 | 0.0919 | 0.1104 |
| 512 | 128 | 0.0681 | 0.0909 | 0.1580 | 0.0349 | 0.1145 | 0.1452 |
| 1024 | 16 | 0.0928 | 0.1233 | 0.2116 | 0.0349 | 0.1525 | 0.1860 |
| 1024 | 32 | 0.0987 | 0.1297 | 0.2210 | 0.0411 | 0.1600 | 0.1990 |
| 1024 | 64 | 0.1173 | 0.1540 | 0.2666 | 0.0534 | 0.2027 | 0.2523 |
| 1024 | 128 | 0.1332 | 0.1942 | 0.3227 | 0.1066 | 0.2601 | 0.3666 |
| 2048 | 16 | 0.3932 | 0.5550 | 0.9497 | 0.0943 | 0.4772 | 0.5673 |
| 2048 | 32 | 0.3932 | 0.5776 | 0.9727 | 0.1189 | 0.4997 | 0.6144 |
| 2048 | 64 | 0.4137 | 0.6098 | 1.0220 | 0.1742 | 0.5799 | 0.7516 |
| 2048 | 128 | 0.4977 | 0.7608 | 1.2529 | 0.3176 | 0.8000 | 1.1121 |
| 4096 | 16 | 2.0431 | 2.9188 | 4.9644 | 0.3094 | 1.8207 | 2.1329 |
| 4096 | 32 | 2.0613 | 2.9573 | 5.0248 | 0.3975 | 1.8719 | 2.2681 |
| 4096 | 64 | 2.2538 | 3.1621 | 5.4370 | 0.5595 | 2.2098 | 2.7621 |
| 4096 | 128 | 2.4694 | 3.6485 | 6.1454 | 1.0792 | 2.8631 | 3.9383 |
| 8192 | 16 | 8.1219 | 11.8923 | 20.0242 | 1.1798 | 6.9714 | 8.1528 |
| 8192 | 32 | 8.1715 | 11.5753 | 19.7907 | 1.5238 | 7.1281 | 8.6569 |
| 8192 | 64 | 9.2619 | 12.7250 | 21.9648 | 2.1028 | 8.6384 | 10.7448 |
| 8192 | 128 | 9.9731 | 14.4833 | 24.4951 | 4.0879 | 11.2332 | 15.3225 |
| 16384 | 16 | 32.3819 | 48.2868 | 82.4676 | 4.5446 | 26.8227 | 31.2938 |
| 16384 | 32 | 32.3940 | 48.3378 | 83.1168 | 5.8348 | 27.0139 | 32.8283 |
| 16384 | 64 | 41.5001 | 57.4540 | 99.2915 | 8.0568 | 37.8528 | 47.2824 |
| 16384 | 128 | 45.7563 | 65.1986 | 111.7887 | 15.9548 | 51.7891 | 68.3571 |
| 32768 | 16 | OOM | OOM | OOM | 17.7664 | OOM | OOM |
| 32768 | 32 | OOM | OOM | OOM | 22.8312 | OOM | OOM |
| 32768 | 64 | OOM | OOM | OOM | 31.8874 | OOM | OOM |
| 32768 | 128 | OOM | OOM | OOM | 65.5329 | OOM | OOM |
| 65536 | 16 | OOM | OOM | OOM | 74.3178 | OOM | OOM |
| 65536 | 32 | OOM | OOM | OOM | 95.8792 | OOM | OOM |
| 65536 | 64 | OOM | OOM | OOM | 133.2265 | OOM | OOM |
| 65536 | 128 | OOM | OOM | OOM | 247.6298 | OOM | OOM |

### 19.3 OOM 分布与容量边界

80 次 OOM 全部集中在 $N=32768$ 和 $65536$。每个长度覆盖两种输入 dtype、四种 d，共 8 个输入配置：普通 PyTorch 的三种模式每个长度产生 24 次 OOM，自定义实现的 B 与 F+B 每个长度产生 16 次 OOM；两个长度合计 $2\times(24+16)=80$ 次。

| 实现 / 模式 | 本次网格内最大成功 N | 本次网格内最小 OOM N |
| --- | ---: | ---: |
| PyTorch F / B / F+B | 16384 | 32768 |
| 自定义 F | 65536 | 本次网格内未出现 |
| 自定义 B / F+B | 16384 | 32768 |

上表边界在本轮两种输入 dtype、四种 d 下相同。65536 是测试上限，不是已测出的自定义前向极限；同样，16384 与 32768 之间的准确容量阈值未被扫描。该边界还受 90% allocator 上限和运行时可用显存影响，不能直接视为设备全部物理显存下的极限。

自定义前向按 tile 计算 attention，保存 Q、K、V、O 和行级 log-sum-exp，不需要在全局显存中物化完整的 score/probability 矩阵。反向则仍使用完整矩阵：代码重算 scores 和 p，并计算 grad_p、grad_scores 等 $N\times N$ 中间量。torch.compile 可以进行编译与算子优化，但当前反向并未实现按 tile 重计算的算法，不能据此宣称其峰值显存已变为线性。

以 batch size=1、FP32 中间量为例，单个方阵的大小为：

$$
M_{\mathrm{matrix}}=N^2\times4\ \mathrm{bytes}.
$$

| N | 单个 FP32 N×N 矩阵 |
| ---: | ---: |
| 16384 | 1 GiB |
| 32768 | 4 GiB |
| 65536 | 16 GiB |

在 $N=32768$、$d=16$、BF16 输入的自定义 B 记录中，异常报告尝试额外申请 4.00 GiB，而当时 PyTorch 已分配约 4.03 GiB，与完整 FP32 中间矩阵的容量瓶颈一致。编译器可能融合部分操作或复用 buffer，因此不能简单把源码中所有矩阵的大小相加当成实测峰值；但完整矩阵重计算和 OOM 日志共同支持反向仍受二次方中间量限制的判断。

普通 PyTorch 在长序列时也显式生成 score、probability 和 mask 等中间量。其 B 模式需要先构造 forward graph；由于对应大长度的 F 本身已 OOM，PyTorch B 标记为 OOM 不一定说明异常发生在反向计时内部。当前日志未记录异常发生阶段，报告不进一步猜测其精确位置。

原始 OOM 文本中出现不合理的超大进程显存数值，不将其用作显存 accounting；本节仅采用申请大小、PyTorch allocated/allowed 等可解释字段及矩阵大小推算。当前数据没有 peak allocated/reserved 字段，因此本节展示的是容量边界，不是实测峰值显存曲线。

### 19.4 性能分析

#### 19.4.1 加速比统计

对相同 dtype、N、d、mode 且两种实现均成功的记录计算：

$$
S=\frac{T_{\mathrm{PyTorch}}}{T_{\mathrm{custom}}}.
$$

$S>1$ 表示自定义实现更快。每个 dtype/mode 有 32 对可比较配置，即 $N=128$ 至 16384 的 8 个长度乘以 4 个 d；任一方 OOM 的配置不参与加速比计算。以下使用未舍入的原始 latency 计算，随后对每组 32 个比值取中位数与最小/最大值。

| 输入 dtype | 模式 | 可比较配置数 | 自定义更快的配置数 | 加速比中位数 | 加速比范围 |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 | F | 32 | 32 | 3.43× | 1.36–7.31× |
| BF16 | B | 32 | 0 | 0.47× | 0.33–0.87× |
| BF16 | F+B | 32 | 10 | 0.91× | 0.51–1.44× |
| FP32 | F | 32 | 32 | 4.16× | 1.25–7.67× |
| FP32 | B | 32 | 19 | 1.08× | 0.75–1.80× |
| FP32 | F+B | 32 | 31 | 1.59× | 0.88–2.64× |

这些中位数描述本轮配置集合的性能分布，不是“所有 PyTorch latency 之和 / 所有自定义 latency 之和”，也不是 Transformer 完整训练的整体加速比。

#### 19.4.2 前向收益与时间缩放

自定义前向在全部 64 对共同成功配置中均更快：BF16 与 FP32 的加速比中位数分别为 3.43×、4.16×。这与分块和融合减少完整 attention 矩阵的 HBM 读写相一致；具体收益中有多少来自访存、kernel 调度或底层矩阵乘策略，本轮未进行单独归因实验。

显存中间量减少并不意味着 attention 的计算量变为线性。例如固定 $d=64$、BF16 时，自定义 F 在 $N=16384,32768,65536$ 下分别为 5.4120、21.2454、88.6569 ms，序列每翻倍，时间约增至 3.93 倍和 4.17 倍，仍呈现接近二次方的计算时间增长。

#### 19.4.3 BF16 反向瓶颈与前后向组合

BF16 的 32 对 B 配置全部比普通 PyTorch 慢，加速比中位数仅 0.47×；对应 F+B 只有 10/32 对更快，中位数为 0.91×。原因分析必须结合实际精度路径：自定义反向将 BF16 输入转为 FP32，并重算完整 scores、probability 及梯度中间矩阵；普通 PyTorch 基线则沿自身 BF16 前向图执行 autograd。这不是“相同 BF16 计算路径下，Triton 反向天然更慢”的证据，因为自定义 B 并非手写 Triton 分块 kernel。

以 $N=1024,d=64$、BF16 为例，F 从 0.0743 ms 降至 0.0390 ms，但 B 从 0.0773 ms 增至 0.2107 ms，独立测得的 F+B 从 0.1498 ms 增至 0.2476 ms。反向增加的工作抵消了前向节省，因此不能只凭前向加速推断整个 attention 前后向都会加速。FP32 重计算、类型转换与完整中间矩阵是代码可见的解释因素，但各项具体耗时仍需要额外 profiling，不能由总时间唯一分解。

FP32 下，B 有 19/32 对配置加速，中位数为 1.08×；F+B 则有 31/32 对加速，中位数为 1.59×。例如 $N=16384,d=64$ 时，F+B 从 99.2915 ms 降至 47.2824 ms，约为 2.10×。因此收益显著依赖 dtype、shape 和测量模式，不能用单个形状的结果概括所有配置。

### 19.5 局限与本节结论

| 局限 | 对结论的约束 |
| --- | --- |
| 使用 RTX 5060，而非作业指定的 B200 | 只能报告本地代理实验，不能外推 B200 性能或容量边界 |
| 自定义反向仍为完整矩阵 FP32 重计算 | 不能声称已完成分块 Triton backward，或完整前后向显存均为线性 |
| 前向固定 16×16 tile | 本轮是当前实现的基线，不代表各形状的最优配置 |
| 只有单轮 sweep 的每 case latency 中位数 | 没有跨运行重复的误差条或置信区间；接近 1× 的差异需要重复实验才能判断稳定性 |
| 每个 case 独立生成随机输入，未固定 seed | 两种实现按 shape/dtype 配对，不是相同输入值的严格配对 |
| JSON 缺少完整运行元数据与峰值显存字段 | 不能据此给出精确显存节省百分比或实测显存复杂度曲线 |
| 基线是显式 PyTorch attention | 结论不等于相对 PyTorch 融合 SDPA 或生产级 FlashAttention 的性能比较 |
| benchmark 成功不等于逐配置数值验证 | 已有单元测试通过，仍不代表 480 个配置均逐项比较输出与梯度 |

本节结论如下：

1. 完成 480 条完整配置记录，400 条成功、80 条 OOM，无重复、缺漏或其他 error。
2. 自定义分块前向在本轮所有测试配置中成功，最长测试序列达到 65536；普通 PyTorch 及自定义反向相关模式的最大成功测试长度为 16384。
3. 自定义前向在共同成功配置上全部加速，BF16/FP32 加速比中位数分别为 3.43×/4.16×。
4. 反向仍是主要限制：完整 FP32 重计算保留了二次方中间量，并使 BF16 前后向组合不一定获益；FP32 前后向组合的加速比中位数为 1.59×。
5. 本轮完成的是 FlashAttention 本地性能评估与报告收尾，不因 OOM 强行扩大硬件规模，也不在本节继续修改 kernel。

进度记录：已完成 FlashAttention 本地 benchmark、Naive 与 Flat-gradient DDP 正确性验证及 CPU/Gloo 性能对照归档，结果见第二十至二十二节。下一小节为 Overlap DDP；指定双 GPU、XL 模型的正式性能实验待补。

原始结果与实现：

- [完整 benchmark JSON](../results/flash_attention/rtx5060_full.json)
- [复查 smoke JSON](../results/flash_attention/rtx5060_smoke_recheck.json)
- [benchmark 脚本](../cs336_systems/flash_attention_benchmark.py)
- [FlashAttention 实现](../cs336_systems/flash_attention.py)

## 二十、Naive DDP 正确性验证

### 20.1 验证对象与配置

本节对应作业 `naive_ddp` 的正确性验证。测试通过 [adapters](../tests/adapters.py) 调用当前 [NaiveDDP](../cs336_systems/ddp.py)，使用官方 [test_ddp.py](../tests/test_ddp.py) 检查普通模型与共享权重模型的训练一致性。测试入口为 `uv run pytest -v tests/test_ddp.py`。

| 项目 | 配置 |
| --- | --- |
| 测试模型 | ToyModel、ToyModelWithTiedWeights |
| 进程数 | 2，使用 spawn 启动 |
| 通信后端 | Gloo |
| 数据 | 官方 fixtures 中的 20 个样本，每个 rank 处理 10 个样本 |
| 损失与优化器 | MSELoss，SGD，学习率 0.1 |
| 训练步数 | 每个用例执行 5 次参数更新 |
| 参考路径 | 单模型处理完整 batch |
| 数值比较 | 使用官方测试中的 torch.allclose 判定 |

上述配置依据当前测试源码。官方测试的设备选择由 `_setup_process_group` 决定：检测到 CUDA 时使用 CUDA，否则使用 CPU。本次反馈未附运行环境信息，因此不将实际设备、操作系统或 Python/PyTorch 版本记为已确认，也不沿用前文 RTX 5060 实验的环境记录。

### 20.2 测试结果与覆盖范围

根据本次运行反馈，官方两个参数化测试用例均通过：

| 用例 | 结果 | 主要覆盖内容 |
| --- | --- | --- |
| test_DistributedDataParallel[ToyModel] | 通过 | 初始参数同步、冻结参数保持不变、多步参数更新与完整 batch 参考路径一致 |
| test_DistributedDataParallel[ToyModelWithTiedWeights] | 通过 | 含共享权重时的初始参数同步，以及多步参数更新与完整 batch 参考路径一致 |

测试先检查不同 rank 的初始参数同步，再将完整数据均分给两个进程进行训练。每次更新后，rank 0 的 DDP 参数与处理完整 batch 的参考模型参数进行比较；同时检查各 rank 的模型状态一致。每步采用相同的随机种子重排数据，覆盖连续 5 步训练。

结果来源为使用者反馈的“两个测试均通过”；本次报告整理未重新执行测试，也未附原始 pytest 日志或耗时。测试直接比较的是模型状态及更新后的参数，未单独逐项比较同步后的梯度张量，因此本节结论限定在官方用例覆盖的训练一致性范围内。

### 20.3 本阶段结论与后续工作

Naive DDP 已完成官方两个用例的正确性验证，并补齐本阶段报告记录。现有实现可作为后续 DDP 实验的基础正确性基线。

本次测试未测量训练吞吐量、梯度通信耗时或通信与计算的重叠情况，也不构成指定双 GPU、XL 模型配置的实验结果。后续仍需完成 Naive DDP benchmark、Flat-gradient DDP 和 Overlap DDP 的实现或实验，以及相应的报告。原始测试输出与实际运行环境信息尚待归档。

## 二十一、Naive DDP CPU/Gloo Benchmark

### 21.1 实验来源与配置

本节归档使用者提供的 benchmark 标准输出，归档日期为 2026-09-16；未单独记录实际运行时间。本次归档未重新执行实验。原始数值保存在 [naive_ddp_cpu_gloo.json](../results/ddp/naive_ddp_cpu_gloo.json)，对应脚本为 [ddp_benchmark.py](../cs336_systems/ddp_benchmark.py)，运行入口为 `uv run python -m cs336_systems.ddp_benchmark`。

| 项目 | 配置 |
| --- | --- |
| 运行环境 | WSL Ubuntu，CPU，Gloo 后端 |
| PyTorch | 2.11.0+cu130；本次计算使用 CPU |
| 进程数 / 每进程 CPU 线程数 | 2 / 1 |
| 模型 | 三层带 bias 的 Linear：128→256→256→32，前两层后接 ReLU |
| 精度 | FP32 |
| 损失 / 优化器 | MSELoss（mean）/ SGD，学习率 0.01 |
| Global / local batch size | 32 / 16 |
| Warmup / 正式测量 | 5 / 20 步 |
| 梯度张量数 / 每 rank 逻辑梯度大小 | 6 / 428160 字节 |

模型、优化器和计时设置依据本次归档时读取的脚本；运行统计依据所提供的 JSON。未记录 CPU 型号、Python 版本和逐步计时样本。

### 21.2 计时口径与结果

每个 step 包含 forward、loss、backward、梯度同步及 optimizer step，排除 zero_grad。输入生成和初始化广播位于计时区间外；预热结束后用 barrier 对齐，正式测量循环中不额外插入阶段 barrier。各 rank 的结果汇总也位于计时循环之外。

同步时间覆盖整个 `synchronize_gradient()` 调用，包括逐参数 all-reduce、梯度平均操作及可能的进程等待。因此本节称其为“梯度同步阶段耗时”，不将其全部解释为数据传输时间。同步占比按每 rank 的平均同步时间除以平均 step 时间计算。

| Rank | Mean step (ms) | Mean sync (ms) | Sync fraction |
| ---: | ---: | ---: | ---: |
| 0 | 3.056945 | 2.745775 | 89.8209% |
| 1 | 3.056745 | 2.742865 | 89.7316% |

两个 rank 平均 step 时间中的最大值为 **3.056945 ms**。该指标是先对各 rank 的 20 步求均值，再取最大值，不是每步取最慢 rank 后再求均值。

三层 Linear 共包含 107040 个参数元素，与 6 个梯度张量一致；FP32 下对应 107040 × 4 = 428160 字节。这是每个 rank 的逻辑梯度大小，不等于通信链路上的实际传输字节数。

### 21.3 结果解释与验证边界

本轮两个 rank 的平均 step 时间接近，梯度同步阶段约占 90%。这是小型 CPU 模型在当前 Gloo 双进程配置下的观测值：同步占比取决于通信数据量、调用次数、带宽与延迟、计算量以及进程等待，不能外推到双 GPU、XL 模型配置。相同模型下，增大 batch size 通常增加计算量，但不会改变参数梯度的逻辑总字节数，因此同步占比可能下降；这属于待验证的趋势解释，本轮未进行 batch sweep。

本脚本验证了完整训练与计时、统计流程能够运行；此前官方 DDP 两个用例通过的记录见第二十节。当前 benchmark 没有单独对照参考模型的参数或梯度，运行成功不构成新的数值等价性测试。由于只保存了单次运行的均值，没有逐步样本、标准差或跨运行重复结果，不能据此断言计时稳定性。

### 21.4 当前进度

已完成 Naive DDP 的官方正确性测试、CPU/Gloo benchmark 流程验证和本次结果归档。讲义要求的单节点双 GPU、XL 模型正式 benchmark 仍待补测。下一小节进入 Flat-gradient DDP，先完成实现与正确性验证，再按相同口径进行本地对照，正式多 GPU 性能对照集中安排。

## 二十二、Flat-gradient DDP CPU/Gloo 对照

### 22.1 验证与实验设置

已保留逐参数同步的 NaiveDDP 和使用者实现的 FlatGradientDDP。整理版本入口后，两种实现分别在 WSL 的 CPU/Gloo 路径通过官方 DDP 两个用例，共 4 项通过，覆盖普通模型与共享权重模型。上述测试在此前版本整理时执行；本次确认代码与已验证版本一致后进行计时。

本轮于 2026-09-16 运行，按 Naive、Flat-gradient 的顺序交替执行三轮，每个版本各三次独立运行。沿用第二十一节的 CPU/Gloo、双进程、每进程 1 线程、FP32、三层 Linear、global/local batch 32/16、SGD 配置；每次均执行 5 步 warmup 和 20 步测量。模型种子为 42，各 rank 输入种子为 1234 + rank，两种版本使用相同初始化与数据生成设置。

每次运行显式指定 --implementation，原始 JSON 同时记录 implementation 与 implementation_class。完整 step 排除 zero_grad；同步阶段包含对应版本所需的数据整理、all-reduce、平均、写回及可能的进程等待。Flat-gradient 的拼接和写回开销未移出计时范围。

### 22.2 逐轮数据及统计口径

每次运行先对各 rank 的 20 个 step 求平均，再取两个 rank 平均值的最大值作为该次 step 指标。同步指标同样取两个 rank 平均同步时间的最大值。两项最大值可能来自不同 rank，不将其比值解释为某个 rank 的实际同步占比。

| 轮次 | 实现 | Step 指标 (ms) | Sync 指标 (ms) |
| ---: | --- | ---: | ---: |
| 1 | naive_ddp | 3.107580 | 2.786340 |
| 1 | flat_gradient_ddp | 0.946575 | 0.610590 |
| 2 | naive_ddp | 3.200805 | 2.886630 |
| 2 | flat_gradient_ddp | 0.959250 | 0.616835 |
| 3 | naive_ddp | 3.053380 | 2.731210 |
| 3 | flat_gradient_ddp | 0.891695 | 0.589055 |

以下均值与样本标准差由每个版本的三个运行级指标计算；标准差不是单次运行内 20 个 step 的标准差。

| 实现 | 三轮 mean step (ms) | 跨运行 sample std (ms) | 三轮 mean sync (ms) |
| --- | ---: | ---: | ---: |
| naive_ddp | 3.120588 | 0.074568 | 2.801393 |
| flat_gradient_ddp | 0.932507 | 0.035908 | 0.605493 |

两种实现平均 step 时间之比为 **3.346×**，Flat-gradient 的平均 step 时间降低约 **70.12%**。这里使用的是均值之比，不是三个逐轮加速比的均值。历史 Naive 数据及此前入口验证的短跑结果未混入本轮统计。

### 22.3 解释与局限

按当前实现，每步梯度同步的 all-reduce 调用由 6 次减少为 1 次，6 个原始参数梯度张量及每 rank 的逻辑梯度大小 428160 字节保持不变。三轮中 Flat-gradient 的完整 step 与同步阶段均更快。结果与减少多次集体通信的固定开销这一解释一致，但没有单独分解数据整理、通信与等待的耗时，不能把收益全部归因于网络传输。

本轮是 CPU/Gloo 小模型对照，不代表双 GPU、XL 模型结果。仅三次独立运行，固定交替顺序且未控制主机并发负载；原始脚本没有保存逐步计时样本，因此不据此建立置信区间或声称普遍加速。正式双 GPU 对照仍待补测。

### 22.4 归档与进度

原始结果目录为 [comparison_20260916_0occay](../results/ddp/comparison_20260916_0occay/)。其中保留六份 JSON，汇总文件 [summary.json](../results/ddp/comparison_20260916_0occay/summary.json) 记录执行顺序、统计定义、结果和本次实现及 benchmark 源码的 SHA-256。

- [naive_ddp_r1.json](../results/ddp/comparison_20260916_0occay/naive_ddp_r1.json)
- [flat_gradient_ddp_r1.json](../results/ddp/comparison_20260916_0occay/flat_gradient_ddp_r1.json)
- [naive_ddp_r2.json](../results/ddp/comparison_20260916_0occay/naive_ddp_r2.json)
- [flat_gradient_ddp_r2.json](../results/ddp/comparison_20260916_0occay/flat_gradient_ddp_r2.json)
- [naive_ddp_r3.json](../results/ddp/comparison_20260916_0occay/naive_ddp_r3.json)
- [flat_gradient_ddp_r3.json](../results/ddp/comparison_20260916_0occay/flat_gradient_ddp_r3.json)

已完成 Flat-gradient 的本地正确性验证、三轮 CPU/Gloo 性能对照和结果归档。下一小节为 Overlap DDP；本节不继续修改同步算法。
