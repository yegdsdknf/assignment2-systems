"""Transformer 训练步骤的可复用基准测试入口。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import math
import cs336_basics.model as basics_model
from einops import einsum
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from contextlib import nullcontext, contextmanager

import torch
import csv
import gc

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW


BenchmarkMode = Literal["forward", "forward_backward", "train"]
Precision = Literal["fp32", "bf16"]


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


@dataclass(frozen=True)
class BenchmarkResult:
    status: Literal["success", "oom"]
    model_size: str
    mode: BenchmarkMode
    device: str
    batch_size: int
    context_length: int
    vocab_size: int
    warmup_steps: int
    measurement_steps: int
    parameter_count: int | None
    mean_ms: float | None
    std_ms: float | None
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None
    oom_stage: str | None
    error: str | None
    precision: Precision
    compiled: bool


def _synchronize(device: torch.device) -> None:
    """让 CPU 计时边界等待 GPU 上已提交的工作完成。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_oom(error: RuntimeError) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _autocast_context(device: torch.device, precision: Precision):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


@contextmanager
def _nvtx_range(
    name: str,
    enable_nvtx: bool,
    device: torch.device,
):
    if not enable_nvtx or device.type != "cuda":
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def annotated_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """与原实现计算相同，仅为 Nsight 增加细粒度 NVTX 区间。"""
    device = Q.device
    d_k = K.shape[-1]

    with _nvtx_range("scaled dot product attention", True, device):
        with _nvtx_range("computing attention score", True, device):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        with _nvtx_range("applying attention mask", True, device):
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))
        with _nvtx_range("computing softmax", True, device):
            attention_weights = softmax(attention_scores, dim=-1)

        with _nvtx_range("final value matmul", True, device):
            return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")


def build_model(
    model_size: str,
    *,
    vocab_size: int,
    context_length: int,
    device: torch.device,
) -> BasicsTransformerLM:
    """按讲义给定的模型规格构造基准模型。"""
    config = MODEL_CONFIGS[model_size]
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    ).to(device)


def _make_step(
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mode: BenchmarkMode,
    device: torch.device,
    precision: Precision,
    enable_nvtx: bool,
) -> Callable[[], None]:
    if mode == "forward":
        model.eval()

        @torch.inference_mode()
        def step() -> None:
            with _nvtx_range("forward", enable_nvtx, device):
                with _autocast_context(device, precision):
                    model(input_ids)

        return step

    model.train()

    def step() -> None:
        with _nvtx_range("zero_grad", enable_nvtx, device):
            optimizer.zero_grad(set_to_none=True)

        with _nvtx_range("forward", enable_nvtx, device):
            with _autocast_context(device, precision):
                logits = model(input_ids)

        # 自定义 cross_entropy 内有 softmax/reduction；
        # 显式转回 FP32，避免低精度累积影响数值稳定性。
        with _nvtx_range("loss", enable_nvtx, device):
            loss = cross_entropy(logits.float(), target_ids)

        with _nvtx_range("backward", enable_nvtx, device):
            loss.backward()

        if mode == "train":
            with _nvtx_range("optimizer_step", enable_nvtx, device):
                optimizer.step()

    return step


def _peak_memory_mib(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    mib = 1024**2
    return torch.cuda.max_memory_allocated(device) / mib, torch.cuda.max_memory_reserved(device) / mib


def benchmark(
    *,
    model_size: str,
    mode: BenchmarkMode,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
    precision: Precision,
    enable_nvtx: bool = False,
    compile_model: bool = False,
) -> BenchmarkResult:
    """运行 warmup 后测量完整的 forward、backward 或训练步骤。"""
    parameter_count: int | None = None
    stage = "model construction"
    try:
        model = build_model(
            model_size,
            vocab_size=vocab_size,
            context_length=context_length,
            device=device,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if compile_model:
            model = torch.compile(model, dynamic=False)
        optimizer = AdamW(model.parameters())

        stage = "input allocation"
        input_ids = torch.randint(vocab_size, (batch_size, context_length), device=device)
        target_ids = torch.randint(vocab_size, (batch_size, context_length), device=device)
        step = _make_step(model, optimizer, input_ids, target_ids, mode, device, precision, enable_nvtx)

        stage = "warmup"
        for _ in range(warmup_steps):
            step()
        _synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        times_ms: list[float] = []
        stage = "measurement"
        for _ in range(measurement_steps):
            _synchronize(device)

            with _nvtx_range("profiled_measurement", enable_nvtx, device):
                # 排除 profiler 启动和停止 capture 的管理开销。
                started_at = time.perf_counter()
                step()
                _synchronize(device)
                elapsed_ms = (time.perf_counter() - started_at) * 1_000
            times_ms.append(elapsed_ms)

        peak_allocated_mib, peak_reserved_mib = _peak_memory_mib(device)
        return BenchmarkResult(
            status="success",
            model_size=model_size,
            mode=mode,
            device=str(device),
            precision=precision,
            batch_size=batch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            warmup_steps=warmup_steps,
            measurement_steps=measurement_steps,
            parameter_count=parameter_count,
            mean_ms=statistics.mean(times_ms),
            std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            peak_allocated_mib=peak_allocated_mib,
            peak_reserved_mib=peak_reserved_mib,
            oom_stage=None,
            error=None,
            compiled=compile_model,
        )
    except RuntimeError as error:
        if not _is_oom(error):
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return BenchmarkResult(
            status="oom",
            model_size=model_size,
            mode=mode,
            device=str(device),
            precision=precision,
            batch_size=batch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            warmup_steps=warmup_steps,
            measurement_steps=measurement_steps,
            parameter_count=parameter_count,
            mean_ms=None,
            std_ms=None,
            peak_allocated_mib=None,
            peak_reserved_mib=None,
            oom_stage=stage,
            error=str(error),
            compiled=compile_model,
        )


def benchmark_warmup_sweep(
    *,
    warmup_steps_list: tuple[int, ...],
    seed: int,
    model_size: int,
    mode: BenchmarkMode,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    measurement_steps: int,
    device: torch.device,
    precision: Precision,
    compiled: bool,
) -> list[BenchmarkResult]:
    """对不同 warmup 次数进行公平比较。"""
    results: list[BenchmarkResult] = []
    for warmup_steps in warmup_steps_list:
        # 每个配置从相同随机状态开始，避免模型初始化和输入数据不同。
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

            # 上一个配置结束后释放不再被引用的缓存，避免影响下一配置的显存统计。
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

        result = benchmark(
            model_size=model_size,
            mode=mode,
            batch_size=batch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            warmup_steps=warmup_steps,
            measurement_steps=measurement_steps,
            device=device,
            precision=precision,
            compile_model=compiled,
        )
        results.append(result)

        status = "完成" if result.status == "success" else f"OOM({result.oom_stage})"
        print(f"warmup={warmup_steps}: {status}")

    return results


def write_results_csv(results: list[BenchmarkResult], output_path: Path) -> None:
    """每种 warmup 配置写入 CSV 的一行。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(results[0]).keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def parse_warmup_sweep(value: str) -> tuple[int, ...]:
    """解析如 '0,1,2,5,10,20' 的 warmup 配置。"""
    try:
        steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("warmup-sweep 必须是逗号分隔的非负整数，例如 0,1,2,5,10") from error
    if not steps or any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("warmup-sweep 至少含一个非负整数")

    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--mode", choices=("forward", "forward_backward", "train"), default="forward")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="可选：写入单条 JSON 结果的文件路径")
    parser.add_argument("--warmup-sweep", type=parse_warmup_sweep, help="逗号分隔的 warmup 次数，例如 0,1,2,5,10,20")
    parser.add_argument("--csv-output", type=Path, help="warmup sweep 的 CSV 输出路径")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--enable-nvtx", action="store_true", help="为 Nsight Systems profiling 添加分阶段 NVTX 标注")
    parser.add_argument("--compile-model", action="store_true", help="使用 torch.compile 编译整个 Transformer")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.context_length <= 0 or args.vocab_size <= 0:
        parser.error("batch-size、context-length 和 vocab-size 必须为正整数")
    if args.warmup_steps < 0 or args.measurement_steps <= 0:
        parser.error("warmup-steps 必须非负，measurement-steps 必须为正整数")
    if args.warmup_sweep is not None and args.csv_output is None:
        parser.error("--warmup-sweep 必须配合 --csv-output 使用")

    if args.warmup_sweep is None and args.csv_output is not None:
        parser.error("--csv-output 仅用于 --warmup-sweep")

    if args.precision == "bf16" and torch.device(args.device).type != "cuda":
        parser.error("bf16 benchmark 当前仅支持 CUDA")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    precision = args.precision
    compiled = args.compile_model
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前 PyTorch 未检测到可用 GPU")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.enable_nvtx:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    if args.warmup_sweep is not None:
        results = benchmark_warmup_sweep(
            warmup_steps_list=args.warmup_sweep,
            seed=args.seed,
            model_size=args.model_size,
            mode=args.mode,
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            measurement_steps=args.measurement_steps,
            device=device,
            precision=precision,
            compiled=compiled,
        )
        write_results_csv(results, args.csv_output)
        print(f"CSV 已写入: {args.csv_output}")
        return
    result = benchmark(
        model_size=args.model_size,
        mode=args.mode,
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        device=device,
        precision=precision,
        enable_nvtx=args.enable_nvtx,
        compile_model=args.compile_model,
    )
    result_json = json.dumps(asdict(result), ensure_ascii=False, indent=2)
    print(result_json)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
