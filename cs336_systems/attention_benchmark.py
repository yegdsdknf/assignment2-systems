from dataclasses import dataclass, asdict
from typing import Literal
from pathlib import Path
from cs336_basics.model import scaled_dot_product_attention

import torch
import statistics
import time
import argparse
import gc
import json

DEFAULT_D_MODELS = (16, 32, 64, 128)
DEFAULT_SEQUENCE_LENGTHS = (256, 1024, 4096, 8192, 16384)


@dataclass(frozen=True)
class AttentionBenchmarkResult:
    status: Literal["success", "oom"]
    implementation: str
    device: str
    batch_size: int
    d_model: int
    sequence_length: int
    warmup_steps: int
    measurement_steps: int

    forward_mean_ms: float | None
    forward_std_ms: float | None
    backward_mean_ms: float | None
    backward_std_ms: float | None

    allocated_before_forward_mib: float | None
    allocated_before_backward_mib: float | None
    forward_allocation_delta_mib: float | None
    peak_allocated_mib: float | None

    oom_stage: str | None
    error: str | None


def benchmark_attention_case(
    *,
    attention_fn,
    implementation: str,
    d_model: int,
    sequence_length: int,
    device: torch.device,
    batch_size: int = 8,
    warmup_steps: int = 5,
    measurement_steps: int = 100,
) -> AttentionBenchmarkResult:
    mib = 1024 * 1024

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def is_oom(error: RuntimeError) -> bool:
        return isinstance(error, torch.cuda.OutOfMemoryError) or ("out of memory" in str(error).lower())

    def current_allocated() -> float | None:
        if device.type != "cuda":
            return None
        return torch.cuda.memory_allocated(device) / mib

    def peak_allocated() -> float | None:
        if device.type != "cuda":
            return None
        return torch.cuda.max_memory_allocated(device) / mib

    def clear_grads() -> None:
        q.grad = None
        k.grad = None
        v.grad = None

    def stats(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        return statistics.fmean(values), statistics.pstdev(values)

    forward_times: list[float] = []
    backward_times: list[float] = []

    allocated_before_forward: float | None = None
    allocated_before_backward: float | None = None
    forward_allocation_delta: float | None = None

    def make_result(
        status: Literal["success", "oom"],
        oom_stage: str | None = None,
        error: str | None = None,
    ) -> AttentionBenchmarkResult:
        forward_mean, forward_std = stats(forward_times)
        backward_mean, backward_std = stats(backward_times)

        return AttentionBenchmarkResult(
            status=status,
            implementation=implementation,
            device=str(device),
            batch_size=batch_size,
            d_model=d_model,
            sequence_length=sequence_length,
            warmup_steps=warmup_steps,
            measurement_steps=measurement_steps,
            forward_mean_ms=forward_mean,
            forward_std_ms=forward_std,
            backward_mean_ms=backward_mean,
            backward_std_ms=backward_std,
            allocated_before_forward_mib=allocated_before_forward,
            allocated_before_backward_mib=allocated_before_backward,
            forward_allocation_delta_mib=forward_allocation_delta,
            peak_allocated_mib=peak_allocated(),
            oom_stage=oom_stage,
            error=error,
        )

    try:
        q = torch.randn(
            batch_size,
            sequence_length,
            d_model,
            device=device,
            requires_grad=True,
        )
        k = torch.randn(
            batch_size,
            sequence_length,
            d_model,
            device=device,
            requires_grad=True,
        )
        v = torch.randn(
            batch_size,
            sequence_length,
            d_model,
            device=device,
            requires_grad=True,
        )
        grad_output = torch.randn_like(q, requires_grad=False)
    except RuntimeError as error:
        if is_oom(error):
            return make_result("oom", "input_allocation", str(error))
        raise

    try:
        for _ in range(warmup_steps):
            output = attention_fn(q, k, v)
            synchronize()

            output.backward(grad_output)
            synchronize()

            clear_grads()
            del output
    except RuntimeError as error:
        if is_oom(error):
            return make_result("oom", "warmup", str(error))
        raise

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    allocated_before_forward = current_allocated()

    try:
        for iteration in range(measurement_steps):
            synchronize()
            start = time.perf_counter()

            output = attention_fn(q, k, v)

            synchronize()
            forward_times.append((time.perf_counter() - start) * 1000)

            del output
    except RuntimeError as error:
        if is_oom(error):
            return make_result("oom", "forward", str(error))
        raise

    try:
        for iteration in range(measurement_steps):
            # 每次反向传播都重新构造计算图。
            output = attention_fn(q, k, v)
            synchronize()

            if iteration == 0:
                allocated_before_backward = current_allocated()

                if allocated_before_forward is not None and allocated_before_backward is not None:
                    forward_allocation_delta = allocated_before_backward - allocated_before_forward

            start = time.perf_counter()

            output.backward(grad_output)

            synchronize()
            backward_times.append((time.perf_counter() - start) * 1000)

            clear_grads()
            del output
    except RuntimeError as error:
        if is_oom(error):
            return make_result("oom", "backward", str(error))
        raise

    return make_result("success")


def make_attention_fn(implementation: str):
    if implementation == "pytorch":
        return scaled_dot_product_attention

    if implementation == "compiled":
        return torch.compile(scaled_dot_product_attention, dynamic=False)
    raise ValueError(f"未知 attention 实现：{implementation}")


def run_attention_sweep(
    *, d_models: list[int], sequence_lengths: list[int], device: torch.device, batch_size: int, warmup_steps: int, measurement_steps: int, output_path: Path, implementation: str
) -> list[dict]:
    results: list[dict] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    attention_fn = make_attention_fn(implementation)

    for d_model in d_models:
        for sequence_length in sequence_lengths:
            result = benchmark_attention_case(
                attention_fn=attention_fn,
                implementation=implementation,
                d_model=d_model,
                sequence_length=sequence_length,
                device=device,
                batch_size=batch_size,
                warmup_steps=warmup_steps,
                measurement_steps=measurement_steps,
            )

            result_dict = asdict(result)
            results.append(result_dict)

            print(json.dumps(result_dict, indent=2))

            # 每完成一组就保存，避免后续 OOM 或中断导致已有结果丢失
            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch attention forward/backward benchmark")

    parser.add_argument("--d-models", nargs="+", type=int, default=list[DEFAULT_D_MODELS], help="attention embedding dimensions")
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=list[DEFAULT_SEQUENCE_LENGTHS], help="sequence lengths")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, default=Path("results/attention/pytorch_attention.json"))
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.90,
    )
    parser.add_argument("--implementation", choices=("pytorch", "compiled"), default="pytorch")

    args = parser.parse_args()

    if any(value <= 0 for value in args.d_models):
        parser.error("--d-models 中的值必须为正数")

    if any(value <= 0 for value in args.sequence_lengths):
        parser.error("--sequence-lengths 中的值必须为正数")

    if args.batch_size <= 0:
        parser.error("--batch-size 必须为正数")

    if args.warmup_steps < 0:
        parser.error("--warmup-steps 不能为负数")

    if args.measurement_steps <= 0:
        parser.error("--measurement-steps 必须为正数")

    if not 0 < args.cuda_memory_fraction <= 1:
        parser.error("--cuda-memory-fraction 必须在 (0, 1] 范围内")

    device = torch.device(args.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("请求了 CUDA，但当前环境不可用")

        # 将未指定索引的 cuda 规范化为 cuda:0。
        if device.index is None:
            device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )

        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device,
        )

    results = run_attention_sweep(
        d_models=args.d_models,
        sequence_lengths=args.sequence_lengths,
        device=device,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        output_path=args.output,
        implementation=args.implementation,
    )

    success_count = sum(result["status"] == "success" for result in results)
    oom_count = sum(result["status"] == "oom" for result in results)

    print(f"完成 {len(results)} 组测试：{success_count} 组成功，{oom_count} 组 OOM")
    print(f"结果已保存到：{args.output}")


if __name__ == "__main__":
    main()
