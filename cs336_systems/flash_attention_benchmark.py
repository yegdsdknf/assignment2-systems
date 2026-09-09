from __future__ import annotations

import argparse
import gc
import json
import math
from itertools import product
from pathlib import Path
from typing import Callable

import torch
import triton

from cs336_systems.flash_attention import FlashAttention2Triton


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

MODES = (
    "forward",
    "backward",
    "forward_backward",
)

def pytorch_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])

    scores = (torch.matmul(q, k.transpose(-2, -1)) * scale)

    if is_causal:
        q_positions = torch.arange(
            q.shape[-2],
            device=q.device,
        )[:, None]

        k_position = torch.arange(
            k.shape[-2],
            device=k.device,
        )[None, :]

        allowed = q_positions >= k_position

        scores = scores.masked_fill(
            ~allowed,
            -1e6,
        )

    probabilities = torch.softmax(
        scores,
        dim=-1,
    )

    return torch.matmul(probabilities, v)


def triton_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,        
) -> torch.Tensor:
    return FlashAttention2Triton.apply(
        q,
        k,
        v,
        is_causal,
    )

# 单个 mode 的计时
def benchmark_mode(
        implementation: Callable,
        mode: str,
        sequence_length: int,
        d_model: int,
        dtype: torch.dtype,
        warmup_ms: int,
        rep_ms: int,
) -> float:
    q = torch.randn(
        1,  #bs
        sequence_length,
        d_model,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    k = torch.randn_like(
        q,
        requires_grad=True,
    )

    v = torch.randn_like(
        q,
        requires_grad=True
    )

    grad_output = torch.randn_like(q)

    if mode == "forward":
        def operation():
            return implementation(q, k, v, True)

    elif mode == "backward":
        # 在计时前构建一次 graph，计时中只执行 backward。
        output = implementation(q, k, v, True)

        def operation():
            return torch.autograd.grad(
                output,
                (q, k, v),
                grad_outputs=grad_output,
                retain_graph=True,
            )

    elif mode == "forward_backward":
        def operation():
            output = implementation(q, k, v, True)

            return torch.autograd.grad(
                output,
                (q, k, v),
                grad_outputs=grad_output
            )

    else:
        raise ValueError(f"未知 benchmark mode：{mode}")

    latency_ms = triton.testing.do_bench(
        operation,
        warmup=warmup_ms,
        rep=rep_ms,
        return_mode="median"
    )

    return float(latency_ms)


# OOM 与错误捕获
def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def run_one_case(
    implementation_name: str,
    implementation: Callable,
    mode: str,
    sequence_length: int,
    d_model: int,
    dtype_name: str,
    warmup_ms: int,
    rep_ms: int,
) -> dict:
    dtype = DTYPES[dtype_name]

    result = {
        "implementation": implementation_name,
        "mode": mode,
        "batch_size": 1,
        "sequence_length": sequence_length,
        "d_model": d_model,
        "dtype": dtype_name,
        "is_causal": True,
        "latency_ms": None,
        "status": "pending",
        "error": None,
    }

    cleanup_cuda()

    try:
        result["latency_ms"] = benchmark_mode(
            implementation=implementation,
            mode=mode,
            sequence_length=sequence_length,
            d_model=d_model,
            dtype=dtype,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
        result["status"] = "ok"

    except torch.cuda.OutOfMemoryError as error:
        result["status"] = "oom"
        result["error"] = str(error)

    except RuntimeError as error:
        message = str(error)

        if "out of memory" in message.lower():
            result["status"] = "oom"
        else:
            result["status"] = "error"

        result["error"] = message

    finally:
        cleanup_cuda()

    return result


FULL_SEQUENCE_LENGTHS = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]

FULL_D_MODELS = [
    16,
    32,
    64,
    128,
]

FULL_DTYPE_NAMES = [
    "bfloat16",
    "float32",
]


SMOKE_CONFIGS = [
    (128, 64, "float32"),
    (256, 64, "bfloat16"),
]


def save_results(
        results: list[dict],
        output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8"
    )


def run_benchmark(
        configs: list[tuple[int, int, str]],
        output_path: Path,
        warmup_ms: int,
        rep_ms: int,
) -> list[dict]:
    implementations = {
        "pytorch": pytorch_attention,
        "triton": triton_attention,
    }

    results: list[dict] = []

    for sequence_length, d_model, dtype_name in configs:
        for implementation_name, implementation in implementations.items():
            for mode in MODES:
                print(
                    f"{implementation_name=} "
                    f"{mode=} "
                    f"{sequence_length=} "
                    f"{d_model=} "
                    f"{dtype_name=}",
                    flush=True,
                )

                result = run_one_case(
                    implementation_name=implementation_name,
                    implementation=implementation,
                    mode=mode,
                    sequence_length=sequence_length,
                    d_model=d_model,
                    dtype_name=dtype_name,
                    warmup_ms=warmup_ms,
                    rep_ms=rep_ms,
                )

                results.append(result)

                # 每完成一项就保存，避免长 sweep 中途失败后丢失结果。
                save_results(
                    results,
                    output_path,
                )

                print(
                    result,
                    flush=True,
                )

    return results


# 用于计算多个序列的笛卡尔积，也就是枚举所有可能的组合。
def make_full_configs() -> list[tuple[int, int, str]]:
    return list(
        product(
            FULL_SEQUENCE_LENGTHS,
            FULL_D_MODELS,
            FULL_DTYPE_NAMES,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # store_true 的含义是：参数出现时存储 True；默认值是 False。
    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/flash_attention/benchmark.json"
        )
    )

    parser.add_argument(
        "--warmup-ms",  # 预热
        type=int,
        default=25,
    )

    parser.add_argument(
        "--rep-ms", #正式测量
        type=int,
        default=100,
    )

    parser.add_argument(
        "--memory-fraction",   #限制显存使用
        type=float,
        default=0.90,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU")

    torch.cuda.set_per_process_memory_fraction(
        args.memory_fraction,
    )

    configs = (
        SMOKE_CONFIGS
        if args.smoke
        else make_full_configs()
    )

    run_benchmark(
        configs=configs,
        output_path=args.output,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms
    )

if __name__ == "__main__":
    main()

