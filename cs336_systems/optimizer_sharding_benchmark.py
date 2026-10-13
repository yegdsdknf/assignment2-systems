"""Compare replicated and sharded AdamW state under distributed training."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import MODEL_CONFIGS, build_model
from cs336_systems.ddp import DDP_IMPLEMENTATIONS
from cs336_systems.sharded_optimizer import ShardedOptimizer


OptimizerImplementation = Literal["replicated_adamw", "sharded_adamw"]
Precision = Literal["fp32", "bf16"]


class SynchronizingModel(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def synchronize_gradient(self) -> None: ...


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_context(device: torch.device, precision: Precision):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _parameter_bytes(module: torch.nn.Module) -> int:
    return sum(_tensor_bytes(parameter) for parameter in module.parameters())


def _gradient_bytes(module: torch.nn.Module) -> int:
    return sum(
        _tensor_bytes(parameter.grad)
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        _tensor_bytes(value)
        for parameter_state in optimizer.state.values()
        for value in parameter_state.values()
        if isinstance(value, torch.Tensor)
    )


def _memory_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }

    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _make_optimizer(
    implementation: OptimizerImplementation,
    parameters,
) -> torch.optim.Optimizer:
    optimizer_kwargs = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
    }
    if implementation == "replicated_adamw":
        return AdamW(parameters, **optimizer_kwargs)
    return ShardedOptimizer(parameters, AdamW, **optimizer_kwargs)


def _train_step(
    *,
    model: SynchronizingModel,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    device: torch.device,
    precision: Precision,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(device, precision):
        logits = model(input_ids)
    loss = cross_entropy(logits.float(), target_ids)
    loss.backward()
    model.synchronize_gradient()
    optimizer.step()


def _worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    torch.set_num_threads(1)
    if args.device == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=10),
    )

    try:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        model = build_model(
            args.model_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            device=device,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        parameter_bytes = _parameter_bytes(model)
        ddp_model = DDP_IMPLEMENTATIONS[args.ddp_implementation](model)
        ddp_model.train()
        optimizer = _make_optimizer(args.implementation, ddp_model.parameters())

        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        after_model_initialization = _memory_snapshot(device)

        local_batch_size = args.global_batch_size // world_size
        torch.manual_seed(args.seed + 1 + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 1 + rank)
        input_ids = torch.randint(
            args.vocab_size,
            (local_batch_size, args.context_length),
            device=device,
        )
        target_ids = torch.randint(
            args.vocab_size,
            (local_batch_size, args.context_length),
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with _autocast_context(device, args.precision):
            logits = ddp_model(input_ids)
        loss = cross_entropy(logits.float(), target_ids)
        loss.backward()
        ddp_model.synchronize_gradient()
        _synchronize(device)
        before_first_optimizer_step = _memory_snapshot(device)
        first_gradient_bytes = _gradient_bytes(ddp_model)
        state_bytes_before_first_step = _optimizer_state_bytes(optimizer)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.step()
        _synchronize(device)
        after_first_optimizer_step = _memory_snapshot(device)
        state_bytes_after_first_step = _optimizer_state_bytes(optimizer)
        state_entry_count = len(optimizer.state)

        for _ in range(args.warmup_steps):
            _train_step(
                model=ddp_model,
                optimizer=optimizer,
                input_ids=input_ids,
                target_ids=target_ids,
                device=device,
                precision=args.precision,
            )
        _synchronize(device)
        dist.barrier()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        step_times_ms: list[float] = []
        for _ in range(args.measurement_steps):
            _synchronize(device)
            started_at = time.perf_counter()
            _train_step(
                model=ddp_model,
                optimizer=optimizer,
                input_ids=input_ids,
                target_ids=target_ids,
                device=device,
                precision=args.precision,
            )
            _synchronize(device)
            step_times_ms.append((time.perf_counter() - started_at) * 1_000)

        measurement_peak = _memory_snapshot(device)
        local_result: dict[str, Any] = {
            "rank": rank,
            "device": str(device),
            "parameter_bytes": parameter_bytes,
            "gradient_bytes_after_first_backward": first_gradient_bytes,
            "optimizer_state_bytes_before_first_step": state_bytes_before_first_step,
            "optimizer_state_bytes_after_first_step": state_bytes_after_first_step,
            "optimizer_state_entry_count": state_entry_count,
            "after_model_initialization": after_model_initialization,
            "before_first_optimizer_step": before_first_optimizer_step,
            "after_first_optimizer_step": after_first_optimizer_step,
            "mean_step_ms": statistics.mean(step_times_ms),
            "std_step_ms": statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0,
            "measurement_peak": measurement_peak,
        }

        all_results: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(all_results, local_result)

        if rank == 0:
            per_rank = [result for result in all_results if result is not None]
            report = {
                "implementation": args.implementation,
                "optimizer_class": AdamW.__name__,
                "ddp_implementation": args.ddp_implementation,
                "device_type": args.device,
                "backend": backend,
                "precision": args.precision,
                "torch_version": str(torch.__version__),
                "world_size": world_size,
                "cpu_threads_per_rank": torch.get_num_threads(),
                "model_size": args.model_size,
                "model_config": asdict(MODEL_CONFIGS[args.model_size]),
                "parameter_count": parameter_count,
                "global_batch_size": args.global_batch_size,
                "local_batch_size": local_batch_size,
                "context_length": args.context_length,
                "vocab_size": args.vocab_size,
                "warmup_steps_after_first_step": args.warmup_steps,
                "measurement_steps": args.measurement_steps,
                "step_includes_zero_grad": True,
                "seed": args.seed,
                "per_rank": per_rank,
                "max_rank_mean_step_ms": max(result["mean_step_ms"] for result in per_rank),
            }
            rendered = json.dumps(report, ensure_ascii=False, indent=2)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as output_file:
                output_file.write(rendered + "\n")
            print(rendered, flush=True)
            print(f"Saved benchmark: {output_path}", flush=True)
    finally:
        dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        choices=("replicated_adamw", "sharded_adamw"),
        required=True,
    )
    parser.add_argument("--ddp-implementation", choices=DDP_IMPLEMENTATIONS, default="overlapping_ddp")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=29_513)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.world_size <= 0:
        parser.error("world-size must be positive")
    if args.global_batch_size <= 0 or args.global_batch_size % args.world_size != 0:
        parser.error("global-batch-size must be positive and divisible by world-size")
    if args.context_length <= 0 or args.vocab_size <= 0:
        parser.error("context-length and vocab-size must be positive")
    if args.warmup_steps < 0 or args.measurement_steps <= 0:
        parser.error("warmup-steps must be non-negative and measurement-steps must be positive")
    if args.precision == "bf16" and args.device != "cuda":
        parser.error("bf16 requires CUDA")
    if args.device == "cuda" and torch.cuda.device_count() < args.world_size:
        parser.error(
            f"requested {args.world_size} CUDA ranks, but only {torch.cuda.device_count()} devices are visible"
        )
    return args


def main() -> None:
    args = _parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = args.output or Path("results/optimizer_sharding") / (
        f"{args.implementation}_{args.device}_{args.model_size}_{timestamp}.json"
    )
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    print(
        f"optimizer={args.implementation}; ddp={args.ddp_implementation}; output={output_path}",
        flush=True,
    )
    mp.spawn(
        _worker,
        args=(args.world_size, args, output_path),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
