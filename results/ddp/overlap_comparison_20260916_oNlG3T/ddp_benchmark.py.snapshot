import argparse
import json
import os
from datetime import datetime, timedelta, UTC
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.ddp import DDP_IMPLEMENTATIONS


def worker(rank: int, world_size: int, implementation: str, output_path: Path) -> None:
    # 避免两个进程各自启用大量 CPU 线程，相互争抢资源。
    torch.set_num_threads(1)

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60)
    )

    try:
        warmup_steps = 5
        measured_steps = 20
        global_batch_size = 32

        assert global_batch_size % world_size == 0
        local_batch_size = global_batch_size // world_size

        # 模型明确位于 CPU，不使用本机 GPU。
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
        ).cpu()

        # 构造时的参数广播不计入稳定训练步。
        ddp_model = DDP_IMPLEMENTATIONS[implementation](model)
        ddp_model.train()

        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss(reduction="mean")

        # 每个 rank 使用不同的局部数据，数据生成不进入计时区间。
        torch.manual_seed(1234 + rank)
        inputs = torch.randn(local_batch_size, 128)
        targets = torch.randn(local_batch_size, 32)

        def train_step() -> tuple[float, float]:
            # 本基线明确不把清空梯度计入 step_ms。
            optimizer.zero_grad(set_to_none=True)

            t0 = perf_counter()

            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            t1 = perf_counter()

            ddp_model.synchronize_gradient()

            t2 = perf_counter()

            optimizer.step()

            t3 = perf_counter()

            step_ms = (t3 - t0) * 1000
            sync_ms = (t2 - t1) * 1000
            return step_ms, sync_ms

        # 预热也进行完整训练，但不纳入统计。
        for _ in range(warmup_steps):
            train_step()

        # 仅在测量开始前对齐，不在每个阶段额外插入 barrier。
        dist.barrier()

        step_times = []
        sync_times = []

        for _ in range(measured_steps):
            step_ms, sync_ms = train_step()
            step_times.append(step_ms)
            sync_times.append(sync_ms)

        mean_step_ms = sum(step_times) / measured_steps
        mean_sync_ms = sum(sync_times) / measured_steps

        local_stats = {
            "rank": rank,
            "mean_step_ms": mean_step_ms,
            "mean_sync_ms": mean_sync_ms,
            "sync_fraction": mean_sync_ms / mean_step_ms
        }

        # 统计通信位于测量循环之外，所有 rank 都必须参与。
        all_stats = [None] * world_size
        dist.all_gather_object(all_stats, local_stats)

        if rank == 0:
            gradients = [
                parameter.grad
                for parameter in ddp_model.parameters()
                if parameter.grad is not None
            ]

            report = {
                "implementation": implementation,
                "implementation_class": type(ddp_model).__name__,
                "device": "cpu",
                "backend": "gloo",
                "dtype": "float32",
                "torch_version": str(torch.__version__),
                "world_size": world_size,
                "cpu_threads_per_rank": torch.get_num_threads(),
                "global_batch_size": global_batch_size,
                "local_batch_size": local_batch_size,
                "warmup_steps": warmup_steps,
                "measured_steps": measured_steps,
                "step_includes_zero_grad": False,
                "gradient_tensor_count": len(gradients),
                # 逻辑梯度大小，不等于通信链路实际传输字节数。
                "gradient_bytes_per_rank": sum(
                    grad.numel() * grad.element_size()
                    for grad in gradients
                ),
                "per_rank": all_stats,
                "max_rank_mean_step_ms": max(
                    stats["mean_step_ms"] for stats in all_stats
                ),
            }

            result_json = json.dumps(report, indent=2)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as output_file:
                output_file.write(result_json + "\n")
            print(result_json, flush=True)
            print(f"Saved benchmark: {output_path}", flush=True)

    finally:
        dist.destroy_process_group()

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare explicit DDP implementations on CPU/Gloo.")
    parser.add_argument("--implementation", choices=DDP_IMPLEMENTATIONS, required=True)
    parser.add_argument("--output", type=Path, help="New result file; existing files are never overwritten.")
    args = parser.parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = args.output or Path("results/ddp") / f"{args.implementation}_cpu_gloo_{timestamp}.json"
    if output_path.exists():
        parser.error(f"Output already exists: {output_path}")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    world_size = 2
    print(f"DDP implementation: {args.implementation}; output: {output_path}", flush=True)
    mp.spawn(
        worker,
        args=(world_size, args.implementation, output_path),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
