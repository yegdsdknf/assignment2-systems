import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
device = torch.device(f"cuda:{local_rank}")

results = []
for size_mb in (1, 10, 100, 1000):
    # float32 每元素 4 字节；这里 MB 按 1,000,000 字节计算。
    tensor = torch.zeros(size_mb * 250_000, dtype=torch.float32, device=device)

    for _ in range(5):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(device)

    local_ms = []
    for _ in range(20):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize(device)
        local_ms.append((time.perf_counter() - start) * 1000)

    all_ranks = [None] * dist.get_world_size()
    dist.all_gather_object(all_ranks, local_ms)
    if rank == 0:
        step_ms = [max(samples[i] for samples in all_ranks) for i in range(20)]
        results.append({
            "size_mb": size_mb,
            "world_size": 2,
            "warmup_steps": 5,
            "measurement_steps": 20,
            "mean_ms": statistics.mean(step_ms),
            "std_ms": statistics.stdev(step_ms),
            "per_rank_mean_ms": [statistics.mean(samples) for samples in all_ranks],
        })
    del tensor

if rank == 0:
    output = Path("/root/autodl-tmp/cs336-results") / (
        f"allreduce_2gpu_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)

dist.destroy_process_group()
