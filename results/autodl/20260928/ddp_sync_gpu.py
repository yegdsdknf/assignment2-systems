import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import DDP_IMPLEMENTATIONS

parser = argparse.ArgumentParser()
parser.add_argument("--implementation", choices=("naive_ddp", "flat_gradient_ddp"), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
torch.set_num_threads(1)
device = torch.device(f"cuda:{local_rank}")
dist.init_process_group("nccl")

torch.manual_seed(0)
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=512,
    d_model=2560,
    num_layers=32,
    num_heads=32,
    d_ff=10240,
).to(device)
ddp_model = DDP_IMPLEMENTATIONS[args.implementation](model)
ddp_model.train()
optimizer = AdamW(ddp_model.parameters())

torch.manual_seed(1234 + rank)
inputs = torch.randint(0, 10000, (2, 512), device=device)
targets = torch.randint(0, 10000, (2, 512), device=device)

def train_step():
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = ddp_model(inputs)
    loss = F.cross_entropy(logits.float().reshape(-1, 10000), targets.reshape(-1))
    loss.backward()
    torch.cuda.synchronize(device)
    sync_start = time.perf_counter()

    ddp_model.synchronize_gradient()
    torch.cuda.synchronize(device)
    sync_end = time.perf_counter()

    optimizer.step()
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    return (end - start) * 1000, (sync_end - sync_start) * 1000

for _ in range(5):
    train_step()

dist.barrier(device_ids=[local_rank])
samples = [train_step() for _ in range(20)]
step_ms = [sample[0] for sample in samples]
sync_ms = [sample[1] for sample in samples]
local_result = {
    "rank": rank,
    "mean_step_ms": statistics.mean(step_ms),
    "mean_sync_ms": statistics.mean(sync_ms),
    "std_sync_ms": statistics.stdev(sync_ms),
    "sync_fraction": statistics.mean(sync_ms) / statistics.mean(step_ms),
}

all_results = [None] * dist.get_world_size()
dist.all_gather_object(all_results, local_result)
if rank == 0:
    report = {
        "implementation": args.implementation,
        "device": "cuda",
        "backend": "nccl",
        "model_size": "xl",
        "precision": "bf16",
        "world_size": 2,
        "global_batch_size": 4,
        "context_length": 512,
        "warmup_steps": 5,
        "measurement_steps": 20,
        "sync_boundary": "after_backward_to_after_synchronize_gradient",
        "per_rank": all_results,
    }
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report, indent=2), flush=True)

dist.destroy_process_group()
