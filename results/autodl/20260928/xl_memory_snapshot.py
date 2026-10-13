import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("forward", "train"), required=True)
parser.add_argument("--context-length", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

if args.output.exists():
    parser.error(f"Output already exists: {args.output}")

torch.set_num_threads(1)
torch.manual_seed(0)
device = torch.device("cuda:0")

model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=args.context_length,
    d_model=2560,
    num_layers=32,
    num_heads=32,
    d_ff=10240,
).to(device)
model.train(args.mode == "train")
optimizer = AdamW(model.parameters()) if args.mode == "train" else None

inputs = torch.randint(0, 10000, (4, args.context_length), device=device)
targets = torch.randint(0, 10000, (4, args.context_length), device=device)

def step():
    if args.mode == "forward":
        with torch.inference_mode():
            model(inputs)
        return

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits.float().reshape(-1, 10000), targets.reshape(-1))
    loss.backward()
    optimizer.step()

# 先完成一次预热；训练模式也由此建立 AdamW 状态。
step()
torch.cuda.synchronize(device)
torch.cuda.reset_peak_memory_stats(device)

torch.cuda.memory._record_memory_history(max_entries=100000)
try:
    step()
    torch.cuda.synchronize(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._dump_snapshot(str(args.output))
finally:
    torch.cuda.memory._record_memory_history(enabled=None)

print(
    f"Saved: {args.output}; "
    f"size_mib={args.output.stat().st_size / 1024**2:.1f}; "
    f"peak_allocated_gib={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}",
    flush=True,
)
