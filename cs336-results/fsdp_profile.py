import argparse
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from cs336_basics.model import BasicsTransformerLM, Embedding, Linear
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import MODEL_CONFIGS
from cs336_systems.fsdp import FSDP


@contextmanager
def nvtx(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def mark_layer_compute(fsdp_model):
    # 在 FSDP 构造之后注册：标记层实际开始执行的位置。
    for name, module in fsdp_model.named_modules():
        if not isinstance(module, (Embedding, Linear)):
            continue

        def before(_module, _inputs, label=name):
            torch.cuda.nvtx.range_push(f"layer_compute/{label}")

        def after(_module, _inputs, _output):
            torch.cuda.nvtx.range_pop()

        module.register_forward_pre_hook(before)
        module.register_forward_hook(after)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=list(MODEL_CONFIGS), default="xl")
    parser.add_argument("--batch-per-rank", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    dist.init_process_group("nccl", device_id=device)

    try:
        config = MODEL_CONFIGS[args.model_size]
        model = BasicsTransformerLM(
            vocab_size=10_000,
            context_length=args.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
        ).to(device)
        model = FSDP(model, compute_dtype=torch.bfloat16)
        mark_layer_compute(model)
        optimizer = AdamW(model.parameters(), lr=1e-4)

        input_ids = torch.randint(
            0, 10_000, (args.batch_per_rank, args.context_length), device=device
        )

        for step in range(args.steps):
            with nvtx(f"step/{step}"):
                optimizer.zero_grad(set_to_none=True)
                with nvtx("forward"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = model(input_ids)
                    loss = output.float().mean()
                with nvtx("backward_and_gradient_sync"):
                    loss.backward()
                    model.finish_gradient_synchronization()
                with nvtx("optimizer_step"):
                    optimizer.step()
                torch.cuda.synchronize(device)

            print(
                f"rank={local_rank} step={step} "
                f"loss={loss.item():.6f} "
                f"peak_allocated_gib={torch.cuda.max_memory_allocated(device) / 2**30:.2f}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
