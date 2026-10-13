from __future__ import annotations

from dataclasses import dataclass, asdict
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Literal
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import build_model, MODEL_CONFIGS
from cs336_systems.checkpointing import CheckpointedTransformerLM
import torch
import argparse
import json
from pathlib import Path


# Memory Profiling：统计 autograd 保存的 tensor
@dataclass
class SavedTensorTracker:
    saved_tensor_count: int = 0
    saved_tensor_bytes: int = 0

    def __post_init__(self) -> None:
        self._storage_bytes: dict[tuple[str, int | None, int], int] = {}

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        self.saved_tensor_count += 1
        self.saved_tensor_bytes += tensor.numel() * tensor.element_size()

        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr())
        self._storage_bytes.setdefault(key, storage.nbytes())
        return tensor

    def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @property
    def unique_storage_bytes(self) -> int:
        return sum(self._storage_bytes.values())


# 将 tracker 封装为 context manager
@contextmanager
def track_saved_tensors() -> Iterator[SavedTensorTracker]:
    tracker = SavedTensorTracker()
    with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
        yield tracker


BenchmarkMode = Literal["forward", "forward_backward", "train"]
Precision = Literal["fp32", "bf16"]


# 定义统一的显存实验结果结构
@dataclass(frozen=True)
class MemoryProfileResult:
    model_size: str
    mode: BenchmarkMode
    precision: Precision
    device: str
    batch_size: int
    context_length: int
    warmup_steps: int
    checkpoint_every: int | None
    vocab_size: int

    parameter_mib: float
    optimizer_state_mib: float
    baseline_allocated_mib: float | None
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None
    step_peak_delta_mib: float | None

    saved_tensor_count: int
    saved_tensor_mib: float
    unique_saved_storage_mib: float
    loss: float | None
    snapshot_path: str | None


# 计算参数与优化器状态的常驻显存
def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _bytes_to_mib(byte_count: int) -> float:
    return byte_count / (1024**2)


def parameter_mib(model: torch.nn.Module) -> float:
    total_bytes = sum(_tensor_nbytes(parameter) for parameter in model.parameters())
    return _bytes_to_mib(total_bytes)


def optimizer_state_mib(optimizer: torch.optim.Optimizer) -> float:
    total_bytes = 0

    for parameter_state in optimizer.state.values():
        for value in parameter_state.values():
            if isinstance(value, torch.Tensor):
                total_bytes += _tensor_nbytes(value)
    return total_bytes / (1024**2)


def _autocast_context(device: torch.device, precision: Precision):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _start_memory_history(snapshot_path: Path | None, device: torch.device) -> bool:
    if snapshot_path is None:
        return False
    if device.type != "cuda":
        raise ValueError("CUDA memory snapshot 仅支持 CUDA 设备")

    torch.cuda.memory._record_memory_history(enabled="all", max_entries=100_000)
    return True


def _finish_memory_history(snapshot_path: Path | None, history_enabled: bool) -> None:
    if not history_enabled or snapshot_path is None:
        return

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


# 三种 step 模式与精度上下文
def _run_step(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, input_ids: torch.Tensor, target_ids: torch.Tensor, mode: BenchmarkMode, device: torch.device, precision: Precision
) -> float | None:
    if mode == "forward":
        model.eval()
        with torch.inference_mode(), _autocast_context(device, precision):
            model(input_ids)
        return None

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with _autocast_context(device, precision):
        logits = model(input_ids)

    # softmax 与 reduction 保持 FP32，避免 BF16 累积误差。
    loss = cross_entropy(logits.float(), target_ids)
    loss.backward()

    if mode == "train":
        optimizer.step()

    return loss.item()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# warmup：建立稳态训练显存
def _warmup(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mode: BenchmarkMode,
    device: torch.device,
    precision: Precision,
    warmup_steps: int,
) -> None:
    for _ in range(warmup_steps):
        _run_step(model=model, optimizer=optimizer, input_ids=input_ids, target_ids=target_ids, mode=mode, device=device, precision=precision)

    if mode != "forward":
        optimizer.zero_grad(set_to_none=True)
    _synchronize(device)


# 实现一次完整 profile_memory
def profile_memory(
    *,
    model_size: str,
    mode: BenchmarkMode,
    precision: Precision,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
    warmup_steps: int,
    snapshot_path: Path | None = None,
    checkpoint_every: int | None,
) -> MemoryProfileResult:
    model = build_model(model_size=model_size, vocab_size=vocab_size, context_length=context_length, device=device)

    if checkpoint_every is not None:
        model = CheckpointedTransformerLM(model, checkpoint_every)

    optimizer = AdamW(model.parameters())
    input_ids = torch.randint(vocab_size, (batch_size, context_length), device=device)
    target_ids = torch.randint(vocab_size, (batch_size, context_length), device=device)
    # 先进行 warmup；train 模式会在这里创建 AdamW 状态。
    _warmup(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        target_ids=target_ids,
        mode=mode,
        device=device,
        precision=precision,
        warmup_steps=warmup_steps,
    )

    parameter_bytes = sum(_tensor_nbytes(parameter) for parameter in model.parameters())
    parameter_mib = parameter_bytes / (1024**2)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

        # 此刻包含参数与已物化的 optimizer state。
        baseline_allocated_mib = torch.cuda.memory_allocated(device) / (1024**2)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_allocated_mib = None

    history_enabled = _start_memory_history(snapshot_path, device)

    try:
        # 正式的一次 step：同时记录 autograd 保存的 tensor。
        with track_saved_tensors() as tracker:
            loss = _run_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                target_ids=target_ids,
                mode=mode,
                device=device,
                precision=precision,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
            peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)
            step_peak_delta_mib = peak_allocated_mib - baseline_allocated_mib
        else:
            peak_allocated_mib = None
            peak_reserved_mib = None
            step_peak_delta_mib = None

    finally:
        _finish_memory_history(snapshot_path, history_enabled)

    return MemoryProfileResult(
        model_size=model_size,
        mode=mode,
        precision=precision,
        device=str(device),
        batch_size=batch_size,
        context_length=context_length,
        warmup_steps=warmup_steps,
        checkpoint_every=checkpoint_every,
        vocab_size=vocab_size,
        parameter_mib=parameter_mib,
        optimizer_state_mib=optimizer_state_mib(optimizer),
        baseline_allocated_mib=baseline_allocated_mib,
        peak_allocated_mib=peak_allocated_mib,
        peak_reserved_mib=peak_reserved_mib,
        step_peak_delta_mib=step_peak_delta_mib,
        saved_tensor_count=tracker.saved_tensor_count,
        saved_tensor_mib=tracker.saved_tensor_bytes / (1024**2),
        unique_saved_storage_mib=tracker.unique_storage_bytes / (1024**2),
        loss=loss,
        snapshot_path=str(snapshot_path) if snapshot_path is not None else None,
    )


# 添加命令行参数解析
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--mode", choices=("forward", "forward_backward", "train"), default="train")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/memory_profiling/profile.json"))
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="可选：输出 CUDA memory snapshot 的 pickle 文件",
    )
    parser.add_argument("--checkpoint-every", type=int, default=None, help="每个 checkpoint 分段包含的 Transformer block 数量")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.context_length <= 0 or args.vocab_size <= 0:
        parser.error("batch-size、context-length 和 vocab-size 必须为正整数")

    if args.warmup_steps < 0:
        parser.error("warmup-steps 必须为非负整数")

    if args.precision == "bf16" and torch.device(args.device).type != "cuda":
        parser.error("BF16 autocast 仅支持 CUDA")

    if args.checkpoint_every is not None and args.checkpoint_every <= 0:
        parser.error("checkpoint-every 必须是正整数")
    return args


# 实现 main()：运行剖析并输出 JSON
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_every = args.checkpoint_every

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前 PyTorch 未检测到可用 GPU")

    result = profile_memory(
        model_size=args.model_size,
        mode=args.mode,
        precision=args.precision,
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=device,
        warmup_steps=args.warmup_steps,
        snapshot_path=args.snapshot,
        checkpoint_every=checkpoint_every,
    )

    rendered = json.dumps(asdict(result), ensure_ascii=False, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
