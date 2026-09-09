"""混合精度数值正确性检查：累积误差与 FP32/BF16 模型输出对比。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.nn_utils import cross_entropy
from cs336_systems.benchmark import MODEL_CONFIGS, build_model


def error_metrics(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """返回标量结果相对 FP32 参考值的误差。"""
    absolute_error = (value - reference).abs().item()
    relative_error = absolute_error / (reference.abs().item() + 1e-12)
    return {
        "value": value.item(),
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }


def accumulation_experiment(device: torch.device, num_elements: int) -> dict[str, object]:
    """比较 FP16 输入下的 FP16 与 FP32 累积误差。"""
    # 使用正数避免参考和接近零，导致相对误差失去意义。
    x_fp32 = torch.linspace(1e-4, 1e-2, num_elements, device=device, dtype=torch.float32)
    x_fp16 = x_fp32.to(torch.float16)

    reference = x_fp32.sum(dtype=torch.float32)
    fp16_accumulation = x_fp16.sum(dtype=torch.float16)
    fp32_accumulation = x_fp16.sum(dtype=torch.float32)

    return {
        "num_elements": num_elements,
        "reference_fp32_input_fp32_accumulation": reference.item(),
        "fp16_input_fp16_accumulation": error_metrics(fp16_accumulation, reference),
        "fp16_input_fp32_accumulation": error_metrics(fp32_accumulation, reference),
    }


def model_precision_experiment(*, device: torch.device, model_size: int, vocab_size: int, batch_size: int, context_length: int, seed: int) -> dict[str, object]:
    """在同一组 FP32 参数和固定输入下比较 FP32/BF16 输出。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = build_model(
        model_size,
        vocab_size=vocab_size,
        context_length=context_length,
        device=device,
    )
    model.eval()

    input_ids = torch.randint(
        vocab_size,
        (batch_size, context_length),
        device=device,
    )
    target_ids = torch.randint(
        vocab_size,
        (batch_size, context_length),
        device=device,
    )

    with torch.inference_mode():
        logits_fp32 = model(input_ids)
        loss_fp32 = cross_entropy(logits_fp32.float(), target_ids)

        # 参数仍为 FP32；只有适合的 CUDA 算子在 BF16 下执行。
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_bf16 = model(input_ids)
        # 自定义 loss 的 softmax/reduction 保持 FP32。
        loss_bf16 = cross_entropy(logits_bf16.float(), target_ids)

    difference = logits_bf16.float() - logits_fp32.float()
    reference_norm = torch.linalg.vector_norm(logits_fp32.float()).item()
    relative_l2_error = torch.linalg.vector_norm(difference).item() / (reference_norm + 1e-12)

    return {
        "model_size": model_size,
        "batch_size": batch_size,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "fp32_logits_finite": torch.isfinite(logits_fp32).all().item(),
        "bf16_logits_finite": torch.isfinite(logits_bf16).all().item(),
        "fp32_loss_finite": torch.isfinite(loss_fp32).item(),
        "bf16_loss_finite": torch.isfinite(loss_bf16).item(),
        "bf16_logits_dtype": str(logits_bf16.dtype),
        "fp32_loss": loss_fp32.item(),
        "bf16_loss": loss_bf16.item(),
        "loss_absolute_error": (loss_bf16 - loss_fp32).abs().item(),
        "logits_max_absolute_error": difference.abs().max().item(),
        "logits_mean_absolute_error": difference.abs().mean().item(),
        "logits_relative_l2_error": relative_l2_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--num-elements", type=int, default=262_144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mixed_precision/numerical_validation.json"),
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.context_length <= 0:
        parser.error("batch-size 和 context-length 必须为正整数")
    if args.vocab_size <= 0 or args.num_elements <= 0:
        parser.error("vocab-size 和 num-elements 必须为正整数")

    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError("该数值验证脚本要求 CUDA，以检查 BF16 autocast。")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 未检测到可用 CUDA GPU。")

    result = {
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "accumulation_experiment": accumulation_experiment(device, args.num_elements),
        "model_precision_experiment": model_precision_experiment(
            device=device,
            model_size=args.model_size,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            context_length=args.context_length,
            seed=args.seed,
        ),
    }

    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    print(result_json)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result_json + "\n", encoding="utf-8")

    checks = result["model_precision_experiment"]
    if not all(
        (
            checks["fp32_logits_finite"],
            checks["bf16_logits_finite"],
            checks["fp32_loss_finite"],
            checks["bf16_loss_finite"],
        )
    ):
        raise RuntimeError("数值验证失败：检测到 NaN 或 Inf。")


if __name__ == "__main__":
    main()
