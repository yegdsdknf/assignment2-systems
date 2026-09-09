import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.checkpointing import CheckpointedTransformerLM


def test_checkpointing_equivalence() -> None:
    torch.manual_seed(0)

    model = BasicsTransformerLM(
        vocab_size=64,
        context_length=8,
        d_model=32,
        num_layers=5,
        num_heads=4,
        d_ff=64,
    )

    input_ids = torch.randint(0, 64, (2, 8))
    target_ids = torch.randint(0, 64, (2, 8))

    # 不执行 optimizer.step()，保证两次前向使用相同参数。
    baseline_logits = model(input_ids)
    baseline_loss = cross_entropy(baseline_logits, target_ids)
    baseline_loss.backward()

    baseline_gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}

    model.zero_grad(set_to_none=True)

    # 5 层按每 2 层一段，能够同时验证尾部不完整分段。
    wrapped_model = CheckpointedTransformerLM(
        model,
        checkpoint_every=2,
    )

    checkpoint_logits = wrapped_model(input_ids)
    checkpoint_loss = cross_entropy(checkpoint_logits, target_ids)
    checkpoint_loss.backward()

    torch.testing.assert_close(
        checkpoint_logits,
        baseline_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        checkpoint_loss,
        baseline_loss,
        rtol=1e-5,
        atol=1e-6,
    )

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad,
            baseline_gradients[name],
            rtol=1e-5,
            atol=1e-6,
        )

    # 同时验证 inference_mode 分支。
    with torch.inference_mode():
        inference_logits = wrapped_model(input_ids)

    torch.testing.assert_close(
        inference_logits,
        baseline_logits,
        rtol=1e-5,
        atol=1e-6,
    )
