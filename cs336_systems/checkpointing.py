from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


class CheckpointedTransformerLM(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, checkpoint_every: int = 1) -> None:
        super().__init__()

        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every 必须是正整数")

        # 注册原模型，使参数仍然由 wrapper.parameters() 管理
        self.model = model
        self.checkpoint_every = checkpoint_every

    def _run_layer_range(
        self,
        x: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        for layer_index in range(start, end):
            x = self.model.layers[layer_index](x)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # embedding 不放入 checkpoint 区间
        x = self.model.token_embeddings(input_ids)

        num_layers = len(self.model.layers)

        for start in range(0, num_layers, self.checkpoint_every):
            end = min(start + self.checkpoint_every, num_layers)

            def run_chunk(
                chunk_input: torch.Tensor,
                start: int = start,
                end: int = end,
            ) -> torch.Tensor:
                return self._run_layer_range(chunk_input, start, end)

            if torch.is_grad_enabled():
                # 非 reentrant 版本可以正确处理 Transformer 参数梯度
                x = checkpoint(run_chunk, x, use_reentrant=False)
            else:
                # inference_mode 下无需 checkpoint
                x = run_chunk(x)

        x = self.model.ln_final(x)
        logits = self.model.lm_head(x)
        return logits
