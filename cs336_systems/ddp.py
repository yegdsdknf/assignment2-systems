import torch
import torch.distributed as dist

class NaiveDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError("创建 NaiveDDP 前必须初始化进程组")

        # 注册原模型，不复制或替换其 Parameter 对象。
        self.module = module

        # 初始化同步不属于模型求导过程。
        with torch.no_grad():
            for parameter in self.module.parameters():
                # 所有 rank 都执行；包括 requires_grad=False 的参数。
                dist.broadcast(parameter, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def synchronize_gradient(self) -> None:
        world_size = dist.get_world_size()

        for parameter in self.module.parameters():
            if parameter.grad is None:
                continue

            # 执行后，每个 rank 都得到所有 rank 梯度之和。
            dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM
            )

            # 等量数据分片 + mean loss 时，应取梯度平均值。
            parameter.grad.div_(world_size)
            