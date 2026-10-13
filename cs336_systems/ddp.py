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


class FlatGradientDDP(NaiveDDP):
    """保留现有集中同步实现，复用相同的初始化与 forward。"""

    @torch.no_grad()
    def synchronize_gradient(self) -> None:
        world_size = dist.get_world_size()

        parameters = [
           p
           for p in self.module.parameters()
           if p.grad is not None
        ]

        if not parameters:
            return

        flat_grad = torch.cat([
            p.grad.reshape(-1)
            for p in parameters]
        )

        dist.all_reduce(
            flat_grad,
            op=dist.ReduceOp.SUM
        )

        flat_grad /= world_size

        offset = 0

        for p in parameters:
            numul = p.grad.numel()

            p.grad.copy_(
                flat_grad[
                    offset:offset+numul
                ].view_as(p.grad)
            )

            offset += numul


class OverlappingDDP(NaiveDDP):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__(module)

        # 保存已经启动、但可能尚未完成的异步通信
        self.pending_works = []

        # 为每个需要梯度的参数注册 hook
        for parameter in self.module.parameters():
            if not parameter.requires_grad:
                continue

            parameter.register_post_accumulate_grad_hook(
                self._gradient_hook
            )

    @torch.no_grad()
    def _gradient_hook(self, parameter: torch.nn.Parameter) -> None:
        """
        某个 parameter 的梯度计算并累积到 parameter.grad 后，
        立即启动异步 all_reduce。
        """

        if parameter.grad is None:
            return

        work = dist.all_reduce(
            parameter.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )

        self.pending_works.append(
            (parameter, work)
        )

    @torch.no_grad()
    def synchronize_gradient(self):
        """
        等待 backward 过程中已经启动的所有异步通信完成，
        然后计算平均梯度。
        """
        world_size = dist.get_world_size()

        for parameter, work in self.pending_works:
            # 等待该 parameter 的 all_reduce 完成
            work.wait()

            parameter.grad.div_(world_size)

        self.pending_works.clear()



DDP_IMPLEMENTATIONS = {
    "naive_ddp": NaiveDDP,
    "flat_gradient_ddp": FlatGradientDDP,
    "overlapping_ddp": OverlappingDDP,
}
