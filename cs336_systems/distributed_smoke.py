import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from cs336_systems.ddp import NaiveDDP
from copy import deepcopy

def worker(rank: int, world_size: int) -> None:
    # 两个进程使用同一联络地址，但各自具有不同的 rank。
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30)
    )

    try:
        # 故意创建不同初值
        torch.manual_seed(42 + rank)

        model = torch.nn.Linear(4, 3, bias=True).cpu()
        model.bias.requires_grad_(False)

        # 必须 clone，否则广播会同时改变你保存的“旧值”。
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        wrapped = NaiveDDP(model)
        reference_model = deepcopy(wrapped.module)

        # 检查源 rank 不变，接收 rank 改变
        if rank == 0:
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(
                    parameter.detach(), before[name], rtol=0, atol=0
                )
        else:
            assert not torch.equal(model.weight.detach(), before["weight"])
            assert not torch.equal(model.bias.detach(), before["bias"])

        assert model.bias.requires_grad is False

        # 检查所有 rank 的参数完全一致
        for name, parameter in model.named_parameters():
            value = parameter.detach()
            gathered = [
                torch.empty_like(value)
                for _ in range(world_size)
            ]

            dist.all_gather(gathered, value)

            for other in gathered:
                torch.testing.assert_close(
                    value, other, rtol=0, atol=0
                )

            print(f"rank={rank}, parameter={name}, sync=passed", flush=True)

        # 检查 forward
        with torch.no_grad():
            x = torch.ones(2, 4)
            output = wrapped(x)

            torch.testing.assert_close(
                output, model(x), rtol=0, atol=0
            )

            gathered_outputs = [
                torch.empty_like(output)
                for _ in range(world_size)
            ]
            dist.all_gather(gathered_outputs, output)

            for other in gathered_outputs:
                torch.testing.assert_close(
                    output, other, rtol=0, atol=0
                )

        print(f"rank={rank}, model sync and forward=passed", flush=True)

        # 验证梯度平均等价于单模型使用完整 batch。
        torch.manual_seed(123)

        global_input = torch.randn(8, 4)
        global_target = torch.randn(8, 3)

        assert global_input.shape[0] % world_size == 0
        local_batch_size = global_input.shape[0] // world_size

        start = rank * local_batch_size
        end = start + local_batch_size

        local_input = global_input[start:end]
        local_target = global_target[start:end]

        loss_fn = torch.nn.MSELoss(reduction="mean")

        ddp_optimizer = torch.optim.SGD(
            wrapped.parameters(),
            lr=0.1
        )

        reference_optimizer = torch.optim.SGD(
            reference_model.parameters(),
            lr=0.1
        )

        # 参考路径：单模型处理完整 batch。
        reference_optimizer.zero_grad(set_to_none=True)

        reference_output = reference_model(global_input)
        reference_loss = loss_fn(reference_output, global_target)
        reference_loss.backward()

        # DDP 路径：每个 rank 只处理自己的数据分片。
        ddp_optimizer.zero_grad(set_to_none=True)

        local_output = wrapped(local_input)
        local_loss = loss_fn(local_output, local_target)
        local_loss.backward()

        # backward 完成后、optimizer step 之前同步并平均梯度。
        wrapped.synchronize_gradient()

        # 先验证梯度。
        for ddp_parameter, reference_parameter in zip(
            wrapped.parameters(), reference_model.parameters()
        ):
            if reference_parameter.grad is None:
                assert ddp_parameter.grad is None
                continue

            torch.testing.assert_close(
                ddp_parameter.grad,
                reference_parameter.grad,
                rtol=1e-5,
                atol=1e-6
            )

        # 再分别更新参数。
        ddp_optimizer.step()
        reference_optimizer.step()

        # 更新后的参数也应一致。
        for ddp_parameter, reference_parameter in zip(
            wrapped.parameters(), reference_model.parameters()
        ):
            torch.testing.assert_close(
                ddp_parameter,
                reference_parameter,
                rtol=1e-5,
                atol=1e-6
            )

        print(f"rank={rank}, gradient averaging and optimizer step=passed",
        flush=True,
        )

        # 显式使用 CPU；两个进程分别创建自己的张量。
        values = [1.0, 2.0] if rank == 0 else [9.0, 9.0]
        tensor = torch.tensor(
            values,
            dtype=torch.float32,
            device="cpu"
        )

        # clone 保存独立副本，避免原地广播覆盖“广播前”的值。
        before = tensor.clone()
        print(f"rank={rank}, before={before.tolist()}", flush=True)

        # 所有 rank 都必须调用；src 只指定数据来源。
        dist.broadcast(tensor, src=0)

        expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
        torch.testing.assert_close(
            tensor,
            expected,
            rtol=0,
            atol=0,
        )
        print(
            f"rank={rank}, after={tensor.tolist()}, check=passed",
            flush=True,
        )
    finally:
        # 正常结束或断言失败，都清理已初始化的进程组。
        dist.destroy_process_group()

def main() -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

    world_size = 2

    # spawn 自动把进程编号作为 worker 的第一个参数。
    mp.spawn(
        worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

    print("双进程广播验证通过。", flush=True)




# 子进程会重新导入模块，因此启动逻辑必须放在入口保护内。
if __name__ == "__main__":
    main()