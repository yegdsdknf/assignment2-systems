from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from cs336_basics.model import Linear, Embedding


@dataclass
class _ShardInfo:
    # 参数名，用于定位和调试。
    name: str

    # 模型中原始的 Parameter 对象。
    parameter: torch.nn.Parameter

    # 参数分片前的原始形状。
    full_shape: torch.Size

    # 原始参数的元素总数。
    full_numel: int

    # 为均匀分片而 padding 后的元素总数。
    padded_numel: int

    # 每个 rank 持有的分片元素数量。
    shard_numel: int

    # 当前 rank 长期保存的参数分片。
    master_shard: torch.Tensor

    # 下面四项仅在一次异步 all-gather 生命周期中存在。

    # 保留通信输入，防止异步通信完成前临时 tensor 被释放。
    gather_input: torch.Tensor | None=None

    # all-gather 的完整输出 buffer。
    # wait 完成后，parameter.data 会指向它的一个 view。
    gather_output: torch.Tensor | None=None

    # async_op=True 返回的通信句柄。
    gather_work: Any | None=None

    # 本次 gather 使用的通信和计算 dtype。
    gather_dtype: torch.dtype | None=None

    # 当前参数在前向静态执行顺序中的位置。
    forward_index: int = -1

    # 当前参数在反向静态执行顺序中的位置。
    backward_index: int = -1


@dataclass
class _PendingCommunication:
    # 通信操作类型，如 all-gather 或 reduce-scatter。
    kind: str

    # 与该通信操作对应的模型参数。
    parameter: torch.nn.Parameter

    # 通信完成后用于保存结果的输出张量。
    output: torch.Tensor

    # 发起通信时使用的输入缓冲区。
    input_buffer: torch.Tensor

    # 异步通信句柄，用于等待通信完成。
    work: Any

class FSDP(torch.nn.Module):
    def __init__(self, 
                 module: torch.nn.Module, 
                 compute_dtype: torch.dtype|None=None
                 ) -> None:
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError("创建 FSDP 前必须初始化进程组")

        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self._shards_by_id: dict[int, _ShardInfo] = {}
        self._shards_by_name: dict[str, _ShardInfo] = {}
        self._pending: list[_PendingCommunication] = []
        self._pending_parameter_ids: set[int] = set()
        self._hook_handles: list[Any] = []
        # 第一次 forward 尚无执行计划；记录实际调用顺序后供后续迭代使用。
        self._forward_plan: list[_ShardInfo] | None=None
        self._forward_observed: list[_ShardInfo] = []
        # 后续迭代中，下一个将执行的受管理层的位置。
        self._forward_cursor = 0
        # 第一次 backward 尚无执行计划；记录实际调用顺序后供后续迭代使用。
        self._backward_plan: list[tuple[_ShardInfo, torch.dtype]] | None=None
        self._backward_observed: list[tuple[_ShardInfo, torch.dtype]] = []
        self._backward_cursor = 0

        # 保存分片前的参数名称，因为之后 parameter.data 的 shape 会改变
        name_by_id = {
            id(parameter): name
            for name, parameter in self.module.named_parameters()
        }

        # 所有 rank 必须从 rank 0 的同一个完整模型开始。
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter.data, src=0)

        # 为 Linear 和 Embedding 建立 weight shard。
        for _, submodule in self.module.named_modules():
            if not isinstance(submodule, (Linear, Embedding)):
                continue

            parameter = submodule.weight
            parameter_id = id(parameter)

            if parameter_id not in self._shards_by_id:
                info = self._shard_parameter(
                    name = name_by_id[parameter_id],
                    parameter = parameter
                )
                self._shards_by_id[parameter_id] = info
                self._shards_by_name[info.name] = info

                # 需要梯度的参数在梯度累积完成后执行分片处理。
                if parameter.requires_grad:
                    handle = parameter.register_post_accumulate_grad_hook(
                        self._make_sharded_gradient_hook(info)
                    )
                    self._hook_handles.append(handle)

            # 共享参数已经分片时，直接复用已有的分片信息。
            else:
                info = self._shards_by_id[parameter_id]

            # 注册模块级 hook，用于参数的 all-gather 和 reshard。
            self._register_module_hooks(submodule, info)

         # Linear/Embedding 之外的参数保持复制，通过 all-reduce 同步梯度。
        for parameter in self.module.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in self._shards_by_id:
                continue

            handle = parameter.register_post_accumulate_grad_hook(
                self._make_replicated_gradient_hook()
            )
            self._hook_handles.append(handle)

    def forward(self, *args, **kwargs):
        if self._forward_plan is None:
            # 第一次迭代：让现有 hook 按需加载参数，同时记录层的实际调用顺序。
            self._forward_observed.clear()
            output = self.module(*args, **kwargs)

            # collective 的调用顺序必须在各 rank 一致，因此先比较记录结果。
            names = [info.name for info in self._forward_observed]
            names_by_rank = [None] * self.world_size
            dist.all_gather_object(names_by_rank, names)
            if any(other != names for other in names_by_rank):
                raise RuntimeError("各 rank 的 forward 层顺序不同")

            # 保存的是每次调用，不按参数去重；同一层可能被调用多次。
            self._forward_plan = list(self._forward_observed)
            return output

        # 后续迭代：从计划开头重新计数。
        self._forward_cursor = 0
        dtype = self.compute_dtype or torch.float32

        # 在模型开始计算前，提前发起前两层的通信；此处不等待。
        for info in self._forward_plan[:2]:
            self._start_all_gather(info, dtype)

        output = self.module(*args, **kwargs)

        # 防止本次调用比首次记录少；多调用会由 pre-hook 检出。
        if self._forward_cursor != len(self._forward_plan):
            raise RuntimeError("本次 forward 的层调用次数与记录不符")
        return output
    #                参数 shard
    #                    │
    #                    ▼
    #           forward_pre_hook
    #                    │
    #               all-gather
    #                    │
    #                    ▼
    #               完整参数
    #                    │
    #                    ▼
    #                 forward
    #                    │
    #                    ▼
    #           forward_post_hook
    #                    │
    #                 reshard
    #                    │
    #                    ▼
    #                参数 shard
    #                    │
    #                    ▼
    #          backward_pre_hook
    #                    │
    #               all-gather
    #                    │
    #                    ▼
    #               完整参数
    #                    │
    #                    ▼
    #                 backward
    #                    │
    #                    ▼
    #             完整梯度产生
    #                    │
    #                    ▼
    #    post_accumulate_grad_hook
    #                    │
    #             reduce-scatter
    #                    │
    #                    ▼
    #               梯度 shard

    #               module hooks
    #                    ↓
    #        管理“参数什么时候 full / shard”

    #            parameter grad hook
    #                    ↓
    #       管理“梯度什么时候 reduce-scatter”


    def _shard_parameter(
            self,
            *,
            name: str,
            parameter: torch.nn.Parameter
    ) -> _ShardInfo:
        full_shape = parameter.data.shape
        full_numel = parameter.data.numel()

        shard_numel = (
            full_numel + self.world_size - 1    #向上取整
        ) // self.world_size
        padded_numel = shard_numel * self.world_size

        padded = torch.zeros(
            padded_numel,
            device=parameter.device,
            dtype=torch.float32
        )
        # 将完整参数展平并转为 FP32，复制到 padding buffer 的有效区域。
        padded[:full_numel].copy_(
            parameter.data.detach().reshape(-1).to(torch.float32)
        )

        start = self.rank * shard_numel
        master_shard = padded[start : start + shard_numel].clone()

        # 原 Parameter 对象保留，但常驻数据变为本 rank 的 FP32 shard。
        parameter.data = master_shard

        return _ShardInfo(
            name=name,
            parameter=parameter,
            full_shape=full_shape,
            full_numel=full_numel,
            padded_numel=padded_numel,
            shard_numel=shard_numel,
            master_shard=master_shard
        )

    def _register_module_hooks(
        self,
        submodule: torch.nn.Module,
        info: _ShardInfo,
) -> None:

        # forward 执行前恢复完整参数。
        def forward_pre_hook(module, inputs):
            # hook 接口要求接收 module 和 inputs，
            # 但这里实际不需要使用它们。
            del module, inputs
            dtype = self.compute_dtype or torch.float32

            if self._forward_plan is None:
                # 第一次 forward：只记录实际经过了哪个受管理层。
                self._forward_observed.append(info)
            else:
                # 后续 forward：确认当前层就是计划中的下一层。
                index = self._forward_cursor
                if index >= len(self._forward_plan) or self._forward_plan[index] is not info:
                    raise RuntimeError("forward 层顺序与记录不符")
                self._forward_cursor += 1

            # 当前层即将计算，必须在这里等待其完整权重可用。
            # 如果通信尚未预先启动，此方法会现场启动并等待。
            self._wait_all_gather(info, dtype)

        # forward 执行完成后重新分片参数。
        def forward_post_hook(module, inputs, output):
            # 这里同样不需要 module 和 inputs。
            del module, inputs

            # 当前层已经计算完，可以恢复为常驻的参数分片。
            self._release_full_weight(info)

            if self._forward_plan is not None:
                # cursor 已指向下一层，因此 cursor + 1 是“再下一层”。
                # 释放当前层后才启动它，使预取窗口向前移动一格。
                next_index = self._forward_cursor + 1
                if next_index < len(self._forward_plan):
                    dtype = self.compute_dtype or torch.float32
                    self._start_all_gather(self._forward_plan[next_index], dtype)

            # forward hook 需要返回原始输出，保持模块输出不变。
            return output

        # backward 开始前重新恢复完整参数。
        def backward_pre_hook(module, grad_output):
            # 当前逻辑不需要 hook 传入的 module 和 grad_output。
            del module, grad_output

            # Linear backward 需要完整 weight 来计算 grad_input，
            # 因此恢复为 forward 计算使用的低精度 dtype。
            if isinstance(submodule, Linear):
                dtype = self.compute_dtype or torch.float32

            # Embedding 等参考实现的 backward 路径使用 FP32 参数。
            else:
                dtype = torch.float32

            if self._backward_plan is None:
                # 首次 backward：记录 hook 的真实触发顺序。
                self._backward_observed.append((info, dtype))
            else:
                index = self._backward_cursor
                if index >= len(self._backward_plan):
                    raise RuntimeError("backward 调用了计划外的层")

                expected_info, expected_dtype = self._backward_plan[index]
                # _ShardInfo 含有 Tensor，因此用对象身份检查，不比较整个 dataclass。
                if expected_info is not info or expected_dtype != dtype:
                    raise RuntimeError("backward 层顺序或 dtype 与记录不符")

                self._backward_cursor += 1
            
            # 在 backward 真正开始前重新 materialize 完整参数。
            self._wait_all_gather(info, dtype)

            if self._backward_plan is not None:
                # 此时当前层尚未开始反向计算；启动下一层通信以争取重叠。
                next_index = self._backward_cursor
                if next_index < len(self._backward_plan):
                    next_info, next_dtype = self._backward_plan[next_index]
                    self._start_all_gather(next_info, next_dtype)

        # forward 前：恢复完整参数。
        self._hook_handles.append(
            submodule.register_forward_pre_hook(forward_pre_hook)
        )

        # forward 后：重新分片并释放完整参数。
        self._hook_handles.append(
            submodule.register_forward_hook(forward_post_hook)
        )

        # backward 前：重新恢复 backward 所需的完整参数。
        self._hook_handles.append(
            submodule.register_full_backward_pre_hook(backward_pre_hook)
        )


    # 启动异步 all-gather
    @torch.no_grad()
    def _start_all_gather(
        self,
        info: _ShardInfo,
        dtype: torch.dtype
    ) -> None:
        """
        启动当前参数 shard 的异步 all-gather，但不等待完成。

        该函数只负责：
        1. 将 FP32 master shard 转成通信 dtype；
        2. 分配 gather 输出；
        3. 发起 collective；
        4. 保存输入、输出和句柄的引用。

        它不会修改 parameter.data。
        """
        # gather_output 非空表示：
        # - 通信正在进行；或
        # - 通信已经完成，完整 weight 正在被模块使用。
        #
        # 同一个参数不能重复发起 all-gather。
        if info.gather_output is not None:
            if info.gather_dtype != dtype:
                raise RuntimeError(
                    f"{info.name} 已经以 {info.gather_dtype} "
                    f"启动 all-gather，不能改为 {dtype}"
                )
            return
        # master shard 永久保持 FP32。
        # 低精度模式只转换临时通信输入。
        local_shard = info.master_shard.to(
            dtype=dtype
        ).contiguous()

        # all_gather_into_tensor 要求输出能够容纳所有 rank 的 shard。
        gathered = torch.empty(
            info.padded_numel,
            device=local_shard.device,
            dtype=dtype
        )

        # async_op=True 立即返回句柄，为通信与其他计算重叠创造条件。
        work = dist.all_gather_into_tensor(
            gathered,
            local_shard,
            async_op=True
        )

        # 异步操作完成之前必须保留输入和输出 tensor。
        info.gather_input = local_shard
        info.gather_output = gathered
        info.gather_work = work
        info.gather_dtype = dtype


    @torch.no_grad()
    def _wait_all_gather(
        self,
        info: _ShardInfo,
        dtype: torch.dtype 
    ) -> None:
        """
        确保当前参数的 all-gather 完成，并让 parameter.data
        临时指向完整 weight。

        如果此前没有预取，该函数会退化为“立即启动，然后等待”。
        """

        # 没有提前启动时，保持正确性优先，现场发起通信。
        if info.gather_output is None:
            self._start_all_gather(info, dtype=dtype)

        if info.gather_dtype != dtype:
            raise RuntimeError(
                f"{info.name} 的 all-gather dtype 不一致："
                f"启动时为 {info.gather_dtype}，"
                f"等待时请求 {dtype}"
            )

        # 如果通信仍在进行，在真正使用 weight 前等待。
        if info.gather_work is not None:
            info.gather_work.wait()
            info.gather_work = None

            # wait 返回后，通信不再依赖输入 shard 的临时副本。
            info.gather_input = None

        if info.gather_output is None:
            raise RuntimeError(
                f"{info.name} 缺少 all-gather 输出"
            )

        # gather_output 包含 padding。
        # 先截取真实元素，再恢复原始二维或多维 shape。
        full_weight = info.gather_output[
            : info.full_numel
        ].view(info.full_shape)

        # 仅临时替换 data，不创建新的 Parameter。
        info.parameter.data = full_weight


    @torch.no_grad()
    def _release_full_weight(
        self,
        info: _ShardInfo
    ) -> None:
        """
        释放完整 weight，并恢复当前 rank 常驻的 FP32 master shard。
        """

        # 不能释放通信仍在写入的输出 buffer。
        if info.gather_work is not None:
            raise RuntimeError(
                f"{info.name} 的 all-gather 尚未完成"
            )

        # optimizer 应始终看到 FP32 shard，而不是完整 weight。
        info.parameter.data = info.master_shard

        # 删除临时引用后，完整 gather buffer 可以被释放。
        info.gather_input = None
        info.gather_output = None
        info.gather_dtype = None

    
    def _make_sharded_gradient_hook(self, info: _ShardInfo):

        @torch.no_grad()
        def hook(parameter: torch.nn.Parameter) -> None:
            # 本次 backward 没有梯度时，只需释放完整参数。
            if parameter.grad is None:
                self._release_full_weight(info)
                return

            parameter_id = id(parameter)

            # 防止同一参数在上一次异步通信完成前再次启动通信。
            if parameter_id in self._pending_parameter_ids:
                raise RuntimeError(
                    f"参数 {info.name} 在上一次通信结束前再次产生梯度"
                )

            # 将完整梯度展平并转换为 FP32。
            full_gradient = (
                parameter.grad.detach()
                .reshape(-1)
                .to(torch.float32)
            )

            # 检查完整梯度大小是否与参数一致。
            if full_gradient.numel() != info.full_numel:
                raise RuntimeError(
                    f"{info.name} 梯度元素数错误："
                    f"期望 {info.full_numel}，"
                    f"得到 {full_gradient.numel()}"
                )

            # 为均匀 reduce-scatter 创建 padding 后的梯度 buffer。
            padded_gradient = torch.zeros(
                info.padded_numel,
                device=full_gradient.device,
                dtype=torch.float32
            )
            padded_gradient[:info.full_numel].copy_(full_gradient)

            # 当前 rank 用于接收梯度 shard 的 buffer。
            gradient_shard = torch.empty(
                info.shard_numel,
                device=full_gradient.device,
                dtype=torch.float32
            )

            # 对所有 rank 的完整梯度求和，并将结果分片到各 rank。
            work = dist.reduce_scatter_tensor(
                gradient_shard,
                padded_gradient,
                op=dist.ReduceOp.SUM,
                async_op=True
            )

            # 完整梯度已经复制到通信 buffer，可以提前释放。
            parameter.grad = None

            # backward 已不再需要完整参数，恢复为 shard 状态。
            self._release_full_weight(info)

            # 记录该参数当前有未完成的异步通信。
            self._pending_parameter_ids.add(parameter_id)

            # 保存通信相关 buffer 和 work，等待后续统一完成。
            self._pending.append(
                _PendingCommunication(
                    kind="sharded",
                    parameter=parameter,
                    output=gradient_shard,
                    input_buffer=padded_gradient,
                    work=work
                )
            )

        return hook

    def _make_replicated_gradient_hook(self):
        @torch.no_grad()
        def hook(parameter: torch.nn.Parameter) -> None:
            if parameter.grad is None:
                return

            parameter_id = id(parameter)
            if parameter_id in self._pending_parameter_ids:
                raise RuntimeError(
                    "复制参数在上一次通信结束前再次产生梯度"
                )

            gradient = parameter.grad
            work = dist.all_reduce(
                gradient,
                op=dist.ReduceOp.SUM,
                async_op=True
            )

            # 记录未完成的梯度同步。
            self._pending_parameter_ids.add(parameter_id)
            self._pending.append(
                _PendingCommunication(
                    kind="replicated",
                    parameter=parameter,
                    output=gradient,
                    input_buffer=gradient,
                    work=work
                )
            )
        return hook


    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
    # 等待所有异步梯度通信完成。
        for pending in self._pending:
            pending.work.wait()

            if pending.kind == "sharded":
                # reduce-scatter 得到当前 rank 的梯度分片。
                gradient = pending.output

                # SUM 转为各 rank 的平均梯度。
                gradient.div_(self.world_size)

                # 转换成当前参数分片使用的 dtype。
                gradient = gradient.to(
                    dtype=pending.parameter.data.dtype
                )

                # 将梯度分片恢复为参数分片的形状并写回 grad。
                pending.parameter.grad = gradient.view_as(
                    pending.parameter.data
                )

            else:
                # replicated 参数的 all-reduce 直接原地修改 grad。
                if pending.parameter.grad is None:
                    raise RuntimeError(
                        "复制参数的梯度在通信完成前被清除"
                    )

                # SUM 转为平均梯度。
                pending.parameter.grad.div_(self.world_size)

        if self._backward_plan is None:
            # 只传可比较的名称和 dtype；_ShardInfo 留在各 rank 本地。
            signature = [
                (info.name, str(dtype))
                for info, dtype in self._backward_observed
            ]
            signature_by_rank = [None] * self.world_size
            dist.all_gather_object(signature_by_rank, signature)

            if any(other != signature for other in signature_by_rank):
                raise RuntimeError("各 rank 的 backward 层顺序不同")

            # 没发生 backward 时保持 None，下次继续尝试记录。
            if self._backward_observed:
                self._backward_plan = list(self._backward_observed)
        else:
            # 有梯度通信说明本轮执行过 backward；检查计划是否走完。
            if self._pending and self._backward_cursor != len(self._backward_plan):
                raise RuntimeError("本次 backward 的层调用次数与记录不符")

        # finish_gradient_synchronization 是两次 backward 之间的状态边界。
        self._backward_observed.clear()
        self._backward_cursor = 0

        # 当前轮所有异步通信已经结束。
        self._pending.clear()
        self._pending_parameter_ids.clear()


    @torch.no_grad()
    def _gather_master_weight(
        self,
        info: _ShardInfo
    ) -> torch.Tensor:
        local_shard = info.master_shard.contiguous()

        gathered = torch.empty(
            info.padded_numel,
            device=local_shard.device,
            dtype=torch.float32
        )
        dist.all_gather_into_tensor(
            gathered,
            local_shard
        )

        return (
            gathered[: info.full_numel]
            .view(info.full_shape)
            .clone()
        )

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        # 保存所有参数对应的完整权重。
        result: dict[str, torch.Tensor] = {}

        for name, parameter in self.module.named_parameters():
            # 查询该参数是否属于分片参数。
            info = self._shards_by_id.get(id(parameter))

            if info is None:
                # 未分片参数直接复制完整权重。
                result[name] = parameter.detach().clone()
            else:
                # 分片参数通过 all-gather 恢复完整 FP32 权重。
                result[name] = self._gather_master_weight(info)

        return result
    