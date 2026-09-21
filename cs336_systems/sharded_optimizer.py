import torch
import torch.distributed as dist
from collections.abc import Iterable
from typing import Any

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, 
                 params: Iterable[torch.Tensor | dict[str, Any]],
                 optimizer_cls: type[torch.optim.Optimizer],
                 **kwargs: Any,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError(
                "创建 ShardedOptimizer 前必须初始化进程组"
            )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = dict(kwargs)

        # 所有 rank 都保存完整参数列表和相同的 owner 映射。
        self._all_params: list[torch.Tensor] = []
        self._owner_by_id: dict[int, int] = {}

        # 用参数元素数量平衡各 rank 的优化器状态。
        self._rank_loads = [0] * self.world_size
        self._rank_parameter_counts = [0] * self.world_size

        self._pending_local_groups = []
        self._local_optimizer = None

        parameter_groups = self._normalize_parameter_groups(params)

        super().__init__(parameter_groups, dict(kwargs))

        if not self._all_params:
            raise ValueError("参数列表不能为空")

        # closure 需要每个 rank 都执行底层 optimizer.step()。
        # 因此初始分片必须保证每个 rank 至少拥有一个参数。
        if any(
            count == 0
            for count in self._rank_parameter_counts
        ):
            raise ValueError(
                "参数数量不足，无法保证每个 rank "
                "至少拥有一个优化器参数"
            )

        self._local_optimizer = optimizer_cls(
            self._pending_local_groups,
            **self._optimizer_kwargs
        )

        # 让外层 optimizer.state 反映本 rank 实际保存的状态。
        self.state = self._local_optimizer.state

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        group = dict(param_group)
        group["params"] = list(group["params"])

        # 外层保留完整参数组和完整参数视图。
        super().add_param_group(group)

        # 使用父类规范化并补充 defaults 后的参数组。
        full_group = self.param_groups[-1]
        local_group = self._partition_parameter_group(full_group)

        if local_group is None:
            return

        if self._local_optimizer is None:
            self._pending_local_groups.append(local_group)
        else:
            self._local_optimizer.add_param_group(local_group)


    def step(self, closure=None, **kwargs):
        loss = self._local_optimizer.step(
            closure=closure,
            **kwargs
        )    

        # 所有 rank 必须执行相同顺序、相同 src 的 broadcast
        with torch.no_grad():
            for parameter in self._all_params:
                owner = self._owner_by_id[id(parameter)]
                dist.broadcast(parameter, src=owner)

        return loss
       

    @staticmethod
    def _normalize_parameter_groups(
        params: Iterable[torch.Tensor | dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if isinstance(params, torch.Tensor):
            materialized: list[Any] = [params]
        else:
            materialized = list(params)

        if not materialized:
            raise ValueError("参数列表不能为空")

        first_item = materialized[0]

        if isinstance(first_item, dict):
            groups: list[dict[str, Any]] = []
            for raw_group in materialized:
                if not isinstance(raw_group, dict):
                    raise TypeError(
                        "参数列表不能混合 Tensor 和参数组"
                    )

                if "params" not in raw_group:
                    raise ValueError(
                        "参数组必须包含 params 字段"
                    )

                group = dict(raw_group)
                group["params"] = list(group["params"])
                groups.append(group)

            return groups

        if any(
            isinstance(item, dict)
            for item in materialized
        ):
            raise TypeError(
                "参数列表不能混合 Tensor 和参数组"
            )

        return [{"params": materialized}]

    def _choose_owner(
            self,
            parameter: torch.Tensor
    ) -> int:
        # 优先选择累计参数元素最少的 rank。
        # 元素数相同时，再比较参数个数和 rank，
        # 从而保证确定性。
        owner = min(
            range(self.world_size),
            key=lambda rank: (
                self._rank_loads[rank],
                self._rank_parameter_counts[rank],
                rank,
            ),
        )

        self._rank_loads[owner] += parameter.numel()
        self._rank_parameter_counts[owner] += 1

        return owner

    def _partition_parameter_group(
            self,
            parameter_group: dict[str, Any],
    ) -> dict[str, Any] | None:
        group = dict(parameter_group)

        if "params" not in group:
            raise ValueError(
                "参数组必须包含 params 字段"
            )

        parameters = list(group["params"])

        seen_in_group: set[int] = set()

        for parameter in parameters:
            if not isinstance(parameter, torch.Tensor):
                raise TypeError("优化器参数必须是 Tensor 或 Parameter")

            parameter_id = id(parameter)

            if parameter_id in seen_in_group:
                raise ValueError("同一个参数在参数组中出现了多次")
            # 跨 parameter group 重复
            if parameter_id in self._owner_by_id:
                raise ValueError("同一个参数不能属于多个参数组")

            seen_in_group.add(parameter_id)
        # 先分配大参数，使各 rank 的状态大小更均衡。
        indexed_parameters = list(enumerate(parameters))
        indexed_parameters.sort(
            key=lambda item: (
                -item[1].numel(),
                item[0]
            )
        )

        owners: dict[int, int] = {}

        for _, parameter in indexed_parameters:
            owners[id(parameter)] = self._choose_owner(
                parameter
            )

        local_parameters: list[torch.Tensor] = []

        # 参数和 collective 顺序保持调用者提供的原始顺序。
        for parameter in parameters:
            parameter_id = id(parameter)
            owner = owners[parameter_id]

            self._all_params.append(parameter)
            self._owner_by_id[parameter_id] = owner

            if owner == self.rank:
                local_parameters.append(parameter)

        if not local_parameters:
            return None

        group["params"] = local_parameters
        return group
