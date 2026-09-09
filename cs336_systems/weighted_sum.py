from __future__ import annotations
import triton
import torch
import triton.language as tl


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,
    output_ptr,
    x_stride_row,
    x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 用 FP32 累加，提高数值稳定性。
    accumulator = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        ).to(tl.float32)

        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        ).to(tl.float32)

        accumulator += tl.sum(
            row * weight[None, :],
            axis=1,
        )

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    tl.store(
        output_block_ptr,
        accumulator,
        boundary_check=(0,),
    )


# def weighted_sum(
#         x: torch.Tensor,
#         weight: torch.Tensor
# ) -> torch.Tensor:
#     if not x.is_cuda or not weight.is_cuda:
#         raise ValueError("weighted_sum 只支持 CUDA tensor")
#     if x.ndim < 1:
#         raise ValueError("x 至少需要有一个维度")

#     if weight.ndim != 1:
#         raise ValueError("weight 必须是一维 tensor")

#     d = x.shape[-1]

#     if weight.shape[0] != d:
#         raise ValueError(
#             f"weight 长度必须等于 x 的最后一维："
#             f"{weight.shape[0]} != {d}"
#         )

#     # 统一为二维 [NUM_ROWS, D]，并保证指针访问连续。
#     input_shape = x.shape
#     x_2d = x.reshape(-1, d).contiguous()
#     weight_1d = weight.contiguous()

#     num_rows = x_2d.shape[0]
#     output = torch.empty(
#         (num_rows,),
#         device=x.device,
#         dtype=torch.float32,
#     )

#     rows_tile_size = 16
#     d_tile_size = max(
#         1,
#         triton.next_power_of_2(d) // 16,
#     )

#     grid = (
#         triton.cdiv(num_rows, rows_tile_size),
#     )

#     weighted_sum_fwd[grid](
#         x_2d,
#         weight_1d,
#         output,
#         x_2d.stride(0),
#         x_2d.stride(1),
#         weight_1d.stride(0),
#         output.stride(0),
#         NUM_ROWS=num_rows,
#         D=d,
#         ROWS_TILE_SIZE=rows_tile_size,
#         D_TILE_SIZE=d_tile_size,
#     )

#     return output.reshape(input_shape[:-1])


# # 与 PDF 示例中的命名保持一致。
# f_weightedsum = weighted_sum


@triton.jit
def weighted_sum_backward(
    x_ptr,
    weight_ptr,  # Input
    grad_output_ptr,  # Grad input
    grad_x_ptr,
    partial_grad_weight_ptr,  # Grad outputs
    stride_xr,
    stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr,
    stride_gxd,
    stride_gwb,
    stride_gwd,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)
    # Inputs
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(
            NUM_ROWS,
            D,
        ),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(
            NUM_ROWS,
            D,
        ),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(
            n_row_tiles,
            D,
        ),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    # 上游梯度只与 row tile 有关，不需要在 D 循环中重复加载。
    grad_output = tl.load(
        grad_output_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
    )

    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )

        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        # grad_x[i, j] = grad_output[i] * weight[j]
        grad_x_row = grad_output[:, None] * weight[None, :]

        tl.store(
            grad_x_block_ptr,
            grad_x_row,
            boundary_check=(0, 1),
        )

        # 当前 row tile 对 grad_weight 的局部贡献。
        partial_grad_weight = tl.sum(
            row * grad_output[:, None],
            axis=0,
            keep_dims=True,
        )

        tl.store(
            partial_grad_weight_block_ptr,
            partial_grad_weight,
            boundary_check=(1,),
        )

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))


class WeightSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if not x.is_cuda or not weight.is_cuda:
            raise ValueError("WeightedSum 只支持 CUDA tensor")

        if x.ndim < 1:
            raise ValueError("x 至少需要有一个维度")

        if weight.ndim != 1:
            raise ValueError("weight 必须是一维 tensor")
        d = x.shape[-1]

        if weight.shape[0] != d:
            raise ValueError(f"weight 长度必须等于 x 的最后一维：{weight.shape[0]} != {d}")

        if x.device != weight.device:
            raise ValueError("x 和 weight 必须位于同一设备")

        input_shape = x.shape

        # Triton kernel 统一把输入看成 [NUM_ROWS, D]。
        x_2d = x.reshape(-1, d).contiguous()
        weight_1d = weight.contiguous()

        num_rows = x_2d.shape[0]

        row_tile_size = 16
        d_tile_size = max(1, triton.next_power_of_2(d) // 16)

        # forward kernel 使用 FP32 accumulator。
        output = torch.empty((num_rows,), device=x.device, dtype=torch.float32)

        grid = (triton.cdiv(num_rows, row_tile_size),)

        weighted_sum_fwd[grid](
            x_2d,
            weight_1d,
            output,
            x_2d.stride(0),
            x_2d.stride(1),
            weight_1d.stride(0),
            output.stride(0),
            NUM_ROWS=num_rows,
            D=d,
            ROWS_TILE_SIZE=row_tile_size,
            D_TILE_SIZE=d_tile_size,
        )

        # Tensor 应使用 save_for_backward 保存。
        ctx.save_for_backward(x_2d, weight_1d)

        # 普通 Python 元数据可以直接保存在 ctx 上。
        ctx.input_shape = input_shape
        ctx.rows_tile_size = row_tile_size
        ctx.d_tile_size = d_tile_size

        return output.reshape(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_2d, weight = ctx.saved_tensors

        input_shape = ctx.input_shape
        rows_tile_size = ctx.rows_tile_size
        d_tile_size = ctx.d_tile_size

        num_rows, d = x_2d.shape

        # Autograd 传来的梯度不保证连续。
        grad_output_1d = grad_output.reshape(-1).contiguous()

        n_row_tiles = triton.cdiv(num_rows, rows_tile_size)

        grad_x = torch.empty_like(x_2d)

        # 每个 Triton program 写入一份局部 weight 梯度。
        partial_grad_weight = torch.empty((n_row_tiles, d), device=x_2d.device, dtype=x_2d.dtype)

        grid = (n_row_tiles,)

        weighted_sum_backward[grid](
            x_2d,
            weight,
            grad_output_1d,
            grad_x,
            partial_grad_weight,
            x_2d.stride(0),
            x_2d.stride(1),
            weight.stride(0),
            grad_output_1d.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            NUM_ROWS=num_rows,
            D=d,
            ROWS_TILE_SIZE=rows_tile_size,
            D_TILE_SIZE=d_tile_size,
        )

        # 汇总所有 row tile 对 weight 梯度的贡献。
        grad_weight = partial_grad_weight.sum(dim=0)

        # backward 返回值必须与 forward 的输入一一对应：
        # forward(ctx, x, weight) -> backward 返回 grad_x, grad_weight。
        return (grad_x.reshape(input_shape), grad_weight)


f_weightedsum = WeightSumFunc.apply


if __name__ == "__main__":
    # 测试代码
    x = torch.randn(100, 37, device="cuda", requires_grad=True)

    weight = torch.randn(37, device="cuda", requires_grad=True)

    grad_output = torch.randn(100, device="cuda")

    actual_y = f_weightedsum(x, weight)

    actual_grad_x, actual_grad_weight = torch.autograd.grad(actual_y, (x, weight), grad_output)

    expect_y = (x * weight).sum(dim=-1)

    expect_grad_x, expect_grad_weight = torch.autograd.grad(expect_y, (x, weight), grad_output)

    torch.testing.assert_close(actual_y, expect_y)
    torch.testing.assert_close(actual_grad_x, expect_grad_x)
    torch.testing.assert_close(actual_grad_weight, expect_grad_weight)

    print("WeightedSum forward/backward PASS")
