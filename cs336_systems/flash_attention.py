from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# pytorch,triton 共享 backward 函数
def _flash_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_q_dtype = q.dtype
    original_k_dtype = k.dtype
    original_v_dtype = v.dtype

    # 使用 FP32 重算，提高 BF16/FP16 情况下的稳定性。
    q_float = q.float()
    k_float = k.float()
    v_float = v.float()
    output_float = output.float()
    grad_output_float = grad_output.float()
    logsumexp_float = logsumexp.float()

    scale = 1.0 / math.sqrt(q.shape[-1])

    # S = QK^T / sqrt(d)
    scores = torch.matmul(q_float, k_float.transpose(-2, -1)) * scale

    if is_causal:
        query_indices = torch.arange(q.shape[-2], device=q.device)[:, None]

        key_indices = torch.arange(k.shape[-2], device=q.device)[None, :]

        allowed = query_indices >= key_indices

        scores = torch.where(allowed, scores, -1e6)

    # P = exp(S - L)，不需要再次调用 softmax。
    p = torch.exp(scores - logsumexp_float.unsqueeze(-1)).to(q.device)

    # D_i = sum_d O_id * dO_id
    d_row = (output_float * grad_output_float).sum(
        dim=-1,
        keepdim=True,
    )

    # dV = P^T dO
    grad_v = torch.matmul(p.transpose(-2, -1), grad_output_float)

    # dP = dO V^T
    grad_p = torch.matmul(
        grad_output_float,
        v_float.transpose(-2, -1),
    )

    # dS = P * (dP - D)
    grad_scores = p * (grad_p - d_row)

    # S = QK^T / sqrt(d)
    grad_q = torch.matmul(grad_scores, k_float) * scale

    grad_k = (
        torch.matmul(
            grad_scores.transpose(-2, -1),
            q_float,
        )
        * scale
    )
    return (
        grad_q.to(original_q_dtype),
        grad_k.to(original_k_dtype),
        grad_v.to(original_v_dtype),
    )


# 这里编译的是普通 PyTorch 函数，不是 Triton kernel
compiled_flash_backward = torch.compile(_flash_attention_backward)


class FlashAttention2PyTorch(torch.autograd.Function):
    Q_BLOCK_SIZE = 16
    KV_BLOCK_SIZE = 16

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError("q、k、v 至少需要两个维度")

        if q.shape[:-2] != k.shape[:-2]:
            raise ValueError("q 和 k 的 batch/head 维必须一致")

        if k.shape[:-2] != v.shape[:-2]:
            raise ValueError("k 和 v 的 batch/head 维必须一致")

        if q.shape[-1] != k.shape[-1]:
            raise ValueError("q 和 k 的特征维必须一致")

        if k.shape[-2] != v.shape[-2]:
            raise ValueError("k 和 v 的序列长度必须一致")

        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        d = q.shape[-1]
        d_value = v.shape[-1]

        q_block_size = FlashAttention2PyTorch.Q_BLOCK_SIZE
        kv_block_size = FlashAttention2PyTorch.KV_BLOCK_SIZE

        scale = 1.0 / math.sqrt(d)

        # 在线 softmax 的状态使用 FP32。
        q_float = q.float()
        k_float = k.float()
        v_float = v.float()

        output_float = torch.empty(
            (*q.shape[:-2], n_queries, d_value),
            device=q.device,
            dtype=torch.float32,
        )

        # 保存每个 query row 的 logsumexp，供 backward 重建 P。
        logsumexp = torch.empty(
            (*q.shape[:-2], n_queries),
            device=q.device,
            dtype=torch.float32,
        )

        for query_start in range(
            0,
            n_queries,
            q_block_size,
        ):
            query_end = min(
                query_start + q_block_size,
                n_queries,
            )

            q_i = q_float[..., query_start:query_end, :]

            query_block_length = query_end - query_start

            # m_i：当前已处理 key blocks 的最大 score。
            m_i = torch.full(
                (*q.shape[:-2], query_block_length),
                -torch.inf,
                device=q.device,
                dtype=torch.float32,
            )

            # l_i：经过最大值平移后的指数和。
            l_i = torch.zeros_like(m_i)

            # acc_i：尚未除以 l_i 的输出分子。
            acc_i = torch.zeros(
                (
                    *q.shape[:-2],
                    query_block_length,
                    d_value,
                ),
                device=q.device,
                dtype=torch.float32,
            )

            for key_start in range(
                0,
                n_keys,
                kv_block_size,
            ):
                key_end = min(
                    key_start + kv_block_size,
                    n_keys,
                )

                k_j = k_float[..., key_start:key_end, :]

                v_j = v_float[..., key_start:key_end, :]

                scores = (
                    torch.matmul(
                        q_i,
                        k_j.transpose(-2, -1),
                    )
                    * scale
                )

                if is_causal:
                    query_indices = torch.arange(
                        query_start,
                        query_end,
                        device=q.device,
                    )

                    key_indices = torch.arange(
                        key_start,
                        key_end,
                        device=q.device,
                    )

                    causal_mask = key_indices[None, :] > query_indices[:, None]

                    scores = scores.masked_fill(
                        causal_mask,
                        -torch.inf,
                    )

                block_max = scores.max(dim=-1).values
                m_new = torch.maximum(m_i, block_max)

                # 将旧分块的累加结果重标定到新的最大值。
                alpha = torch.exp(m_i - m_new)

                # 当前 key block 的未归一化 softmax。
                p_ij = torch.exp(scores - m_new.unsqueeze(-1))

                l_new = alpha * l_i + p_ij.sum(dim=-1)

                acc_i = alpha.unsqueeze(-1) * acc_i + torch.matmul(p_ij, v_j)

                m_i = m_new
                l_i = l_new

            output_i = acc_i / l_i.unsqueeze(-1)

            output_float[..., query_start:query_end, :] = output_i

            logsumexp[..., query_start:query_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(
            q,
            k,
            v,
            output_float,
            logsumexp,
        )

        ctx.is_causal = bool(is_causal)
        ctx.scale = scale
        ctx.q_block_size = q_block_size
        ctx.kv_block_size = kv_block_size

        return output_float.to(q.dtype)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        (
            q,
            k,
            v,
            output,
            logsumexp,
        ) = ctx.saved_tensors

        grad_q, grad_k, grad_v = compiled_flash_backward(
            q,
            k,
            v,
            output,
            grad_output.contiguous(),
            logsumexp,
            ctx.is_causal,
        )

        # 若 forward 调用时显式传入了 is_causal，
        # backward 需要为这个非 Tensor 参数返回 None。
        if len(ctx.needs_input_grad) == 4:
            return grad_q, grad_k, grad_v, None

        # 支持 FlashAttention.apply(q, k, v) 的默认参数调用。
        return grad_q, grad_k, grad_v


@triton.jit
def flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    """
    这里的两个 program id 对应 launch grid：
    grid = (
        triton.cdiv(N_QUERIES, Q_TILE_SIZE),
        batch_index
    )
    例如：
    query_tile_index = 3
    batch_index = 2
    Q_TILE_SIZE = 16
    该 program 处理第 2 个 batch 中 Q 的第 48:64 行。
    """

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(
            query_tile_index * Q_TILE_SIZE,
            0,
        ),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # offsets 表示这个 block 在完整矩阵中的“起始坐标”。
    # k,v都要从头遍历，[K_TILE_SIZE, D]取0,0
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(
            query_tile_index * Q_TILE_SIZE,
            0,
        ),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # L 每个 query row 只有一个值，因此是一维 tile：
    l_block_ptr = tl.make_block_ptr(
        base=l_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    # Q tile 在整个 kernel 中只加载一次：
    q_i = tl.load(
        q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )

    # 每一行当前见过的最大 attention score
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    # 每一行当前的 softmax 分母
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    # 尚未除以 softmax 分母的输出分子
    acc_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # 遍历kv tiles
    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        key_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

        causal_allowed = query_offsets[:, None] >= key_offsets[None, :]

        k_j = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

        v_j = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        scores = (
            tl.dot(
                q_i,
                tl.trans(k_j),
            )
            * scale
        )

        if is_causal:
            scores += tl.where(
                causal_allowed,
                0.0,
                -1e6,
            )

        block_max = tl.max(
            scores,
            axis=1,
        )

        m_new = tl.maximum(m_i, block_max)

        alpha = tl.exp(m_i - m_new)

        # 当前 tile 的未归一化概率
        p_ij = tl.exp(scores - m_new[:, None])

        # 更新分母
        l_i = alpha * l_i + tl.sum(p_ij, axis=1)

        # 更新输出分子
        acc_i = alpha[:, None] * acc_i

        acc_i = tl.dot(p_ij.to(v_j.dtype), v_j, acc=acc_i)

        # 更新状态与 pointers
        m_i = m_new

        k_block_ptr = k_block_ptr.advance((K_TILE_SIZE, 0))

        v_block_ptr = v_block_ptr.advance((K_TILE_SIZE, 0))

    output = acc_i / l_i[:, None]

    logsumexp = m_i + tl.log(l_i)

    tl.store(o_block_ptr, output.to(o_block_ptr.type.element_ty), boundary_check=(0, 1))

    tl.store(l_block_ptr, logsumexp, boundary_check=(0,))


class FlashAttention2Triton(torch.autograd.Function):
    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if q.ndim != 3:
            raise ValueError("q 必须为 [B, Nq, D]")

        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("k、v 必须为 [B, Nk, D]")

        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            raise ValueError("q、k、v 必须位于 CUDA")

        if q.device != k.device or q.device != v.device:
            raise ValueError("q、k、v 必须位于同一设备")

        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("q、k、v 的 dtype 必须一致")

        batch_size, n_queries, d = q.shape

        if k.shape[0] != batch_size:
            raise ValueError("q 和 k 的 batch size 不一致")

        if v.shape[0] != batch_size:
            raise ValueError("q 和 v 的 batch size 不一致")

        n_keys = k.shape[1]

        if v.shape[1] != n_keys:
            raise ValueError("k 和 v 的序列长度不一致")

        if k.shape[2] != d:
            raise ValueError("q 和 k 的特征维不一致")

        if v.shape[2] != d:
            raise ValueError("当前 kernel 要求 v 的特征维等于 D")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        output = torch.empty_like(q)

        logsumexp = torch.empty(
            (batch_size, n_queries),
            device=q.device,
            dtype=torch.float32,
        )

        q_tile_size = FlashAttention2Triton.Q_TILE_SIZE

        k_tile_size = FlashAttention2Triton.K_TILE_SIZE

        grid = (
            triton.cdiv(
                n_queries,
                q_tile_size,
            ),
            batch_size,
        )

        scale = 1.0 / math.sqrt(d)

        flash_fwd_kernel[grid](
            q,
            k,
            v,
            output,
            logsumexp,
            # Q strides
            q.stride(0),
            q.stride(1),
            q.stride(2),
            # K strides
            k.stride(0),
            k.stride(1),
            k.stride(2),
            # V strides
            v.stride(0),
            v.stride(1),
            v.stride(2),
            # O strides
            output.stride(0),
            output.stride(1),
            output.stride(2),
            # L strides
            logsumexp.stride(0),
            logsumexp.stride(1),
            # 运行时参数
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=scale,
            # 编译期常量
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            is_causal=is_causal,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            output,
            logsumexp,
        )

        ctx.is_causal = bool(is_causal)

        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        (
            q,
            k,
            v,
            output,
            logsumexp,
        ) = ctx.saved_tensors

        grad_q, grad_k, grad_v = compiled_flash_backward(
            q,
            k,
            v,
            output,
            grad_output.contiguous(),
            logsumexp,
            ctx.is_causal,
        )

        # 若 forward 调用时显式传入了 is_causal，
        # backward 需要为这个非 Tensor 参数返回 None。
        if len(ctx.needs_input_grad) == 4:
            return grad_q, grad_k, grad_v, None

        # 支持 FlashAttention.apply(q, k, v) 的默认参数调用。
        return grad_q, grad_k, grad_v
