# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import warnings

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except (ImportError, OSError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ImportError, OSError) as exc:
    FLASH_ATTN_2_AVAILABLE = False
    if not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            f"FlashAttention unavailable ({exc}); using PyTorch SDPA.",
            RuntimeWarning,
        )

# FLASH_ATTN_3_AVAILABLE = False

__all__ = [
    'flash_attention',
    'attention',
]


def _sdpa_attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
                    q_scale, causal, window_size, dtype):
    """Preserve FlashAttention lengths, scale and bottom-right mask alignment."""
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    target_dtype = v.dtype if v.dtype in half_dtypes else dtype
    q, k, v = (x.transpose(1, 2).to(target_dtype) for x in (q, k, v))
    if q_scale is not None:
        q = q * q_scale
    if q.size(1) != k.size(1):
        repeats = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    if q_lens is None and k_lens is None and not causal and window_size == (-1, -1):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, scale=softmax_scale)
    else:
        out = q.new_zeros((*q.shape[:-1], v.size(-1)))
        for i in range(q.size(0)):
            lq = q.size(2) if q_lens is None else int(q_lens[i])
            lk = k.size(2) if k_lens is None else int(k_lens[i])
            if lq == 0 or lk == 0:
                continue
            mask = None
            if causal or window_size != (-1, -1):
                row = torch.arange(lq, device=q.device)[:, None] + lk - lq
                col = torch.arange(lk, device=q.device)[None, :]
                mask = torch.ones((lq, lk), dtype=torch.bool, device=q.device)
                if causal:
                    mask &= col <= row
                left, right = window_size
                if left >= 0:
                    mask &= col >= row - left
                if right >= 0:
                    mask &= col <= row + right
            out[i:i + 1, :, :lq] = torch.nn.functional.scaled_dot_product_attention(
                q[i:i + 1, :, :lq], k[i:i + 1, :, :lk], v[i:i + 1, :, :lk],
                attn_mask=mask, dropout_p=dropout_p, scale=softmax_scale)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256
    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
        return _sdpa_attention(q, k, v, q_lens, k_lens, dropout_p,
                               softmax_scale, q_scale, causal, window_size, dtype)

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    return flash_attention(
        q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens,
        dropout_p=dropout_p, softmax_scale=softmax_scale, q_scale=q_scale,
        causal=causal, window_size=window_size, deterministic=deterministic,
        dtype=dtype, version=fa_version,
    )
