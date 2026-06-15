from __future__ import annotations

# Portions of this optional Wan fast-kernel helper module are adapted from
# Sparse-VideoGen (Apache-2.0, Copyright 2024 MIT HAN Lab):
# https://github.com/svg-project/Sparse-VideoGen
# The code here has been modified for this repository's Diffusers-based Wan
# runtime integration, backend selection, and fallback behavior.

import os
import sys
import warnings
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
    _TRITON_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - runtime-dependent
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = exc


_FAST_QK_NORM_ENABLED = False
_FAST_ROTARY_ENABLED = False
_FAST_KERNELS_MODULE = None
_FAST_KERNELS_IMPORT_ATTEMPTED = False
_FAST_KERNELS_IMPORT_ERROR: Optional[Exception] = None


def flatten_if_batched(*tensors):
    if not tensors:
        raise ValueError("At least one tensor must be provided.")

    first = tensors[0]
    if first.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D tensors, got ndim={first.ndim}")

    if first.ndim == 3:
        batch_size = first.shape[0]
        seq_len = first.shape[1]
        for tensor in tensors:
            if tensor.shape[:2] != (batch_size, seq_len):
                raise ValueError("All tensors must share batch and sequence dimensions.")
        return [tensor.reshape(-1, tensor.shape[-1]).contiguous() for tensor in tensors], True, batch_size

    return [tensor.contiguous() for tensor in tensors], False, None


def _prepare_batchwise_vector(x: torch.Tensor, batch_size: int) -> Optional[torch.Tensor]:
    if x.ndim == 3:
        if x.shape[1] != 1:
            return None
        x = x[:, 0, :]
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 2:
        return None

    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1)
    if x.shape[0] != batch_size:
        return None
    return x.contiguous()


def set_fast_qk_norm_enabled(enabled: bool) -> None:
    global _FAST_QK_NORM_ENABLED
    _FAST_QK_NORM_ENABLED = bool(enabled)


def _candidate_fast_kernel_build_dirs() -> list[str]:
    candidates = []

    env_path = os.environ.get("SINKATTENTION_FAST_KERNELS_BUILD")
    if env_path:
        candidates.append(env_path)

    unique_candidates = []
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized not in unique_candidates:
            unique_candidates.append(normalized)
    return unique_candidates


def _try_import_fast_kernels():
    global _FAST_KERNELS_MODULE
    global _FAST_KERNELS_IMPORT_ATTEMPTED
    global _FAST_KERNELS_IMPORT_ERROR

    if _FAST_KERNELS_IMPORT_ATTEMPTED:
        return _FAST_KERNELS_MODULE

    _FAST_KERNELS_IMPORT_ATTEMPTED = True
    last_error = None
    for build_dir in _candidate_fast_kernel_build_dirs():
        if not os.path.isdir(build_dir):
            continue
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)
        try:
            import _kernels as kernels

            _FAST_KERNELS_MODULE = kernels
            _FAST_KERNELS_IMPORT_ERROR = None
            return _FAST_KERNELS_MODULE
        except Exception as exc:  # pragma: no cover - runtime-dependent
            last_error = exc

    _FAST_KERNELS_IMPORT_ERROR = last_error
    return None


def set_fast_rotary_enabled(enabled: bool) -> None:
    global _FAST_ROTARY_ENABLED
    _FAST_ROTARY_ENABLED = bool(enabled)


def get_fast_rotary_backend() -> str:
    if not _FAST_ROTARY_ENABLED:
        return "disabled"
    if _try_import_fast_kernels() is not None:
        return "cuda_ext"
    if _HAS_TRITON:
        return "triton"
    return "torch_real"


def _split_rotary_freqs(freqs) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(freqs, (tuple, list)) and len(freqs) == 2:
        return freqs[0], freqs[1]
    if isinstance(freqs, torch.Tensor) and torch.is_complex(freqs):
        return freqs.real, freqs.imag
    raise TypeError(f"Unsupported rotary embedding type: {type(freqs)!r}")


def _apply_wan_tuple_rotary_emb(hidden_states: torch.Tensor, freqs) -> torch.Tensor:
    freqs_cos, freqs_sin = _split_rotary_freqs(freqs)

    if hidden_states.ndim != 4 or freqs_cos.ndim != 4 or freqs_sin.ndim != 4:
        raise TypeError(
            "Wan tuple rotary expects hidden_states [B, L/H, H/L, D] and freqs [1, L, 1, D]."
        )

    if hidden_states.shape[1] == freqs_cos.shape[1]:
        aligned_cos = freqs_cos
        aligned_sin = freqs_sin
    elif hidden_states.shape[2] == freqs_cos.shape[1]:
        aligned_cos = freqs_cos.transpose(1, 2)
        aligned_sin = freqs_sin.transpose(1, 2)
    else:
        raise ValueError(
            f"Cannot align Wan tuple rotary freqs {tuple(freqs_cos.shape)} "
            f"with hidden states {tuple(hidden_states.shape)}."
        )

    x1, x2 = hidden_states.float().unflatten(-1, (-1, 2)).unbind(-1)
    cos = aligned_cos[..., 0::2].to(torch.float32)
    sin = aligned_sin[..., 1::2].to(torch.float32)

    out = torch.empty_like(hidden_states, dtype=torch.float32)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.to(hidden_states.dtype)


def _reference_apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    if isinstance(freqs, (tuple, list)):
        return _apply_wan_tuple_rotary_emb(hidden_states, freqs)
    dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
    x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
    return x_out.type_as(hidden_states)


def _torch_real_apply_rotary_emb(hidden_states: torch.Tensor, freqs) -> torch.Tensor:
    if isinstance(freqs, (tuple, list)):
        return _apply_wan_tuple_rotary_emb(hidden_states, freqs)
    freqs_real, freqs_imag = _split_rotary_freqs(freqs)

    pairs = hidden_states.float().unflatten(3, (-1, 2))
    output = torch.empty_like(pairs, dtype=torch.float32)
    output[..., 0] = pairs[..., 0] * freqs_real - pairs[..., 1] * freqs_imag
    output[..., 1] = pairs[..., 0] * freqs_imag + pairs[..., 1] * freqs_real
    return output.flatten(3, 4).to(hidden_states.dtype)


def _fast_cuda_ext_apply_rotary_qk(query: torch.Tensor, key: torch.Tensor, freqs) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _try_import_fast_kernels()
    if kernels is None:
        raise RuntimeError(f"Fast Wan kernels are unavailable: {_FAST_KERNELS_IMPORT_ERROR!r}")

    freqs_real, freqs_imag = _split_rotary_freqs(freqs)
    if freqs_real.ndim == 4 and freqs_real.shape[:2] == (1, 1):
        freqs_real = freqs_real[0, 0]
        freqs_imag = freqs_imag[0, 0]

    query_out = query.contiguous()
    key_out = key.contiguous()
    kernels.apply_qk_rope_inplace_cossin_complex(
        query_out,
        key_out,
        freqs_real.contiguous().to(torch.float32),
        freqs_imag.contiguous().to(torch.float32),
        0,
    )
    return query_out, key_out


def apply_fast_rotary_qk(query: torch.Tensor, key: torch.Tensor, freqs) -> tuple[torch.Tensor, torch.Tensor]:
    if freqs is None:
        return query, key
    if not _FAST_ROTARY_ENABLED:
        return _reference_apply_rotary_emb(query, freqs), _reference_apply_rotary_emb(key, freqs)

    if _try_import_fast_kernels() is not None:
        try:
            return _fast_cuda_ext_apply_rotary_qk(query, key, freqs)
        except Exception:
            pass

    if _HAS_TRITON:
        try:
            return triton_rotary_qk_forward(query, key, freqs)
        except Exception:
            pass

    return _torch_real_apply_rotary_emb(query, freqs), _torch_real_apply_rotary_emb(key, freqs)


if _HAS_TRITON:

    @triton.jit
    def _layer_norm_param_fwd_fused(
        x_ptr,
        y_ptr,
        w_ptr,
        b_ptr,
        mean_ptr,
        rstd_ptr,
        x_stride,
        y_stride,
        hidden_dim: tl.constexpr,
        hidden_dim_padded: tl.constexpr,
        eps,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, hidden_dim_padded)
        mask = cols < hidden_dim

        x_offsets = x_ptr + rows[:, None] * x_stride + cols[None, :]
        y_offsets = y_ptr + rows[:, None] * y_stride + cols[None, :]

        x = tl.load(x_offsets, mask=mask[None, :], other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1, keep_dims=True) / hidden_dim
        var = tl.sum((x - mean) * (x - mean), axis=1, keep_dims=True) / hidden_dim
        rstd = 1 / tl.sqrt(var + eps)

        tl.store(mean_ptr + rows, tl.reshape(mean, (BLOCK_M,)))
        tl.store(rstd_ptr + rows, tl.reshape(rstd, (BLOCK_M,)))

        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = ((x - mean) * rstd) * w + b
        tl.store(y_offsets, y.to(y_ptr.type.element_ty), mask=mask[None, :])

    @triton.jit
    def _layer_norm_noparam_fwd_fused(
        x_ptr,
        y_ptr,
        mean_ptr,
        rstd_ptr,
        x_stride,
        y_stride,
        hidden_dim: tl.constexpr,
        hidden_dim_padded: tl.constexpr,
        eps,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, hidden_dim_padded)
        mask = cols < hidden_dim

        x_offsets = x_ptr + rows[:, None] * x_stride + cols[None, :]
        y_offsets = y_ptr + rows[:, None] * y_stride + cols[None, :]

        x = tl.load(x_offsets, mask=mask[None, :], other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1, keep_dims=True) / hidden_dim
        var = tl.sum((x - mean) * (x - mean), axis=1, keep_dims=True) / hidden_dim
        rstd = 1 / tl.sqrt(var + eps)

        tl.store(mean_ptr + rows, tl.reshape(mean, (BLOCK_M,)))
        tl.store(rstd_ptr + rows, tl.reshape(rstd, (BLOCK_M,)))

        y = (x - mean) * rstd
        tl.store(y_offsets, y.to(y_ptr.type.element_ty), mask=mask[None, :])

    @triton.jit
    def _rms_norm_fwd_fused(
        x_ptr,
        y_ptr,
        w_ptr,
        rstd_ptr,
        x_stride,
        y_stride,
        total_rows: tl.constexpr,
        hidden_dim: tl.constexpr,
        hidden_dim_padded: tl.constexpr,
        eps,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, hidden_dim_padded)

        row_mask = rows < total_rows
        col_mask = cols < hidden_dim
        mask = row_mask[:, None] & col_mask[None, :]

        x_offsets = x_ptr + rows[:, None] * x_stride + cols[None, :]
        y_offsets = y_ptr + rows[:, None] * y_stride + cols[None, :]

        x = tl.load(x_offsets, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / hidden_dim
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(rstd_ptr + rows, rstd, mask=row_mask)
        rstd = tl.reshape(rstd, (BLOCK_M, 1))

        w = tl.load(w_ptr + cols, mask=col_mask, other=1.0).to(tl.float32)
        y = x * rstd * w
        tl.store(y_offsets, y.to(y_ptr.type.element_ty), mask=mask)

    @triton.jit
    def _modulate_shift_fwd_fused(
        x_ptr,
        scale_ptr,
        shift_ptr,
        y_ptr,
        x_stride_row,
        x_stride_col,
        scale_stride_batch,
        scale_stride_col,
        shift_stride_batch,
        shift_stride_col,
        y_stride_row,
        y_stride_col,
        rows_per_batch: tl.constexpr,
        total_rows: tl.constexpr,
        hidden_dim: tl.constexpr,
        hidden_dim_padded: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, hidden_dim_padded)

        row_mask = rows < total_rows
        col_mask = cols < hidden_dim
        mask = row_mask[:, None] & col_mask[None, :]

        x_offsets = x_ptr + rows[:, None] * x_stride_row + cols[None, :] * x_stride_col
        x = tl.load(x_offsets, mask=mask, other=0.0).to(tl.float32)

        batch_ids = rows // rows_per_batch
        scale_offsets = batch_ids[:, None] * scale_stride_batch + cols[None, :] * scale_stride_col
        shift_offsets = batch_ids[:, None] * shift_stride_batch + cols[None, :] * shift_stride_col
        scale = tl.load(scale_ptr + scale_offsets, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + shift_offsets, mask=mask, other=0.0).to(tl.float32)

        y = x * (1 + scale) + shift
        tl.store(
            y_ptr + rows[:, None] * y_stride_row + cols[None, :] * y_stride_col,
            y.to(y_ptr.type.element_ty),
            mask=mask,
        )

    @triton.jit
    def _gate_residual_fwd_fused(
        residual_ptr,
        x_ptr,
        gate_ptr,
        y_ptr,
        residual_stride_row,
        residual_stride_col,
        x_stride_row,
        x_stride_col,
        gate_stride_batch,
        gate_stride_col,
        y_stride_row,
        y_stride_col,
        rows_per_batch: tl.constexpr,
        total_rows: tl.constexpr,
        hidden_dim: tl.constexpr,
        hidden_dim_padded: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, hidden_dim_padded)

        row_mask = rows < total_rows
        col_mask = cols < hidden_dim
        mask = row_mask[:, None] & col_mask[None, :]

        residual_offsets = residual_ptr + rows[:, None] * residual_stride_row + cols[None, :] * residual_stride_col
        x_offsets = x_ptr + rows[:, None] * x_stride_row + cols[None, :] * x_stride_col

        residual = tl.load(residual_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_offsets, mask=mask, other=0.0).to(tl.float32)

        batch_ids = rows // rows_per_batch
        gate_offsets = batch_ids[:, None] * gate_stride_batch + cols[None, :] * gate_stride_col
        gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)

        y = residual + x * gate
        tl.store(
            y_ptr + rows[:, None] * y_stride_row + cols[None, :] * y_stride_col,
            y.to(y_ptr.type.element_ty),
            mask=mask,
        )

    @triton.jit
    def _rotary_qk_fwd_fused(
        q_ptr,
        k_ptr,
        freqs_real_ptr,
        freqs_imag_ptr,
        q_out_ptr,
        k_out_ptr,
        q_stride_row,
        k_stride_row,
        freq_stride_row,
        q_out_stride_row,
        k_out_stride_row,
        total_rows: tl.constexpr,
        seq_len: tl.constexpr,
        half_head_dim: tl.constexpr,
        half_head_dim_padded: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        pairs = tl.arange(0, half_head_dim_padded)

        row_mask = rows < total_rows
        pair_mask = pairs < half_head_dim
        mask = row_mask[:, None] & pair_mask[None, :]

        seq_idx = rows % seq_len
        even_cols = pairs * 2
        odd_cols = even_cols + 1

        freqs_real = tl.load(
            freqs_real_ptr + seq_idx[:, None] * freq_stride_row + pairs[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        freqs_imag = tl.load(
            freqs_imag_ptr + seq_idx[:, None] * freq_stride_row + pairs[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        q_even = tl.load(q_ptr + rows[:, None] * q_stride_row + even_cols[None, :], mask=mask, other=0.0).to(tl.float32)
        q_odd = tl.load(q_ptr + rows[:, None] * q_stride_row + odd_cols[None, :], mask=mask, other=0.0).to(tl.float32)
        k_even = tl.load(k_ptr + rows[:, None] * k_stride_row + even_cols[None, :], mask=mask, other=0.0).to(tl.float32)
        k_odd = tl.load(k_ptr + rows[:, None] * k_stride_row + odd_cols[None, :], mask=mask, other=0.0).to(tl.float32)

        q_out_even = q_even * freqs_real - q_odd * freqs_imag
        q_out_odd = q_even * freqs_imag + q_odd * freqs_real
        k_out_even = k_even * freqs_real - k_odd * freqs_imag
        k_out_odd = k_even * freqs_imag + k_odd * freqs_real

        tl.store(
            q_out_ptr + rows[:, None] * q_out_stride_row + even_cols[None, :],
            q_out_even.to(q_out_ptr.type.element_ty),
            mask=mask,
        )
        tl.store(
            q_out_ptr + rows[:, None] * q_out_stride_row + odd_cols[None, :],
            q_out_odd.to(q_out_ptr.type.element_ty),
            mask=mask,
        )
        tl.store(
            k_out_ptr + rows[:, None] * k_out_stride_row + even_cols[None, :],
            k_out_even.to(k_out_ptr.type.element_ty),
            mask=mask,
        )
        tl.store(
            k_out_ptr + rows[:, None] * k_out_stride_row + odd_cols[None, :],
            k_out_odd.to(k_out_ptr.type.element_ty),
            mask=mask,
        )

else:
    _layer_norm_param_fwd_fused = None
    _layer_norm_noparam_fwd_fused = None
    _rms_norm_fwd_fused = None
    _modulate_shift_fwd_fused = None
    _gate_residual_fwd_fused = None
    _rotary_qk_fwd_fused = None


def triton_layernorm_forward(
    x: torch.Tensor,
    w: Optional[torch.Tensor],
    b: Optional[torch.Tensor],
    eps: float,
    *,
    elementwise_affine: bool = True,
) -> torch.Tensor:
    if not _HAS_TRITON:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR!r}")

    [x_2d], batched, batch_size = flatten_if_batched(x)
    total_rows, hidden_dim = x_2d.shape
    y_2d = torch.empty_like(x_2d, dtype=torch.float32)
    mean = torch.empty((total_rows,), dtype=torch.float32, device=x_2d.device)
    rstd = torch.empty((total_rows,), dtype=torch.float32, device=x_2d.device)

    hidden_dim_padded = triton.next_power_of_2(hidden_dim)
    block_m = 32 if hidden_dim <= 512 else 1

    if elementwise_affine:
        if w is None or b is None:
            raise ValueError("LayerNorm with affine=True requires weight and bias.")
        _layer_norm_param_fwd_fused[(triton.cdiv(total_rows, block_m),)](
            x_2d,
            y_2d,
            w.contiguous(),
            b.contiguous(),
            mean,
            rstd,
            x_2d.stride(0),
            y_2d.stride(0),
            hidden_dim,
            hidden_dim_padded,
            eps,
            num_warps=8,
            BLOCK_M=block_m,
        )
    else:
        _layer_norm_noparam_fwd_fused[(triton.cdiv(total_rows, block_m),)](
            x_2d,
            y_2d,
            mean,
            rstd,
            x_2d.stride(0),
            y_2d.stride(0),
            hidden_dim,
            hidden_dim_padded,
            eps,
            num_warps=8,
            BLOCK_M=block_m,
        )

    if batched:
        return y_2d.reshape(batch_size, -1, hidden_dim)
    return y_2d


def triton_rmsnorm_forward(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    if not _HAS_TRITON:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR!r}")

    [x_2d], batched, batch_size = flatten_if_batched(x)
    total_rows, hidden_dim = x_2d.shape
    y_2d = torch.empty_like(x_2d, dtype=x_2d.dtype)
    rstd = torch.empty((total_rows,), dtype=torch.float32, device=x_2d.device)

    hidden_dim_padded = triton.next_power_of_2(hidden_dim)
    block_m = 32 if hidden_dim <= 512 else 1

    _rms_norm_fwd_fused[(triton.cdiv(total_rows, block_m),)](
        x_2d,
        y_2d,
        w.contiguous(),
        rstd,
        x_2d.stride(0),
        y_2d.stride(0),
        total_rows,
        hidden_dim,
        hidden_dim_padded,
        eps,
        num_warps=8,
        BLOCK_M=block_m,
    )

    if batched:
        return y_2d.reshape(batch_size, -1, hidden_dim)
    return y_2d


def triton_modulate_shift_forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = x.dtype
    if not _HAS_TRITON:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR!r}")
    if x.ndim != 3:
        raise ValueError(f"Expected 3D x tensor, got shape {tuple(x.shape)}")

    batch_size, rows_per_batch, hidden_dim = x.shape
    scale_2d = _prepare_batchwise_vector(scale, batch_size)
    shift_2d = _prepare_batchwise_vector(shift, batch_size)
    if scale_2d is None or shift_2d is None:
        raise ValueError(
            f"Unsupported scale/shift shapes {tuple(scale.shape)} and {tuple(shift.shape)} for x={tuple(x.shape)}"
        )
    if scale_2d.shape[1] != hidden_dim or shift_2d.shape[1] != hidden_dim:
        raise ValueError("Scale/shift hidden dimension must match x hidden dimension.")

    x_2d = x.contiguous().reshape(-1, hidden_dim)
    y_2d = torch.empty_like(x_2d, dtype=output_dtype)

    hidden_dim_padded = triton.next_power_of_2(hidden_dim)
    block_m = 32 if hidden_dim <= 512 else 1

    _modulate_shift_fwd_fused[(triton.cdiv(x_2d.shape[0], block_m),)](
        x_2d,
        scale_2d,
        shift_2d,
        y_2d,
        x_2d.stride(0),
        x_2d.stride(1),
        scale_2d.stride(0),
        scale_2d.stride(1),
        shift_2d.stride(0),
        shift_2d.stride(1),
        y_2d.stride(0),
        y_2d.stride(1),
        rows_per_batch,
        x_2d.shape[0],
        hidden_dim,
        hidden_dim_padded,
        num_warps=8,
        BLOCK_M=block_m,
    )

    return y_2d.reshape(batch_size, rows_per_batch, hidden_dim)


def triton_modulate_gate_residual_forward(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    *,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = residual.dtype
    if not _HAS_TRITON:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR!r}")
    if residual.ndim != 3 or x.ndim != 3:
        raise ValueError(f"Expected 3D residual/x tensors, got {tuple(residual.shape)} and {tuple(x.shape)}")
    if residual.shape != x.shape:
        raise ValueError(f"Residual/X shape mismatch: {tuple(residual.shape)} vs {tuple(x.shape)}")

    batch_size, rows_per_batch, hidden_dim = residual.shape
    gate_2d = _prepare_batchwise_vector(gate, batch_size)
    if gate_2d is None or gate_2d.shape[1] != hidden_dim:
        raise ValueError(
            f"Unsupported gate shape {tuple(gate.shape)} for residual shape {tuple(residual.shape)}"
        )

    residual_2d = residual.contiguous().reshape(-1, hidden_dim)
    x_2d = x.contiguous().reshape(-1, hidden_dim)
    y_2d = torch.empty_like(x_2d, dtype=output_dtype)

    hidden_dim_padded = triton.next_power_of_2(hidden_dim)
    block_m = 32 if hidden_dim <= 512 else 1

    _gate_residual_fwd_fused[(triton.cdiv(residual_2d.shape[0], block_m),)](
        residual_2d,
        x_2d,
        gate_2d,
        y_2d,
        residual_2d.stride(0),
        residual_2d.stride(1),
        x_2d.stride(0),
        x_2d.stride(1),
        gate_2d.stride(0),
        gate_2d.stride(1),
        y_2d.stride(0),
        y_2d.stride(1),
        rows_per_batch,
        residual_2d.shape[0],
        hidden_dim,
        hidden_dim_padded,
        num_warps=8,
        BLOCK_M=block_m,
    )

    return y_2d.reshape(batch_size, rows_per_batch, hidden_dim)


def _normalize_rotary_freqs_2d(freqs) -> tuple[torch.Tensor, torch.Tensor]:
    freqs_real, freqs_imag = _split_rotary_freqs(freqs)
    if freqs_real.ndim == 4 and freqs_real.shape[:2] == (1, 1):
        freqs_real = freqs_real[0, 0]
        freqs_imag = freqs_imag[0, 0]
    if freqs_real.ndim != 2 or freqs_imag.ndim != 2:
        raise ValueError(
            f"Expected rotary frequencies with shape [seq_len, head_dim/2], got {tuple(freqs_real.shape)}"
        )
    return freqs_real.contiguous().to(torch.float32), freqs_imag.contiguous().to(torch.float32)


def triton_rotary_qk_forward(query: torch.Tensor, key: torch.Tensor, freqs) -> tuple[torch.Tensor, torch.Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR!r}")
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError(f"Expected 4D query/key tensors, got {tuple(query.shape)} and {tuple(key.shape)}")
    if query.shape != key.shape:
        raise ValueError(f"Query/key shape mismatch: {tuple(query.shape)} vs {tuple(key.shape)}")

    batch_size, num_heads, seq_len, head_dim = query.shape
    if head_dim % 2 != 0:
        raise ValueError(f"Expected an even head_dim, got {head_dim}")

    freqs_real, freqs_imag = _normalize_rotary_freqs_2d(freqs)
    half_head_dim = head_dim // 2
    if freqs_real.shape != (seq_len, half_head_dim):
        raise ValueError(
            f"Rotary frequency shape mismatch: expected {(seq_len, half_head_dim)}, got {tuple(freqs_real.shape)}"
        )

    q_2d = query.contiguous().reshape(batch_size * num_heads * seq_len, head_dim)
    k_2d = key.contiguous().reshape(batch_size * num_heads * seq_len, head_dim)
    q_out_2d = torch.empty_like(q_2d)
    k_out_2d = torch.empty_like(k_2d)
    half_head_dim_padded = triton.next_power_of_2(half_head_dim)
    block_m = 8 if head_dim <= 128 else 4
    num_warps = 4 if head_dim <= 128 else 8

    _rotary_qk_fwd_fused[(triton.cdiv(q_2d.shape[0], block_m),)](
        q_2d,
        k_2d,
        freqs_real,
        freqs_imag,
        q_out_2d,
        k_out_2d,
        q_2d.stride(0),
        k_2d.stride(0),
        freqs_real.stride(0),
        q_out_2d.stride(0),
        k_out_2d.stride(0),
        q_2d.shape[0],
        seq_len,
        half_head_dim,
        half_head_dim_padded,
        num_warps=num_warps,
        BLOCK_M=block_m,
    )
    return (
        q_out_2d.reshape(batch_size, num_heads, seq_len, head_dim),
        k_out_2d.reshape(batch_size, num_heads, seq_len, head_dim),
    )


def fused_gate_residual_forward(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    *,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = residual.dtype
    try:
        return triton_modulate_gate_residual_forward(residual, x, gate, output_dtype=output_dtype)
    except Exception:
        return (residual.float() + x.float() * gate.float()).to(output_dtype)


def fused_modulate_shift_forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = x.dtype
    try:
        return triton_modulate_shift_forward(x, scale, shift, output_dtype=output_dtype)
    except Exception:
        return (x.float() * (1 + scale.float()) + shift.float()).to(output_dtype)


def fused_layernorm_forward(norm_module, x: torch.Tensor) -> torch.Tensor:
    try:
        return triton_layernorm_forward(
            x,
            getattr(norm_module, "weight", None),
            getattr(norm_module, "bias", None),
            getattr(norm_module, "eps", 1e-6),
            elementwise_affine=bool(getattr(norm_module, "elementwise_affine", True)),
        )
    except Exception:
        return norm_module(x.float())


def _reshape_norm_tensor_for_kernel(tensor: torch.Tensor) -> tuple[torch.Tensor, Optional[tuple[int, ...]]]:
    if tensor.ndim == 4:
        batch_size, num_heads, seq_len, hidden_dim = tensor.shape
        reshaped = tensor.contiguous().reshape(batch_size * num_heads, seq_len, hidden_dim)
        return reshaped, tuple(tensor.shape)
    if tensor.ndim in (2, 3):
        return tensor, None
    raise ValueError(f"Unsupported tensor rank for fast norm kernel: {tuple(tensor.shape)}")


def _restore_norm_tensor_shape(tensor: torch.Tensor, original_shape: Optional[tuple[int, ...]]) -> torch.Tensor:
    if original_shape is None:
        return tensor
    return tensor.reshape(*original_shape)


def apply_fast_qk_norm(attn, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    def maybe_apply(norm_module, tensor):
        if norm_module is None:
            return tensor
        if _FAST_QK_NORM_ENABLED and _HAS_TRITON:
            weight = getattr(norm_module, "weight", None)
            bias = getattr(norm_module, "bias", None)
            eps = getattr(norm_module, "eps", None)
            if weight is not None and eps is not None:
                try:
                    tensor_reshaped, original_shape = _reshape_norm_tensor_for_kernel(tensor)
                    class_name = str(norm_module.__class__.__name__).lower()
                    if "layernorm" in class_name or bias is not None:
                        output = triton_layernorm_forward(
                            tensor_reshaped,
                            weight,
                            bias,
                            eps,
                            elementwise_affine=bool(getattr(norm_module, "elementwise_affine", True)),
                        )
                    else:
                        output = triton_rmsnorm_forward(tensor_reshaped, weight, eps)
                    output = _restore_norm_tensor_shape(output, original_shape)
                    return output.to(dtype=tensor.dtype)
                except Exception:
                    pass
        return norm_module(tensor)

    query = maybe_apply(getattr(attn, "norm_q", None), query)
    key = maybe_apply(getattr(attn, "norm_k", None), key)
    return query, key


def _original_wan_block_forward():
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock

    original_forward = getattr(WanTransformerBlock, "_sink_original_forward", None)
    if original_forward is None:
        original_forward = WanTransformerBlock.forward
        WanTransformerBlock._sink_original_forward = original_forward
    return WanTransformerBlock, original_forward


def enable_wan_fast_misc_fusion(model) -> dict:
    WanTransformerBlock, _ = _original_wan_block_forward()

    if getattr(WanTransformerBlock, "_sink_fast_misc_fusion_enabled", False):
        setattr(model, "_sink_fast_misc_fusion_enabled", True)
        setattr(model, "_sink_fast_rotary_enabled", False)
        set_fast_qk_norm_enabled(False)
        set_fast_rotary_enabled(False)
        return {
            "enabled": True,
            "has_triton": _HAS_TRITON,
            "already_enabled": True,
            "qk_norm": False,
            "layernorm": False,
            "modulate": False,
            "rotary": False,
            "rotary_backend": "disabled",
        }

    if not _HAS_TRITON:
        warnings.warn(
            f"Cannot enable fast Wan misc fusion because Triton is unavailable: {_TRITON_IMPORT_ERROR!r}"
        )
        setattr(model, "_sink_fast_misc_fusion_enabled", False)
        setattr(model, "_sink_fast_rotary_enabled", False)
        set_fast_qk_norm_enabled(False)
        set_fast_rotary_enabled(False)
        return {
            "enabled": False,
            "has_triton": False,
            "already_enabled": False,
            "qk_norm": False,
            "layernorm": False,
            "modulate": False,
            "rotary": False,
            "rotary_backend": "disabled",
        }

    def misc_only_forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = fused_gate_residual_forward(hidden_states, attn_output, gate_msa, output_dtype=hidden_states.dtype)

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = fused_gate_residual_forward(hidden_states, ff_output, c_gate_msa, output_dtype=hidden_states.dtype)
        return hidden_states

    WanTransformerBlock.forward = misc_only_forward
    WanTransformerBlock._sink_fast_misc_fusion_enabled = True
    WanTransformerBlock._sink_fast_kernels_enabled = False
    WanTransformerBlock._sink_fast_rotary_enabled = False
    set_fast_qk_norm_enabled(False)
    set_fast_rotary_enabled(False)
    setattr(model, "_sink_fast_misc_fusion_enabled", True)
    setattr(model, "_sink_fast_kernels_enabled", False)
    setattr(model, "_sink_fast_rotary_enabled", False)
    return {
        "enabled": True,
        "has_triton": True,
        "already_enabled": False,
        "qk_norm": False,
        "layernorm": False,
        "modulate": False,
        "rotary": False,
        "rotary_backend": "disabled",
    }


def enable_wan_fast_kernels(model) -> dict:
    WanTransformerBlock, _ = _original_wan_block_forward()

    if getattr(WanTransformerBlock, "_sink_fast_kernels_enabled", False):
        setattr(model, "_sink_fast_misc_fusion_enabled", True)
        setattr(model, "_sink_fast_kernels_enabled", True)
        setattr(model, "_sink_fast_rotary_enabled", True)
        set_fast_qk_norm_enabled(True)
        set_fast_rotary_enabled(True)
        return {
            "enabled": True,
            "has_triton": _HAS_TRITON,
            "already_enabled": True,
            "qk_norm": True,
            "layernorm": True,
            "modulate": True,
            "rotary": True,
            "rotary_backend": get_fast_rotary_backend(),
        }

    if not _HAS_TRITON:
        warnings.warn(
            f"Cannot enable fast Wan kernels because Triton is unavailable: {_TRITON_IMPORT_ERROR!r}"
        )
        setattr(model, "_sink_fast_misc_fusion_enabled", False)
        setattr(model, "_sink_fast_kernels_enabled", False)
        setattr(model, "_sink_fast_rotary_enabled", False)
        set_fast_qk_norm_enabled(False)
        set_fast_rotary_enabled(False)
        return {
            "enabled": False,
            "has_triton": False,
            "already_enabled": False,
            "qk_norm": False,
            "layernorm": False,
            "modulate": False,
            "rotary": False,
            "rotary_backend": "disabled",
        }

    def fast_forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        norm_hidden_states = fused_layernorm_forward(self.norm1, hidden_states)
        norm_hidden_states = fused_modulate_shift_forward(
            norm_hidden_states,
            scale_msa,
            shift_msa,
            output_dtype=hidden_states.dtype,
        )
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = fused_gate_residual_forward(
            hidden_states,
            attn_output,
            gate_msa,
            output_dtype=hidden_states.dtype,
        )

        if getattr(self.norm2, "__class__", None).__name__ == "Identity":
            norm_hidden_states = hidden_states
        else:
            norm_hidden_states = fused_layernorm_forward(self.norm2, hidden_states).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        norm_hidden_states = fused_layernorm_forward(self.norm3, hidden_states)
        norm_hidden_states = fused_modulate_shift_forward(
            norm_hidden_states,
            c_scale_msa,
            c_shift_msa,
            output_dtype=hidden_states.dtype,
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = fused_gate_residual_forward(
            hidden_states,
            ff_output,
            c_gate_msa,
            output_dtype=hidden_states.dtype,
        )
        return hidden_states

    WanTransformerBlock.forward = fast_forward
    WanTransformerBlock._sink_fast_misc_fusion_enabled = True
    WanTransformerBlock._sink_fast_kernels_enabled = True
    WanTransformerBlock._sink_fast_rotary_enabled = True
    set_fast_qk_norm_enabled(True)
    set_fast_rotary_enabled(True)
    setattr(model, "_sink_fast_misc_fusion_enabled", True)
    setattr(model, "_sink_fast_kernels_enabled", True)
    setattr(model, "_sink_fast_rotary_enabled", True)
    return {
        "enabled": True,
        "has_triton": True,
        "already_enabled": False,
        "qk_norm": True,
        "layernorm": True,
        "modulate": True,
        "rotary": True,
        "rotary_backend": get_fast_rotary_backend(),
    }


def enable_wan_fast_rotary(model=None) -> dict:
    set_fast_rotary_enabled(True)
    if model is not None:
        setattr(model, "_sink_fast_rotary_enabled", True)
    return {
        "enabled": True,
        "rotary": True,
        "rotary_backend": get_fast_rotary_backend(),
        "cuda_ext_available": _try_import_fast_kernels() is not None,
    }


def disable_wan_fast_misc_fusion(model=None) -> bool:
    WanTransformerBlock, original_forward = _original_wan_block_forward()
    WanTransformerBlock.forward = original_forward
    WanTransformerBlock._sink_fast_misc_fusion_enabled = False
    WanTransformerBlock._sink_fast_kernels_enabled = False
    WanTransformerBlock._sink_fast_rotary_enabled = False
    set_fast_qk_norm_enabled(False)
    set_fast_rotary_enabled(False)
    if model is not None:
        setattr(model, "_sink_fast_misc_fusion_enabled", False)
        setattr(model, "_sink_fast_kernels_enabled", False)
        setattr(model, "_sink_fast_rotary_enabled", False)
    return True


def disable_wan_fast_kernels(model=None) -> bool:
    return disable_wan_fast_misc_fusion(model=model)
