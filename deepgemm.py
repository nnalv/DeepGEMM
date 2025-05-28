import torch
import deep_gemm
from typing import Tuple
from deep_gemm import ceil_div


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)

    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)

def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def gemm(x_fp8, y_fp8):
    (m, k) = x_fp8[0].shape
    (n, k) = y_fp8[0].shape
    out = torch.empty((m, n), device=x_fp8[0].device, dtype=torch.bfloat16)
    deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
    return out


def wgrad_gemm(x_fp8, y_fp8):
    (m, k) = x_fp8[0].shape
    (n, k) = y_fp8[0].shape
    out = torch.empty((m, n), device=x_fp8[0].device, dtype=torch.float)
    deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt(x_fp8, y_fp8, out)
    return out