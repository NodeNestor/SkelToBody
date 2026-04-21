"""
Conditioning adapter for the Sparse-Structure (SS) flow stage.

Input  : skeleton SS latent of shape (B, 8, 16, 16, 16)
Output : cond tokens (B, N, 1024) consumable as cross-attention context
         by TRELLIS.2's SparseStructureFlowModel (replaces DinoV3 features).

The 16^3 grid is flattened into 4096 tokens; each token carries an 8-channel
latent vector enriched with a learned 3D sinusoidal positional embedding, then
projected up to 1024-dim via a small MLP.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


def sinusoidal_pos_3d_grid(res: int, dim: int, device, dtype) -> torch.Tensor:
    """Return (res^3, dim) positional embedding covering a res^3 grid."""
    assert dim % 6 == 0
    per_axis = dim // 3
    half = per_axis // 2
    freqs = torch.exp(
        -math.log(1e4) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    coords = torch.arange(res, device=device, dtype=torch.float32) / res
    out = []
    for _ in range(3):
        a = coords[:, None] * freqs[None]
        out.append(torch.cat([torch.sin(a), torch.cos(a)], dim=-1))
    ex, ey, ez = out
    pe = (ex[:, None, None, :] + ey[None, :, None, :] + ez[None, None, :, :]).reshape(res ** 3, per_axis)
    # expand to full dim by tiling across three axes
    pe_full = torch.cat([ex[:, None, None, :].expand(res, res, res, per_axis).reshape(-1, per_axis),
                         ey[None, :, None, :].expand(res, res, res, per_axis).reshape(-1, per_axis),
                         ez[None, None, :, :].expand(res, res, res, per_axis).reshape(-1, per_axis)], dim=-1)
    return pe_full.to(dtype)


class SSConditioner(nn.Module):
    """Dense 3D latent -> token sequence for cross-attention."""
    def __init__(self, in_channels: int = 8, hidden: int = 512,
                 out_channels: int = 1024, grid_res: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.out_channels = out_channels
        self.grid_res = grid_res
        # pe dim must be divisible by 6
        pe_dim = (hidden // 6) * 6
        self.pe_dim = pe_dim
        self.in_proj = nn.Linear(in_channels, hidden)
        self.pe_proj = nn.Linear(pe_dim, hidden, bias=False)
        self.mid = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.out_proj = nn.Linear(hidden, out_channels)
        self._pe_cache = {}

    def _pe(self, device, dtype):
        key = (device, dtype)
        if key not in self._pe_cache:
            self._pe_cache[key] = sinusoidal_pos_3d_grid(self.grid_res, self.pe_dim, device, dtype)
        return self._pe_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 8, 16, 16, 16) -> (B, 4096, 1024)"""
        B, C, D, H, W = x.shape
        assert D == H == W == self.grid_res
        t = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)   # (B, N, C)
        h = self.in_proj(t) + self.pe_proj(self._pe(x.device, x.dtype))
        h = h + self.mid(h)
        return self.out_proj(h)
