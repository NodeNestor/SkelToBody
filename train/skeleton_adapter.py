"""
Skeleton conditioning adapter.

Input  : sparse skeleton SLAT  (coords: N_s x 3 int, feats: N_s x 32 float)
Output : cond token sequence   (N_s x cond_channels), consumable as the
         replacement for DinoV3 image features inside TRELLIS.2's flow DiTs.

The adapter is a small native-3D transformer with 3D sinusoidal positional
embeddings derived from the sparse voxel coordinates. Output is then linearly
projected into the flow DiT's cond_channels (1024 by default).
"""

from __future__ import annotations
from dataclasses import dataclass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pos_3d(coords: torch.Tensor, dim: int, max_period: float = 1e4) -> torch.Tensor:
    """coords: (N, 3) int. returns (N, dim) with 3 interleaved sinusoidal bands."""
    assert dim % 6 == 0, 'dim must be divisible by 6'
    per_axis = dim // 3
    half = per_axis // 2
    device = coords.device
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    out = []
    for ax in range(3):
        a = coords[:, ax:ax + 1].float() * freqs[None]
        out.append(torch.cat([torch.sin(a), torch.cos(a)], dim=-1))
    return torch.cat(out, dim=-1)


class SelfAttnBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3, bias=True)
        self.proj = nn.Linear(hidden, hidden, bias=True)
        self.norm2 = nn.LayerNorm(hidden)
        mlp_h = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_h), nn.GELU(), nn.Linear(mlp_h, hidden)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        h = self.n_heads
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h, N, d)
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, N) bool, True = pad
            attn_mask = key_padding_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=~attn_mask if attn_mask is not None else None,
                                             dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class AdapterConfig:
    in_channels: int = 32           # matches shape_enc_next_dc_f16c32_fp16 latent channels
    hidden: int = 384               # tuned down for 4060 8GB
    out_channels: int = 1024        # matches TRELLIS.2 cond_channels
    n_blocks: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0
    max_tokens: int = 8192          # safety cap for memory; skeleton is small, so fine
    grid_res: int = 32              # SLAT sparse coord resolution


class SkeletonAdapter(nn.Module):
    def __init__(self, cfg: AdapterConfig | None = None):
        super().__init__()
        self.cfg = cfg or AdapterConfig()
        c = self.cfg
        self.in_proj = nn.Linear(c.in_channels, c.hidden, bias=True)
        # 3D pos-embed dim must be divisible by 6; snap hidden to nearest valid
        pos_dim = (c.hidden // 6) * 6
        self.pos_proj = nn.Linear(pos_dim, c.hidden, bias=False)
        self.blocks = nn.ModuleList([
            SelfAttnBlock(c.hidden, c.n_heads, c.mlp_ratio) for _ in range(c.n_blocks)
        ])
        self.norm_out = nn.LayerNorm(c.hidden)
        self.out_proj = nn.Linear(c.hidden, c.out_channels, bias=True)
        self._pos_dim = pos_dim

    def forward(self,
                feats_list: list[torch.Tensor],      # list of (N_i, in_channels)
                coords_list: list[torch.Tensor]      # list of (N_i, 3) int
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cond_tokens: B x N_max x out_channels, mask: B x N_max bool where True=valid)."""
        assert len(feats_list) == len(coords_list)
        B = len(feats_list)
        lengths = [f.shape[0] for f in feats_list]
        N_max = max(lengths)
        device = feats_list[0].device
        dtype = feats_list[0].dtype

        x = torch.zeros(B, N_max, self.cfg.hidden, device=device, dtype=dtype)
        valid = torch.zeros(B, N_max, device=device, dtype=torch.bool)

        adapter_dtype = self.in_proj.weight.dtype
        for i, (f, c) in enumerate(zip(feats_list, coords_list)):
            n = f.shape[0]
            pe = sinusoidal_pos_3d(c.float() / self.cfg.grid_res, self._pos_dim).to(adapter_dtype)
            x[i, :n] = self.in_proj(f.to(adapter_dtype)) + self.pos_proj(pe)
            valid[i, :n] = True

        pad = ~valid  # True = pad for attention mask
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad)
        x = self.norm_out(x)
        cond = self.out_proj(x)  # (B, N_max, out_channels)
        return cond, valid
