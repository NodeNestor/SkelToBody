"""
Train skeleton -> body conditioning for TRELLIS.2's shape-flow DiT.

Strategy:
  - Frozen: TRELLIS.2 ElasticSLatFlowModel in NF4 (bitsandbytes 4-bit).
  - Trainable: LoRA rank 32 on DiT cross-attention QKV + MLP.
  - Trainable: SkeletonAdapter -> (N_s, 1024) cond tokens (replaces DinoV3 features).
  - Loss: rectified flow matching on body SLAT feats (denoise conditioned on skeleton cond).
  - Precision: bf16 activations, 8-bit AdamW, grad checkpointing.

Usage (inside the Docker container):
    python train/train.py \
        --pairs_dir /workspace/pairs/slats_512 \
        --output_dir /workspace/checkpoints/run1 \
        --steps 30000
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# TRELLIS.2 imports
import trellis2.models as models
import trellis2.modules.sparse as sp

# Our modules
from skeleton_adapter import SkeletonAdapter, AdapterConfig
from dataset import SlatPairDataset, collate


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps a frozen (possibly quantized) Linear with a trainable low-rank update."""
    def __init__(self, base: nn.Linear, rank: int = 32, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f = base.in_features
        out_f = base.out_features
        # Match device/dtype of the base weight so LoRA adapters land on GPU too.
        base_weight = getattr(base, 'weight', None)
        dev = base_weight.device if base_weight is not None else torch.device('cpu')
        dtype = torch.bfloat16  # LoRA adapters always bf16 regardless of base precision
        self.lora_a = nn.Linear(in_f, rank, bias=False, device=dev, dtype=dtype)
        self.lora_b = nn.Linear(rank, out_f, bias=False, device=dev, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        out = self.base(x)
        ldtype = self.lora_a.weight.dtype
        # Handle TRELLIS.2's sparse/varlen tensor wrappers — they expose .feats and .replace().
        if hasattr(x, 'feats') and hasattr(x, 'replace'):
            lora_feats = self.lora_b(self.lora_a(x.feats.to(ldtype))) * self.scale
            return out.replace(out.feats + lora_feats.to(out.feats.dtype))
        lora_out = self.lora_b(self.lora_a(x.to(ldtype))) * self.scale
        return out + lora_out.to(out.dtype)


def inject_lora(model: nn.Module, target_substrings=('qkv', 'proj', 'mlp'), rank: int = 32) -> list[nn.Parameter]:
    """Replace Linear modules whose name contains any target substring with LoRA-wrapped versions.
    Returns list of newly created trainable parameters."""
    trainable = []
    # collect first to avoid mutating during iteration
    to_wrap = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(t in name.lower() for t in target_substrings):
            to_wrap.append(name)
    for full in to_wrap:
        parent_name, _, leaf = full.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, leaf)
        wrapped = LoRALinear(base, rank=rank)
        setattr(parent, leaf, wrapped)
        trainable.extend([p for p in wrapped.parameters() if p.requires_grad])
    print(f'[LoRA] wrapped {len(to_wrap)} Linear layers with rank={rank}')
    return trainable


# ---------------------------------------------------------------------------
# Flow matching loss
# ---------------------------------------------------------------------------
def rectified_flow_loss(
    denoiser: nn.Module,
    body_feats_list: list[torch.Tensor],
    body_coords_list: list[torch.Tensor],
    cond: torch.Tensor,
    cond_mask: torch.Tensor,
) -> torch.Tensor:
    """
    body_feats_list: list of (N_i, 32) normalized SLAT features (the target).
    body_coords_list: list of (N_i, 3) int coords.
    cond: (B, M, 1024) from skeleton adapter.
    cond_mask: (B, M) bool, True = valid.
    """
    device = cond.device
    B = len(body_feats_list)
    losses = []
    for i in range(B):
        f = body_feats_list[i].to(device)       # (N, 32) float
        c = body_coords_list[i].to(device)      # (N, 3) int
        N = f.shape[0]

        # Build SparseTensor for the body target
        coords4 = torch.cat([torch.zeros_like(c[:, 0:1]), c], dim=-1).int()
        body = sp.SparseTensor(f.to(torch.float32), coords4)

        # Rectified flow: x_t = (1-t) * noise + t * x1, target velocity = x1 - noise
        t = torch.rand(1, device=device)
        noise = torch.randn_like(body.feats)
        x_t_feats = (1 - t) * noise + t * body.feats
        x_t = body.replace(x_t_feats)
        target = body.feats - noise

        # Classifier-free dropout on cond (10%)
        cond_i = cond[i:i+1]
        mask_i = cond_mask[i:i+1]
        if torch.rand(1, device=device).item() < 0.1:
            cond_i = torch.zeros_like(cond_i)
            mask_i = torch.zeros_like(mask_i)

        # Denoiser forward: sparse body tokens attend to cond tokens.
        # ElasticSLatFlowModel signature: forward(x: SparseTensor, t: Tensor, cond: Tensor)
        t_in = t.expand(1)
        pred = denoiser(x_t, t_in, cond_i)
        losses.append(F.mse_loss(pred.feats, target))

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs_dir', required=True, help='dir of <specimen>.npz SLAT pairs')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--denoiser_pretrained', type=str,
                    default='microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16')
    ap.add_argument('--steps', type=int, default=30_000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lora_rank', type=int, default=16)
    ap.add_argument('--grad_accum', type=int, default=8)
    ap.add_argument('--log_every', type=int, default=25)
    ap.add_argument('--save_every', type=int, default=2500)
    ap.add_argument('--quantize', action='store_true', help='load denoiser in NF4 via bitsandbytes')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda'

    # Dataset
    ds = SlatPairDataset(args.pairs_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
    print(f'[data] {len(ds)} pairs')

    # Denoiser (frozen, optionally NF4-quantized)
    print(f'[model] loading denoiser: {args.denoiser_pretrained}')
    denoiser = models.from_pretrained(args.denoiser_pretrained).eval()
    if args.quantize:
        try:
            from bitsandbytes.nn import Linear4bit
            _quantize_linears(denoiser)
            print('[model] denoiser quantized to NF4')
        except Exception as e:
            print(f'[warn] NF4 quantization failed ({e}); continuing in bf16')
    denoiser = denoiser.to(device)
    # Enable grad checkpointing on EVERY block, not just the top module.
    # TRELLIS.2's blocks check self.use_checkpoint individually.
    if hasattr(denoiser, 'blocks'):
        for blk in denoiser.blocks:
            if hasattr(blk, 'use_checkpoint'):
                blk.use_checkpoint = True
    if hasattr(denoiser, 'use_checkpoint'):
        denoiser.use_checkpoint = True
    for p in denoiser.parameters():
        p.requires_grad_(False)

    # Inject LoRA (remains in bf16 even when base is 4-bit)
    lora_params = inject_lora(denoiser, rank=args.lora_rank)

    # Skeleton adapter
    adapter = SkeletonAdapter(AdapterConfig()).to(device).to(torch.bfloat16)

    trainable = lora_params + list(adapter.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f'[train] trainable params: {n_trainable / 1e6:.2f} M')

    # Optimizer: 8-bit AdamW if available
    try:
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
        print('[opt] AdamW8bit')
    except Exception:
        optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
        print('[opt] AdamW (fp32 states)')

    scaler = torch.amp.GradScaler(device='cuda', enabled=False)  # bf16 mode, no scaler needed

    step = 0
    data_iter = iter(dl)
    adapter.train()
    running_loss = 0.0
    while step < args.steps:
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)

            sk_f = [x.to(device).to(torch.bfloat16) for x in batch['skel_feats']]
            sk_c = [x.to(device) for x in batch['skel_coords']]

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                cond, mask = adapter(sk_f, sk_c)
                loss = rectified_flow_loss(
                    denoiser,
                    batch['body_feats'], batch['body_coords'],
                    cond.to(torch.float32), mask,
                ) / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        running_loss = 0.9 * running_loss + 0.1 * accum_loss if step > 0 else accum_loss
        step += 1

        if step % args.log_every == 0:
            print(f'step {step:6d} | loss {running_loss:.4f} | last {accum_loss:.4f}')

        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                'step': step,
                'adapter': adapter.state_dict(),
                'lora': {n: p.detach().cpu() for n, p in denoiser.named_parameters() if p.requires_grad},
            }
            out = Path(args.output_dir) / f'ckpt_{step:06d}.pt'
            torch.save(ckpt, out)
            print(f'[save] {out}')


def _quantize_linears(model: nn.Module):
    """Replace nn.Linear with bitsandbytes Linear4bit (NF4), in-place."""
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    to_swap = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not isinstance(mod, Linear4bit):
            to_swap.append(name)
    for full in to_swap:
        parent_name, _, leaf = full.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, leaf)
        new = Linear4bit(
            base.in_features, base.out_features, bias=base.bias is not None,
            compute_dtype=torch.bfloat16, quant_type='nf4', compress_statistics=True,
        )
        with torch.no_grad():
            new.weight = bnb.nn.Params4bit(base.weight.data, requires_grad=False, quant_type='nf4')
            if base.bias is not None:
                new.bias = nn.Parameter(base.bias.data, requires_grad=False)
        setattr(parent, leaf, new)


if __name__ == '__main__':
    main()
