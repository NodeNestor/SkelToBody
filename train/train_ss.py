"""
Train the Sparse-Structure (SS) flow stage to predict body voxel occupancy
from skeleton voxel occupancy.

Pairs the pretrained TRELLIS.2 SparseStructureFlowModel (1.3B) with
a trainable SSConditioner + rank-16 LoRA on attention/MLP Linears.

Loss is rectified flow matching over the 16^3 dense latent with 10% CFG
dropout. bf16 activations, 8-bit AdamW, grad checkpointing on each block.
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import trellis2.models as models

from ss_adapter import SSConditioner

# LoRA helpers are identical to train.py; re-import them.
from train import LoRALinear, inject_lora


class SSPairDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(Path(root).glob('*.npz'))
        # keep only files that actually have SS fields
        self.files = [f for f in self.files if 'skel_ss' in np.load(f).files]
        if not self.files:
            raise RuntimeError(f'No .npz files with skel_ss/body_ss in {root}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        return {
            'skel_ss': torch.from_numpy(d['skel_ss']).float(),  # (8, 16, 16, 16)
            'body_ss': torch.from_numpy(d['body_ss']).float(),
            'name': self.files[idx].stem,
        }


def collate(batch):
    return {
        'skel_ss': torch.stack([b['skel_ss'] for b in batch]),
        'body_ss': torch.stack([b['body_ss'] for b in batch]),
        'names':   [b['name'] for b in batch],
    }


def ss_flow_loss(denoiser, body_ss: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Rectified flow loss on dense 16^3 latent."""
    B = body_ss.shape[0]
    device = body_ss.device
    t = torch.rand(B, device=device)
    noise = torch.randn_like(body_ss)
    x_t = (1 - t.view(B, 1, 1, 1, 1)) * noise + t.view(B, 1, 1, 1, 1) * body_ss
    target = body_ss - noise
    # CFG dropout — zero cond 10% of the time
    drop = (torch.rand(B, device=device) < 0.1).view(B, 1, 1)
    cond = cond.masked_fill(drop, 0.0)
    pred = denoiser(x_t, t, cond)
    return F.mse_loss(pred, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--denoiser_pretrained', type=str,
                    default='microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16')
    ap.add_argument('--steps', type=int, default=10_000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lora_rank', type=int, default=16)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--log_every', type=int, default=25)
    ap.add_argument('--save_every', type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda'

    ds = SSPairDataset(args.pairs_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                    num_workers=2, pin_memory=True)
    print(f'[data] {len(ds)} pairs with SS latents')

    print(f'[model] loading {args.denoiser_pretrained}')
    denoiser = models.from_pretrained(args.denoiser_pretrained).eval().to(device)
    if hasattr(denoiser, 'blocks'):
        for blk in denoiser.blocks:
            if hasattr(blk, 'use_checkpoint'):
                blk.use_checkpoint = True
    for p in denoiser.parameters():
        p.requires_grad_(False)

    lora_params = inject_lora(denoiser, rank=args.lora_rank)
    conditioner = SSConditioner().to(device).to(torch.bfloat16)

    trainable = lora_params + list(conditioner.parameters())
    print(f'[train] trainable params: {sum(p.numel() for p in trainable) / 1e6:.2f} M')

    try:
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
        print('[opt] AdamW8bit')
    except Exception:
        optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
        print('[opt] AdamW')

    step = 0
    data_iter = iter(dl)
    conditioner.train()
    ema_loss = 0.0
    while step < args.steps:
        optim.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)
            skel_ss = batch['skel_ss'].to(device).to(torch.bfloat16)
            body_ss = batch['body_ss'].to(device).to(torch.float32)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                cond = conditioner(skel_ss).float()
                loss = ss_flow_loss(denoiser, body_ss, cond) / args.grad_accum
            loss.backward()
            accum += loss.item()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        ema_loss = 0.9 * ema_loss + 0.1 * accum if step > 0 else accum
        step += 1

        if step % args.log_every == 0:
            print(f'step {step:6d} | loss {ema_loss:.4f} | last {accum:.4f}')
        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                'step': step,
                'conditioner': conditioner.state_dict(),
                'lora': {n: p.detach().cpu() for n, p in denoiser.named_parameters() if p.requires_grad},
            }
            out = Path(args.output_dir) / f'ss_ckpt_{step:06d}.pt'
            torch.save(ckpt, out)
            print(f'[save] {out}')


if __name__ == '__main__':
    main()
