"""Paired skeleton/body SLAT dataset (loaded from .npz files written by data/mesh_to_slat.py)."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SHAPE_LATENT_MEAN = np.array([
    0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
    -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
    0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
    -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665,
], dtype=np.float32)
SHAPE_LATENT_STD = np.array([
    5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
    5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
    4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
    5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691,
], dtype=np.float32)


def normalize(feats: np.ndarray) -> np.ndarray:
    return (feats - SHAPE_LATENT_MEAN) / SHAPE_LATENT_STD


def denormalize(feats: np.ndarray) -> np.ndarray:
    return feats * SHAPE_LATENT_STD + SHAPE_LATENT_MEAN


class SlatPairDataset(Dataset):
    def __init__(self, root: str, max_tokens: int = 2048):
        self.files = sorted(Path(root).glob('*.npz'))
        if not self.files:
            raise RuntimeError(f'No .npz files in {root}')
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        sk_f = normalize(d['skel_feats'])
        bo_f = normalize(d['body_feats'])
        sk_c = d['skel_coords'].astype(np.int64)
        bo_c = d['body_coords'].astype(np.int64)

        # random truncate for memory safety on huge specimens
        if sk_f.shape[0] > self.max_tokens:
            sel = np.random.choice(sk_f.shape[0], self.max_tokens, replace=False)
            sk_f, sk_c = sk_f[sel], sk_c[sel]
        if bo_f.shape[0] > self.max_tokens:
            sel = np.random.choice(bo_f.shape[0], self.max_tokens, replace=False)
            bo_f, bo_c = bo_f[sel], bo_c[sel]

        return {
            'skel_feats': torch.from_numpy(sk_f).float(),
            'skel_coords': torch.from_numpy(sk_c).long(),
            'body_feats': torch.from_numpy(bo_f).float(),
            'body_coords': torch.from_numpy(bo_c).long(),
            'name': self.files[idx].stem,
        }


def collate(batch: list[dict]) -> dict:
    return {
        'skel_feats':  [b['skel_feats'] for b in batch],
        'skel_coords': [b['skel_coords'] for b in batch],
        'body_feats':  [b['body_feats'] for b in batch],
        'body_coords': [b['body_coords'] for b in batch],
        'names':       [b['name'] for b in batch],
    }
