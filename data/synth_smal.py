"""
Generate synthetic paired (skeleton, body) meshes by sampling random poses and
shapes from the SMAL parametric quadruped model.

SMAL weights (~10 MB) require a free registration at https://smal.is.tue.mpg.de/
Download `smal_CVPR2017.pkl` and place it at:
    /workspace/smal_weights/smal_CVPR2017.pkl

Each sample produces:
    body.glb      — deformed outer SMAL mesh (3889 vertices)
    skeleton.glb  — thin capsule-mesh along each rigid bone axis
Output is normalized into [-0.5, 0.5]^3 to match the TRELLIS.2 pipeline.
"""
from __future__ import annotations
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import trimesh


def load_smal(path: str):
    with open(path, 'rb') as f:
        dd = pickle.load(f, encoding='latin1')
    v_template = torch.tensor(dd['v_template'], dtype=torch.float32)
    shapedirs  = torch.tensor(np.array(dd['shapedirs']), dtype=torch.float32)
    faces      = np.asarray(dd['f'], dtype=np.int64)
    J_reg      = torch.tensor(np.array(dd['J_regressor'].todense()) if hasattr(dd['J_regressor'], 'todense') else np.array(dd['J_regressor']), dtype=torch.float32)
    kintree    = np.asarray(dd['kintree_table'], dtype=np.int64)  # (2, J)
    weights    = torch.tensor(np.array(dd['weights']), dtype=torch.float32)
    posedirs   = torch.tensor(np.array(dd['posedirs']), dtype=torch.float32)
    return {
        'v_template': v_template, 'shapedirs': shapedirs, 'faces': faces,
        'J_regressor': J_reg, 'kintree': kintree, 'weights': weights, 'posedirs': posedirs,
    }


def rodrigues(r: torch.Tensor) -> torch.Tensor:
    """r: (N, 3) axis-angle → (N, 3, 3) rotation matrices."""
    theta = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = r / theta
    K = torch.stack([
        torch.stack([torch.zeros_like(k[:,0]), -k[:,2], k[:,1]], dim=-1),
        torch.stack([ k[:,2], torch.zeros_like(k[:,0]), -k[:,0]], dim=-1),
        torch.stack([-k[:,1], k[:,0], torch.zeros_like(k[:,0])], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=r.device).expand(K.shape)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    return I + s * K + (1 - c) * (K @ K)


def smal_forward(smal: dict, beta: torch.Tensor, pose: torch.Tensor):
    """Simplified SMAL forward (shape + per-joint rotation). Returns (v, J).
    beta: (B,) shape coeffs; pose: (J, 3) axis-angle."""
    vT = smal['v_template'] + (smal['shapedirs'] @ beta)
    J = smal['J_regressor'] @ vT
    kintree = smal['kintree']
    R = rodrigues(pose)  # (J, 3, 3)

    # Build per-joint world transforms via tree walk
    n_joints = R.shape[0]
    world_R = [R[0]]
    world_t = [J[0]]
    for j in range(1, n_joints):
        parent = kintree[0, j]
        world_R.append(world_R[parent] @ R[j])
        world_t.append(world_R[parent] @ (J[j] - J[parent]) + world_t[parent])
    world_R = torch.stack(world_R)  # (J, 3, 3)
    world_t = torch.stack(world_t)  # (J, 3)

    # LBS
    weights = smal['weights']  # (V, J)
    v_rot = torch.einsum('jab,vb->vja', world_R, vT) + world_t  # (V, J, 3)
    v_posed = (weights.unsqueeze(-1) * v_rot).sum(dim=1)        # (V, 3)
    return v_posed, world_t  # joints in world space


def joints_to_skeleton_mesh(joints: np.ndarray, kintree: np.ndarray, radius: float = 0.01) -> trimesh.Trimesh:
    """Thin capsule along each parent→child segment."""
    segments = []
    for j in range(1, joints.shape[0]):
        parent = kintree[0, j]
        a = joints[parent]; b = joints[j]
        L = np.linalg.norm(b - a)
        if L < 1e-4:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=L, sections=8)
        axis = (b - a) / L
        z = np.array([0., 0., 1.])
        if np.linalg.norm(axis - z) < 1e-6:
            R = np.eye(3)
        else:
            v = np.cross(z, axis)
            s = np.linalg.norm(v); c = axis @ z
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + K + K @ K * ((1 - c) / (s*s + 1e-12))
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = (a + b) / 2
        cyl.apply_transform(T)
        segments.append(cyl)
    return trimesh.util.concatenate(segments) if segments else trimesh.Trimesh()


def normalize_and_export(body_v: np.ndarray, faces: np.ndarray, skel: trimesh.Trimesh, out_dir: Path):
    body = trimesh.Trimesh(vertices=body_v, faces=faces, process=True)
    mn, mx = body.vertices.min(0), body.vertices.max(0)
    c = (mn + mx) / 2; s = 0.99 / float((mx - mn).max())
    for m in (body, skel):
        if len(m.vertices):
            m.vertices = (m.vertices - c) * s
    out_dir.mkdir(parents=True, exist_ok=True)
    body.export(out_dir / 'body.glb')
    skel.export(out_dir / 'skeleton.glb')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='/workspace/smal_weights/smal_CVPR2017.pkl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_samples', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit(
            f'SMAL weights not found at {args.weights}.  Register at '
            f'https://smal.is.tue.mpg.de/ and drop `smal_CVPR2017.pkl` there.'
        )

    smal = load_smal(args.weights)
    rng = np.random.default_rng(args.seed)

    J_count = smal['J_regressor'].shape[0]
    out_root = Path(args.out)
    for i in range(args.n_samples):
        beta = torch.tensor(rng.normal(0, 0.8, size=smal['shapedirs'].shape[-1]), dtype=torch.float32)
        # random pose with mild rotations on major joints
        pose = torch.tensor(rng.normal(0, 0.15, size=(J_count, 3)), dtype=torch.float32)
        pose[0] = 0  # keep root unrotated

        v_posed, joints = smal_forward(smal, beta, pose)
        v_np = v_posed.detach().cpu().numpy()
        j_np = joints.detach().cpu().numpy()
        skel = joints_to_skeleton_mesh(j_np, smal['kintree'], radius=0.01)
        normalize_and_export(v_np, smal['faces'], skel, out_root / f'smal_{i:05d}')
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{args.n_samples} generated')

    print(f'Done. {args.n_samples} SMAL pairs written to {out_root}')


if __name__ == '__main__':
    main()
