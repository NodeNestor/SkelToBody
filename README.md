<div align="center">

# SkelToBody

**Native-3D skeleton → body generative model**
*built on [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) via QLoRA-style conditioning*

![status](https://img.shields.io/badge/status-research%20%26%20training-orange)
![license](https://img.shields.io/badge/license-MIT-green)
![hardware](https://img.shields.io/badge/gpu-8GB%20VRAM%20min-blue)

</div>

> ⚠️ **This project is in research & training stage.** The architecture is
> working end-to-end on real data, but the model has not been trained long
> enough to produce reliable results yet. Expect undertrained/noisy output if
> you run the public checkpoint. See **[Status](#status)** below for the
> current training state and known limitations.

---

## What it does

Give it a skeleton mesh. It guesses a plausible body that could wrap that
skeleton.

```
skeleton.glb  ─►  [SkelToBody]  ─►  body.glb
```

Think of it as a statistical paleoartist: learn the skeleton→body mapping
across thousands of vertebrates, then ask it to extrapolate to species it's
never seen — extinct ones, hypothetical ones, a skeleton you drew yourself.

## Why it might work

Unlike prior approaches that feed rendered multiview images into an
image-to-3D generator, SkelToBody keeps everything in 3D the whole way:

1. The skeleton is encoded into the **same SLAT latent space** that TRELLIS.2
   uses natively for shapes. No modality gap, no 2D→3D fuzziness.
2. A small trainable **conditioning adapter** (a 3D transformer with
   sinusoidal positional embeddings over voxel coordinates) maps skeleton
   SLAT → the 1024-dim cond tokens TRELLIS.2's shape-flow DiT expects.
3. **LoRA** (rank 16) adapters on the frozen 3B DiT let it specialize for
   vertebrate bodies without forgetting the massive 3D prior it learned from
   Objaverse-XL pretraining.

## Architecture

```
skeleton mesh
     │
     ▼   TRELLIS.2 SC-VAE shape encoder (frozen)
skeleton SLAT  (sparse voxel grid, 32 channels per voxel)
     │
     ▼   SkeletonAdapter (trainable, ~3M params)
cond tokens  (N × 1024; replaces DinoV3 image features)
     │
     ▼   TRELLIS.2 shape-flow DiT, 3B params (frozen) + LoRA rank 16 (trainable)
body SLAT  (rectified-flow sample with CFG)
     │
     ▼   TRELLIS.2 SC-VAE shape decoder (frozen)
body mesh
```

---

## Status

**Current training: in progress on 1228 TotalSegmentator human CTs × 20 000 steps.**

| Piece | Status |
|---|---|
| Docker env (CUDA 12.4 + torch 2.6 + flash-attn 2.7.3 + o_voxel + flex_gemm) | ✅ builds clean |
| CT volume → watertight (skeleton, body) mesh pair | ✅ verified on synthetic + real |
| Mesh → SLAT via frozen SC-VAE | ✅ verified |
| Adapter + LoRA training loop (flow matching + CFG + 8-bit AdamW + grad ckpt) | ✅ working |
| Checkpoint save / restore | ✅ 184 LoRA params round-tripped |
| Inference (skeleton → body mesh) | ✅ runs; output currently spatially off |
| **Sparse Structure (SS) flow conditioning** | ❌ skipped — inference dilates skeleton voxels instead |
| **Training convergence on diverse species** | ❌ only humans so far |
| Real animal-CT diversity | ❌ blocked on MorphoSource account |

**Known limitations (be honest):**
- The model is deeply **undertrained** at the public checkpoint — loss is still
  descending when snapshots are taken.
- We **skipped** training a proper SS flow model. Inference dilates skeleton
  voxels outward to give the body room to grow, but this is a workaround.
- Trained **only on humans** so far. Extrapolation to dinosaurs or other
  vertebrates will hallucinate human-ish shapes until we add non-human data.
- "Fluid-preserved" CT specimens (oVert-style) will produce wet-specimen
  bodies, not living-animal shapes, unless augmented.

---

## Quick start

```bash
git clone https://github.com/NodeNestor/SkelToBody.git
cd SkelToBody
./setup.sh           # clones TRELLIS.2 + builds the Docker image (~20 min)
docker compose run --rm skel2body bash
```

Inside the container:

```bash
# Sanity-check the whole pipeline on synthetic data
bash test_pipeline.sh

# Fetch every dataset you have access to (see data/sources.md)
bash fetch_all.sh

# Train
python train/train.py \
    --pairs_dir real_pairs/slats_128 \
    --output_dir checkpoints/my_run \
    --steps 20000 --lr 1e-4 --lora_rank 16

# Infer
python infer.py \
    --skeleton path/to/skeleton.glb \
    --ckpt     checkpoints/my_run/ckpt_020000.pt \
    --out      body.glb \
    --cfg 3.0 --dilate 3
```

---

## Hardware

- GPU with **≥ 8 GB VRAM**, compute capability ≥ 8.0 (RTX 3060 / 4060 / 4070 /
  4080 / A6000 …). Tested on RTX 4060.
- `TORCH_CUDA_ARCH_LIST` is pinned to `8.9` — change in the Dockerfile for
  other cards. **Blackwell (sm_120 / RTX 5060 Ti)** needs torch 2.7+ which
  requires a different base image.
- Linux in the container; Windows + Docker Desktop works via WSL2 backend.

### Memory budget on 8 GB (bf16 + LoRA + grad ckpt)
| Component | VRAM |
|---|---|
| 3B shape-flow DiT (bf16) | ~6 GB |
| LoRA rank 16 | ~40 MB |
| Skeleton adapter | ~60 MB |
| Activations (grad ckpt, 2048 tokens) | ~1 GB |
| 8-bit AdamW state | ~50 MB |
| **Total** | **~7.2 GB** |

---

## Datasets

See [data/sources.md](data/sources.md) for the full registry. Summary:

| Source | Size | Auth | Fetcher |
|---|---|---|---|
| [TotalSegmentator-CT-Lite](https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-CT-Lite) | 1228 human CTs, 117 labels | none | `data/fetch_totalsegmentator.py` |
| [CADS-dataset](https://huggingface.co/datasets/mrmrx/CADS-dataset) | Multi-source human CT (superset of above) | none | `data/fetch_cads.py` |
| [Objaverse-XL](https://objaverse.allenai.org/) (animal/skeleton subset) | ~500 artist-made | none | `data/fetch_objaverse_animals.py` |
| [SMAL](https://smal.is.tue.mpg.de/) synthetic quadruped pairs | unlimited | free MPI registration | `data/synth_smal.py` |
| [MorphoSource / oVert](https://www.morphosource.org/) | **13K vertebrate CTs** | free account + token | `data/download_morphosource.py` |

## Recommendations if you want to actually use this

1. Start with TotalSegmentator training to validate the pipeline runs.
2. Register free at **MorphoSource** → get API token → bulk download oVert.
   This is the single biggest diversity multiplier available.
3. Register free at **MPI / SMAL** → drop `smal_CVPR2017.pkl` into
   `smal_weights/` → run `data/synth_smal.py` to add unlimited synthetic
   quadruped pairs.
4. Train at least 50 000 – 100 000 steps.
5. For eventual dinosaur inference, source skeleton meshes from
   [Smithsonian 3D](https://3d.si.edu), [NHM UK](https://data.nhm.ac.uk/),
   [African Fossils](https://africanfossils.org/), or the
   [ESRF Paleo Database](https://paleo.esrf.fr/).

---

## Acknowledgements

Built on top of:

- [**Microsoft TRELLIS.2**](https://github.com/microsoft/TRELLIS.2) — 4B-param
  native-3D generative model (MIT license). We load its frozen SC-VAE +
  shape-flow DiT and inject a small conditioning head.
- [**TotalSegmentator**](https://github.com/wasserth/TotalSegmentator) — 117
  anatomical structures from CT (Wasserthal et al., Apache 2.0). Dataset
  mirror from `YongchengYAO/TotalSegmentator-CT-Lite` on Hugging Face
  (CC-BY-4.0).
- [**oVert / openVertebrate**](https://www.floridamuseum.ufl.edu/overt/) —
  the broader vertebrate CT dataset on MorphoSource that this project is
  ultimately aimed at.
- [**SMAL**](https://smal.is.tue.mpg.de/) — parametric quadruped body model
  (Zuffi, Kanazawa, Jacobs, Black 2017).
- [**SKEL**](https://skel.is.tue.mpg.de/) — biomechanical skeleton-inside-skin
  model. Conceptually informs our skeleton→body goal.

## License

MIT (see [LICENSE](LICENSE)). External dependencies retain their own
licenses; note TRELLIS.2's own MIT terms and the various non-commercial /
CC-BY-NC terms of some data sources (oVert is CC-BY-4.0, African Fossils is
CC-BY-NC-SA, etc.).

## Citation

If this work is ever useful to you, please also cite the upstream TRELLIS.2
paper and the TotalSegmentator paper whose data made the base training
possible.
