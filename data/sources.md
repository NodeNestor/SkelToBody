# Paired (skeleton, body) data sources — ingestion plan

## ✅ Auth-free, HF-native (we can pull today)

### TotalSegmentator-CT-Lite  *(already pulled, 50/1228 used)*
- HF: `YongchengYAO/TotalSegmentator-CT-Lite`
- Fetcher: `data/fetch_totalsegmentator.py`
- Humans only. 1228 whole-body CTs with 117 labels incl. full skeleton.
- **TODO: scale from 50 → all 1228.**

### CADS-dataset
- HF: `mrmrx/CADS-dataset`
- Largely supersets TotalSegmentator + adds MSD / Amos22 / KiTS etc — all human CT with some bone+body labels.
- Needs per-subset label mapping (bone IDs differ).
- Fetcher: `data/fetch_cads.py` *(to write)*

### Objaverse-XL animal+skeleton subset
- Via `objaverse` Python package (free, no auth).
- LVIS annotations include categories: `animal`, `skeleton`, `dinosaur`, `skull`, `lizard`, `bird`, etc.
- Artist-made 3D models — NOT native CT. Many have internal bone rigs (skinned meshes).
- Can extract skeleton via bone-rig inspection → pair with outer skin mesh.
- Fetcher: `data/fetch_objaverse_animals.py` *(to write)*

### facebook/sam-3d-body-dataset
- HF: `facebook/sam-3d-body-dataset`
- Humans with MHR rig (decoupled skeleton + skin surface).
- Direct download.

### BioAMASS (via SKEL model)
- SKEL repo on GitHub has BioAMASS — 113 subjects × 2198 motion sequences of paired SMPL body + BSM biomechanical skeleton.
- Humans only but MASSIVE number of poses.

## 🔑 Free account needed (user registers, we bulk-download)

### MorphoSource / oVert *(biggest prize — real vertebrate diversity)*
- ~13K vertebrate CT scans, 80%+ of living genera.
- Free MorphoSource account → API token → bulk script pulls Open-Access records.
- Fetcher: `data/download_morphosource.py` *(exists, needs token)*

### SMAL (parametric quadruped body model)
- `smal.is.tue.mpg.de`  (MPI, free registration)
- Unlimited synthetic (body, skeleton) pairs by sampling shape/pose params.
- Fetcher: `data/synth_smal.py` *(to write; needs user's downloaded weights)*

### SKEL / SMPL
- MPI account. Human parametric model.

### ESRF Paleo Database
- `paleo.esrf.fr`  free registration.
- Fossil synchrotron µCT, including dinosaurs.

### TCIA CT-ORG
- `cancerimagingarchive.net/collection/ct-org`  free account.
- 140 CTs, 6 organs incl. bones.

## 📦 Surface-only (would need a second surface→volume step to pair)

- Phenome10k — thousands of surface STLs
- Smithsonian 3D — 2000 CC0 objects
- NHM UK Data Portal — dinosaurs incl. Mantellisaurus
- African Fossils — Kenya fossils
- MorphoMuseuM — peer-reviewed mammals
- DigiMorph — ~50 STLs
- iDigFossils — fossil STLs

These give us *input skeletons* for inference on new species (e.g. a fossil dino skeleton → body prediction), but don't contribute paired training data unless we manually match skeleton & body meshes for the same specimen.

## 🎯 Plan of attack (by hands-free-ness)

1. Scale TotalSegmentator to 1228 specimens (no user action)
2. Add CADS-dataset (no user action, HF free)
3. Add SAM-3D-Body dataset (no user action, HF free)
4. Generate SMAL synthetic quadruped pairs — needs you to register at smal.is.tue.mpg.de and drop the weights file in `/workspace/smal_weights/`
5. Bulk-download Objaverse-XL animal/skeleton filter subset (no user action)
6. When ready: MorphoSource account → oVert bulk download
