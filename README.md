# Reproducing DVGO, and testing it on new data

This repository is a reproduction study of **Direct Voxel Grid Optimization (DVGO)**
(Sun, Sun and Chen, CVPR 2022), carried out for COMP8240 at Macquarie University.

It is a fork of the authors' code, [sunset1995/DirectVoxGO](https://github.com/sunset1995/DirectVoxGO).
All method code is theirs. What this fork adds:

1. Patches to run the 2022 code on a 2026 software stack
2. Two loader bug fixes found while reproducing
3. Converters that bring four new data sources into the format DVGO reads
4. Configs and results for every run reported in the final report

Full numbers are in [`results.md`](results.md).

---

## Results at a glance

| Source | Type | Scenes | Mean test PSNR | Paper |
|---|---|---|---|---|
| Synthetic-NeRF | original | 8 | 31.90 | 31.95 |
| Tanks and Temples | original | 5 | 28.31 | 28.41 |
| DTU (IDR/NeuS subset) | new, existing | 5 | 26.65 | n/a |
| DL3DV bounded | new, existing | 4 | 17.67 | n/a |
| Blender scenes | new, constructed | 3 | 41.38 | n/a |
| Phone capture | new, constructed | 1 | 22.18 | n/a |

SSIM and LPIPS (VGG and AlexNet) for every scene are in `results.md`.

---

## What changed from the original code

**Environment patches** (needed for PyTorch 2.10 / CUDA 12.8):

| File | Change |
|---|---|
| `run.py` | `mmcv.Config` replaced with `mmengine.Config` |
| `lib/cuda/*.cu` | `AT_DISPATCH` uses `.scalar_type()` instead of the removed `.type()` |
| `lib/utils.py`, `lib/grid.py` | `torch.load(..., weights_only=False)` |
| `lib/load_llff.py` | imageio dropped `ignoregamma`, replaced with `apply_gamma` |

**Bug fixes:**

| File | Bug |
|---|---|
| `run.py` | Ray-sampling indices from `torch.randint` land on CUDA while images stay on CPU under `load2gpu_on_the_fly=True`. Crashes on Tanks and Temples. |
| `lib/load_tankstemple.py` | Ignatius ships `intrinsics.txt` in the NSVF row format, the other four scenes ship a 4x4 matrix. `np.loadtxt` rejects the ragged file. |

**Evaluation:** `run.py` computes SSIM and LPIPS by default and writes an
`eval.txt` next to the test renders.

---

## Environment

Tested on:

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA GeForce RTX 4090 (driver 580.178.04) |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |

```bash
conda create -n dvgo python=3.10
conda activate dvgo
pip install -r requirements-repro.txt
```

`requirements-repro.txt` is the exact `pip freeze` of the environment used.
The upstream `requirements.txt` is kept unchanged for reference.

The CUDA extensions in `lib/cuda/` compile on first run.

---

## Data

No data is stored in this repository. Place each dataset under `data/`.

| Dataset | Source |
|---|---|
| Synthetic-NeRF | The official link in the upstream README is dead. Used a Kaggle mirror: TODO add link |
| Tanks and Temples (NSVF masked subset) | Linked from the upstream README |
| DTU (IDR/NeuS preprocessed, 15 scans) | TODO add link |
| DL3DV-10K Benchmark | Hugging Face, requires accepting the terms of use. `download.py` fetches selected scenes. TODO verify link |
| Blender models | CC0 models, TODO add source links |
| Phone capture | Captured for this project, TODO decide whether to release |

---

## Reproducing

### Original data

```bash
python run.py --config configs/nerf/lego.py --render_test --dump_images
python run.py --config configs/tankstemple/Ignatius.py --render_test --dump_images
```

Replace with any scene under `configs/nerf/` or `configs/tankstemple/`.

### DTU

```bash
python scripts/dtu_to_tnt.py data/DTU/dtu_scan114 data/dtu_tnt/dtu_scan114
python run.py --config configs/dtu/scan114.py --render_test --dump_images
```

Scans: 69, 97, 105, 114, 118.

DTU is converted to the Tanks and Temples layout, not the Blender layout. The
Blender loader stores only a focal length and assumes the principal point is at
the image centre. DTU's is not, and the Blender path cost 6.4 dB on scan114.

### DL3DV

```bash
python download.py   # see file for scene selection
python scripts/dl3dv_to_tnt.py DL3DV-10K-Benchmark/<hash>/nerfstudio data/dl3dv_<name>
python run.py --config configs/dl3dv/<name>.py --render_test --dump_images
```

| Config | Scene | Note |
|---|---|---|
| `nonreflect.py` | a401469c | |
| `nonreflect2.py` | 35317e62 | |
| `reflect1.py` | d1b3a0b3 | |
| `reflect2.py` | d9f4c746 | |
| `nonreflect_high.py` | a401469c | ablation, 256³ grid |
| `scene1.py` | 0a485338 | discarded, labelled unbounded |

Full hashes are in `results.md`.

### Phone capture

```bash
colmap automatic_reconstructor --workspace_path ./capture \
    --image_path ./capture/images --single_camera 1 --sparse 1 --dense 0 \
    --quality extreme
colmap image_undistorter --image_path capture/images \
    --input_path capture/sparse/0 --output_path capture/undistorted \
    --output_type COLMAP --max_image_size 1046
python scripts/sam2_masks.py capture/undistorted/images capture/masks \
    --ckpt ~/sam2_ckpt/sam2.1_hiera_large.pt --point 388 500
python scripts/colmap_to_tnt.py capture/undistorted data/real_can \
    --masks capture/masks
python run.py --config configs/real/can.py --render_test --dump_images
```

### Blender scenes

```bash
blender -b -P scripts/blender_scene.py -- --model <model.blend> \
    --dst data/synth_cat --views 78 --radius 1.4
python run.py --config configs/synth/cat.py --render_test --dump_images
```

Scenes: `cat`, `propane`, `lion`.

---

## Repository map

| Path | Contents |
|---|---|
| `run.py`, `lib/` | DVGO, with the patches above |
| `configs/nerf/`, `configs/tankstemple/` | upstream configs, unchanged |
| `configs/dtu/`, `configs/dl3dv/`, `configs/real/`, `configs/synth/` | configs for new data |
| `scripts/dtu_to_tnt.py` | DTU (IDR/NeuS) to Tanks and Temples layout |
| `scripts/dl3dv_to_tnt.py` | DL3DV (nerfstudio) to Tanks and Temples layout |
| `scripts/colmap_to_tnt.py` | COLMAP to Tanks and Temples layout, centres on masked points |
| `scripts/sam2_masks.py` | object masks with SAM 2.1 video propagation |
| `scripts/blender_scene.py` | headless Blender render with exact camera poses |
| `download.py` | DL3DV scene downloader |
| `results.md` | every number reported |

---

## Citation

```bibtex
@inproceedings{sun2022direct,
  title     = {Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction},
  author    = {Sun, Cheng and Sun, Min and Chen, Hwann-Tzong},
  booktitle = {CVPR},
  year      = {2022}
}
```

Licence: see [`LICENSE`](LICENSE), inherited from the upstream repository.
