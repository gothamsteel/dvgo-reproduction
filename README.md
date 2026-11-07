# Reproducing DVGO, and testing it on new data

This repository is a reproduction study of **Direct Voxel Grid Optimization (DVGO)**
(Sun, Sun and Chen, CVPR 2022), carried out for COMP8240 at Macquarie University.

It is a fork of the authors' code, [sunset1995/DirectVoxGO](https://github.com/sunset1995/DirectVoxGO).
All method code is theirs. What this fork adds:

1. Patches to run the 2022 code on a 2026 software stack
2. Two loader bug fixes found while reproducing
3. Converters that bring four new data sources into the format DVGO reads
4. An LLM-as-judge evaluation of the renders, in [`judge/`](judge)
5. Configs, results, figure scripts and test-path videos for every run in the final report

Full numbers are in [`results.md`](results.md). The authors' original README is
kept as [`README_upstream.md`](README_upstream.md).

---

## Results at a glance

| Source | Type | Scenes | Mean test PSNR | Paper |
|---|---|---|---|---|
| Synthetic-NeRF | original | 8 | 31.90 | 31.95 |
| Synthetic-NSVF | original | 8 | 34.93 | 35.08 |
| BlendedMVS | original | 4 | 28.08 | 28.02 |
| Tanks and Temples | original | 5 | 28.31 | 28.41 |
| DeepVoxels | original | 4 | 45.51 | 45.83 |
| DTU (IDR/NeuS subset) | new, existing | 5 | 26.65 | n/a |
| DL3DV (bounded scenes) | new, existing | 4 | 17.67 | n/a |
| Blender scenes | new, built here | 3 | 41.38 | n/a |
| Phone capture | new, built here | 1 | 22.18 | n/a |

SSIM and LPIPS for every scene are in `results.md`.

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
| GPU | NVIDIA GeForce RTX 4090, 24 GB (driver 580.178.04) |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |

```bash
conda create -n dvgo python=3.10
conda activate dvgo
export PYTHONNOUSERSITE=1   # ignore packages in ~/.local, if any
pip install -r requirements-repro.txt
```

`requirements-repro.txt` pins every package DVGO imports to the version used
for the report; `torch_scatter` comes from the PyG wheel index named in the
file. The upstream `requirements.txt` is kept unchanged for reference.
The CUDA extensions in `lib/cuda/` compile on first run, which needs `nvcc`
from a CUDA toolkit (12.8 or newer; 13.0 was used).

Checked from a fresh clone and a fresh environment on 7 November 2026:
`lego` scored 34.673 dB (34.641 in the report; IMPROVING_LOG.md notes about
0.2 dB drift between setups).

The LLM judge uses a second environment, described in
[`judge/README.md`](judge/README.md).

---

## Data

No data is stored in this repository. Place each dataset under `data/`, at the
path the configs expect.

| Dataset | Source | Path |
|---|---|---|
| Synthetic-NeRF | The official link in the upstream README is dead. Kaggle mirror: <https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset> | `data/nerf_synthetic/<scene>` |
| Synthetic-NSVF | <https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip> | `data/Synthetic_NSVF/<scene>` |
| BlendedMVS (NSVF subset) | <https://dl.fbaipublicfiles.com/nsvf/dataset/BlendedMVS.zip> | `data/BlendedMVS/<scene>` |
| Tanks and Temples (NSVF masked subset) | <https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip> | `data/TanksAndTemple/<scene>` |
| DeepVoxels | `synthetic_scenes.zip` from <https://drive.google.com/drive/folders/1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH>. It also holds `bus` and `shoe`, which the paper does not use. | `data/deepvoxels/[train,validation,test]/<scene>` |
| DTU (IDR/NeuS preprocessed, 15 scans) | NeuS authors' Dropbox, linked from the NeuS repository: <https://www.dropbox.com/scl/fo/um3wj3ctiuoottbfmqmgb/ABZRltszDvWHJ824UL6DHw0?rlkey=3vjok0aivnoiaf8z5j6w05k92&dl=0> | `data/DTU/dtu_scan<N>`, converted to `data/dtu_tnt/` |
| DL3DV-10K Benchmark | <https://huggingface.co/datasets/DL3DV/DL3DV-10K-Benchmark>. Accept the terms of use on Hugging Face first. | `DL3DV-10K-Benchmark/`, converted to `data/dl3dv_<name>` |
| Blender models (CC0) | Poly Haven: [concrete cat statue](https://polyhaven.com/a/concrete_cat_statue), [propane tank](https://polyhaven.com/a/propane_tank), [lion head](https://polyhaven.com/a/lion_head) | rendered to `data/synth_<name>` |
| Phone capture | Taken for this project. The processed capture (78 masked PNGs, poses, intrinsics) is in [`phone_capture/real_can_sam_v2`](phone_capture/real_can_sam_v2); the images carry no metadata. Copy or link it to the path on the right. | `data/real_can_sam_v2` |

---

## Reproducing

All commands run from the repository root in the `dvgo` environment unless
stated otherwise.

### Original data

```bash
python run.py --config configs/nerf/lego.py --render_test --dump_images
python run.py --config configs/nsvf/Bike.py --render_test --dump_images
python run.py --config configs/blendedmvs/Jade.py --render_test --dump_images
python run.py --config configs/tankstemple/Ignatius.py --render_test --dump_images
python run.py --config configs/deepvoxels/armchair.py --render_test --dump_images
```

Replace with any scene under `configs/nerf/`, `configs/nsvf/`,
`configs/blendedmvs/`, `configs/tankstemple/` or `configs/deepvoxels/`.
These configs are the authors', unchanged. Without `--dump_images` the test
views are scored but not saved; `--render_only` re-renders a trained scene
from its checkpoint.

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
python download.py --subset hash --only_level4 --hash <full scene hash>
python scripts/dl3dv_to_tnt.py DL3DV-10K-Benchmark/<hash>/nerfstudio data/dl3dv_<name>
python run.py --config configs/dl3dv/<name>.py --render_test --dump_images
```

`download.py` is the DL3DV authors' download script. Full hashes are at the
end of `results.md`.

| Config | Data folder | Scene | Note |
|---|---|---|---|
| `nonreflect.py` | `dl3dv_non_reflect` | a401469c | |
| `nonreflect2.py` | `dl3dv_non_reflect2` | 35317e62 | |
| `reflect1.py` | `dl3dv_reflect1` | d1b3a0b3 | |
| `reflect2.py` | `dl3dv_reflect2` | d9f4c746 | |
| `nonreflect_high.py` | `dl3dv_non_reflect` | a401469c | ablation, 256³ grid |
| `scene1.py` | `dl3dv_scene1` | 0a485338 | discarded, labelled unbounded |

### Blender scenes

```bash
blender -b -P scripts/blender_scene.py -- \
    --model data/cat_blender/concrete_cat_statue_2k.blend \
    --dst data/synth_cat --views 78 --radius 1.4
python run.py --config configs/synth/cat.py --render_test --dump_images
```

Scenes: `cat`, `propane`, `lion`. The script reads every camera pose directly
from Blender, so the poses are exact.

### Phone capture

To retrain from the released capture only:

```bash
mkdir -p data && ln -s ../phone_capture/real_can_sam_v2 data/real_can_sam_v2
python run.py --config configs/real/can_sam_v2.py --render_test --dump_images
```

Rebuilding it from the raw photos (not released) needs
[COLMAP](https://colmap.github.io/) and
[SAM 2](https://github.com/facebookresearch/sam2) with the
`sam2.1_hiera_large.pt` checkpoint.

```bash
colmap automatic_reconstructor --workspace_path ./capture \
    --image_path ./capture/images --single_camera 1 --sparse 1 --dense 0
colmap image_undistorter --image_path capture/images \
    --input_path capture/sparse/0 --output_path capture/undistorted \
    --output_type COLMAP --max_image_size 1046
colmap model_converter --input_path capture/undistorted/sparse \
    --output_path capture/undistorted/sparse_txt --output_type TXT
python scripts/sam2_masks.py capture/undistorted/images capture/masks \
    --ckpt ~/sam2_ckpt/sam2.1_hiera_large.pt --point 388 500
python scripts/colmap_to_tnt.py capture/undistorted data/real_can_sam_v2 \
    --masks capture/masks
python run.py --config configs/real/can_sam_v2.py --render_test --dump_images
```

`--point` is one click on the object in the first frame. With masks given,
`colmap_to_tnt.py` recentres and rescales the scene on the object's own sparse
points. Adding `--quality extreme` to the first command gives the
extreme-quality variant in `results.md`.

### Controlled pose test (Blender cat through COLMAP)

The 78 cat renders, composited onto white, go through the same COLMAP steps as
the phone capture, with masks taken from the render alpha. `--keep-split` keeps
the original train/test split so the test views match the exact-pose run.

```bash
colmap automatic_reconstructor --workspace_path ~/cat_colmap \
    --image_path ~/cat_colmap/images --single_camera 1 --sparse 1 --dense 0
colmap image_undistorter --image_path ~/cat_colmap/images \
    --input_path ~/cat_colmap/sparse/0 --output_path ~/cat_colmap/undistorted \
    --output_type COLMAP --max_image_size 1046
colmap model_converter --input_path ~/cat_colmap/undistorted/sparse \
    --output_path ~/cat_colmap/undistorted/sparse_txt --output_type TXT
python scripts/colmap_to_tnt.py ~/cat_colmap/undistorted data/synth_cat_colmap \
    --masks ~/cat_colmap/masks --keep-split
python run.py --config configs/synth/cat_colmap.py --render_test --dump_images
```

---

## LLM-as-judge

A vision-language model (Qwen2.5-VL-7B-Instruct, run locally) answers one
yes/no question per render: does it show reconstruction artefacts? On 40
held-out renders it agreed with the author at Cohen's kappa 0.38, about as
well as a second human labeller did (0.42), and with that labeller at 0.78.
Code, prompts, labels and every verdict are in [`judge/`](judge), explained in
[`judge/README.md`](judge/README.md).

---

## Repository map

| Path | Contents |
|---|---|
| `run.py`, `lib/` | DVGO, with the patches above |
| `configs/nerf/`, `configs/blendedmvs/`, `configs/tankstemple/`, `configs/nsvf/`, `configs/deepvoxels/` | upstream configs, unchanged |
| `configs/dtu/`, `configs/dl3dv/`, `configs/synth/`, `configs/real/` | configs for the new data |
| `scripts/dtu_to_tnt.py` | DTU (IDR/NeuS) to Tanks and Temples layout |
| `scripts/dl3dv_to_tnt.py` | DL3DV (nerfstudio) to Tanks and Temples layout |
| `scripts/colmap_to_tnt.py` | COLMAP to Tanks and Temples layout, centres on masked points |
| `scripts/sam2_masks.py` | object masks with SAM 2.1 video propagation |
| `scripts/blender_scene.py` | headless Blender render with exact camera poses |
| `scripts/exposure_test.py` | per-frame colour correction test for the DL3DV failure |
| `results/exposure_test.csv` | its output |
| `download.py` | DL3DV scene downloader, from the DL3DV authors |
| `judge/` | LLM-as-judge code, prompts, labels and verdicts |
| `media/videos/` | DVGO renderings along the test camera path, one video per new-data scene |
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
