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

SSIM and LPIPS (VGG and AlexNet) for every scene are in `results.md`.

**LLM judge** (Qwen2.5-VL-7B-Instruct, prompt v2) against the author's labels:
accuracy 0.80 and Cohen's kappa 0.41 on 10 held-out renders, where always
answering "no" already scores 0.70. See `results.md`, section 5.

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
pip install -r requirements-repro.txt
```

`requirements-repro.txt` is the exact `pip freeze` of the environment used.
The upstream `requirements.txt` is kept unchanged for reference.
The CUDA extensions in `lib/cuda/` compile on first run.

The LLM judge runs in a second environment, because it needs a recent
`transformers`:

```bash
conda create -n judge python=3.10
conda activate judge
pip install torch==2.10.0 torchvision==0.25.0 transformers==5.20.2 \
    accelerate scikit-learn scipy pandas pillow
```

The judge model (about 16 GB in bfloat16) downloads from Hugging Face on first
run and fits on the 24 GB GPU above.

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
| Phone capture | Taken for this project. Not included in this repository. | `data/real_can_sam_v2` |

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

`download.py` is the DL3DV authors' download script. Full hashes are in
`results.md`.

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

Needs [COLMAP](https://colmap.github.io/) and
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
python scripts/colmap_to_tnt.py ~/cat_colmap/undistorted data/synth_cat_colmap \
    --masks ~/cat_colmap/masks --keep-split
python run.py --config configs/synth/cat_colmap.py --render_test --dump_images
```

The full command list is in `results.md`, section 4.

---

## LLM-as-judge (`judge/`)

PSNR, SSIM and LPIPS say how far a render is from the real image, but not what
is wrong with it. The judge adds one yes/no question per render: does it
contain visible reconstruction artefacts (floaters, specks, ragged edges,
holes or broken geometry)?

- **Model:** Qwen2.5-VL-7B-Instruct, run locally in bfloat16. Following the
  unit's week 9 practical, the verdict is the probability that the next token
  is *Yes* rather than *No*; above 0.5 counts as yes.
- **Input:** the render only. The judge never sees the real image.
- **Human labels:** 30 renders, 5 per source, drawn at random with a fixed seed
  and shown under anonymous ids, so the labeller could not see the source or
  PSNR. Labelled by the author with [`LABELLING_GUIDE.md`](judge/LABELLING_GUIDE.md)
  (13 yes, 17 no). 20 were used to develop the prompt and 10 were held out.
- **Prompts:** `prompt_v1.txt` matches the labelling guide. `prompt_v2.txt`
  was written after the development round and is the one reported. The held-out
  10 were judged once, with v2, and no further changes.
- **Second judge:** GPT-6 in ChatGPT (medium reasoning), given `prompt_v2.txt`
  and one image per new chat. Its answers are in `labels_chatgpt.csv`.

Every verdict is committed as a CSV, so every agreement number can be checked
without a GPU:

```bash
cd judge
python - <<'EOF'
import csv
from sklearn.metrics import accuracy_score, cohen_kappa_score
human = {r['id']: r['artefacts'] for r in csv.DictReader(open('labels_eric.csv'))}
gpt = {r['id']: r['artefacts'] for r in csv.DictReader(open('labels_chatgpt.csv'))}
held = list(csv.DictReader(open('judge_prompt_v2_heldout.csv')))
ids = sorted(human)
h, g = [human[i] for i in ids], [gpt[i] for i in ids]
print('GPT-6, all 30:   acc %.2f  kappa %.2f' % (accuracy_score(h, g), cohen_kappa_score(h, g)))
h, q = [r['human'] for r in held], [r['judge'] for r in held]
print('Qwen, held out:  acc %.2f  kappa %.2f' % (accuracy_score(h, q), cohen_kappa_score(h, q)))
EOF
```

To rerun the judge itself (`judge` environment, from `judge/`):

```bash
python per_image_psnr.py per_image_psnr.csv <log folders>   # per-render PSNR
python judge.py --prompt prompt_v1.txt --split dev
python judge.py --prompt prompt_v2.txt --split dev
python judge.py --prompt prompt_v2.txt --split heldout
python scale.py        # prompt v2 on every test render, writes judge_scale_v2.csv
python label.py labels_eric.csv   # the labelling tool: y = artefacts, n = clean, b = back
```

Three things to know before rerunning:

- `judge.py` reads the 30 images from `judge/samples/`. These are copies of test
  renders, so they are not committed; they exist once the scenes are trained.
- `sample.py` draws the 30 images. **Running it again overwrites
  `samples_key.csv` and resets `labels_eric.csv` to blank**, so do not run it
  to check the reported numbers.
- `per_image_psnr.csv` and `samples_key.csv` store absolute paths from the
  machine used, and `per_image_psnr.py` assumes the repository is at
  `~/DirectVoxGO`. Regenerate `per_image_psnr.csv` from your own log folders
  before running `scale.py`.

| File | Contents |
|---|---|
| `LABELLING_GUIDE.md` | the guide the human labeller followed |
| `prompt_v1.txt`, `prompt_v2.txt` | the two judge prompts |
| `samples_key.csv` | the 30 sampled renders, their split, source and PSNR |
| `labels_eric.csv`, `labels_chatgpt.csv` | human labels and GPT-6 answers |
| `judge_prompt_*_{dev,heldout}.csv` | Qwen verdicts with `p_yes` per image |
| `judge_scale_v2.csv` | Qwen verdicts on all 495 test renders |
| `per_image_psnr.csv` | PSNR of every test render, from the saved images |
| `smoke_test.py` | checks the model loads and answers simple questions |
| `make_results_fig.py`, `make_scene_fig.py` | build the report's render figures and copy the test-path videos |

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
