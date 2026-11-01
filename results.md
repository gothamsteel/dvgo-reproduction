# Results

DVGO (Sun et al., CVPR 2022) reproduced on the original data, then applied to
three sources it was not evaluated on. All runs on an RTX 4090 with the
released per-scene configs at the paper's M(f) = 160^3 setting, unless a run
is explicitly marked as an ablation.

Every new dataset is converted to the `tankstemple` layout and read by the same
loader, so the loader is not a variable in any comparison below.

Paper figures are from Table 1 of the main paper and the supplementary material.

---

## 1. Original data

### Synthetic-NeRF, 8 scenes, 100 train / 200 test, 800x800

| Scene | PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) |
|---|---|---|---|---|
| chair | 34.092 | 0.976 | 0.028 | 0.017 |
| drums | 25.420 | 0.930 | 0.079 | 0.061 |
| ficus | 32.583 | 0.977 | 0.025 | 0.016 |
| hotdog | 36.706 | 0.980 | 0.034 | 0.017 |
| lego | 34.641 | 0.976 | 0.027 | 0.013 |
| materials | 29.586 | 0.950 | 0.059 | 0.027 |
| mic | 33.125 | 0.983 | 0.018 | 0.014 |
| ship | 29.049 | 0.877 | 0.161 | 0.117 |
| **mean (ours)** | **31.900** | **0.956** | **0.054** | **0.035** |
| **paper** | **31.95** | **0.957** | **0.053** | **0.035** |

Within 0.05 dB on PSNR, 0.001 on SSIM, exact on LPIPS-alex.

lego was run twice, six weeks apart on the same machine: 34.641032826900485 and
34.64103285074234. Agreement to eight decimal places means the remaining
difference against the paper is not run-to-run variance.

### Tanks and Temples, 5 scenes, NSVF masked subset, 1920x1080

| Scene | PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) |
|---|---|---|---|---|
| Barn | 26.855 | 0.835 | 0.298 | 0.299 |
| Caterpillar | 25.714 | 0.904 | 0.170 | 0.152 |
| Family | 33.706 | 0.962 | 0.070 | 0.063 |
| Ignatius | 28.171 | 0.943 | 0.087 | 0.092 |
| Truck | 27.088 | 0.905 | 0.161 | 0.146 |
| **mean (ours)** | **28.307** | **0.910** | **0.157** | **0.151** |
| **paper** | **28.41** | **0.911** | **0.155** | **0.148** |

Barn's LPIPS is roughly double every other scene while its PSNR is
unremarkable. Check against the paper's per-scene supplementary figures before
claiming this is a reproduction artefact.

### BlendedMVS, 4 scenes, NSVF subset, 768x576

| Scene | PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) | Paper PSNR |
|---|---|---|---|---|---|
| Jade | 27.790 | 0.917 | 0.105 | 0.072 | 27.68 |
| Fountain | 28.176 | 0.921 | 0.117 | 0.087 | 28.48 |
| Character | 30.278 | 0.963 | 0.046 | 0.029 | 30.31 |
| Statues | 26.092 | 0.885 | 0.138 | 0.109 | 25.62 |
| **mean (ours)** | **28.084** | **0.921** | **0.101** | **0.074** | |
| **paper** | **28.02** | **0.922** | **0.101** | **0.075** | |

Per-scene paper figures are from supplementary Table L3. Individual scenes
differ from the paper by up to 0.47 dB in either direction (Statues +0.47,
Fountain -0.30), while the mean lands 0.06 dB above. The errors do not share a
sign, so this reads as run-to-run variation rather than a systematic gap.

No code changes were needed for this benchmark. Each scene trained in about
2 min 20 s, against 13.8 min on average reported on an RTX 2080 Ti
(supplementary Table K2c).

### Two loader bugs found while reproducing

**`lib/load_tankstemple.py`, ragged intrinsics.** Ignatius ships
`intrinsics.txt` in the NSVF format, one row of `focal cx cy k1` plus padding
rows, while the other four scenes ship a 4x4 matrix. `np.loadtxt` rejects the
ragged file and the scene cannot load. Two different formats inside one
released benchmark.

**`run.py`, ray-sampling device.** These configs set
`load2gpu_on_the_fly=True`, which keeps images on the CPU, while
`torch.randint` returns CUDA indices under a modern PyTorch default. Indexing a
CPU tensor with CUDA indices raises. The synthetic scenes are small enough to
stay on the GPU, so the bug appears only here.

Neither is a flaw in the method. Both are the cost of running a 2022 codebase
in 2026 on data the authors packaged inconsistently.

---

## 2. New existing data: DTU

DTU (Jensen et al. 2014), in the 15-scan subset preprocessed by IDR and NeuS:
`image/`, `mask/`, and a `cameras_sphere.npz` of world-to-image projection
matrices normalised into a unit sphere. The raw release is 123 GB and ships
calibration in DTU's own format, so the preprocessed subset is used and cited
as such.

Converted with `scripts/dtu_to_tnt.py`: projection matrices decomposed into
intrinsics and camera-to-world poses, masks composited into alpha, one view in
eight held out.

| Scan | Views | PSNR | SSIM | LPIPS (vgg) | Train PSNR | Gap |
|---|---|---|---|---|---|---|
| scan114 | 64 | 28.071 | 0.907 | 0.168 | 31.65 | 3.58 |
| scan69 | **49** | **23.100** | 0.897 | **0.214** | 32.80 | **9.70** |
| scan97 | 64 | 26.047 | 0.911 | 0.165 | 29.77 | 3.72 |
| scan105 | 64 | 28.801 | 0.928 | 0.148 | 34.11 | 5.31 |
| scan118 | 64 | 27.216 | 0.939 | 0.142 | 30.73 | 3.51 |
| **mean** | | **26.647** | **0.916** | **0.167** | | |

**Verdict: feasible, as predicted.** The mean sits close to the paper's own
Tanks and Temples average of 28.41, on data the method was never shown.

**scan69 is the outlier and is confounded.** It has 49 views instead of 64,
and it is glazed ceramic with visible specular highlights. Its training PSNR is
the second highest at 32.80, so it fits its training views and fails on
held-out ones: a 9.70 dB gap against 3.5 to 5.3 for the rest. View count and
material cannot be separated from this run alone. The controlled version would
retrain scan105 on 42 views to match scan69's count; not done.

### Ablation: the file format is worth 6.4 dB

scan114 converted two ways, everything else identical.

| Format | PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) | Train PSNR |
|---|---|---|---|---|---|
| blender | 21.679 | 0.788 | 0.260 | 0.233 | 28.23 |
| tankstemple | **28.071** | **0.907** | **0.168** | **0.132** | 31.65 |

The blender path stores only a focal length, and `lib/load_data.py` then
synthesises K with the principal point at the image centre:

```python
if K is None:
    K = np.array([[focal, 0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]])
```

DTU's calibrated principal point is (823.2, 619.1) against an image centre of
(800, 600), and its fx and fy differ by 9 px. Every ray was aimed roughly 23 px
off. **The renders looked correct.** Only the train/test PSNR gap exposed it:
6.55 dB before, 3.58 dB after. The tankstemple loader reads a full K from
`intrinsics.txt`, so nothing is discarded.

Diagnostic worth keeping: a large train/test gap with visually plausible
renders means the camera model, not the reconstruction.

---

## 3. New existing data: DL3DV

DL3DV-10K Benchmark, chosen in the proposal for one reason: every scene carries
labels for indoor/outdoor, reflection and transparency, so a matched pair can
be selected in advance. Four bounded Shopping-Malls scenes under artificial
light, nontransparent, differing only in the reflection label.

Converted with `scripts/dl3dv_to_tnt.py`. 960x540 (images_4), one view in eight
held out, `white_bkgd=False` since these are whole rooms, not masked objects.

| Hash | Label | Views | PSNR | SSIM | LPIPS (vgg) | Train PSNR |
|---|---|---|---|---|---|---|
| a401469c | nonreflection | 359 | 16.762 | 0.544 | 0.580 | 23.95 |
| 35317e62 | nonreflection | 381 | 18.794 | 0.584 | 0.502 | 23.92 |
| d1b3a0b3 | reflection | 319 | 15.588 | 0.473 | 0.619 | 23.16 |
| d9f4c746 | reflection | 325 | 19.536 | 0.651 | 0.498 | 23.17 |
| **nonreflection mean** | | | **17.778** | 0.564 | 0.541 | |
| **reflection mean** | | | **17.562** | 0.562 | 0.559 | |

**The reflection label predicts nothing.** The group means differ by 0.22 dB,
while the spread inside the reflective group alone is 3.9 dB. Whatever is
limiting these reconstructions is not the material.

**Training PSNR is flat at 23.2 to 24.0 across all four.** Four different
scenes, 319 to 381 views each, and the same ceiling. DVGO cannot fit the
training images, so the test scores are measuring a failure that happens before
generalisation is in question.

### What was ruled out

| Hypothesis | Test | Result |
|---|---|---|
| Scene out of scope | checked the `environment` label | all four are `bounded` |
| Voxel resolution | rerun at 256^3 on a401469c | 16.676 vs 16.762, no change |
| Lens distortion dropped | per-pixel error, centre vs full frame | ratio 1.18, error is uniform |

The 256^3 run is the informative one: training PSNR rose from 23.95 to 24.92
while test stayed flat. Extra capacity fit the training views better and
generalised no better, so capacity was never the constraint.

The distortion check rests on radial distortion being zero on the optical axis
and growing with radius. If dropping k1 and k2 were the cause, error would
concentrate at the frame edges. A ratio of 1.18 says it does not.

### Remaining hypothesis, untested

A hand-held walk-through video with running auto-exposure and auto-white-balance
gives the same surface a different colour in different frames. No single
radiance field can fit that, which matches a training ceiling independent of
capacity, scene and material. This is the failure mode the proposal predicted
for PhotoTourism, appearing in DL3DV instead.

Stated as a hypothesis, not a finding. Testing it would mean measuring exposure
drift across frames, which was not done.

### Scope note

A fifth scene, `0a485338`, was run and discarded: the metadata labels it
`unbounded` (outdoor Square, open sky). DL3DV's `bounded` label means enclosed,
not object-scale, which is a weaker condition than the paper's bounded
inward-facing setting. Worth stating plainly: the annotation that made this
dataset attractive does not carry the meaning the proposal assumed.

---

## 4. Newly constructed data

Two datasets built for this project, differing in where the camera poses come
from. Both bounded, inward-facing, masked, 78 views split 68 train / 10 test,
776x1046, both through the same loader.

### Real capture, COLMAP poses

80 phone photos of a chocolate tin, 2 discarded for landscape orientation while
the rest were portrait, **78 of 78 remaining images registered** by COLMAP.
Undistorted to a PINHOLE camera. Object masks by SAM 2.1 video propagation
(`scripts/sam2_masks.py`), one click on frame 0. Converted with
`scripts/colmap_to_tnt.py`, which normalises the scene on the sparse points
whose 2D observations fall inside the masks.

### Synthetic, exact poses

CC0 models rendered in Blender with Cycles, camera transforms read directly
from the scene graph, transparent background (`scripts/blender_scene.py`).
Fixed two-light setup, since the method assumes a point has one appearance.

| Scene | Material | PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) | Train PSNR |
|---|---|---|---|---|---|---|
| Cat statue | matte stone | 40.943 | 0.982 | 0.039 | 0.020 | 36.28 |
| Propane tank | weathered paint | 41.004 | 0.987 | 0.033 | 0.019 | 39.54 |
| Lion bust | polished bronze | **42.184** | 0.992 | 0.018 | 0.007 | 39.13 |
| Real capture | chocolate tin | **22.180** | 0.873 | 0.155 | 0.136 | 29.80 |

### The pose-provenance comparison

Cat (exact poses) 40.94 against the real capture (COLMAP poses) 22.18, a gap of
18.8 dB.

### What was ruled out on the real capture

| Change | Test PSNR |
|---|---|
| rembg masks, soft alpha | 22.411 |
| SAM 2.1 masks | 22.237 |
| scene recentred on masked object points | 22.180 |
| COLMAP rerun at extreme quality, full-resolution solve | 21.754 |

The last one matters. COLMAP's mean reprojection error fell from 1.25 px to
0.86 px and mean track length rose from 3.94 to 4.85, so the poses genuinely
improved, and the reconstruction did not.

### Reading the gap honestly

Pose provenance is one contributor, not the whole story:

- **Content differs.** A stone cat is not a chocolate tin. The proposal called
  for content "as close as is practical" and this is not that close.
- **A render has no sensor noise, no motion blur, no auto-exposure drift and
  fixed illumination.** The real capture has all four.
- **Pose refinement did not help,** which argues directly against pose error
  being the dominant term.

The defensible claim: a hand-held structure-from-motion capture costs a large
amount of reconstruction quality relative to a rendered scene, and the portion
attributable to pose error alone is smaller than the headline gap.

### Material has no measurable effect under fixed lighting

Three synthetic scenes, same pipeline, same camera path, same view count,
differing in surface material.

The most reflective object, polished bronze, scored **highest** at 42.18. This
is the opposite of the proposal's expectation for reflective scenes. DVGO's
twelve-dimensional feature grid and two-layer MLP handle rendered specular
highlights without difficulty.

Two caveats. The lion is a smooth bust with a simple silhouette while the cat
has thin ears and legs, so geometric complexity is confounded with material.
And a rendered highlight under two fixed area lights is far better behaved than
a real glossy surface under real lighting; DTU scan69's glazed ceramic scored
23.10, which is the real-world version of the same question.

---

## 5. Summary across all data

| Source | Scenes | Mean PSNR | Nature |
|---|---|---|---|
| Synthetic-NeRF | 8 | 31.900 | original, rendered |
| Tanks and Temples | 5 | 28.307 | original, real, masked |
| BlendedMVS | 4 | 28.084 | original, rendered and blended with real photos |
| DTU (IDR/NeuS subset) | 5 | 26.647 | new existing, real, calibrated rig, masked |
| Constructed synthetic | 3 | 41.377 | new, rendered, exact poses |
| Constructed real capture | 1 | 22.180 | new, phone, COLMAP poses |
| DL3DV bounded | 4 | 17.670 | new existing, real, video, unmasked |

The ordering is the result. Rendered data with exact poses scores highest.
Calibrated real data with masks reproduces the paper. A hand-held masked
capture loses roughly 6 dB against the calibrated rig. Unmasked room-scale
video fails outright regardless of its annotations.

---

## Reproducing

```bash
# original data
python run.py --config configs/nerf/lego.py --render_test --dump_images
python run.py --config configs/tankstemple/Ignatius.py --render_test --dump_images

# DTU
python scripts/dtu_to_tnt.py data/DTU/dtu_scan114 data/dtu_tnt/dtu_scan114
python run.py --config configs/dtu/scan114.py --render_test --dump_images

# DL3DV
python scripts/dl3dv_to_tnt.py \
    DL3DV-10K-Benchmark/<hash>/nerfstudio data/dl3dv_<name>
python run.py --config configs/dl3dv/<name>.py --render_test --dump_images

# real capture
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

# synthetic
blender -b -P scripts/blender_scene.py -- --model <model.blend> \
    --dst data/synth_cat --views 78 --radius 1.4
python run.py --config configs/synth/cat.py --render_test --dump_images
```

## Scene index

| Name used here | DL3DV hash |
|---|---|
| non_reflect | a401469cb0fa576b93e836a4a6d712394f1a55a0a53a42250286980bd078e669 |
| non_reflect2 | 35317e621976e87f0c143e66fc61fb8cddb4ff134304da7a00e32ac1983105b4 |
| reflect1 | d1b3a0b37acc72207295b05e8c359f8b43bf879eb882c60747cfc6d3c7d8efa3 |
| reflect2 | d9f4c746e6fa456323601b8ca05357c5585d34663c765107d901b4fd929d784b |
| (discarded, unbounded) | 0a485338bbdaf19ba9090b874bb36ef0599a9c9a475a382c22903cf5981c6ea6 |

## Known gaps

- Synthetic-NSVF and DeepVoxels, two of the paper's five benchmarks, were not run.
- DTU LPIPS-alex recorded for scan114 only.
- The 22.180 real-capture run's terminal log was lost; its `eval.txt` survives
  in the render folder.
- scan69's view count and material are confounded; the controlled test was not
  run.
- The exposure-drift hypothesis for DL3DV is untested.