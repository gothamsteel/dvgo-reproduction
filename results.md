# Results

Every number in the final report, per scene. The analysis is in the report;
the commands are in [`README.md`](README.md).

All runs: RTX 4090, the authors' released configs, grid size M(f) = 160^3,
unless marked as an ablation. Every new dataset is converted to the Tanks and
Temples layout and read by the same loader. Paper values are from the paper's
supplementary Tables L1 to L5 (per scene) and K2 (training time, RTX 2080 Ti).
Test-path videos for every new-data scene are in `media/videos/`.

## 1. Original data

### Synthetic-NeRF (8 scenes, 100 train / 200 test, 800x800)

| Scene | PSNR | Paper PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) |
|---|---|---|---|---|---|
| chair | 34.092 | 34.09 | 0.976 | 0.028 | 0.017 |
| drums | 25.420 | 25.44 | 0.930 | 0.079 | 0.061 |
| ficus | 32.583 | 32.78 | 0.977 | 0.025 | 0.016 |
| hotdog | 36.706 | 36.74 | 0.980 | 0.034 | 0.017 |
| lego | 34.641 | 34.64 | 0.976 | 0.027 | 0.013 |
| materials | 29.586 | 29.57 | 0.950 | 0.059 | 0.027 |
| mic | 33.125 | 33.20 | 0.983 | 0.018 | 0.014 |
| ship | 29.049 | 29.13 | 0.877 | 0.161 | 0.117 |
| **mean (ours)** | **31.900** | | **0.956** | **0.054** | **0.035** |
| **paper** | | **31.95** | **0.957** | **0.053** | **0.035** |

lego was trained twice, six weeks apart: 34.641032826900485 and
34.64103285074234, equal to seven decimal places, so training is effectively
deterministic on this machine.

### Synthetic-NSVF (8 scenes, 100 train / 200 test, 800x800)

| Scene | PSNR | Paper PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) | Train time |
|---|---|---|---|---|---|---|
| Wineholder | 30.220 | 30.26 | 0.949 | 0.055 | 0.038 | 1:48 |
| Steamtrain | 36.361 | 36.56 | 0.988 | 0.019 | 0.009 | 2:14 |
| Toad | 32.988 | 33.10 | 0.965 | 0.046 | 0.030 | 1:54 |
| Robot | 36.256 | 36.36 | 0.992 | 0.013 | 0.005 | 1:58 |
| Bike | 38.128 | 38.33 | 0.991 | 0.012 | 0.004 | 1:50 |
| Palace | 34.276 | 34.49 | 0.961 | 0.044 | 0.028 | 2:30 |
| Spaceship | 37.524 | 37.71 | 0.987 | 0.020 | 0.009 | 2:25 |
| Lifestyle | 33.709 | 33.79 | 0.965 | 0.054 | 0.027 | 1:57 |
| **mean (ours)** | **34.933** | | **0.975** | **0.033** | **0.019** | 2:04 |
| **paper** | | **35.08** | **0.975** | **0.033** | **0.019** | 13.2 min |

### BlendedMVS (4 scenes, NSVF subset, 768x576)

| Scene | PSNR | Paper PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) |
|---|---|---|---|---|---|
| Jade | 27.790 | 27.68 | 0.917 | 0.105 | 0.072 |
| Fountain | 28.176 | 28.48 | 0.921 | 0.117 | 0.087 |
| Character | 30.278 | 30.31 | 0.963 | 0.046 | 0.029 |
| Statues | 26.092 | 25.62 | 0.885 | 0.138 | 0.109 |
| **mean (ours)** | **28.084** | | **0.921** | **0.101** | **0.074** |
| **paper** | | **28.02** | **0.922** | **0.101** | **0.075** |

About 2 min 20 s of training per scene (paper: 13.8 min).

### Tanks and Temples (5 scenes, NSVF masked subset, 1920x1080)

| Scene | PSNR | Paper PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) |
|---|---|---|---|---|---|
| Barn | 26.855 | 27.01 | 0.835 | 0.298 | 0.299 |
| Caterpillar | 25.714 | 26.00 | 0.904 | 0.170 | 0.152 |
| Family | 33.706 | 33.75 | 0.962 | 0.070 | 0.063 |
| Ignatius | 28.171 | 28.16 | 0.943 | 0.087 | 0.092 |
| Truck | 27.088 | 27.15 | 0.905 | 0.161 | 0.146 |
| **mean (ours)** | **28.307** | | **0.910** | **0.157** | **0.151** |
| **paper** | | **28.41** | **0.911** | **0.155** | **0.148** |

Barn's high LPIPS matches the paper's own (0.294 VGG, 0.290 AlexNet).

### DeepVoxels (4 scenes, 479 train / 1000 test, 512x512)

| Scene | Paper's name | PSNR | Paper PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) | Train time |
|---|---|---|---|---|---|---|---|
| armchair | Chair | 48.390 | 48.48 | 0.997 | 0.017 | 0.007 | 2:15 |
| greek | Pedestal | 48.517 | 48.51 | 0.998 | 0.003 | 0.001 | 2:10 |
| cube | Cube | 43.256 | 43.77 | 0.998 | 0.002 | 0.002 | 2:00 |
| vase | Vase | 41.890 | 42.54 | 0.997 | 0.005 | 0.002 | 2:19 |
| **mean (ours)** | | **45.513** | | **0.998** | **0.007** | **0.003** | 2:11 |
| **paper** | | | **45.83** | **0.998** | **0.006** | **0.002** | 14.3 min |

DVGO printed cube's training time as `00:01:60`, a rounding quirk in its timer.
Synthetic-NSVF and DeepVoxels were re-rendered from their checkpoints with
`--render_only --render_test --dump_images`; all 12 scenes gave the same PSNR.

## 2. New existing data

### DTU (IDR/NeuS preprocessed subset, 1600x1200, 1 view in 8 held out)

| Scan | Views | PSNR | SSIM | LPIPS (vgg) | Train PSNR | Gap |
|---|---|---|---|---|---|---|
| scan69 | 49 | 23.100 | 0.897 | 0.214 | 32.80 | 9.70 |
| scan97 | 64 | 26.047 | 0.911 | 0.165 | 29.77 | 3.72 |
| scan105 | 64 | 28.801 | 0.928 | 0.148 | 34.11 | 5.31 |
| scan114 | 64 | 28.071 | 0.907 | 0.168 | 31.65 | 3.58 |
| scan118 | 64 | 27.216 | 0.939 | 0.142 | 30.73 | 3.51 |
| **mean** | | **26.647** | **0.916** | **0.167** | | |

**Data format ablation (scan114, everything else identical):**

| Format | PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) | Train PSNR |
|---|---|---|---|---|---|
| Synthetic-NeRF (Blender) | 21.679 | 0.788 | 0.260 | 0.233 | 28.23 |
| Tanks and Temples | **28.071** | **0.907** | **0.168** | **0.132** | 31.65 |

The Blender format stores only a focal length, so the loader puts the
principal point at the image centre, (800, 600). DTU's calibrated principal
point is (823.2, 619.1).

### DL3DV (960x540, 1 view in 8 held out, no masks)

| Scene | Label | Views | PSNR | SSIM | LPIPS (vgg) | Train PSNR |
|---|---|---|---|---|---|---|
| a401469c | non-reflective | 359 | 16.762 | 0.544 | 0.580 | 23.95 |
| 35317e62 | non-reflective | 381 | 18.794 | 0.584 | 0.502 | 23.92 |
| d1b3a0b3 | reflective | 319 | 15.588 | 0.473 | 0.619 | 23.16 |
| d9f4c746 | reflective | 325 | 19.536 | 0.651 | 0.498 | 23.17 |
| **mean** | | | **17.670** | | | |

| Tested cause | Test | Result |
|---|---|---|
| Voxel resolution | 256^3 grid on a401469c | test 16.676 (was 16.762), train 24.92 (was 23.95) |
| Lens distortion | per-pixel error, image edges vs centre | ratio 1.18, error spread evenly |
| Changing exposure | per-frame gain and offset per channel, training views | +0.01 to +0.04 dB (DTU control: +0.02 to +0.12 dB) |

Exposure test (`scripts/exposure_test.py`, output in
`results/exposure_test.csv`). Training PSNR here is over the whole image,
from the saved 8-bit renders; on DTU only the object is scored.

| Scene | Train views | Train PSNR | Colour-corrected | Gain |
|---|---|---|---|---|
| DL3DV a401469c | 314 | 24.13 | 24.17 | +0.04 |
| DL3DV 35317e62 | 333 | 24.04 | 24.07 | +0.03 |
| DL3DV d1b3a0b3 | 279 | 23.28 | 23.32 | +0.04 |
| DL3DV d9f4c746 | 284 | 23.30 | 23.31 | +0.01 |
| DTU scan69 | 42 | 31.68 | 31.70 | +0.02 |
| DTU scan97 | 56 | 28.87 | 28.92 | +0.05 |
| DTU scan105 | 56 | 32.61 | 32.66 | +0.05 |
| DTU scan114 | 56 | 30.42 | 30.48 | +0.06 |
| DTU scan118 | 56 | 30.99 | 31.11 | +0.12 |

The correction barely helps DL3DV, no more than DTU, so exposure changes do
not explain the failure.

A fifth scene, 0a485338, was discarded because DL3DV labels it unbounded.

## 3. Constructed data (78 views, 68 train / 10 test, 776x1046)

| Scene | Material | PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) | Train PSNR |
|---|---|---|---|---|---|---|
| Concrete cat statue | matte concrete | 40.943 | 0.982 | 0.039 | 0.020 | 36.28 |
| Propane tank | weathered paint | 41.004 | 0.987 | 0.033 | 0.019 | 39.54 |
| Lion head | polished bronze | 42.184 | 0.992 | 0.018 | 0.007 | 39.13 |
| **Blender mean** | | **41.377** | | | | |
| Phone capture | chocolate tin | 22.180 | 0.873 | 0.155 | 0.136 | 29.80 |

**Controlled pose test:** the same 78 cat renders, with poses from COLMAP
instead of Blender, and the same test views.

| | Exact poses | COLMAP poses |
|---|---|---|
| Registered images | 78 / 78 | 78 / 78 |
| Mean reprojection error | n/a | 0.47 px |
| Test PSNR | 40.943 | 41.201 |
| SSIM | 0.982 | 0.987 |
| LPIPS (vgg) | 0.039 | 0.035 |
| Train PSNR | 36.28 | 37.78 |

**Phone capture variants:**

| Variant | Reprojection error | Test PSNR |
|---|---|---|
| SAM 2.1 masks, scene centred by COLMAP | 1.25 px | 22.237 |
| SAM 2.1 masks, scene centred on the object (final) | 1.25 px | 22.180 |
| As above, COLMAP at extreme quality | 0.86 px | 21.754 |

## 4. LLM judge

Round 1: 30 renders, 5 per source, labelled by the author without seeing
the source or PSNR (13 yes, 17 no); 20 for prompt development, 10 held out.
Round 2: 30 more drawn with a new seed and labelled the same way (19 yes),
all held out, so 40 held out in total. A second labeller, given only the
labelling guide, labelled all 60 in shuffled order. Details in
[`judge/README.md`](judge/README.md).

| Judge | Prompt | Images | Accuracy | Cohen's kappa |
|---|---|---|---|---|
| Qwen2.5-VL-7B | v1 | 20 development | 0.85 | 0.70 |
| Qwen2.5-VL-7B | v2 | 20 development | 0.85 | 0.70 |
| Qwen2.5-VL-7B | v2 | 10 held out (round 1) | 0.80 | 0.41 |
| Qwen2.5-VL-7B | v2 | 30 held out (round 2) | 0.63 | 0.35 |
| **Qwen2.5-VL-7B** | **v2** | **40 held out** | **0.68** | **0.38** |
| Always "yes" | | 40 held out | 0.55 | 0.00 |
| GPT-6 (ChatGPT, medium) | v2 | all 30 | 0.63 | 0.32 |
| GPT-6 (ChatGPT, medium) | v2 | 10 held out | 0.50 | 0.19 |

On the first 30: Qwen missed 5 of 13 artefacts with no false alarms; GPT-6
missed none but flagged 11 of 17 clean renders. Qwen and GPT-6 agree at
kappa 0.17. On the 40 held out, all 11 of Qwen's errors are misses.

**Against the second labeller:**

| Pair | Images | Agreement | Cohen's kappa |
|---|---|---|---|
| Author vs second labeller | all 60 | 0.70 | 0.42 |
| Qwen vs second labeller | all 60 | 0.93 | 0.82 |
| Qwen vs second labeller | 40 held out | 0.93 | 0.78 |

Renders labelled as having artefacts:

| Source | Author, round 1 | Author, round 2 | Second labeller | Qwen |
|---|---|---|---|---|
| Synthetic-NeRF | 0 of 5 | 0 of 5 | 0 of 10 | 0 of 10 |
| BlendedMVS | 0 of 5 | 4 of 5 | 0 of 10 | 0 of 10 |
| DTU | 3 of 5 | 5 of 5 | 0 of 10 | 1 of 10 |
| DL3DV | 5 of 5 | 5 of 5 | 10 of 10 | 10 of 10 |
| Blender scenes | 0 of 5 | 0 of 5 | 0 of 10 | 0 of 10 |
| Phone capture | 5 of 5 | 5 of 5 | 4 of 10 | 5 of 10 |

The author flags very small floaters, the second labeller only clear damage,
and Qwen sides with the second labeller.

**Qwen (prompt v2) on every test render:**

| Source | Renders | Mean PSNR | Flagged | Mean p(yes) | Spearman (PSNR, p) |
|---|---|---|---|---|---|
| Synthetic-NeRF (lego only) | 200 | 34.65 | 0% | 0.07 | -0.29 |
| BlendedMVS | 42 | 28.40 | 0% | 0.25 | -0.03 |
| DTU | 39 | 26.74 | 18% | 0.34 | -0.42 |
| DL3DV | 174 | 17.70 | 99% | 0.74 | -0.17 |
| Blender scenes | 30 | 41.30 | 0% | 0.09 | +0.21 |
| Phone capture | 10 | 22.17 | 50% | 0.51 | -0.60 |

Spearman over all 495 renders: -0.86. PSNR here is computed from the saved
8-bit images, so it differs slightly from the tables above.

## 5. Summary

| Source | Scenes | Mean PSNR | Data |
|---|---|---|---|
| DeepVoxels | 4 | 45.51 | original, rendered |
| Blender scenes | 3 | 41.38 | constructed, rendered, exact poses |
| Synthetic-NSVF | 8 | 34.93 | original, rendered |
| Synthetic-NeRF | 8 | 31.90 | original, rendered |
| Tanks and Temples | 5 | 28.31 | original, real, masked |
| BlendedMVS | 4 | 28.08 | original, rendered meshes blended with photos |
| DTU | 5 | 26.65 | existing, real, calibrated rig, masked |
| Phone capture | 1 | 22.18 | constructed, real, COLMAP poses, masked |
| DL3DV | 4 | 17.67 | existing, real video, no masks |

## DL3DV scene hashes

| Config | Hash |
|---|---|
| `nonreflect.py` | a401469cb0fa576b93e836a4a6d712394f1a55a0a53a42250286980bd078e669 |
| `nonreflect2.py` | 35317e621976e87f0c143e66fc61fb8cddb4ff134304da7a00e32ac1983105b4 |
| `reflect1.py` | d1b3a0b37acc72207295b05e8c359f8b43bf879eb882c60747cfc6d3c7d8efa3 |
| `reflect2.py` | d9f4c746e6fa456323601b8ca05357c5585d34663c765107d901b4fd929d784b |
| `scene1.py` (discarded) | 0a485338bbdaf19ba9090b874bb36ef0599a9c9a475a382c22903cf5981c6ea6 |

## Known gaps

- DTU LPIPS (alex) was recorded for scan114 only.
- The final phone run's terminal log was lost; its `eval.txt` survives.
- scan69's view count and glazed surface are confounded.
- The cause of the DL3DV failure is still unknown; exposure changes are ruled out.
- The second labeller labelled once; there is no third labeller to settle borderline cases.
- GPT-6 was only run on the first 30 renders.
