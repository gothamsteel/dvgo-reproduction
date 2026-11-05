# Results

DVGO (Sun et al., CVPR 2022) reproduced on the original data, then applied to
four data sources it was not evaluated on, and checked with an LLM judge. All
runs on an RTX 4090 with the released per-scene configs at the paper's
M(f) = 160^3 setting, unless a run is explicitly marked as an ablation.

Every new dataset is converted to the `tankstemple` layout and read by the same
loader, so the loader is not a variable in any comparison below.

Paper figures are from Table 1 of the main paper and the per-scene tables of
the supplementary material (Tables L1 to L5). Commands to reproduce every run
are in [`README.md`](README.md).

---

## 1. Original data

### Synthetic-NeRF, 8 scenes, 100 train / 200 test, 800x800

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

Within 0.05 dB on PSNR, 0.001 on SSIM, exact on LPIPS-alex. Per-scene paper
values are from supplementary Table L1.

lego was run twice, six weeks apart on the same machine: 34.641032826900485 and
34.64103285074234. Agreement to seven decimal places means the remaining
difference against the paper is not run-to-run variance on our side.

### Tanks and Temples, 5 scenes, NSVF masked subset, 1920x1080

| Scene | PSNR | Paper PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) |
|---|---|---|---|---|---|
| Barn | 26.855 | 27.01 | 0.835 | 0.298 | 0.299 |
| Caterpillar | 25.714 | 26.00 | 0.904 | 0.170 | 0.152 |
| Family | 33.706 | 33.75 | 0.962 | 0.070 | 0.063 |
| Ignatius | 28.171 | 28.16 | 0.943 | 0.087 | 0.092 |
| Truck | 27.088 | 27.15 | 0.905 | 0.161 | 0.146 |
| **mean (ours)** | **28.307** | | **0.910** | **0.157** | **0.151** |
| **paper** | | **28.41** | **0.911** | **0.155** | **0.148** |

Four of five scenes are slightly below the paper, by at most 0.29 dB
(Caterpillar). Per-scene paper values are from supplementary Table L4.

Barn's LPIPS is roughly double every other scene. This is a property of the
scene, not of the reproduction: the paper's own Barn LPIPS is 0.294 (VGG) and
0.290 (AlexNet).

### BlendedMVS, 4 scenes, NSVF subset, 768x576

| Scene | PSNR | Paper PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) |
|---|---|---|---|---|---|
| Jade | 27.790 | 27.68 | 0.917 | 0.105 | 0.072 |
| Fountain | 28.176 | 28.48 | 0.921 | 0.117 | 0.087 |
| Character | 30.278 | 30.31 | 0.963 | 0.046 | 0.029 |
| Statues | 26.092 | 25.62 | 0.885 | 0.138 | 0.109 |
| **mean (ours)** | **28.084** | | **0.921** | **0.101** | **0.074** |
| **paper** | | **28.02** | **0.922** | **0.101** | **0.075** |

Per-scene paper values are from supplementary Table L3. Individual scenes
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
in 2026, on a benchmark release whose files are not packaged consistently.

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
| scan69 | **49** | **23.100** | 0.897 | **0.214** | 32.80 | **9.70** |
| scan97 | 64 | 26.047 | 0.911 | 0.165 | 29.77 | 3.72 |
| scan105 | 64 | 28.801 | 0.928 | 0.148 | 34.11 | 5.31 |
| scan114 | 64 | 28.071 | 0.907 | 0.168 | 31.65 | 3.58 |
| scan118 | 64 | 27.216 | 0.939 | 0.142 | 30.73 | 3.51 |
| **mean** | | **26.647** | **0.916** | **0.167** | | |

**Verdict: feasible, as predicted.** The mean is 1.7 dB below Tanks and Temples
(28.31 here), the paper's only fully real benchmark, on data the method was
never tested on.

**scan69 is the outlier and is confounded.** It has 49 views instead of 64,
and it is glazed ceramic with visible specular highlights. Its training PSNR is
the second highest at 32.80, so it fits its training views and fails on
held-out ones: a 9.70 dB gap against 3.5 to 5.3 for the rest. View count and
material cannot be separated from this run alone. The controlled version would
retrain scan105 on scan69's training-view count; not done.

### Ablation: the file format is worth 6.4 dB

scan114 converted two ways, everything else identical.

| Format | PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) | Train PSNR |
|---|---|---|---|---|---|
| blender | 21.679 | 0.788 | 0.260 | 0.233 | 28.23 |
| tankstemple | **28.071** | **0.907** | **0.168** | **0.132** | 31.65 |

The blender path stores only a focal length, and `lib/load_data.py` then
builds K with the principal point at the image centre:

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
renders points to the camera model, not the reconstruction.

Test-path videos for every scan are in `media/videos/dtu__*.mp4`.

---

## 3. New existing data: DL3DV

DL3DV-10K Benchmark, chosen in the proposal for one reason: every scene carries
labels for indoor/outdoor, reflection and transparency, so a matched set can
be selected in advance. Four bounded shopping-mall scenes with the same
lighting label, nontransparent, two labelled reflective and two not.

Converted with `scripts/dl3dv_to_tnt.py`. 960x540 (images_4), one view in eight
held out, `white_bkgd=False` since these are whole scenes, not masked objects.

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
| Lens distortion dropped | per-pixel error, edges vs centre | ratio 1.18, error is spread evenly |

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
`unbounded` (an outdoor square, open sky). DL3DV's `bounded` label means
enclosed, not object-scale, which is a weaker condition than the paper's
bounded inward-facing setting. The annotation that made this dataset attractive
does not carry the meaning the proposal assumed.

Test-path videos for every scene are in `media/videos/dl3dv__*.mp4`.

---

## 4. Newly constructed data

Two datasets built for this project, differing in where the camera poses come
from. Both bounded, inward-facing, masked, 78 views split 68 train / 10 test,
776x1046, both through the same loader.

### Phone capture, COLMAP poses

80 phone photos of a chocolate tin, 2 discarded for landscape orientation while
the rest were portrait, **78 of 78 remaining images registered** by COLMAP.
Undistorted to a PINHOLE camera. Object masks by SAM 2.1 video propagation
(`scripts/sam2_masks.py`), one click on frame 0. Converted with
`scripts/colmap_to_tnt.py`, which normalises the scene on the sparse points
whose 2D observations fall inside the masks. Config: `configs/real/can_sam_v2.py`.

### Blender scenes, exact poses

CC0 models from Poly Haven rendered in Blender with Cycles, camera transforms
read directly from the scene graph, transparent background
(`scripts/blender_scene.py`). Fixed two-light setup, since the method assumes
a point has one appearance.

| Scene | Material | PSNR | SSIM | LPIPS (vgg) | LPIPS (alex) | Train PSNR |
|---|---|---|---|---|---|---|
| Concrete cat statue | matte concrete | 40.943 | 0.982 | 0.039 | 0.020 | 36.28 |
| Propane tank | weathered paint | 41.004 | 0.987 | 0.033 | 0.019 | 39.54 |
| Lion head | polished bronze | **42.184** | 0.992 | 0.018 | 0.007 | 39.13 |
| **Blender mean** | | **41.377** | | | | |
| Phone capture | chocolate tin | **22.180** | 0.873 | 0.155 | 0.136 | 29.80 |

### The pose-provenance comparison

Cat (exact poses) 40.94 against the phone capture (COLMAP poses) 22.18, a gap
of 18.8 dB. This comparison changes two things at once, the pose source and the
content, so it cannot attribute the gap to either.

### Controlled test: the same renders, with COLMAP poses

The 78 Blender cat renders were composited onto white and passed through the
same COLMAP pipeline as the phone capture (default quality, single camera,
undistorted to PINHOLE). Masks come from the render alpha. The converter keeps
the original 68/10 train/test split, so the test views are identical.

| | exact poses | COLMAP poses |
|---|---|---|
| registered images | 78 / 78 | 78 / 78 |
| mean reprojection error | n/a | 0.47 px |
| mean track length | n/a | 8.60 |
| test PSNR | 40.943 | **41.201** |
| SSIM | 0.982 | 0.987 |
| LPIPS (vgg) | 0.039 | 0.035 |
| LPIPS (alex) | 0.020 | 0.021 |
| train PSNR | 36.28 | 37.78 |

**Estimated poses cost nothing on identical content.** The 0.26 dB difference
is within run-to-run noise. Undistortion cropped the images from 776x1046 to
773x1042, as it does for the phone capture, so the two runs are evaluated on
very slightly different pixels.

Scope of the claim: COLMAP reached 0.47 px on clean renders against 1.25 px on
the phone photos. This test shows that COLMAP-quality poses on clean images do
not limit DVGO. It does not by itself show that the phone's poses cost nothing;
the extreme-quality rerun below covers that case.

Commands for this test:

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

### Phone capture variants

| Variant | Test PSNR |
|---|---|
| SAM 2.1 masks, scene centred by COLMAP | 22.237 |
| SAM 2.1 masks, scene recentred on the object's points (final) | 22.180 |
| As above, COLMAP rerun at extreme quality, full-resolution solve | 21.754 |

The last one matters. COLMAP's mean reprojection error fell from 1.25 px to
0.86 px and mean track length rose from 3.94 to 4.85, so the poses genuinely
improved, and the reconstruction did not.

### Reading the gap honestly

Pose provenance is one contributor, not the whole story:

- **Content differs.** A concrete cat is not a chocolate tin. The proposal
  called for content "as close as is practical" and this is not that close.
- **A render has no sensor noise, no motion blur, no auto-exposure drift and
  fixed illumination.** The phone capture has all four.
- **Better poses did not help,** which argues directly against pose error
  being the dominant term.
- **On identical content, COLMAP poses matched exact poses** (41.20 against
  40.94), so pose estimation as a process is not what costs quality.

The defensible claim: a hand-held capture costs a large amount of
reconstruction quality relative to a rendered scene, and two independent tests
indicate that pose estimation is not the main cause. The remaining candidates
are the capture itself (sensor noise, motion blur, exposure drift, real
lighting) and the different object.

### Material has no measurable effect under fixed lighting

Three synthetic scenes, same pipeline, same camera path, same view count,
differing in surface material.

The most reflective object, polished bronze, scored **highest** at 42.18. This
is the opposite of the proposal's expectation for reflective scenes. DVGO's
twelve-dimensional feature grid and two-layer MLP handle rendered specular
highlights without difficulty.

Two caveats. The lion head is a smooth shape while the cat has thin ears and
legs, so geometric complexity is confounded with material. And a rendered
highlight under two fixed lights is far better behaved than a real glossy
surface under real lighting; DTU scan69's glazed ceramic scored 23.10, which is
the real-world version of the same question.

Test-path videos are in `media/videos/blender__*.mp4` and
`media/videos/phone__real_can_sam_v2.mp4`.

---

## 5. LLM judge

PSNR, SSIM and LPIPS measure how far a render is from the real image, not what
is wrong with it. The judge answers one question per render: does it contain
visible reconstruction artefacts? Setup, files and commands are in
[`README.md`](README.md#llm-as-judge-judge).

### Validation against human labels

30 renders, 5 per source, labelled by the author without seeing the source or
PSNR: 13 yes, 17 no. 20 were used to develop the prompt, 10 were held out and
judged once.

| Judge | Prompt | Images | Accuracy | Cohen's kappa |
|---|---|---|---|---|
| Qwen2.5-VL-7B | v1 | 20 development | 0.85 | 0.70 |
| Qwen2.5-VL-7B | v2 | 20 development | 0.85 | 0.70 |
| **Qwen2.5-VL-7B** | **v2** | **10 held out** | **0.80** | **0.41** |
| Always "no" | | 10 held out | 0.70 | 0.00 |
| GPT-6 (ChatGPT, medium) | v2 | all 30 | 0.63 | 0.32 |
| GPT-6 (ChatGPT, medium) | v2 | 10 held out | 0.50 | 0.19 |

Kappa 0.41 counts as moderate agreement (Landis and Koch, 1977). With only 10
held-out images, 3 of them positive, it is an uncertain estimate.

The two judges fail in opposite directions. Across all 30 images, Qwen missed
5 of the 13 renders the human labelled yes and raised no false alarms. GPT-6
said yes to 24 of 30: it caught all 13, but also flagged 11 of the 17 the human
labelled clean, three of them Blender renders above 40 dB. The two judges agree
with each other at kappa 0.17.

Prompt v2 differs from v1 in two places: it adds a "ragged edges" item, and
instead of telling the judge to ignore "the plain white background itself", it
says an empty white background is fine but anything in it counts. The change
was made because v1 appeared to make the judge ignore specks in the background.

### The judge on every test render (prompt v2)

| Source | Renders | Mean PSNR | Flagged | Mean p(yes) | Spearman (PSNR, p) |
|---|---|---|---|---|---|
| Synthetic-NeRF (lego only) | 200 | 34.65 | 0% | 0.07 | -0.29 |
| BlendedMVS | 42 | 28.40 | 0% | 0.25 | -0.03 |
| DTU | 39 | 26.74 | 18% | 0.34 | -0.42 |
| DL3DV | 174 | 17.70 | 99% | 0.74 | -0.17 |
| Blender scenes | 30 | 41.30 | 0% | 0.09 | +0.21 |
| Phone capture | 10 | 22.17 | 50% | 0.51 | -0.60 |

Across all 495 renders, Spearman(PSNR, p) = -0.86. Within each source the link
is weak. PSNR here is computed from the saved 8-bit images, so it differs
slightly from sections 1 to 4. lego is the one Synthetic-NeRF scene whose real
test images were still available.

The useful result is one comparison: DTU and BlendedMVS have similar PSNR
(26.7 and 28.4 dB), yet the judge flags 18% of DTU renders and none of
BlendedMVS. Their errors differ in kind, which PSNR alone does not show.
Because the judge misses subtle artefacts, the flagged rates are lower bounds.

---

## 6. Summary across all data

| Source | Scenes | Mean PSNR | Nature |
|---|---|---|---|
| Synthetic-NeRF | 8 | 31.900 | original, rendered |
| Tanks and Temples | 5 | 28.307 | original, real, masked |
| BlendedMVS | 4 | 28.084 | original, rendered and blended with real photos |
| DTU (IDR/NeuS subset) | 5 | 26.647 | new existing, real, calibrated rig, masked |
| Blender scenes | 3 | 41.377 | new, rendered, exact poses |
| Phone capture | 1 | 22.180 | new, phone, COLMAP poses |
| DL3DV bounded | 4 | 17.670 | new existing, real, video, unmasked |

The ordering is the result. Rendered data with exact poses scores highest.
Calibrated real data with masks comes within 1.7 dB of the paper's real
benchmark. A hand-held masked capture is about 6 dB below Tanks and Temples and
4.5 dB below DTU. Unmasked scene-scale video fails outright regardless of its
annotations.

---

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
- The 22.180 phone-capture run's terminal log was lost; its `eval.txt` survives
  in the render folder.
- scan69's view count and material are confounded; the controlled test was not
  run.
- The exposure-drift hypothesis for DL3DV is untested.
- The LLM judge was validated on 10 held-out images with one human labeller, so
  its agreement figure is uncertain.
