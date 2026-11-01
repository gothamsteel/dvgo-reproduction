#!/usr/bin/env python3
"""Convert a DL3DV nerfstudio scene into tankstemple format for DVGO.

DL3DV ships Nerfstudio-style transforms.json. Two reasons not to feed that
straight to DVGO's blender loader:

  - the intrinsics in transforms.json describe the full-resolution frames
    (3840x2160), while the downloadable images_4 folder is 960x540, so
    everything must be scaled
  - fl_x and fl_y differ slightly, and the blender path stores a single
    focal length

tankstemple carries a full K, so both are handled. DVGO has no distortion
model, so the OPENCV k1/k2/p1/p2 terms are dropped; on the scenes checked
they are small (k1 ~ 0.003) but this is an approximation worth stating.

Input:
    <scene>/nerfstudio/transforms.json
    <scene>/nerfstudio/images_4/frame_00001.png ...

Output (tankstemple layout):
    <dst>/rgb/0_000.png      0_ = train, 1_ = test
    <dst>/pose/0_000.txt     4x4 camera-to-world, OpenCV convention
    <dst>/intrinsics.txt     4x4 K

Config must set dataset_type='tankstemple', inverse_y=True, white_bkgd=False
(these are whole scenes, not masked objects).

Usage:
    python dl3dv_to_tnt.py \\
        DL3DV-10K-Benchmark/<hash>/nerfstudio data/dl3dv_reflect
"""

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="scene's nerfstudio folder")
    ap.add_argument("dst", help="output folder")
    ap.add_argument("--images", default="images_4",
                    help="image subfolder actually present (default images_4)")
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap the number of frames, 0 for all")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.src, "transforms.json")))
    frames = meta["frames"]
    frames.sort(key=lambda f: f["file_path"])
    if args.max_frames:
        step = max(1, len(frames) // args.max_frames)
        frames = frames[::step][:args.max_frames]
    print(f"{len(frames)} frames")

    img_dir = os.path.join(args.src, args.images)
    first = os.path.basename(frames[0]["file_path"])
    probe = os.path.join(img_dir, first)
    if not os.path.exists(probe):
        raise SystemExit(f"image not found: {probe}")
    W, H = Image.open(probe).size

    # transforms.json intrinsics describe the full-resolution frames
    sx, sy = W / meta["w"], H / meta["h"]
    fx, fy = meta["fl_x"] * sx, meta["fl_y"] * sy
    cx, cy = meta["cx"] * sx, meta["cy"] * sy
    print(f"full res {meta['w']}x{meta['h']} -> using {W}x{H} (scale {sx:.4f})")
    print(f"fx {fx:.2f}  fy {fy:.2f}")
    print(f"cx {cx:.1f} (W/2 {W/2:.1f})   cy {cy:.1f} (H/2 {H/2:.1f})")

    dist = [meta.get(k, 0.0) for k in ("k1", "k2", "p1", "p2")]
    if any(abs(d) > 1e-4 for d in dist):
        print(f"NOTE: dropping distortion {dict(zip('k1 k2 p1 p2'.split(), dist))}; "
              "DVGO assumes a pinhole camera")

    os.makedirs(os.path.join(args.dst, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.dst, "pose"), exist_ok=True)

    test_idx = set(range(0, len(frames), args.test_every))
    radii = []
    n_train = n_test = 0

    for i, fr in enumerate(frames):
        name = os.path.basename(fr["file_path"])
        src_img = os.path.join(img_dir, name)
        if not os.path.exists(src_img):
            raise SystemExit(f"missing {src_img}")

        pose = np.array(fr["transform_matrix"], dtype=np.float64)
        # Nerfstudio is OpenGL (+Y up, -Z forward). inverse_y=True wants
        # OpenCV (+Y down, +Z forward): negate the 2nd and 3rd basis columns.
        pose[:3, 1:3] *= -1

        split = 1 if i in test_idx else 0
        n_test += split
        n_train += 1 - split
        stem = f"{split}_{i:03d}"

        im = Image.open(src_img)
        if im.size != (W, H):
            raise SystemExit(f"{name} is {im.size}, expected {(W, H)}")
        if im.mode != "RGB":
            im = im.convert("RGB")
            im.save(os.path.join(args.dst, "rgb", f"{stem}.png"))
        else:
            shutil.copy(src_img, os.path.join(args.dst, "rgb", f"{stem}.png"))

        np.savetxt(os.path.join(args.dst, "pose", f"{stem}.txt"), pose, fmt="%.10f")
        radii.append(float(np.linalg.norm(pose[:3, 3])))

    K4 = np.eye(4)
    K4[0, 0], K4[1, 1], K4[0, 2], K4[1, 2] = fx, fy, cx, cy
    np.savetxt(os.path.join(args.dst, "intrinsics.txt"), K4, fmt="%.10f")

    r = np.array(radii)
    print(f"camera distance from origin: min {r.min():.3f} "
          f"mean {r.mean():.3f} max {r.max():.3f}")
    print(f"train {n_train}, test {n_test}, resolution {W}x{H}")
    print(f"wrote to {args.dst}")


if __name__ == "__main__":
    main()
