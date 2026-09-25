#!/usr/bin/env python3
"""Convert an IDR/NeuS-preprocessed DTU scan into tankstemple format for DVGO.

Why tankstemple and not blender: the blender path stores only a focal length
and DVGO then assumes the principal point sits at (W/2, H/2). DTU's calibrated
cx, cy are off centre, so that assumption shifts every ray. The tankstemple
loader reads a full K from intrinsics.txt and uses it unchanged.

Input (NeuS layout):
    <src>/cameras_sphere.npz
    <src>/image/000.png ...
    <src>/mask/000.png  ...

Output (tankstemple layout):
    <dst>/rgb/0_000.png      0_ = train, 1_ = test
    <dst>/pose/0_000.txt     4x4 camera-to-world, OpenCV convention
    <dst>/intrinsics.txt     4x4 K

Config must set dataset_type='tankstemple', inverse_y=True, white_bkgd=True.

Usage:
    python dtu_to_tnt.py data/DTU/dtu_scan114 data/dtu_tnt/dtu_scan114
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image


def load_K_Rt_from_P(P):
    """Decompose a 3x4 projection matrix into intrinsics and camera-to-world pose.

    Same decomposition IDR and NeuS use (models/dataset.py: load_K_Rt_from_P).
    """
    out = cv2.decomposeProjectionMatrix(P)
    K, R, t = out[0], out[1], out[2]
    K = K / K[2, 2]

    pose = np.eye(4)
    pose[:3, :3] = R.T
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return K, pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="NeuS scan folder")
    ap.add_argument("dst", help="output folder")
    ap.add_argument("--test-every", type=int, default=8,
                    help="hold out every Nth view for test (default 8)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.dst, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.dst, "pose"), exist_ok=True)

    cam = np.load(os.path.join(args.src, "cameras_sphere.npz"))
    n = sum(1 for k in cam.files
            if k.startswith("world_mat_") and "inv" not in k)
    print(f"found {n} views")

    test_idx = set(range(0, n, args.test_every))
    Ks, radii = [], []
    n_train = n_test = 0

    for i in range(n):
        P = (cam[f"world_mat_{i}"] @ cam[f"scale_mat_{i}"])[:3, :4]
        K, pose = load_K_Rt_from_P(P)
        # no axis flip: inverse_y=True expects the OpenCV convention these
        # poses are already in.

        img_path = os.path.join(args.src, "image", f"{i:06d}.png")
        msk_path = os.path.join(args.src, "mask", f"{i:03d}.png")

        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.array(Image.open(msk_path).convert("L"))
        if msk.shape != img.shape[:2]:
            raise SystemExit(f"view {i}: mask {msk.shape} != image {img.shape[:2]}")

        split = 1 if i in test_idx else 0
        if split:
            n_test += 1
        else:
            n_train += 1
        stem = f"{split}_{i:03d}"

        Image.fromarray(np.dstack([img, msk])).save(
            os.path.join(args.dst, "rgb", f"{stem}.png"))
        np.savetxt(os.path.join(args.dst, "pose", f"{stem}.txt"), pose, fmt="%.10f")

        Ks.append(K)
        radii.append(float(np.linalg.norm(pose[:3, 3])))

    H, W = img.shape[:2]
    Ks = np.stack(Ks)

    # tankstemple stores one shared K. Verify the views actually agree.
    spread = np.abs(Ks - Ks.mean(0)).max()
    print(f"max per-view deviation in K: {spread:.4f}")
    if spread > 1.0:
        print("WARNING: intrinsics vary across views; a single intrinsics.txt "
              "cannot represent this capture")

    K = Ks.mean(0)
    K4 = np.eye(4)
    K4[:3, :3] = K
    np.savetxt(os.path.join(args.dst, "intrinsics.txt"), K4, fmt="%.10f")

    print(f"fx {K[0,0]:.2f}  fy {K[1,1]:.2f}")
    print(f"cx {K[0,2]:.1f} (W/2 {W/2:.1f})   cy {K[1,2]:.1f} (H/2 {H/2:.1f})")
    print(f"camera distance from origin: min {min(radii):.3f} "
          f"mean {np.mean(radii):.3f} max {max(radii):.3f}")
    print(f"train {n_train}, test {n_test}, resolution {W}x{H}")
    print(f"wrote to {args.dst}")


if __name__ == "__main__":
    main()