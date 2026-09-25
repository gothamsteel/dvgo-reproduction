#!/usr/bin/env python3
"""Convert an undistorted COLMAP reconstruction into tankstemple format for DVGO.

Input:
    <colmap>/images/<name>.jpg            undistorted images
    <colmap>/sparse_txt/cameras.txt       must be a single PINHOLE camera
    <colmap>/sparse_txt/images.txt        world-to-camera poses + 2D observations
    <colmap>/sparse_txt/points3D.txt      sparse point cloud
    <masks>/<name>.png                    object mask, white on black (optional)

Output (tankstemple layout):
    <dst>/rgb/0_000.png      0_ = train, 1_ = test
    <dst>/pose/0_000.txt     4x4 camera-to-world, OpenCV convention
    <dst>/intrinsics.txt     4x4 K

Config must set dataset_type='tankstemple', inverse_y=True, white_bkgd=True.

COLMAP's world origin and scale are arbitrary. With masks supplied, the scene
is recentred and rescaled on the sparse points whose 2D observations land
inside the masks, i.e. the object's own points, so the object ends up roughly
inside a unit sphere the way the IDR/NeuS DTU scans are. Without masks it
falls back to all points.

Usage:
    python colmap_to_tnt.py capture2/undistorted data/real_can \\
        --masks capture2/masks_sam
"""

import argparse
import os
from collections import defaultdict

import numpy as np
from PIL import Image


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def read_cameras(path):
    cams = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if not p:
                continue
            cams[int(p[0])] = dict(model=p[1], w=int(p[2]), h=int(p[3]),
                                   params=[float(v) for v in p[4:]])
    return cams


def read_images(path):
    """Return [(name, qvec, tvec, camera_id, obs)] where obs is [(x, y, pt3d_id)]."""
    out = []
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        p = lines[i].split()
        if not p:
            continue
        q = np.array([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        q2 = lines[i + 1].split()
        obs = [(float(q2[j]), float(q2[j + 1]), int(q2[j + 2]))
               for j in range(0, len(q2) - 2, 3)]
        out.append((p[9], q, t, int(p[8]), obs))
    return out


def read_points(path):
    ids, xyz = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) > 3:
                ids.append(int(p[0]))
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(ids), np.array(xyz)


def load_mask(masks_dir, name, W, H):
    stem = os.path.splitext(name)[0]
    for ext in (".png", ".jpg", ".jpeg"):
        cand = os.path.join(masks_dir, stem + ext)
        if os.path.exists(cand):
            m = Image.open(cand).convert("L")
            if m.size != (W, H):
                m = m.resize((W, H))
            return np.array(m) > 127
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("colmap")
    ap.add_argument("dst")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--sparse", default="sparse_txt")
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--object-radius", type=float, default=1.0,
                    help="scale so the object's 90th-percentile point radius "
                         "equals this (default 1.0, matching the DTU scans)")
    args = ap.parse_args()

    model = os.path.join(args.colmap, args.sparse)
    cams = read_cameras(os.path.join(model, "cameras.txt"))
    imgs = read_images(os.path.join(model, "images.txt"))
    imgs.sort(key=lambda r: r[0])

    if len(cams) != 1:
        raise SystemExit(f"expected 1 camera, found {len(cams)}; "
                         "rerun COLMAP with --single_camera 1")
    cam = next(iter(cams.values()))
    if cam["model"] != "PINHOLE":
        raise SystemExit(f"camera model is {cam['model']}, not PINHOLE; "
                         "run colmap image_undistorter first")

    fx, fy, cx, cy = cam["params"]
    W, H = cam["w"], cam["h"]
    print(f"{len(imgs)} images, {W}x{H}")
    print(f"fx {fx:.2f}  fy {fy:.2f}")
    print(f"cx {cx:.1f} (W/2 {W/2:.1f})   cy {cy:.1f} (H/2 {H/2:.1f})")

    poses = []
    for name, q, t, _, _ in imgs:
        R = qvec2rotmat(q)
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        poses.append(c2w)
    poses = np.stack(poses)

    pt_ids, pts = read_points(os.path.join(model, "points3D.txt"))
    id_to_row = {pid: i for i, pid in enumerate(pt_ids)}

    if args.masks:
        inside = defaultdict(int)
        total = defaultdict(int)
        missing = 0
        for name, _, _, _, obs in imgs:
            m = load_mask(args.masks, name, W, H)
            if m is None:
                missing += 1
                continue
            for x, y, pid in obs:
                if pid < 0:
                    continue
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < W and 0 <= yi < H:
                    total[pid] += 1
                    if m[yi, xi]:
                        inside[pid] += 1
        if missing:
            print(f"WARNING: {missing} images had no mask")
        keep = [id_to_row[pid] for pid in total
                if pid in id_to_row and inside[pid] > 0.5 * total[pid]]
        obj_pts = pts[keep] if keep else pts
        print(f"{len(obj_pts)} of {len(pts)} sparse points are on the object")
    else:
        obj_pts = pts
        print(f"no masks given, normalising on all {len(pts)} points")

    centre = np.median(obj_pts, axis=0)
    r = np.linalg.norm(obj_pts - centre, axis=1)
    scale = args.object_radius / np.percentile(r, 90)

    poses[:, :3, 3] = (poses[:, :3, 3] - centre) * scale
    radii = np.linalg.norm(poses[:, :3, 3], axis=1)
    print(f"recentred on {centre.round(3)}, scaled by {scale:.4f}")
    print(f"object point radius: median {np.median(r)*scale:.3f} "
          f"p90 {np.percentile(r, 90)*scale:.3f} max {r.max()*scale:.3f}")
    print(f"camera distance from origin: min {radii.min():.3f} "
          f"mean {radii.mean():.3f} max {radii.max():.3f}")

    os.makedirs(os.path.join(args.dst, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.dst, "pose"), exist_ok=True)

    test_idx = set(range(0, len(imgs), args.test_every))
    n_train = n_test = 0

    for i, ((name, _, _, _, _), pose) in enumerate(zip(imgs, poses)):
        img = Image.open(os.path.join(args.colmap, "images", name)).convert("RGB")
        if (img.width, img.height) != (W, H):
            raise SystemExit(f"{name} is {img.size}, expected {(W, H)}")
        arr = np.array(img)

        alpha = np.full(arr.shape[:2], 255, np.uint8)
        if args.masks:
            m = load_mask(args.masks, name, W, H)
            if m is not None:
                alpha = (m * 255).astype(np.uint8)

        split = 1 if i in test_idx else 0
        n_test += split
        n_train += 1 - split
        stem = f"{split}_{i:03d}"

        Image.fromarray(np.dstack([arr, alpha])).save(
            os.path.join(args.dst, "rgb", f"{stem}.png"))
        np.savetxt(os.path.join(args.dst, "pose", f"{stem}.txt"), pose, fmt="%.10f")

    K4 = np.eye(4)
    K4[0, 0], K4[1, 1], K4[0, 2], K4[1, 2] = fx, fy, cx, cy
    np.savetxt(os.path.join(args.dst, "intrinsics.txt"), K4, fmt="%.10f")

    print(f"train {n_train}, test {n_test}")
    print(f"wrote to {args.dst}")


if __name__ == "__main__":
    main()
