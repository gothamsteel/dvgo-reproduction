#!/usr/bin/env python3
"""Per-image test PSNR for DVGO render folders.

Reads datadir, dataset_type and white_bkgd from the config.py that DVGO saved
in each log folder, loads the test ground truth the way run.py does, and
compares it with render_test_fine_last/NNN.png. Prints each scene's mean next
to eval.txt as a correctness check.

Usage: python per_image_psnr.py out.csv LOGDIR [LOGDIR ...]
"""
import csv, glob, json, os, re, sys
import numpy as np
from PIL import Image

ROOT = os.path.expanduser('~/DirectVoxGO')  # configs use paths relative to this


def cfg(txt, key):
    m = re.search(rf"{key}\s*=\s*['\"]?([^'\",\)\n]+)", txt)
    return m.group(1).strip() if m else None


def load(path):
    a = np.asarray(Image.open(path)).astype(np.float32) / 255.
    return np.stack([a] * 3, -1) if a.ndim == 2 else a


def gt_rgb(a, white):
    if a.shape[-1] == 4:  # same compositing as run.py
        rgb, al = a[..., :3], a[..., 3:]
        return rgb * al + (1. - al) if white else rgb * al
    return a[..., :3]


def test_files(datadir, dtype):
    if dtype == 'blender':
        meta = json.load(open(os.path.join(datadir, 'transforms_test.json')))
        return [os.path.join(datadir, f['file_path'] + '.png') for f in meta['frames']]
    # tankstemple / blendedmvs loaders: prefix 1_ is the test split
    return [p for p in sorted(glob.glob(os.path.join(datadir, 'rgb', '*png')))
            if os.path.basename(p).startswith('1_')]


out, logs = sys.argv[1], sys.argv[2:]
rows = []
for log in sorted(logs):
    rdir = os.path.join(log, 'render_test_fine_last')
    if not os.path.isfile(os.path.join(log, 'config.py')) or not os.path.isdir(rdir):
        continue
    txt = open(os.path.join(log, 'config.py')).read()
    datadir = os.path.normpath(os.path.join(ROOT, cfg(txt, 'datadir')))
    dtype, white = cfg(txt, 'dataset_type'), cfg(txt, 'white_bkgd') == 'True'
    name = f"{os.path.basename(os.path.dirname(log))}/{os.path.basename(log)}"
    if not os.path.isdir(datadir):
        print(f"{name:40s} SKIP, no data at {datadir}")
        continue
    renders = sorted(glob.glob(os.path.join(rdir, '[0-9][0-9][0-9].png')))
    gts = test_files(datadir, dtype)
    if len(renders) != len(gts):
        print(f"{name:40s} SKIP, {len(renders)} renders vs {len(gts)} ground truth")
        continue
    ps = []
    for i, (r, g) in enumerate(zip(renders, gts)):
        R, G = load(r)[..., :3], gt_rgb(load(g), white)
        if R.shape != G.shape:
            G = np.asarray(Image.fromarray((G * 255).astype(np.uint8)).resize(
                (R.shape[1], R.shape[0]), Image.BOX)).astype(np.float32) / 255.
        p = -10. * np.log10(np.mean((R - G) ** 2))
        ps.append(p)
        rows.append(dict(source=os.path.basename(os.path.dirname(log)),
                         scene=os.path.basename(log), index=i, psnr=round(p, 3),
                         render=r, gt=g))
    ev = open(os.path.join(rdir, 'eval.txt')).read() if os.path.isfile(os.path.join(rdir, 'eval.txt')) else ''
    m = re.search(r"psnr\s+([\d.]+)", ev)
    print(f"{name:40s} n={len(ps):3d} mean={np.mean(ps):7.3f} eval.txt={m.group(1) if m else '?'}")

with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {out}")
