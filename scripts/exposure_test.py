#!/usr/bin/env python3
"""Does changing exposure explain DVGO's failure on DL3DV?

A hand-held video camera adjusts exposure and white balance as it moves, so
the same surface can have different colours in different frames. DVGO stores
one colour per point, so it cannot fit both. If that is the problem, a simple
per-frame colour correction (gain and offset per channel, fitted between the
render and the photo) should raise PSNR far more on DL3DV than on DTU, which
was captured in a lab under fixed lighting.

Needs the renders saved by
    python run.py --config <cfg> --render_only --render_train --render_test --dump_images
Run from the repository root:
    python scripts/exposure_test.py configs/dl3dv/nonreflect.py ... configs/dtu/scan114.py ...
Writes results/exposure_test.csv and prints one line per scene and split.
"""
import csv, glob, os, sys
import numpy as np
import imageio.v2 as imageio
from mmengine import Config


def gt_images(datadir, prefix, white_bkgd):
    """Photos of one split, in the order DVGO renders them (sorted file names)."""
    paths = sorted(glob.glob(os.path.join(datadir, 'rgb', '*png')))
    out = []
    for p in paths:
        if os.path.basename(p)[0] != prefix:
            continue
        im = imageio.imread(p).astype(np.float32) / 255.
        if im.shape[-1] == 4:
            a = im[..., 3:]
            im = im[..., :3] * a + (1. - a) if white_bkgd else im[..., :3] * a
        out.append(im[..., :3])
    return out


def psnr(a, b):
    return -10. * np.log10(np.mean((a - b) ** 2))


def fit_gain_offset(render, gt):
    """Least-squares gain and offset per colour channel: gt ~ g * render + b."""
    g, b = np.ones(3), np.zeros(3)
    for c in range(3):
        x, y = render[:, c], gt[:, c]
        A = np.stack([x, np.ones_like(x)], 1)
        (g[c], b[c]), *_ = np.linalg.lstsq(A, y, rcond=None)
    return g, b


def run(cfg_path, split):
    cfg = Config.fromfile(cfg_path)
    white = bool(cfg.data.white_bkgd)
    prefix = '0' if split == 'train' else '1'
    gts = gt_images(cfg.data.datadir, prefix, white)
    rdir = os.path.join(cfg.basedir, cfg.expname, f'render_{split}_fine_last')
    rpaths = sorted(glob.glob(os.path.join(rdir, '[0-9][0-9][0-9].png')))
    if len(rpaths) != len(gts):
        sys.exit(f'{rdir}: {len(rpaths)} renders but {len(gts)} photos; render with --dump_images first')
    raw, cor, gains = [], [], []
    for rp, gt in zip(rpaths, gts):
        r = imageio.imread(rp).astype(np.float32)[..., :3] / 255.
        # On white-background data, score only the object, so the white
        # background does not pin the fit to "no change".
        m = ~np.all(gt >= 0.999, -1) if white else np.ones(gt.shape[:2], bool)
        rv, gv = r[m], gt[m]
        g, b = fit_gain_offset(rv, gv)
        raw.append(psnr(rv, gv))
        cor.append(psnr(np.clip(rv * g + b, 0, 1), gv))
        gains.append(g)
    gains = np.array(gains)
    stops = np.log2(np.clip(gains.mean(1), 1e-3, None))      # brightness change per frame
    wb = np.log2(np.clip(gains[:, 0] / gains[:, 2], 1e-3, None))  # red/blue balance per frame
    return dict(config=os.path.basename(os.path.dirname(cfg_path)) + '/' + os.path.basename(cfg_path),
                split=split, frames=len(gts), white_bkgd=white,
                psnr_raw=round(float(np.mean(raw)), 2), psnr_corrected=round(float(np.mean(cor)), 2),
                gain_db=round(float(np.mean(cor) - np.mean(raw)), 2),
                exposure_spread_stops=round(float(stops.std()), 3),
                white_balance_spread_stops=round(float(wb.std()), 3))


if __name__ == '__main__':
    rows = []
    for cfg_path in sys.argv[1:]:
        for split in ('train', 'test'):
            r = run(cfg_path, split)
            rows.append(r)
            print(f"{r['config']:26s} {split:5s} n={r['frames']:3d}  PSNR {r['psnr_raw']:6.2f} -> {r['psnr_corrected']:6.2f}"
                  f"  (+{r['gain_db']:.2f} dB)  exposure spread {r['exposure_spread_stops']:.3f} stops"
                  f"  white balance spread {r['white_balance_spread_stops']:.3f} stops")
    os.makedirs('results', exist_ok=True)
    with open('results/exposure_test.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('saved results/exposure_test.csv')
