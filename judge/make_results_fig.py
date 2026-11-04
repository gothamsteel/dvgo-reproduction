#!/usr/bin/env python3
"""Pick one representative test view per data source and export the real
image and the DVGO render side by side, for the report's results figure.

The view is the one whose PSNR is closest to the median PSNR of that source,
so the figure shows a typical result, not the best one. Ground truth is
composited onto white where it has an alpha channel, as in run.py.

Usage (from ~/DirectVoxGO/judge, in the judge env):
    python make_results_fig.py
Output: fig_res/<source>_gt.jpg, fig_res/<source>_render.jpg, fig_res/picked.csv
"""
import csv, os
import numpy as np
from PIL import Image

SOURCES = {
    'synthetic_nerf': ['dvgo_lego'],
    'blendedmvs': ['dvgo_Character', 'dvgo_Fountain', 'dvgo_Jade', 'dvgo_Statues'],
    'tanks_and_temples': ['dvgo_Barn', 'dvgo_Caterpillar', 'dvgo_Family',
                          'dvgo_Ignatius', 'dvgo_Truck'],
    'dtu': ['dvgo_dtu_scan69', 'dvgo_dtu_scan97', 'dvgo_dtu_scan105',
            'dvgo_dtu_scan114', 'dvgo_dtu_scan118'],
    'dl3dv': ['dvgo_dl3dv_non_reflect', 'dvgo_dl3dv_non_reflect2',
              'dvgo_dl3dv_reflect1', 'dvgo_dl3dv_reflect2'],
    'blender': ['dvgo_synth_cat', 'dvgo_synth_lion', 'dvgo_synth_propane'],
    'phone': ['dvgo_real_can_sam_v2'],
}
OUT, MAXSIDE, MARGIN = 'fig_res', 600, 25


def rgb_on_white(path):
    im = Image.open(path)
    if im.mode in ('RGBA', 'LA'):
        im = im.convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, im).convert('RGB')
    return im.convert('RGB')


def content_box(*ims, thr=245):
    """Bounding box of non-white pixels over all images, plus a margin."""
    mask = np.zeros(np.asarray(ims[0]).shape[:2], bool)
    for im in ims:
        mask |= (np.asarray(im) < thr).any(2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    w, h = ims[0].size
    return (max(0, xs.min() - MARGIN), max(0, ys.min() - MARGIN),
            min(w, xs.max() + MARGIN), min(h, ys.max() + MARGIN))


rows, seen = [], set()
for r in csv.DictReader(open('per_image_psnr.csv')):
    k = (r['scene'], r['index'])
    if k not in seen:
        seen.add(k)
        rows.append(r)

os.makedirs(OUT, exist_ok=True)
picked = []
for src, scenes in SOURCES.items():
    cand = [r for r in rows if r['scene'] in scenes]
    if not cand:
        print(f'{src:18s} SKIP: no rows in per_image_psnr.csv')
        continue
    med = float(np.median([float(r['psnr']) for r in cand]))
    best = min(cand, key=lambda r: abs(float(r['psnr']) - med))
    gt, rd = rgb_on_white(best['gt']), rgb_on_white(best['render'])
    if gt.size != rd.size:
        gt = gt.resize(rd.size, Image.BOX)
    box = content_box(gt, rd)
    if box:
        gt, rd = gt.crop(box), rd.crop(box)
    for im, tag in ((gt, 'gt'), (rd, 'render')):
        im = im.copy()
        im.thumbnail((MAXSIDE, MAXSIDE))
        im.save(f'{OUT}/{src}_{tag}.jpg', quality=90)
    picked.append(dict(source=src, scene=best['scene'], index=best['index'],
                       psnr=best['psnr'], source_median=round(med, 3), n=len(cand)))
    print(f"{src:18s} {best['scene']:28s} view {best['index']:>3s}  "
          f"PSNR {float(best['psnr']):6.2f}  (source median {med:6.2f}, n={len(cand)})")

with open(f'{OUT}/picked.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(picked[0]))
    w.writeheader()
    w.writerows(picked)
print(f'wrote {2 * len(picked)} images to {OUT}/')
