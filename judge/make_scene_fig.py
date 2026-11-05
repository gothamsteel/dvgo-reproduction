#!/usr/bin/env python3
"""Representative test views for the appendix figures.

Replication datasets: one render per scene, the view whose PSNR is closest to
the scene's median. Synthetic-NeRF scenes other than lego have no ground
truth on disk, so all eight use one fixed test view (FIXED_NERF_VIEW).

New data: three views per scene, the lowest, median and highest PSNR view,
each as real image and render cropped identically, so the figure shows the
spread of quality from several viewpoints rather than one chosen view.

Also copies each new-data scene's test-path video (video.rgb.mp4) to
VIDEO_DIR, for the repository.

Usage (from judge/, in the judge env):
    python make_scene_fig.py                              # every source
    python make_scene_fig.py synthetic_nsvf deepvoxels    # only these sources
Logs are searched in the repository's logs/ folder. Logs kept elsewhere can be
added with DVGO_EXTRA_LOGS=/path/to/logs (several paths separated by ':').
Output: fig_scenes/<source>__<scene>_<lo|med|hi>_{render,gt}.jpg, list.csv
"""
import csv, glob, os, re, shutil, sys
import numpy as np
from PIL import Image

# repository root (the folder above judge/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# log folders to search: the repository's logs/, plus any extra folders listed
# in DVGO_EXTRA_LOGS (separated by ':'), e.g. logs moved to another disk
LOG_ROOTS = [os.path.join(ROOT, 'logs')] + [
    p for p in os.environ.get('DVGO_EXTRA_LOGS', '').split(os.pathsep) if p]
SOURCES = {
    'synthetic_nerf': ['dvgo_chair', 'dvgo_drums', 'dvgo_ficus', 'dvgo_hotdog',
                       'dvgo_lego', 'dvgo_materials', 'dvgo_mic', 'dvgo_ship'],
    'synthetic_nsvf': ['dvgo_Wineholder', 'dvgo_Steamtrain', 'dvgo_Toad', 'dvgo_Robot',
                       'dvgo_Bike', 'dvgo_Palace', 'dvgo_Spaceship', 'dvgo_Lifestyle'],
    'blendedmvs': ['dvgo_Jade', 'dvgo_Fountain', 'dvgo_Character', 'dvgo_Statues'],
    'tanks_and_temples': ['dvgo_Barn', 'dvgo_Caterpillar', 'dvgo_Family',
                          'dvgo_Ignatius', 'dvgo_Truck'],
    'dtu': ['dvgo_dtu_scan69', 'dvgo_dtu_scan97', 'dvgo_dtu_scan105',
            'dvgo_dtu_scan114', 'dvgo_dtu_scan118'],
    'dl3dv': ['dvgo_dl3dv_non_reflect', 'dvgo_dl3dv_non_reflect2',
              'dvgo_dl3dv_reflect1', 'dvgo_dl3dv_reflect2'],
    'blender': ['dvgo_synth_cat', 'dvgo_synth_propane', 'dvgo_synth_lion'],
    'phone': ['dvgo_real_can_sam_v2'],
    'deepvoxels': ['dvgo_armchair', 'dvgo_greek', 'dvgo_cube', 'dvgo_vase'],
}
NEW_DATA = {'dtu', 'dl3dv', 'blender', 'phone'}
FIXED_NERF_VIEW = 182          # lego's median view, used for every NeRF scene
OUT, MAXSIDE, MARGIN = 'fig_scenes', 420, 20
VIDEO_DIR = os.path.join(ROOT, 'media', 'videos')


def find_log(scene):
    for root in LOG_ROOTS:
        for d in glob.glob(f'{root}/*/{scene}'):
            if os.path.isdir(f'{d}/render_test_fine_last'):
                return d
    return None


def scene_psnr(logdir):
    p = f'{logdir}/render_test_fine_last/eval.txt'
    if os.path.isfile(p):
        m = re.search(r'psnr\s+([\d.]+)', open(p).read())
        if m:
            return float(m.group(1))
    return None


def rgb_on_white(path):
    im = Image.open(path)
    if im.mode in ('RGBA', 'LA'):
        im = im.convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, im).convert('RGB')
    return im.convert('RGB')


def content_box(*ims, thr=245):
    mask = np.zeros(np.asarray(ims[0]).shape[:2], bool)
    for im in ims:
        mask |= (np.asarray(im) < thr).any(2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    w, h = ims[0].size
    return (max(0, xs.min() - MARGIN), max(0, ys.min() - MARGIN),
            min(w, xs.max() + MARGIN), min(h, ys.max() + MARGIN))


rows, seen = {}, set()
for r in csv.DictReader(open('per_image_psnr.csv')):
    k = (r['scene'], r['index'])
    if k not in seen:
        seen.add(k)
        rows.setdefault(r['scene'], []).append(r)

os.makedirs(OUT, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
out = []


def export(src, sc, tag, log, idx, vpsnr, gt_path, sp):
    rd = rgb_on_white(f'{log}/render_test_fine_last/{idx:03d}.png')
    gt = None
    if gt_path:
        gt = rgb_on_white(gt_path)
        if gt.size != rd.size:
            gt = gt.resize(rd.size, Image.BOX)
    box = content_box(*([rd, gt] if gt is not None else [rd]))
    stem = f'{OUT}/{src}__{sc.replace("dvgo_", "")}_{tag}'
    for im, kind in ((rd, 'render'), (gt, 'gt')):
        if im is None:
            continue
        im = im.crop(box) if box else im.copy()
        im.thumbnail((MAXSIDE, MAXSIDE))
        im.save(f'{stem}_{kind}.jpg', quality=88)
    out.append(dict(source=src, scene=sc.replace('dvgo_', ''), pick=tag, view=idx,
                    view_psnr='' if vpsnr is None else round(vpsnr, 2),
                    scene_psnr='' if sp is None else round(sp, 2),
                    has_gt=gt is not None))
    print(f"{src:18s} {sc:28s} {tag:3s} view {idx:3d}  PSNR "
          f"{'   n/a' if vpsnr is None else f'{vpsnr:6.2f}'}")


ONLY = set(sys.argv[1:])  # optional: limit to the named sources
for src, scenes in SOURCES.items():
    if ONLY and src not in ONLY:
        continue
    for sc in scenes:
        log = find_log(sc)
        if log is None:
            print(f'{src:18s} {sc:28s} SKIP: no log folder')
            continue
        sp = scene_psnr(log)
        if src == 'synthetic_nerf':
            export(src, sc, 'med', log, FIXED_NERF_VIEW, None, None, sp)
            continue
        cand = sorted(rows.get(sc, []), key=lambda r: float(r['psnr']))
        if not cand:
            print(f'{src:18s} {sc:28s} SKIP: no per-image PSNR')
            continue
        med = float(np.median([float(r['psnr']) for r in cand]))
        mid = min(cand, key=lambda r: abs(float(r['psnr']) - med))
        if src in NEW_DATA:
            picks = [('lo', cand[0]), ('med', mid), ('hi', cand[-1])]
            vid = f'{log}/render_test_fine_last/video.rgb.mp4'
            if os.path.isfile(vid):
                shutil.copy(vid, f'{VIDEO_DIR}/{src}__{sc.replace("dvgo_", "")}.mp4')
        else:
            picks = [('med', mid)]
        for tag, r in picks:
            export(src, sc, tag, log, int(r['index']), float(r['psnr']),
                   r['gt'] if src in NEW_DATA else None, sp)

with open(f'{OUT}/list.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
n_img = len(glob.glob(f'{OUT}/*.jpg'))
n_vid = len(glob.glob(f'{VIDEO_DIR}/*.mp4'))
print(f'{len(out)} views, {n_img} images in {OUT}/, {n_vid} videos in {VIDEO_DIR}/')
