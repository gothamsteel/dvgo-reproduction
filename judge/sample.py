#!/usr/bin/env python3
"""Pick 30 test renders for the LLM judge: 5 per source, random with a fixed
seed, spread across scenes. Copies them under anonymous shuffled ids so
labellers cannot see source or PSNR, and assigns a 20/10 dev/held-out split."""
import csv, random, shutil, os
from collections import defaultdict

SEED, PER_SOURCE = 1, 5
SOURCES = {
    'synthetic_nerf': ['dvgo_lego'],
    'blendedmvs': ['dvgo_Character', 'dvgo_Fountain', 'dvgo_Jade', 'dvgo_Statues'],
    'dtu': ['dvgo_dtu_scan69', 'dvgo_dtu_scan97', 'dvgo_dtu_scan105',
            'dvgo_dtu_scan114', 'dvgo_dtu_scan118'],
    'dl3dv': ['dvgo_dl3dv_non_reflect', 'dvgo_dl3dv_non_reflect2',
              'dvgo_dl3dv_reflect1', 'dvgo_dl3dv_reflect2'],
    'blender': ['dvgo_synth_cat', 'dvgo_synth_lion', 'dvgo_synth_propane'],
    'phone': ['dvgo_real_can_sam_v2'],
}

rows, seen = defaultdict(list), set()
for r in csv.DictReader(open('per_image_psnr.csv')):
    key = (r['scene'], r['index'])
    if key in seen:
        continue
    seen.add(key)
    rows[r['scene']].append(r)

rng = random.Random(SEED)
picked = []
for source, scenes in SOURCES.items():
    pools = {s: rng.sample(rows[s], len(rows[s])) for s in scenes}
    order = rng.sample(scenes, len(scenes))
    k = 0
    while len([p for p in picked if p['source_group'] == source]) < PER_SOURCE:
        s = order[k % len(order)]
        if pools[s]:
            r = dict(pools[s].pop())
            r['source_group'] = source
            picked.append(r)
        k += 1

rng.shuffle(picked)
split = ['dev'] * 20 + ['heldout'] * 10
rng.shuffle(split)
os.makedirs('samples', exist_ok=True)
with open('samples_key.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'split', 'source', 'scene', 'index', 'psnr', 'render'])
    for i, (r, sp) in enumerate(zip(picked, split), 1):
        sid = f's{i:02d}'
        shutil.copy(r['render'], f'samples/{sid}.png')
        w.writerow([sid, sp, r['source_group'], r['scene'], r['index'], r['psnr'], r['render']])
for who in ('eric', 'friend'):
    with open(f'labels_{who}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'artefacts'])
        for i in range(1, len(picked) + 1):
            w.writerow([f's{i:02d}', ''])
print(f'{len(picked)} samples written')
