#!/usr/bin/env python3
"""Draw 30 more held-out renders for the LLM judge, after prompt v2 was fixed.

Same rule as sample.py (5 per source, random with a fixed seed, spread across
scenes), but a new seed, and never a render already in samples_key.csv.
Copies them to samples/ as s31 to s60 and writes two NEW files, so the
reported files (samples_key.csv, labels_eric.csv) are never touched:
    samples_key_extra.csv   which render each new id is (split 'extra')
    labels_eric_extra.csv   blank labels for label.py

Usage (from judge/):  python sample_extra.py
"""
import csv, os, random, shutil, sys
from collections import defaultdict
from sources import SOURCES

SEED, PER_SOURCE, FIRST_ID = 2, 5, 31
KEY, LABELS = 'samples_key_extra.csv', 'labels_eric_extra.csv'
DRY = '--dry-run' in sys.argv

for f in (KEY, LABELS):
    if os.path.exists(f) and not DRY:
        sys.exit(f'{f} already exists; delete it first if you really want to redraw')

used = {(r['scene'], r['index']) for r in csv.DictReader(open('samples_key.csv'))}
rows, seen = defaultdict(list), set()
for r in csv.DictReader(open('per_image_psnr.csv')):
    k = (r['scene'], r['index'])
    if k in seen or k in used:
        continue
    seen.add(k)
    rows[r['scene']].append(r)

rng = random.Random(SEED)
picked = []
for source, scenes in SOURCES.items():
    pools = {s: rng.sample(rows[s], len(rows[s])) for s in scenes}
    order = rng.sample(scenes, len(scenes))
    k = 0
    while sum(p['source_group'] == source for p in picked) < PER_SOURCE:
        if not any(pools.values()):
            sys.exit(f'not enough unused renders for {source}')
        s = order[k % len(order)]
        if pools[s]:
            r = dict(pools[s].pop())
            r['source_group'] = source
            picked.append(r)
        k += 1
rng.shuffle(picked)

ids = [f's{i}' for i in range(FIRST_ID, FIRST_ID + len(picked))]

# Renders may have moved since per_image_psnr.csv was written. Look for each one
# under ../logs and under every folder in DVGO_EXTRA_LOGS (colon separated).
ROOTS = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')]
ROOTS += [d for d in os.environ.get('DVGO_EXTRA_LOGS', '').split(':') if d]
def locate(path):
    if os.path.exists(path):
        return path
    tail = path.split('/logs/', 1)[-1]
    for root in ROOTS:
        cand = os.path.join(root, tail)
        if os.path.exists(cand):
            return cand
    return None
missing = []
for r in picked:
    found = locate(r['render'])
    if found is None:
        missing.append(r['render'])
    else:
        r['render'] = found
if missing:
    sys.exit(f'{len(missing)} renders not found, nothing written. First one: {missing[0]}\n'
             'Set DVGO_EXTRA_LOGS to the folder that now holds your logs.')
if DRY:
    for sid, r in zip(ids, picked):
        print(sid, r['source_group'], r['scene'], r['index'], r['psnr'])
    sys.exit(0)
os.makedirs('samples', exist_ok=True)
with open(KEY, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'split', 'source', 'scene', 'index', 'psnr', 'render'])
    for sid, r in zip(ids, picked):
        shutil.copy(r['render'], f'samples/{sid}.png')
        w.writerow([sid, 'extra', r['source_group'], r['scene'], r['index'], r['psnr'], r['render']])
with open(LABELS, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'artefacts'])
    for sid in ids:
        w.writerow([sid, ''])
print(f'{len(picked)} new samples: {ids[0]} to {ids[-1]}; label them with: python label.py {LABELS}')
