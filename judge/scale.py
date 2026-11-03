#!/usr/bin/env python3
"""Run the frozen judge (prompt v2) on every test render of the six sources.
Held-out agreement was moderate with no false positives, so the per-source
'yes' rate is reported as a lower bound on the artefact rate."""
import csv, torch
import pandas as pd
from scipy.stats import spearmanr
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from sources import SOURCES  # same scene lists as the 30-sample draw

MODEL, PROMPT = 'Qwen/Qwen2.5-VL-7B-Instruct', open('prompt_v2.txt').read().strip()
scene2src = {s: src for src, scenes in SOURCES.items() for s in scenes}

seen, todo = set(), []
for r in csv.DictReader(open('per_image_psnr.csv')):
    k = (r['scene'], r['index'])
    if r['scene'] in scene2src and k not in seen:
        seen.add(k); todo.append(r)

proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda').eval()
tok = proc.tokenizer
ids = lambda w: list({tok.encode(x, add_special_tokens=False)[0] for x in (w, ' ' + w)})
YES, NO = ids('Yes'), ids('No')

@torch.no_grad()
def p_yes(img):
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + 'Answer:'
    inp = proc(text=[text], images=[img], return_tensors='pt').to(model.device)
    p = torch.softmax(model(**inp).logits[0, -1].float(), -1)
    y, n = p[YES].sum().item(), p[NO].sum().item()
    return y / (y + n)

out = []
for k, r in enumerate(todo, 1):
    img = Image.open(r['render']).convert('RGB'); img.thumbnail((1000, 800))
    py = p_yes(img)
    out.append(dict(source=scene2src[r['scene']], scene=r['scene'], index=r['index'],
                    psnr=float(r['psnr']), p_yes=round(py, 3), judge='yes' if py > 0.5 else 'no'))
    if k % 50 == 0: print(f'{k}/{len(todo)}', flush=True)

df = pd.DataFrame(out); df.to_csv('judge_scale_v2.csv', index=False)
order = list(SOURCES)
t = df.groupby('source').agg(n=('psnr', 'size'), mean_psnr=('psnr', 'mean'),
                             yes_rate=('judge', lambda s: (s == 'yes').mean()),
                             mean_p_yes=('p_yes', 'mean')).reindex(order).round(3)
print(t.to_string())
rho, p = spearmanr(df.psnr, df.p_yes)
print(f'\nall renders: Spearman(PSNR, p_yes) = {rho:.3f}  (p = {p:.1e}, n = {len(df)})')
for s in order:
    d = df[df.source == s]
    if d.psnr.nunique() > 2:
        print(f'  within {s:15s} rho = {spearmanr(d.psnr, d.p_yes)[0]:+.3f}  (n = {len(d)})')
