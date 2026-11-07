#!/usr/bin/env python3
"""LLM-as-judge for DVGO renders, following the week 9 practical: the verdict is
read from the next-token probabilities of Yes and No, so every answer is valid
and comes with a confidence. Compares against human labels with accuracy and
Cohen's kappa, plus a lazy annotator that always gives the majority label.

Usage: python judge.py --prompt prompt_v1.txt --split dev
       python judge.py --prompt prompt_v2.txt --split extra \
           --key samples_key_extra.csv --labels labels_eric_extra.csv
"""
import argparse, csv, os, torch
from collections import Counter
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from transformers import AutoProcessor, AutoModelForImageTextToText

ap = argparse.ArgumentParser()
ap.add_argument('--prompt', default='prompt_v1.txt')
ap.add_argument('--split', default='dev', choices=['dev', 'heldout', 'extra', 'all'])
ap.add_argument('--key', default='samples_key.csv')
ap.add_argument('--labels', default='labels_eric.csv')
ap.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
args = ap.parse_args()

key = [r for r in csv.DictReader(open(args.key))
       if args.split == 'all' or r['split'] == args.split]
human = {r['id']: r['artefacts'] for r in csv.DictReader(open(args.labels))}
prompt = open(args.prompt).read().strip()

proc = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(
    args.model, dtype=torch.bfloat16, device_map='cuda').eval()
tok = proc.tokenizer


def ids(word):
    return list({tok.encode(w, add_special_tokens=False)[0] for w in (word, ' ' + word)})


@torch.no_grad()
def p_yes(img):
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + 'Answer:'
    inp = proc(text=[text], images=[img], return_tensors='pt').to(model.device)
    p = torch.softmax(model(**inp).logits[0, -1].float(), -1)
    y, n = p[ids('Yes')].sum().item(), p[ids('No')].sum().item()
    return y / (y + n)


rows = []
for r in key:
    img = Image.open(f"samples/{r['id']}.png").convert('RGB')
    img.thumbnail((1000, 800))  # the same size the human labeller saw
    py = p_yes(img)
    rows.append(dict(id=r['id'], split=r['split'], source=r['source'], psnr=r['psnr'],
                     human=human[r['id']], p_yes=round(py, 3), judge='yes' if py > 0.5 else 'no'))
    x = rows[-1]
    print(f"{x['id']}  {x['source']:15s} psnr {float(x['psnr']):6.2f}  human {x['human']:3s}"
          f"  judge {x['judge']:3s} (p_yes {x['p_yes']:.3f})" + ('' if x['human'] == x['judge'] else '   <-- disagree'))

tag = os.path.splitext(os.path.basename(args.prompt))[0]
with open(f'judge_{tag}_{args.split}.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

h, j = [x['human'] for x in rows], [x['judge'] for x in rows]
lazy = [Counter(h).most_common(1)[0][0]] * len(h)
print(f"\n{args.split}: n={len(h)}")
print(f"judge  accuracy {accuracy_score(h, j):.3f}  kappa {cohen_kappa_score(h, j):.3f}")
print(f"lazy   accuracy {accuracy_score(h, lazy):.3f}  kappa 0.000  (always '{lazy[0]}')")
print("confusion, rows = human [no, yes], cols = judge [no, yes]")
print(confusion_matrix(h, j, labels=['no', 'yes']))
