# LLM-as-judge

Results are summarised in [`../results.md`](../results.md), section 4. All
commands below run from this folder.

## Environment

The judge needs a recent `transformers`, so it has its own environment:

```bash
conda create -n judge python=3.10
conda activate judge
pip install torch==2.10.0 torchvision==0.25.0 transformers==5.20.2 \
    accelerate scikit-learn scipy pandas pillow
```

The model (about 16 GB in bfloat16) downloads from Hugging Face on first run
and fits on a 24 GB GPU.

## Design

PSNR, SSIM and LPIPS say how far a render is from the real image, but not what
is wrong with it. The judge adds one yes/no question per render: does it
contain visible reconstruction artefacts (floaters, specks, ragged edges,
holes or broken geometry)?

- **Model:** Qwen2.5-VL-7B-Instruct, run locally in bfloat16. Following the
  unit's week 9 practical, the verdict is the probability that the next token
  is *Yes* rather than *No*; above 0.5 counts as yes.
- **Input:** the render only. The judge never sees the real image.
- **Human labels:** 30 renders, 5 per source, drawn at random with a fixed seed
  and shown under anonymous ids, so the labeller could not see the source or
  PSNR. Labelled by the author with [`LABELLING_GUIDE.md`](LABELLING_GUIDE.md)
  (13 yes, 17 no). 20 were used to develop the prompt and 10 were held out.
- **Prompts:** `prompt_v1.txt` matches the labelling guide. `prompt_v2.txt`
  was written after the development round and is the one reported. The held-out
  10 were judged once, with v2, and no further changes.
- **Second judge:** GPT-6 in ChatGPT (medium reasoning), given `prompt_v2.txt`
  and one image per new chat. Its answers are in `labels_chatgpt.csv`.

## Checking the numbers

Every verdict is committed as a CSV, so every agreement number can be checked
without a GPU:

```bash
python - <<'EOF'
import csv
from sklearn.metrics import accuracy_score, cohen_kappa_score
human = {r['id']: r['artefacts'] for r in csv.DictReader(open('labels_eric.csv'))}
gpt = {r['id']: r['artefacts'] for r in csv.DictReader(open('labels_chatgpt.csv'))}
held = list(csv.DictReader(open('judge_prompt_v2_heldout.csv')))
ids = sorted(human)
h, g = [human[i] for i in ids], [gpt[i] for i in ids]
print('GPT-6, all 30:   acc %.2f  kappa %.2f' % (accuracy_score(h, g), cohen_kappa_score(h, g)))
h, q = [r['human'] for r in held], [r['judge'] for r in held]
print('Qwen, held out:  acc %.2f  kappa %.2f' % (accuracy_score(h, q), cohen_kappa_score(h, q)))
EOF
```

## Rerunning

To rerun the judge itself (`judge` environment):

```bash
python per_image_psnr.py per_image_psnr.csv <log folders>   # per-render PSNR
python judge.py --prompt prompt_v1.txt --split dev
python judge.py --prompt prompt_v2.txt --split dev
python judge.py --prompt prompt_v2.txt --split heldout
python scale.py        # prompt v2 on every test render, writes judge_scale_v2.csv
python label.py labels_eric.csv   # the labelling tool: y = artefacts, n = clean, b = back
```

Three things to know before rerunning:

- `judge.py` reads the 30 images from `samples/`. These are copies of test
  renders, so they are not committed; they exist once the scenes are trained.
- `sample.py` draws the 30 images. **Running it again overwrites
  `samples_key.csv` and resets `labels_eric.csv` to blank**, so do not run it
  to check the reported numbers.
- `per_image_psnr.csv` and `samples_key.csv` store absolute paths from the
  machine used. Regenerate `per_image_psnr.csv` from your own log folders
  before running `scale.py`. The scripts find the repository from their own
  location, so it can be cloned anywhere; logs kept outside `logs/` can be
  added for `make_scene_fig.py` with `DVGO_EXTRA_LOGS=/path/to/logs`.

## Files

| File | Contents |
|---|---|
| `LABELLING_GUIDE.md` | the guide the human labeller followed |
| `prompt_v1.txt`, `prompt_v2.txt` | the two judge prompts |
| `samples_key.csv` | the 30 sampled renders, their split, source and PSNR |
| `labels_eric.csv`, `labels_chatgpt.csv` | human labels and GPT-6 answers |
| `judge_prompt_*_{dev,heldout}.csv` | Qwen verdicts with `p_yes` per image |
| `judge_scale_v2.csv` | Qwen verdicts on all 495 test renders |
| `per_image_psnr.csv` | PSNR of every test render, from the saved images |
| `smoke_test.py` | checks the model loads and answers simple questions |
| `make_results_fig.py`, `make_scene_fig.py` | build the report's render figures and copy the test-path videos |
