import sys, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
tok = proc.tokenizer

def ids(word):
    return list({tok.encode(w, add_special_tokens=False)[0] for w in (word, " " + word)})

@torch.no_grad()
def ask(img, question, options=("Yes", "No")):
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer:"
    inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    p = torch.softmax(model(**inp).logits[0, -1].float(), -1)
    s = [p[ids(o)].sum().item() for o in options]
    return {o: round(v / sum(s), 3) for o, v in zip(options, s)}

img = Image.open(sys.argv[1]).convert("RGB")
print("cat? ", ask(img, "Is there a cat in this image? Answer Yes or No."))
print("car? ", ask(img, "Is there a car in this image? Answer Yes or No."))

msgs = [{"role": "user", "content": [{"type": "image"},
        {"type": "text", "text": "Describe this image in one sentence."}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
out = model.generate(**inp, max_new_tokens=40, do_sample=False)
print("desc: ", proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0])
print("statue?", ask(img, "Is there a statue or sculpture of a cat in this image? Answer Yes or No."))
