#!/usr/bin/env python3
"""Show each sample, press y (artefacts) or n (clean). b = back, q = quit.
Saves after every key, so it can be stopped and resumed."""
import csv, sys, tkinter as tk
from PIL import Image, ImageTk

out = sys.argv[1] if len(sys.argv) > 1 else 'labels_eric.csv'
rows = list(csv.DictReader(open(out)))
i = next((k for k, r in enumerate(rows) if not r['artefacts']), 0)

root = tk.Tk()
info = tk.Label(root, font=('Arial', 16)); info.pack()
pic = tk.Label(root); pic.pack()

def save():
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'artefacts'])
        w.writeheader(); w.writerows(rows)

def show():
    if i >= len(rows):
        info.config(text='Done. Close the window.'); pic.config(image=''); return
    im = Image.open(f"samples/{rows[i]['id']}.png").convert('RGB')
    im.thumbnail((1000, 800))
    pic.img = ImageTk.PhotoImage(im); pic.config(image=pic.img)
    info.config(text=f"{rows[i]['id']}  ({i+1}/{len(rows)})   artefacts?  y = yes, n = no, b = back"
                     f"   [current: {rows[i]['artefacts'] or '-'}]")

def key(e):
    global i
    if e.char in ('y', 'n') and i < len(rows):
        rows[i]['artefacts'] = 'yes' if e.char == 'y' else 'no'; save(); i += 1
    elif e.char == 'b' and i > 0:
        i -= 1
    elif e.char == 'q':
        root.destroy(); return
    show()

root.bind('<Key>', key); show(); root.mainloop()
