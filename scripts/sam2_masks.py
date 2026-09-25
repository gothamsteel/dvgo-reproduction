#!/usr/bin/env python3
"""Segment one object across an ordered image folder with SAM 2.1.

You click the object once (as an x,y point on the first frame) and SAM 2
propagates the mask through the rest of the sequence.

SAM 2's video predictor requires frames named 00000.jpg, 00001.jpg, ...
so this script builds a numbered working folder first and maps the masks
back to the original filenames afterwards.

Usage:
    python sam2_masks.py capture2/undistorted/images capture2/masks_sam \\
        --ckpt ~/sam2_ckpt/sam2.1_hiera_large.pt \\
        --cfg configs/sam2.1/sam2.1_hiera_l.yaml \\
        --point 388 500

Check the first frame's point lands on the object. Default is image centre.
Add more points with repeated --point, and negative points with --neg-point
(background, tells SAM what NOT to include).
"""

import argparse
import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", help="folder of ordered images")
    ap.add_argument("dst", help="output mask folder")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--point", nargs=2, type=int, action="append", default=None,
                    metavar=("X", "Y"), help="foreground point on frame 0")
    ap.add_argument("--neg-point", nargs=2, type=int, action="append", default=None,
                    metavar=("X", "Y"), help="background point on frame 0")
    args = ap.parse_args()

    from sam2.build_sam import build_sam2_video_predictor

    names = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        raise SystemExit(f"no images in {args.images}")
    print(f"{len(names)} frames, first is {names[0]}")

    w, h = Image.open(os.path.join(args.images, names[0])).size
    pts = args.point or [[w // 2, h // 2]]
    labels = [1] * len(pts)
    if args.neg_point:
        pts = pts + args.neg_point
        labels = labels + [0] * len(args.neg_point)
    print(f"frame size {w}x{h}, points {pts}, labels {labels}")

    tmp = tempfile.mkdtemp(prefix="sam2frames_")
    try:
        for i, n in enumerate(names):
            im = Image.open(os.path.join(args.images, n)).convert("RGB")
            im.save(os.path.join(tmp, f"{i:05d}.jpg"), quality=95)

        predictor = build_sam2_video_predictor(args.cfg, args.ckpt)
        os.makedirs(args.dst, exist_ok=True)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=tmp)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=np.array(pts, dtype=np.float32),
                labels=np.array(labels, dtype=np.int32),
            )

            done = 0
            for idx, obj_ids, logits in predictor.propagate_in_video(state):
                m = (logits[0] > 0.0).cpu().numpy().squeeze()
                out = os.path.splitext(names[idx])[0] + ".png"
                Image.fromarray((m * 255).astype(np.uint8)).save(
                    os.path.join(args.dst, out))
                done += 1

        print(f"wrote {done} masks to {args.dst}")
        if done < len(names):
            print(f"WARNING: {len(names) - done} frames got no mask")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
