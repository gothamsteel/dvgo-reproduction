#!/usr/bin/env python3
"""Render a synthetic scene in tankstemple format, with exact camera poses.

Imports a .blend model (e.g. a Poly Haven CC0 asset), normalises it into a
unit-ish sphere at the origin, and renders it from views circling the object.
Pairs with the COLMAP real capture: same format, same loader, same view count,
so the substantial difference between the two datasets is where the poses come
from — exact here, estimated there.

Output (tankstemple layout):
    <dst>/rgb/0_000.png      RGBA, transparent background
    <dst>/pose/0_000.txt     4x4 camera-to-world, OpenCV convention
    <dst>/intrinsics.txt     4x4 K

Config must set dataset_type='tankstemple', inverse_y=True, white_bkgd=True.

Run headless:
    blender -b -P blender_scene.py -- \\
        --model data/cat_blender/concrete_cat_statue_2k.blend \\
        --dst data/synth_cat --views 78

Options after the bare -- :
    --model PATH      .blend to import (omit for a procedural cylinder)
    --dst PATH        output folder
    --views N         number of views (default 78)
    --test-every N    hold out every Nth view (default 8)
    --res W H         resolution (default 776 1046)
    --radius R        camera distance from origin (default 2.1)
    --obj-radius R    normalise the model to this radius (default 0.55)
    --elev LO HI      elevation range in degrees (default 5 55)
    --samples N       Cycles samples (default 64)
"""

import argparse
import math
import os
import sys

import bpy
import mathutils


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--dst", default="data/synth_scene")
    ap.add_argument("--views", type=int, default=78)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--res", nargs=2, type=int, default=[776, 1046])
    ap.add_argument("--radius", type=float, default=2.1)
    ap.add_argument("--obj-radius", type=float, default=0.55)
    ap.add_argument("--elev", nargs=2, type=float, default=[5.0, 55.0])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_model(path):
    """Append every object from the .blend into the current scene."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise SystemExit(f"model not found: {path}")

    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = list(src.objects)

    linked = []
    for obj in dst.objects:
        if obj is None:
            continue
        if obj.type in {"LIGHT", "CAMERA"}:
            continue          # we supply our own
        bpy.context.collection.objects.link(obj)
        linked.append(obj)

    if not linked:
        raise SystemExit(f"no usable objects found in {path}")
    print(f"imported {len(linked)} objects: {[o.name for o in linked]}")
    return linked


def make_cylinder():
    bpy.ops.mesh.primitive_cylinder_add(radius=0.38, depth=1.0, vertices=128)
    obj = bpy.context.active_object
    mat = bpy.data.materials.new("Mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 22.0
    tex.inputs["Detail"].default_value = 6.0
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    obj.data.materials.append(mat)
    return [obj]


def normalise(objects, target_radius):
    """Recentre on the combined bounding box and scale to target_radius."""
    dg = bpy.context.evaluated_depsgraph_get()
    lo = mathutils.Vector((float("inf"),) * 3)
    hi = -lo.copy()
    for obj in objects:
        if obj.type != "MESH":
            continue
        ev = obj.evaluated_get(dg)
        for corner in ev.bound_box:
            p = ev.matrix_world @ mathutils.Vector(corner)
            lo = mathutils.Vector(map(min, lo, p))
            hi = mathutils.Vector(map(max, hi, p))

    centre = (lo + hi) / 2.0
    extent = (hi - lo)
    radius = max(extent) / 2.0
    scale = target_radius / radius
    print(f"model bbox {tuple(round(v,3) for v in extent)}, "
          f"centre {tuple(round(v,3) for v in centre)}, scale {scale:.4f}")

    pivot = bpy.data.objects.new("Pivot", None)
    bpy.context.collection.objects.link(pivot)
    for obj in objects:
        if obj.parent is None:
            obj.parent = pivot
            obj.matrix_parent_inverse = pivot.matrix_world.inverted()

    pivot.location = -centre * scale
    pivot.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    return pivot


def make_lights():
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.38, 0.39, 0.42, 1.0)
    bg.inputs["Strength"].default_value = 1.0

    for i, (loc, energy) in enumerate([((3, -2, 4), 400.0), ((-3, 2, 3), 200.0)]):
        ld = bpy.data.lights.new(f"L{i}", type="AREA")
        ld.energy = energy
        ld.size = 2.5
        lo = bpy.data.objects.new(f"L{i}", ld)
        lo.location = loc
        lo.rotation_euler = mathutils.Vector(loc).to_track_quat("Z", "Y").to_euler()
        bpy.context.collection.objects.link(lo)


def make_camera(W):
    cd = bpy.data.cameras.new("Cam")
    cd.lens = 35.0
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
    cam = bpy.data.objects.new("Cam", cd)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    fx = W * cd.lens / cd.sensor_width
    return cam, fx, fx


def setup_render(W, H, samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.render.resolution_x = W
    sc.render.resolution_y = H
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dev_type in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = dev_type
                prefs.get_devices()
                n = 0
                for d in prefs.devices:
                    if d.type != "CPU":
                        d.use = True
                        n += 1
                if n:
                    sc.cycles.device = "GPU"
                    print(f"cycles device: {dev_type}, {n} devices")
                    break
            except Exception:
                continue
    except Exception as e:
        print(f"GPU setup skipped: {e}")


def camera_positions(n, radius, elev_lo, elev_hi, seed):
    """One full azimuth sweep, elevation oscillating across the band.

    Azimuth is linear in index so that holding out every Nth view samples
    the whole circle rather than one arc.
    """
    out = []
    for i in range(n):
        az = 2.0 * math.pi * i / n + seed
        t = (i + 0.5) / n
        s = 0.5 * (1 - math.cos(2 * math.pi * 3 * t))
        el = math.radians(elev_lo + (elev_hi - elev_lo) * s)
        out.append((radius * math.cos(el) * math.cos(az),
                    radius * math.cos(el) * math.sin(az),
                    radius * math.sin(el)))
    return out


def main():
    args = parse_args()
    W, H = args.res

    clear_scene()
    objs = import_model(args.model) if args.model else make_cylinder()
    normalise(objs, args.obj_radius)
    make_lights()
    cam, fx, fy = make_camera(W)
    setup_render(W, H, args.samples)

    rgb_dir = os.path.join(args.dst, "rgb")
    pose_dir = os.path.join(args.dst, "pose")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    locs = camera_positions(args.views, args.radius,
                            args.elev[0], args.elev[1], args.seed)
    test_idx = set(range(0, args.views, args.test_every))
    n_train = n_test = 0

    for i, loc in enumerate(locs):
        cam.location = loc
        cam.rotation_euler = (-mathutils.Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()

        split = 1 if i in test_idx else 0
        n_test += split
        n_train += 1 - split
        stem = f"{split}_{i:03d}"

        bpy.context.scene.render.filepath = os.path.join(rgb_dir, f"{stem}.png")
        bpy.ops.render.render(write_still=True)

        # Blender: -Z forward, +Y up (OpenGL). DVGO inverse_y=True: +Z forward,
        # +Y down (OpenCV). Negate the second and third basis columns.
        c2w = [list(r) for r in cam.matrix_world]
        for r in range(3):
            c2w[r][1] = -c2w[r][1]
            c2w[r][2] = -c2w[r][2]

        with open(os.path.join(pose_dir, f"{stem}.txt"), "w") as f:
            for row in c2w:
                f.write(" ".join(f"{v:.10f}" for v in row) + "\n")

        print(f"[{i+1}/{args.views}] {stem}")

    K = [[fx, 0.0, W / 2.0, 0.0],
         [0.0, fy, H / 2.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    with open(os.path.join(args.dst, "intrinsics.txt"), "w") as f:
        for row in K:
            f.write(" ".join(f"{v:.10f}" for v in row) + "\n")

    print(f"fx {fx:.2f}  fy {fy:.2f}  cx {W/2:.1f}  cy {H/2:.1f}")
    print(f"camera radius {args.radius}, object radius {args.obj_radius}")
    print(f"train {n_train}, test {n_test}, resolution {W}x{H}")
    print(f"wrote to {args.dst}")


if __name__ == "__main__":
    main()
