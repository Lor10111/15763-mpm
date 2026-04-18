"""
npz_to_bgeo.py — Convert per-frame .npz particle dumps to .bgeo for Houdini.

Works on any output directory from `test_tasty_meal_water.py` (MLS-MPM) or
`simulator_3d_tasty_meal_water.py` (basic-MPM). Each .npz is expected to
contain a 'position' (N,3) array and optionally a 'velocity' (N,3) array;
these become point attributes on the output .bgeo file.

Usage:
    python npz_to_bgeo.py <input_dir> [--out-dir <path>] [--pattern "*.npz"]

Examples:
    # In-place conversion (writes .bgeo next to each .npz)
    python npz_to_bgeo.py output/tasty_meal_water_mlsmpm

    # Separate output directory
    python npz_to_bgeo.py output/tasty_meal_water_mlsmpm --out-dir bgeo_out

    # Only convert water frames
    python npz_to_bgeo.py output/tasty_meal_water_mlsmpm --pattern "water_*.npz"

Requires:
    pip install partio

In Houdini, drop a File SOP and point it at e.g. `water_$F4.bgeo` — Houdini
will auto-pick up the frame number from $F4 and animate.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import partio
except ImportError:
    print("ERROR: partio not installed.")
    print("       install with:  pip install partio")
    sys.exit(1)


def npz_to_bgeo(npz_path: Path, bgeo_path: Path) -> int:
    """Convert one .npz file to .bgeo. Returns particle count."""
    d = np.load(npz_path)
    if "position" not in d.files:
        raise KeyError(f"{npz_path}: no 'position' key (found: {d.files})")
    pos = d["position"].astype(np.float32)
    has_vel = "velocity" in d.files
    vel = d["velocity"].astype(np.float32) if has_vel else None

    p = partio.create()
    pos_attr = p.addAttribute("position", partio.VECTOR, 3)
    vel_attr = p.addAttribute("velocity", partio.VECTOR, 3) if has_vel else None

    n = pos.shape[0]
    for i in range(n):
        idx = p.addParticle()
        p.set(pos_attr, idx, (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
        if has_vel:
            p.set(vel_attr, idx, (float(vel[i, 0]), float(vel[i, 1]), float(vel[i, 2])))

    bgeo_path.parent.mkdir(parents=True, exist_ok=True)
    partio.write(str(bgeo_path), p)
    return n


def main():
    ap = argparse.ArgumentParser(description="Convert .npz particle dumps to .bgeo.")
    ap.add_argument("input_dir", type=Path, help="Directory containing .npz files.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: in place, same as input_dir).")
    ap.add_argument("--pattern", type=str, default="*.npz",
                    help="Glob pattern (default: *.npz).")
    args = ap.parse_args()

    in_dir = args.input_dir.resolve()
    if not in_dir.is_dir():
        print(f"ERROR: {in_dir} is not a directory.")
        sys.exit(1)

    out_dir = args.out_dir.resolve() if args.out_dir is not None else in_dir
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"No files matching '{args.pattern}' under {in_dir}")
        sys.exit(1)

    print(f"[npz_to_bgeo]  {len(files)} files   {in_dir}  ->  {out_dir}")
    total_particles = 0
    for i, npz in enumerate(files):
        bgeo = out_dir / (npz.stem + ".bgeo")
        try:
            n = npz_to_bgeo(npz, bgeo)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {npz.name}  FAILED: {e}")
            continue
        total_particles += n
        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {npz.name}  ({n} particles)")

    print(f"[npz_to_bgeo]  done.  wrote {len(files)} .bgeo files, {total_particles} particles total.")
    print(f"[npz_to_bgeo]  in Houdini: File SOP -> e.g. {out_dir}/water_$F4.bgeo")


if __name__ == "__main__":
    main()
