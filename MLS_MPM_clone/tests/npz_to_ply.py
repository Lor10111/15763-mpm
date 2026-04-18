"""
npz_to_ply.py — Convert per-frame .npz particle dumps to binary .ply.

Works on any output directory from `test_tasty_meal_water.py` (MLS-MPM) or
`simulator_3d_tasty_meal_water.py` (basic-MPM). Each .npz is expected to
contain a 'position' (N,3) array and optionally a 'velocity' (N,3) array;
these are written as per-vertex attributes on the output .ply file.

Binary little-endian .ply loads fast and is natively supported by
Houdini (File SOP), Blender (File > Import > PLY), MeshLab, CloudCompare.

No extra dependencies — just numpy.

Usage:
    python npz_to_ply.py <input_dir> [--out-dir <path>] [--pattern "*.npz"]

Examples:
    # In-place conversion (writes .ply next to each .npz)
    python npz_to_ply.py output/tasty_meal_water_mlsmpm

    # Separate output directory
    python npz_to_ply.py output/tasty_meal_water_mlsmpm --out-dir ply_out

    # Only convert water frames
    python npz_to_ply.py output/tasty_meal_water_mlsmpm --pattern "water_*.npz"
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def write_binary_ply(ply_path: Path, pos: np.ndarray, vel: np.ndarray | None = None) -> None:
    """Write binary little-endian .ply with x/y/z + optional velocity vx/vy/vz."""
    pos = np.ascontiguousarray(pos, dtype=np.float32)
    n   = pos.shape[0]

    header_lines = [
        b"ply",
        b"format binary_little_endian 1.0",
        f"element vertex {n}".encode(),
        b"property float x",
        b"property float y",
        b"property float z",
    ]
    if vel is not None:
        vel = np.ascontiguousarray(vel, dtype=np.float32)
        header_lines += [
            b"property float vx",
            b"property float vy",
            b"property float vz",
        ]
    header_lines.append(b"end_header")
    header = b"\n".join(header_lines) + b"\n"

    if vel is None:
        body = pos
    else:
        body = np.empty((n, 6), dtype=np.float32)
        body[:, :3] = pos
        body[:, 3:] = vel

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ply_path, "wb") as f:
        f.write(header)
        f.write(body.tobytes())


def npz_to_ply(npz_path: Path, ply_path: Path) -> int:
    d = np.load(npz_path)
    if "position" not in d.files:
        raise KeyError(f"{npz_path}: no 'position' key (found: {d.files})")
    pos = d["position"]
    vel = d["velocity"] if "velocity" in d.files else None
    write_binary_ply(ply_path, pos, vel)
    return int(pos.shape[0])


def main():
    ap = argparse.ArgumentParser(description="Convert .npz particle dumps to binary .ply.")
    ap.add_argument("input_dir", type=Path, help="Directory containing .npz files.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: in place).")
    ap.add_argument("--pattern", type=str, default="*.npz",
                    help="Glob pattern (default: *.npz).")
    args = ap.parse_args()

    in_dir = args.input_dir.resolve()
    if not in_dir.is_dir():
        print(f"ERROR: {in_dir} is not a directory.")
        sys.exit(1)

    out_dir = args.out_dir.resolve() if args.out_dir is not None else in_dir
    files   = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"No files matching '{args.pattern}' under {in_dir}")
        sys.exit(1)

    print(f"[npz_to_ply]  {len(files)} files   {in_dir}  ->  {out_dir}")
    total = 0
    for i, npz in enumerate(files):
        ply = out_dir / (npz.stem + ".ply")
        try:
            n = npz_to_ply(npz, ply)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {npz.name}  FAILED: {e}")
            continue
        total += n
        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {npz.name}  ({n} particles)")

    print(f"[npz_to_ply]  done.  {len(files)} .ply files, {total} particles total.")
    print(f"[npz_to_ply]  Houdini:  File SOP -> {out_dir}/water_$F4.ply")
    print(f"[npz_to_ply]  Blender:  File > Import > PLY (pick any single frame)")


if __name__ == "__main__":
    main()
