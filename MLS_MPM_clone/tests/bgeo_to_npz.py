"""
bgeo_to_npz.py -- Convert Houdini/Partio .bgeo particle dumps to .npz.

This is mainly for CKMPM outputs such as:
    model_0_particle_frame_12.bgeo
    model_1_particle_frame_12.bgeo

The converter stores particle positions under both keys:
    position : (N, 3) float32
    pos      : (N, 3) float32

Usage:
    # Convert one file next to itself
    python bgeo_to_npz.py path/to/model_0_particle_frame_0.bgeo

    # Convert every .bgeo in a directory, one .npz per .bgeo
    python bgeo_to_npz.py path/to/output_dir

    # Write to a separate directory
    python bgeo_to_npz.py path/to/output_dir --out-dir path/to/npz_out

    # CKMPM convenience: combine model_*/frame_* files by frame
    python bgeo_to_npz.py path/to/output_dir --combine-models --out-dir path/to/npz_out
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bgeo_reader import read_bgeo_positions


CKMPM_PATTERN = re.compile(r"^model_(\d+)_particle_frame_(\d+)\.bgeo(?:\.gz)?$")


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def convert_one(bgeo_path: Path, out_dir: Path | None = None) -> tuple[Path, int]:
    pos = read_bgeo_positions(bgeo_path)
    out_path = (out_dir if out_dir is not None else bgeo_path.parent) / f"{bgeo_path.stem}.npz"
    if bgeo_path.name.endswith(".bgeo.gz"):
        out_path = (out_dir if out_dir is not None else bgeo_path.parent) / f"{bgeo_path.name[:-8]}.npz"
    save_npz(out_path, position=pos, pos=pos)
    return out_path, pos.shape[0]


def discover_bgeo(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is neither a file nor a directory")
    return sorted(path.glob(pattern))


def convert_individual(input_path: Path, out_dir: Path | None, pattern: str) -> None:
    files = discover_bgeo(input_path, pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} under {input_path}")

    total = 0
    print(f"[bgeo_to_npz] converting {len(files)} file(s)")
    for i, bgeo in enumerate(files, 1):
        out_path, n = convert_one(bgeo, out_dir)
        total += n
        print(f"  [{i}/{len(files)}] {bgeo.name} -> {out_path.name}  ({n} particles)")
    print(f"[bgeo_to_npz] done: {len(files)} npz file(s), {total} particles total")


def convert_combined_by_frame(input_dir: Path, out_dir: Path, pattern: str) -> None:
    files = discover_bgeo(input_dir, pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} under {input_dir}")

    by_frame: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    skipped = []
    for path in files:
        match = CKMPM_PATTERN.match(path.name)
        if match is None:
            skipped.append(path.name)
            continue
        model_id = int(match.group(1))
        frame_id = int(match.group(2))
        by_frame[frame_id].append((model_id, path))

    if not by_frame:
        raise ValueError(
            "No CKMPM-style files found. Expected names like "
            "model_0_particle_frame_12.bgeo"
        )

    if skipped:
        print(f"[bgeo_to_npz] skipped {len(skipped)} non-CKMPM filename(s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    total_particles = 0
    frames = sorted(by_frame)
    print(f"[bgeo_to_npz] combining {len(frames)} frame(s) -> {out_dir}")
    for k, frame_id in enumerate(frames, 1):
        chunks = []
        model_ids = []
        for model_id, path in sorted(by_frame[frame_id]):
            pos = read_bgeo_positions(path)
            chunks.append(pos)
            model_ids.append(np.full(pos.shape[0], model_id, dtype=np.int32))

        position = np.concatenate(chunks, axis=0)
        model_id_arr = np.concatenate(model_ids, axis=0)
        out_path = out_dir / f"frame_{frame_id:04d}.npz"
        save_npz(out_path, position=position, pos=position, model_id=model_id_arr, frame=np.array(frame_id))
        total_particles += position.shape[0]
        print(f"  [{k}/{len(frames)}] frame {frame_id} -> {out_path.name}  ({position.shape[0]} particles)")

    print(f"[bgeo_to_npz] done: {len(frames)} combined frame(s), {total_particles} particles total")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .bgeo particle dumps to .npz.")
    parser.add_argument("input", type=Path, help="Input .bgeo file or directory.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--pattern", default="*.bgeo", help="Glob pattern for directory input.")
    parser.add_argument(
        "--combine-models",
        action="store_true",
        help="Combine CKMPM model_*_particle_frame_*.bgeo files into one npz per frame.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else None

    try:
        if args.combine_models:
            if not input_path.is_dir():
                raise ValueError("--combine-models requires a directory input")
            convert_combined_by_frame(input_path, out_dir if out_dir is not None else input_path / "npz", args.pattern)
        else:
            convert_individual(input_path, out_dir, args.pattern)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
