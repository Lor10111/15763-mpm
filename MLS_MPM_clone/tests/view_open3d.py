"""
view_open3d.py — Interactive Open3D viewer for MLS-MPM .npz particle dumps.

Reads per-frame .npz files directly (no conversion step) and animates them
in an interactive 3D window. Filename prefix (cup / water / ice / …) is
auto-detected and each group gets its own color.

Controls:
    SPACE        play / pause
    RIGHT        next frame
    LEFT         previous frame
    R            reset to frame 0
    mouse drag   rotate / pan / zoom
    Q / ESC      quit

Expected filename pattern:  <group>_<NNNN>.npz
    water_0000.npz, water_0001.npz, …
    cup_0000.npz,   cup_0001.npz,   …
    ice_0000.npz,   ice_0001.npz,   …

Each .npz must contain a 'position' (N,3) float array. 'velocity' is
optional and currently unused for rendering.

Usage:
    python view_open3d.py <dir>
    python view_open3d.py <dir> --fps 30
    python view_open3d.py <dir> --groups water ice
    python view_open3d.py <dir> --point-size 4

Install:
    pip install open3d numpy
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d not installed.")
    print("       install with:  pip install open3d")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Per-group color palette. Extra groups fall back to a cycling palette.
# Colors tuned for a dark background.
DEFAULT_COLORS = {
    "cup":    (0.60, 0.60, 0.64),   # light gray
    "water":  (0.20, 0.55, 0.95),   # blue
    "ice":    (0.88, 0.94, 1.00),   # near-white w/ blue tint
    "honey":  (0.95, 0.70, 0.15),   # amber
    "slush":  (0.70, 0.82, 1.00),   # pale blue
    "slushy": (0.70, 0.82, 1.00),
}
FALLBACK_COLORS = [
    (0.90, 0.40, 0.40),
    (0.40, 0.90, 0.50),
    (0.95, 0.85, 0.30),
    (0.85, 0.40, 0.95),
    (0.40, 0.85, 0.95),
]


def discover_frames(in_dir: Path, groups_filter):
    """Scan directory for <group>_NNNN.npz files.

    Returns
    -------
    out : dict[str, list[Path]]
        Per-group ordered list of file paths (one per frame).
    frame_indices : list[int]
        Common frame indices shared by every group.
    """
    pattern = re.compile(r"^([A-Za-z][A-Za-z0-9]*)_(\d+)\.npz$")
    buckets = defaultdict(dict)
    for f in in_dir.glob("*.npz"):
        m = pattern.match(f.name)
        if not m:
            continue
        name, idx = m.group(1), int(m.group(2))
        if groups_filter is not None and name not in groups_filter:
            continue
        buckets[name][idx] = f

    if not buckets:
        raise RuntimeError(
            f"no matching .npz files found in {in_dir}\n"
            f"    expected pattern: <group>_NNNN.npz"
        )

    frame_sets = [set(d.keys()) for d in buckets.values()]
    common = sorted(set.intersection(*frame_sets))
    if not common:
        raise RuntimeError(
            "no frames shared across all groups; "
            f"per-group frame counts: { {k: len(v) for k,v in buckets.items()} }"
        )

    out = {name: [buckets[name][i] for i in common] for name in buckets}
    return out, common


def load_positions(path: Path) -> np.ndarray:
    d = np.load(path)
    if "position" not in d.files:
        raise KeyError(f"{path}: no 'position' key (found: {d.files})")
    return np.asarray(d["position"], dtype=np.float64)


def color_for(name: str, fallback_idx: int):
    if name in DEFAULT_COLORS:
        return DEFAULT_COLORS[name]
    return FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]


class Viewer:
    def __init__(self, groups, frames, fps, point_size):
        self.groups = groups        # dict[str, list[Path]]
        self.frames = frames        # list[int]
        self.n_frames = len(frames)
        self.fps = float(fps)
        self.point_size = float(point_size)
        self.current = 0
        self.playing = True
        self._last_advance = time.time()

        # One PointCloud per group, initialized to frame 0.
        self.pcds = {}
        for i, name in enumerate(groups):
            pc = o3d.geometry.PointCloud()
            pos = load_positions(groups[name][0])
            pc.points = o3d.utility.Vector3dVector(pos)
            pc.paint_uniform_color(color_for(name, i))
            self.pcds[name] = pc

    # -- frame stepping -----------------------------------------------------
    def set_frame(self, k: int):
        k = max(0, min(self.n_frames - 1, k))
        self.current = k
        for name, pc in self.pcds.items():
            pos = load_positions(self.groups[name][k])
            pc.points = o3d.utility.Vector3dVector(pos)

    # -- main loop ----------------------------------------------------------
    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="MLS-MPM viewer", width=1280, height=800)

        # Coordinate frame at origin for orientation (5 cm axes).
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        vis.add_geometry(axes)
        for pc in self.pcds.values():
            vis.add_geometry(pc)

        opt = vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.array([0.04, 0.04, 0.07])

        # --- key callbacks (GLFW key codes) ---
        def toggle_play(_v):
            self.playing = not self.playing
            self._last_advance = time.time()
            print(f"[viewer] {'PLAY' if self.playing else 'PAUSE'}  frame {self.current}/{self.n_frames-1}")
            return False

        def step(delta):
            def fn(v):
                self.playing = False
                self.set_frame(self.current + delta)
                for pc in self.pcds.values():
                    v.update_geometry(pc)
                print(f"[viewer] PAUSE  frame {self.current}/{self.n_frames-1}")
                return False
            return fn

        def reset(v):
            self.playing = False
            self.set_frame(0)
            for pc in self.pcds.values():
                v.update_geometry(pc)
            print(f"[viewer] RESET  frame 0")
            return False

        vis.register_key_callback(32,  toggle_play)   # SPACE
        vis.register_key_callback(262, step(+1))      # RIGHT
        vis.register_key_callback(263, step(-1))      # LEFT
        vis.register_key_callback(264, step(-10))     # DOWN (jump back 10)
        vis.register_key_callback(265, step(+10))     # UP   (jump ahead 10)
        vis.register_key_callback(82,  reset)         # R

        print(f"[viewer] {self.n_frames} frames   groups: {list(self.pcds)}")
        print(f"[viewer] SPACE=play/pause  LEFT/RIGHT=step  UP/DOWN=±10  R=reset  Q=quit")

        # Manual poll loop so we can drive time-based playback.
        while True:
            now = time.time()
            if self.playing and (now - self._last_advance) >= 1.0 / self.fps:
                nxt = self.current + 1
                if nxt >= self.n_frames:
                    nxt = 0          # loop
                self.set_frame(nxt)
                for pc in self.pcds.values():
                    vis.update_geometry(pc)
                self._last_advance = now

            if not vis.poll_events():
                break
            vis.update_renderer()

        vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(description="Open3D viewer for MLS-MPM .npz dumps.")
    ap.add_argument("input_dir", type=Path,
                    help="Directory containing <group>_NNNN.npz files.")
    ap.add_argument("--fps", type=float, default=48.0,
                    help="Playback frame rate (default: 48).")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="Only show these groups (e.g. --groups water ice). "
                         "Default: show every group found.")
    ap.add_argument("--point-size", type=float, default=3.5,
                    help="Rendered point size in pixels (default: 3.5).")
    args = ap.parse_args()

    in_dir = args.input_dir.resolve()
    if not in_dir.is_dir():
        print(f"ERROR: {in_dir} is not a directory.")
        sys.exit(1)

    try:
        groups, frames = discover_frames(in_dir, args.groups)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    counts = ", ".join(f"{k}={len(v)}" for k, v in groups.items())
    print(f"[viewer] input: {in_dir}")
    print(f"[viewer] {counts}   common frames: {len(frames)}   fps: {args.fps}")

    Viewer(groups, frames, args.fps, args.point_size).run()


if __name__ == "__main__":
    main()
