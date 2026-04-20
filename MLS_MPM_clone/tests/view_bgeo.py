"""
view_bgeo.py — Interactive Open3D viewer for CKMPM .bgeo dumps.

Mirrors view_open3d.py's UX but reads .bgeo via partio instead of .npz.

Expected filename pattern (CKMPM C++ output):
    model_<G>_particle_frame_<N>.bgeo
where <G> is the model index (group) and <N> is the frame number.

Controls:
    SPACE        play / pause
    LEFT / RIGHT step ± 1 frame
    DOWN / UP    step ± 10 frames
    R            reset to frame 0
    mouse drag   rotate / pan / zoom
    Q / ESC      quit

Install:
    pip install open3d partio
"""
import argparse, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("ERROR: pip install open3d")

from bgeo_reader import read_bgeo_positions

# ----- per-group color palette (for model_0 / model_1 / ...) ---------
PALETTE = [
    (0.20, 0.55, 0.95),  # blue
    (0.95, 0.55, 0.25),  # orange
    (0.40, 0.85, 0.50),  # green
    (0.90, 0.40, 0.85),  # magenta
    (0.95, 0.85, 0.30),  # yellow
]

PATTERN = re.compile(r"^model_(\d+)_particle_frame_(\d+)\.bgeo$")


def discover(in_dir: Path):
    buckets = defaultdict(dict)
    for f in in_dir.glob("*.bgeo"):
        m = PATTERN.match(f.name)
        if not m:
            continue
        g, idx = int(m.group(1)), int(m.group(2))
        buckets[f"model_{g}"][idx] = f
    if not buckets:
        sys.exit(f"no model_*_particle_frame_*.bgeo files in {in_dir}")
    sets = [set(d.keys()) for d in buckets.values()]
    common = sorted(set.intersection(*sets))
    if not common:
        sys.exit(f"no shared frames across groups: { {k: len(v) for k,v in buckets.items()} }")
    return {k: [buckets[k][i] for i in common] for k in sorted(buckets)}, common


def load_positions(path: Path) -> np.ndarray:
    return read_bgeo_positions(path).astype(np.float64)


class Viewer:
    def __init__(self, groups, frames, fps, point_size):
        self.groups = groups
        self.frames = frames
        self.n = len(frames)
        self.fps = float(fps)
        self.point_size = float(point_size)
        self.cur = 0
        self.play = True
        self.t_last = time.time()
        self.pcds = {}
        for i, name in enumerate(groups):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(load_positions(groups[name][0]))
            pc.paint_uniform_color(PALETTE[i % len(PALETTE)])
            self.pcds[name] = pc

    def set_frame(self, k):
        k = max(0, min(self.n - 1, k))
        self.cur = k
        for name, pc in self.pcds.items():
            pc.points = o3d.utility.Vector3dVector(load_positions(self.groups[name][k]))

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="CKMPM bgeo viewer", width=1280, height=800)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(axes)
        for pc in self.pcds.values():
            vis.add_geometry(pc)

        opt = vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.array([0.04, 0.04, 0.07])

        def toggle(_): self.play = not self.play; self.t_last = time.time(); return False
        def step(d):
            def fn(v):
                self.play = False
                self.set_frame(self.cur + d)
                for pc in self.pcds.values(): v.update_geometry(pc)
                return False
            return fn
        def reset(v):
            self.play = False; self.set_frame(0)
            for pc in self.pcds.values(): v.update_geometry(pc)
            return False

        vis.register_key_callback(32, toggle)        # SPACE
        vis.register_key_callback(262, step(+1))     # RIGHT
        vis.register_key_callback(263, step(-1))     # LEFT
        vis.register_key_callback(264, step(-10))    # DOWN
        vis.register_key_callback(265, step(+10))    # UP
        vis.register_key_callback(82,  reset)        # R

        print(f"[viewer] {self.n} frames, groups: {list(self.pcds)}")
        print(f"[viewer] SPACE=play LEFT/RIGHT=±1 UP/DOWN=±10 R=reset Q=quit")

        while True:
            now = time.time()
            if self.play and (now - self.t_last) >= 1.0 / self.fps:
                nxt = (self.cur + 1) % self.n
                self.set_frame(nxt)
                for pc in self.pcds.values(): vis.update_geometry(pc)
                self.t_last = now
            if not vis.poll_events(): break
            vis.update_renderer()
        vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--fps", type=float, default=48.0)
    ap.add_argument("--point-size", type=float, default=3.0)
    args = ap.parse_args()

    in_dir = args.input_dir.resolve()
    if not in_dir.is_dir():
        sys.exit(f"{in_dir} is not a directory")

    groups, frames = discover(in_dir)
    print(f"[viewer] {in_dir}")
    for k, v in groups.items():
        print(f"  {k}: {len(v)} frames")
    Viewer(groups, frames, args.fps, args.point_size).run()


if __name__ == "__main__":
    main()
