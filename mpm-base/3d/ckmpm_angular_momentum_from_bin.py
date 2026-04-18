"""
ckmpm_angular_momentum_from_bin.py
===================================
Reads CKMPM's per-frame conservation metric binary files for dual_rotation
and produces an .npz + plot compatible with the other solvers.

Binary format (13 float32 per frame per model):
    [0]     timestamp  (s)
    [1..3]  Eulerian linear momentum    (not used)
    [4..6]  Eulerian angular momentum   (not used)
    [7..9]  Lagrangian velocity SUM     (linear)
    [10..12] Lagrangian (r×v) SUM       ← multiply by particle_mass → L_z at index 12

Usage:
    python ckmpm_angular_momentum_from_bin.py [--result_dir PATH] [--out_npz PATH]

Default result_dir: ../../CKMPM/result/dual_rotation
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Physical constants (must match test_dual_rotation.cuh) ────────────────────
DX            = 1.0 / 64.0
RHO           = 1000.0
PPC           = 8.0
PARTICLE_VOL  = DX**3 / PPC
PARTICLE_MASS = PARTICLE_VOL * RHO

RECORD_FLOATS = 13
RECORD_BYTES  = RECORD_FLOATS * 4


def read_bin(path):
    size = os.path.getsize(path)
    if size % RECORD_BYTES != 0:
        raise ValueError(f"{path}: size {size} not multiple of {RECORD_BYTES}")
    n = size // RECORD_BYTES
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype="<f4").reshape(n, RECORD_FLOATS)


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--out_npz",    default=None)
    parser.add_argument("--particle_mass", type=float, default=None)
    args = parser.parse_args()

    if args.result_dir is None:
        result_dir = os.path.normpath(
            os.path.join(here, "..", "..", "CKMPM", "result", "dual_rotation"))
    else:
        result_dir = args.result_dir

    if args.out_npz is None:
        out_dir = os.path.join(here, "output")
        out_npz = os.path.join(out_dir, "dual_rotation_Lz_ckmpm.npz")
    else:
        out_npz = args.out_npz
        out_dir = os.path.dirname(out_npz)

    pm = args.particle_mass if args.particle_mass is not None else PARTICLE_MASS

    bin0 = os.path.join(result_dir, "conservationMetric_model_0.bin")
    bin1 = os.path.join(result_dir, "conservationMetric_model_1.bin")

    for p in (bin0, bin1):
        if not os.path.isfile(p):
            print(f"[ckmpm-rot] ERROR: not found: {p}")
            print("  Run: cd CKMPM/build && make mpm_test_dual_rotation -j")
            print("       ./tests/mpm_test_dual_rotation")
            sys.exit(1)

    d0 = read_bin(bin0)
    d1 = read_bin(bin1)
    n  = min(len(d0), len(d1))
    d0, d1 = d0[:n], d1[:n]

    times  = d0[:, 0].astype(np.float64)

    # index 12 = Lagrangian (r×v) sum, z-component
    # multiply by particle_mass to get L_z (kg·m²/s)
    Lz0    = (d0[:, 12] * pm).astype(np.float64)
    Lz1    = (d1[:, 12] * pm).astype(np.float64)
    Lz_tot = Lz0 + Lz1

    os.makedirs(out_dir, exist_ok=True)
    np.savez(out_npz, times=times, Lz0=Lz0, Lz1=Lz1, Lz_total=Lz_tot)

    print(f"[ckmpm-rot] {n} frames  particle_mass={pm:.4e} kg")
    print(f"[ckmpm-rot] Lz0: init={Lz0[0]:.4e}  final={Lz0[-1]:.4e}  "
          f"decay={100*(1-abs(Lz0[-1]/Lz0[0])):.1f}%")
    print(f"[ckmpm-rot] |ΔL_total|_max = {float(np.max(np.abs(Lz_tot))):.4e} kg·m²/s")
    print(f"[ckmpm-rot] Saved → {out_npz}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_png = out_npz.replace(".npz", ".png")
    drift   = float(np.max(np.abs(Lz_tot)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, Lz0,    lw=1.5, color="#e74c3c", label="Cube 0 (+ω)")
    ax.plot(times, Lz1,    lw=1.5, color="#f39c12", label="Cube 1 (−ω)")
    ax.plot(times, Lz_tot, lw=1.5, color="#2980b9", label="Total")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.text(0.97, 0.05, f"|ΔL_total|_max = {drift:.2e} kg·m²/s",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#2980b9",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2980b9", alpha=0.7))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("L_z  (kg·m²/s)")
    ax.set_title("Angular Momentum Conservation – CKMPM, Dual Rotation")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[ckmpm-rot] Plot saved → {out_png}")


if __name__ == "__main__":
    main()
