"""
ckmpm_momentum_from_bin.py
==========================
Reads CKMPM's per-frame conservation metric binary files and produces an .npz
compatible with plot_momentum_comparison.py.

CKMPM writes one binary file per model when collectConservationMetric=True:
    result/colliding_cubes/conservationMetric_model_0.bin
    result/colliding_cubes/conservationMetric_model_1.bin

Each file contains N_frames records of 13 floats (little-endian float32):
    [0]     timestamp  (seconds since simulation start)
    [1..3]  Eulerian linear momentum  (grid-based, kg·m/s if mass-weighted internally)
    [4..6]  Eulerian angular momentum
    [7..9]  Lagrangian linear VELOCITY SUM  ← must multiply by particle_mass
    [10..12] Lagrangian angular velocity sum

The particle mass for colliding_cubes matches both Python solvers:
    rho = 1000 kg/m³,  dx = 1/64,  ppc = 8
    particle_vol  = dx³ / ppc = (1/64)³ / 8 ≈ 3.8147e-6 m³
    particle_mass = particle_vol × rho        ≈ 3.8147e-3 kg

Usage:
    python ckmpm_momentum_from_bin.py [--result_dir PATH] [--out_npz PATH]

Default paths (relative to this script's directory):
    result_dir : ../../CKMPM/result/colliding_cubes
    out_npz    : output/colliding_cubes_momentum_ckmpm.npz
"""

import os
import sys
import argparse
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── physical constants (must match test_colliding_cubes.cuh) ──────────────────
DX             = 1.0 / 64.0
RHO            = 1000.0
PPC            = 8.0
PARTICLE_VOL   = DX ** 3 / PPC
PARTICLE_MASS  = PARTICLE_VOL * RHO

RECORD_FLOATS  = 13          # floats per frame per model
RECORD_BYTES   = RECORD_FLOATS * 4


def read_bin(path: str) -> np.ndarray:
    """Return (N, 13) float32 array from a conservation metric .bin file."""
    size = os.path.getsize(path)
    if size % RECORD_BYTES != 0:
        raise ValueError(
            f"{path}: file size {size} is not a multiple of {RECORD_BYTES} bytes."
        )
    n = size // RECORD_BYTES
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype="<f4").reshape(n, RECORD_FLOATS)
    return data


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--result_dir", default=None,
        help="Directory containing conservationMetric_model_*.bin "
             "(default: <script>/../../../CKMPM/result/colliding_cubes)")
    parser.add_argument(
        "--out_npz", default=None,
        help="Output .npz path (default: <script>/output/colliding_cubes_momentum_ckmpm.npz)")
    parser.add_argument(
        "--particle_mass", type=float, default=None,
        help=f"Override particle mass (default: {PARTICLE_MASS:.6e} kg)")
    args = parser.parse_args()

    # ── paths ─────────────────────────────────────────────────────────────────
    if args.result_dir is None:
        result_dir = os.path.join(here, "..", "..", "CKMPM", "result", "colliding_cubes")
    else:
        result_dir = args.result_dir
    result_dir = os.path.normpath(result_dir)

    if args.out_npz is None:
        out_dir = os.path.join(here, "output")
        out_npz = os.path.join(out_dir, "colliding_cubes_momentum_ckmpm.npz")
    else:
        out_npz = args.out_npz
        out_dir = os.path.dirname(out_npz)

    pm = args.particle_mass if args.particle_mass is not None else PARTICLE_MASS

    # ── locate bin files ──────────────────────────────────────────────────────
    bin0 = os.path.join(result_dir, "conservationMetric_model_0.bin")
    bin1 = os.path.join(result_dir, "conservationMetric_model_1.bin")

    for p in (bin0, bin1):
        if not os.path.isfile(p):
            print(f"[ckmpm] ERROR: file not found: {p}")
            print("       Run the CKMPM colliding_cubes test first:")
            print("         cd CKMPM/build && cmake .. -DCMAKE_BUILD_TYPE=Release \\")
            print("           && make mpm_test_colliding_cubes -j$(nproc)")
            print("         ./tests/mpm_test_colliding_cubes")
            sys.exit(1)

    # ── read data ─────────────────────────────────────────────────────────────
    d0 = read_bin(bin0)   # shape (N, 13)
    d1 = read_bin(bin1)

    if len(d0) != len(d1):
        print(f"[ckmpm] WARNING: model 0 has {len(d0)} frames, model 1 has {len(d1)}.")
        n = min(len(d0), len(d1))
        d0, d1 = d0[:n], d1[:n]

    # ── extract quantities ────────────────────────────────────────────────────
    # Column layout:
    #   0        : timestamp (s)
    #   1..3     : Eulerian linear momentum  (mass×vel, summed over grid cells)
    #   4..6     : Eulerian angular momentum
    #   7..9     : Lagrangian velocity SUM (need ×particle_mass for momentum)
    #   10..12   : Lagrangian angular velocity sum

    times   = d0[:, 0].astype(np.float64)

    # Lagrangian linear momentum (x-component), per model
    # The binary stores velocity sum; multiply by particle mass to get momentum.
    px0     = (d0[:, 7] * pm).astype(np.float64)
    px1     = (d1[:, 7] * pm).astype(np.float64)
    px_tot  = px0 + px1

    # ── save .npz ─────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    np.savez(out_npz, times=times, px0=px0, px1=px1, px_total=px_tot)

    print(f"[ckmpm] Loaded {len(times)} frames from {result_dir}")
    print(f"[ckmpm] particle_mass used = {pm:.6e} kg")
    print(f"[ckmpm] |Δp_total|_max = {float(np.max(np.abs(px_tot))):.4e} kg·m/s")
    print(f"[ckmpm] Saved → {out_npz}")

    # ── plot ──────────────────────────────────────────────────────────────────
    out_png = out_npz.replace(".npz", ".png")
    drift   = float(np.max(np.abs(px_tot)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, px0,    lw=1.5, color="#e74c3c", label="Cube 0")
    ax.plot(times, px1,    lw=1.5, color="#f39c12", label="Cube 1")
    ax.plot(times, px_tot, lw=1.5, color="#2980b9", label="Total")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.text(0.97, 0.05, f"|Δp_total|_max = {drift:.2e} kg·m/s",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#2980b9",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2980b9", alpha=0.7))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("X-Component of Linear Momentum\n(kg · m/s)")
    ax.set_title("Linear Momentum Conservation – CKMPM, Colliding Cubes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[ckmpm] momentum plot saved → {out_png}")

    # ── quick sanity print ────────────────────────────────────────────────────
    print(f"\n  Frame  |  t (s)   |  px0 (kg·m/s)  |  px1 (kg·m/s)  |  px_total")
    print(f"  -------|----------|----------------|----------------|----------")
    step = max(1, len(times) // 10)
    for i in range(0, len(times), step):
        print(f"  {i:5d}  | {times[i]:8.4f} | {px0[i]:14.6e} | {px1[i]:14.6e} | {px_tot[i]:+.4e}")


if __name__ == "__main__":
    main()
