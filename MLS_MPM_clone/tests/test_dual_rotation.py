# Dual Rotation – MLS-MPM  (PyTorch / MLS-APIC)
# =================================================
# Two identical elastic cubes given an initial pure rotation about the z-axis.
#   Cube 1: ω = +OMEGA rad/s (counterclockwise when viewed from +z)
#   Cube 2: ω = −OMEGA rad/s (clockwise)
# Total initial angular momentum L_z ≈ 0 → should be conserved.
#
# Shared parameters (identical across CKMPM / MLS-MPM / Basic-MPM-3D):
#   E          = 1e6 Pa   (Fixed Corotated)
#   nu         = 0.4
#   rho        = 1000 kg/m³
#   omega      = ±2.0 rad/s
#   ppc        = 8  (2×2×2 sub-cell sampling)
#   CFL        = 0.5
#   fps        = 48
#   total_time = 5.0 s
#   gravity    = 0.0
#
# Grid resolution for this implementation:
#   grid_size = 64   →   dx = 1/64 m
#
# Cube geometry – fractional positions match CKMPM dual_rotation:
#   Cube 1 centre: (20, 32, 32) cells = (0.3125, 0.5, 0.5) m   ω = +2.0
#   Cube 2 centre: (44, 32, 32) cells = (0.6875, 0.5, 0.5) m   ω = −2.0
#   (CKMPM uses dx=1/256, centres at 80/256=0.3125 and 176/256=0.6875 → identical fractions)

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mpm_pytorch import MPMSolver
from mpm_pytorch.constitutive_models.elasticity import CorotatedElasticity
from mpm_pytorch.constitutive_models.plasticity import IdentityPlasticity

# ── Parameters ────────────────────────────────────────────────────────────────
GRID_SIZE  = 64
DX         = 1.0 / GRID_SIZE
E          = 1e6
NU         = 0.4
DENSITY    = 1000.0
CFL        = 0.5
PPC        = 8
FPS        = 48
TOTAL_TIME = 5.0
OMEGA      = 2.0       # rad/s

C_S           = ((E * (1 - NU)) / ((1 + NU) * (1 - 2*NU) * DENSITY)) ** 0.5
DT            = CFL * DX / C_S
PARTICLE_VOL  = DX**3 / PPC
PARTICLE_MASS = PARTICLE_VOL * DENSITY

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUT_GIF = os.path.join(OUT_DIR, "dual_rotation_mlsmpm.gif")
OUT_LZ  = os.path.join(OUT_DIR, "dual_rotation_Lz_mlsmpm.png")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Particle initialisation ───────────────────────────────────────────────────
def make_rotating_cube(cx_cell, cy_cell, cz_cell, omega_z):
    """
    9×9×9 grid of cells, 2×2×2 sub-cell sampling → 5832 particles.
    Velocity field: v = ω × r, ω = (0, 0, omega_z)
        vx = −omega_z * ry,   vy = +omega_z * rx,   vz = 0
    """
    pos_list, vel_list = [], []
    cx = cx_cell * DX
    cy = cy_cell * DX
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for w in range(8):
                    di = (w & 4) >> 2
                    dj = (w & 2) >> 1
                    dk =  w & 1
                    px = (cx_cell + i - 0.25 + di * 0.5) * DX
                    py = (cy_cell + j - 0.25 + dj * 0.5) * DX
                    pz = (cz_cell + k - 0.25 + dk * 0.5) * DX
                    rx = px - cx
                    ry = py - cy
                    vx = -omega_z * ry
                    vy = +omega_z * rx
                    pos_list.append([px, py, pz])
                    vel_list.append([vx, vy, 0.0])
    return (np.array(pos_list, dtype=np.float32),
            np.array(vel_list, dtype=np.float32))


# ── Build initial state ───────────────────────────────────────────────────────
pos0, vel0 = make_rotating_cube(20, 32, 32, +OMEGA)
pos1, vel1 = make_rotating_cube(44, 32, 32, -OMEGA)
n0 = len(pos0)   # 5832

domain_center = np.array([0.5, 0.5, 0.5])

def np_Lz(pos, vel):
    r = pos - domain_center
    return float(PARTICLE_MASS * np.sum(r[:, 0] * vel[:, 1] - r[:, 1] * vel[:, 0]))

Lz0_init = np_Lz(pos0, vel0)
Lz1_init = np_Lz(pos1, vel1)
print(f"[mlsmpm-rot]  grid={GRID_SIZE}  dx={DX:.4e}  dt={DT:.4e}  c_s={C_S:.4f}")
print(f"[mlsmpm-rot]  particle_vol={PARTICLE_VOL:.4e}  particle_mass={PARTICLE_MASS:.4e}")
print(f"[mlsmpm-rot]  Cube0 init L_z = {Lz0_init:.6e}")
print(f"[mlsmpm-rot]  Cube1 init L_z = {Lz1_init:.6e}")
print(f"[mlsmpm-rot]  Total init L_z = {Lz0_init + Lz1_init:.6e}  (should be ~0)")

all_pos = np.concatenate([pos0, pos1], axis=0)
all_vel = np.concatenate([vel0, vel1], axis=0)
N = len(all_pos)
print(f"[mlsmpm-rot]  Total particles: {N}  (n0={n0})")


# ── Simulation ────────────────────────────────────────────────────────────────
def run_simulation(device):
    x_t = torch.tensor(all_pos, dtype=torch.float32, device=device)
    v_t = torch.tensor(all_vel, dtype=torch.float32, device=device)
    C   = torch.zeros(N, 3, 3, dtype=torch.float32, device=device)
    F   = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(N, -1, -1).clone()

    solver = MPMSolver(
        init_pos=x_t,
        rho=DENSITY,
        num_grids=GRID_SIZE,
        dt=DT,
        gravity=[0.0, 0.0, 0.0],
        clip_bound=0.0,
        damping=1.0,
        device=device,
    )
    solver.vol    = float(PARTICLE_VOL)
    solver.p_mass = float(PARTICLE_MASS)
    solver.pre_particle_process  = []
    solver.post_grid_process     = []

    elasticity = CorotatedElasticity(E=E, nu=NU).to(device)
    plasticity = IdentityPlasticity().to(device)

    frames_0   = []   # cube 1 positions each frame
    frames_1   = []   # cube 2 positions each frame
    Lz_history = []   # total L_z each frame

    target_frame_dt = 1.0 / FPS
    next_frame_time = 0.0
    sim_time        = 0.0

    def record():
        pts = x_t.detach().cpu().numpy()
        vels = v_t.detach().cpu().numpy()
        frames_0.append(pts[:n0].copy())
        frames_1.append(pts[n0:].copy())
        # L_z = Σ m * ( (x−cx)*vy − (y−cy)*vx )
        r    = pts - domain_center
        Lz   = float(PARTICLE_MASS * np.sum(r[:, 0] * vels[:, 1] - r[:, 1] * vels[:, 0]))
        Lz_history.append(Lz)

    record()
    next_frame_time += target_frame_dt

    while sim_time < TOTAL_TIME - 1e-12:
        stress       = elasticity(F)
        x_t, v_t, C, F = solver(x_t, v_t, C, F, stress)
        F            = plasticity(F)
        sim_time    += DT

        if sim_time + 1e-12 >= next_frame_time:
            record()
            if len(frames_0) % 12 == 0:
                print(f"  Frame {len(frames_0):4d}/{int(TOTAL_TIME*FPS)}  "
                      f"t={sim_time:.3f}s  L_z={Lz_history[-1]:.4e}")
            next_frame_time += target_frame_dt

    return frames_0, frames_1, Lz_history


# ── 3-D GIF ───────────────────────────────────────────────────────────────────
def render_gif(frames_0, frames_1):
    fig = plt.figure(figsize=(7.5, 7.0))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0.15, 0.85); ax.set_ylim(0.35, 0.65); ax.set_zlim(0.35, 0.65)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Dual Rotation – MLS-MPM (3D)")
    ax.set_box_aspect((7, 3, 3))
    ax.view_init(elev=20.0, azim=-60.0)

    scat0 = ax.scatter([], [], [], s=1, c="#e74c3c", depthshade=False, label="Cube 1 (+ω)")
    scat1 = ax.scatter([], [], [], s=1, c="#2e86de", depthshade=False, label="Cube 2 (−ω)")
    time_txt = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="upper right", markerscale=5)

    def update(frame_id):
        p0 = frames_0[frame_id]; p1 = frames_1[frame_id]
        scat0._offsets3d = (p0[:, 0], p0[:, 1], p0[:, 2])
        scat1._offsets3d = (p1[:, 0], p1[:, 1], p1[:, 2])
        time_txt.set_text(f"t = {frame_id / FPS:.2f} s")
        return scat0, scat1, time_txt

    ani = animation.FuncAnimation(fig, update, frames=len(frames_0),
                                   interval=1000 // FPS, blit=False)
    ani.save(OUT_GIF, writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"[mlsmpm-rot]  GIF saved → {OUT_GIF}")


# ── Angular momentum plot ─────────────────────────────────────────────────────
def plot_Lz(Lz_history):
    times = np.arange(len(Lz_history)) / FPS
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, Lz_history, lw=1.5, color="#27ae60", label="L_z (total)")
    ax.axhline(0, color="k", ls="--", lw=0.8, label="L_z = 0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("L_z  (kg·m²/s)")
    ax.set_title("Total Angular Momentum Conservation – MLS-MPM, Dual Rotation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_LZ, dpi=120)
    plt.close(fig)
    print(f"[mlsmpm-rot]  L_z plot saved → {OUT_LZ}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mlsmpm-rot]  device={device}")
    frames_0, frames_1, Lz_history = run_simulation(device)
    print(f"[mlsmpm-rot]  simulation done, rendering …")
    render_gif(frames_0, frames_1)
    plot_Lz(Lz_history)
    Lz_arr = np.array(Lz_history)
    print(f"[mlsmpm-rot]  L_z: mean={Lz_arr.mean():.4e}  std={Lz_arr.std():.4e}  "
          f"max|L_z|={np.abs(Lz_arr).max():.4e}")
