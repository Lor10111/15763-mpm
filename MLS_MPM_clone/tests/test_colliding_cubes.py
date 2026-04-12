import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpm_pytorch import MPMSolver
from mpm_pytorch.constitutive_models.elasticity import CorotatedElasticity
from mpm_pytorch.constitutive_models.plasticity import IdentityPlasticity


GRID_SIZE = 64
DX = 1.0 / GRID_SIZE
E = 1e6
NU = 0.4
DENSITY = 1000.0
CFL = 0.5
PPC = 8
FPS = 48
TOTAL_TIME = 5.0
V0 = 0.1

C_S = ((E * (1.0 - NU)) / ((1.0 + NU) * (1.0 - 2.0 * NU) * DENSITY)) ** 0.5
DT = CFL * DX / C_S
PARTICLE_VOL = DX ** 3 / PPC
PARTICLE_MASS = PARTICLE_VOL * DENSITY
STEPS_PER_FRAME = max(1, int(round(1.0 / (FPS * DT))))
TOTAL_FRAMES = int(round(TOTAL_TIME * FPS))

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUT_GIF = os.path.join(OUT_DIR, "colliding_cubes.gif")
os.makedirs(OUT_DIR, exist_ok=True)


def make_cube(cx_cell, cy_cell, cz_cell, vx):
    positions = []
    velocities = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for w in range(8):
                    di = (w & 4) >> 2
                    dj = (w & 2) >> 1
                    dk = w & 1
                    positions.append(
                        [
                            (cx_cell + i + 0.25 + di * 0.5) * DX,
                            (cy_cell + j + 0.25 + dj * 0.5) * DX,
                            (cz_cell + k + 0.25 + dk * 0.5) * DX,
                        ]
                    )
                    velocities.append([vx, 0.0, 0.0])
    return np.asarray(positions, dtype=np.float32), np.asarray(velocities, dtype=np.float32)


def build_initial_state():
    pos0, vel0 = make_cube(16, 32, 32, +V0)
    pos1, vel1 = make_cube(48, 32, 32, -V0)
    all_pos = np.concatenate([pos0, pos1], axis=0)
    all_vel = np.concatenate([vel0, vel1], axis=0)
    return pos0, pos1, all_pos, all_vel


def run_simulation(device):
    pos0, pos1, all_pos, all_vel = build_initial_state()
    n0 = len(pos0)
    x_t = torch.tensor(all_pos, device=device)
    v_t = torch.tensor(all_vel, device=device)

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
    solver.vol = float(PARTICLE_VOL)
    solver.p_mass = float(PARTICLE_MASS)
    solver.pre_particle_process = []
    solver.post_grid_process = []

    elasticity = CorotatedElasticity(E=E, nu=NU).to(device)
    plasticity = IdentityPlasticity().to(device)

    num_particles = all_pos.shape[0]
    C = torch.zeros(num_particles, 3, 3, device=device)
    F = torch.eye(3, device=device).unsqueeze(0).expand(num_particles, -1, -1).clone()

    frames_0 = []
    frames_1 = []

    def record():
        positions = x_t.detach().cpu().numpy()
        frames_0.append(positions[:n0].copy())
        frames_1.append(positions[n0:].copy())

    record()

    for _ in tqdm(range(1, TOTAL_FRAMES), desc="Simulating"):
        for _ in range(STEPS_PER_FRAME):
            stress = elasticity(F)
            x_t, v_t, C, F = solver(x_t, v_t, C, F, stress)
            F = plasticity(F)
        record()

    return frames_0, frames_1


def render_gif(frames_0, frames_1):
    fig = plt.figure(figsize=(7.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0.25, 0.75)
    ax.set_ylim(0.25, 0.75)
    ax.set_zlim(0.25, 0.75)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Colliding Cubes - MLS-MPM (3D)")
    ax.set_proj_type("persp")
    ax.view_init(elev=20.0, azim=-65.0)
    ax.view_init(elev=20.0, azim=-90.0)
    ax.grid(False)

    scat0 = ax.scatter([], [], [], s=1, c="#e74c3c", depthshade=False, label="Cube 1")
    scat1 = ax.scatter([], [], [], s=1, c="#2e86de", depthshade=False, label="Cube 2")
    time_txt = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="upper right", markerscale=5)

    def update(frame_id):
        pts0 = frames_0[frame_id]
        pts1 = frames_1[frame_id]
        scat0._offsets3d = (pts0[:, 0], pts0[:, 1], pts0[:, 2])
        scat1._offsets3d = (pts1[:, 0], pts1[:, 1], pts1[:, 2])
        time_txt.set_text(f"t = {frame_id / FPS:.3f} s")
        return scat0, scat1, time_txt

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames_0),
        interval=1000 // FPS,
        blit=False,
    )
    ani.save(OUT_GIF, writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mlsmpm] grid_size={GRID_SIZE} dx={DX:.4e} dt={DT:.4e} total_frames={TOTAL_FRAMES}")
    print(f"[mlsmpm] particle_vol={PARTICLE_VOL:.3e} particle_mass={PARTICLE_MASS:.3e}")
    frames_0, frames_1 = run_simulation(device)
    print(f"[mlsmpm] simulation done, rendering 3D GIF -> {OUT_GIF}")
    render_gif(frames_0, frames_1)
    print(f"[mlsmpm] GIF saved -> {OUT_GIF}")
