"""
Test 2: Material block dropping on 3 horizontal cylinder filters (3D)
Run: python test2_filter_drop.py --config examples/water.yaml
     python test2_filter_drop.py --config examples/sand.yaml
     python test2_filter_drop.py --config examples/snow.yaml
     python test2_filter_drop.py --config examples/jelly.yaml
Metrics recorded: particle positions per frame, total kinetic energy per frame
"""
from typing import *
import argparse, os, csv
from omegaconf import OmegaConf
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

from mpm_pytorch import MPMSolver, set_boundary_conditions, get_constitutive


def get_cube(center, size, num, add_noise=False, device=torch.device("cuda")):
    start = torch.tensor(center) - torch.tensor(size) / 2
    end   = torch.tensor(center) + torch.tensor(size) / 2
    x = torch.linspace(start[0], end[0], num)
    y = torch.linspace(start[1], end[1], num)
    z = torch.linspace(start[2], end[2], num)
    cube = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).view(-1, 3)
    if add_noise:
        noisy = start + torch.rand_like(cube) * (end - start)
        cube = torch.cat([cube, noisy], dim=0)
    return cube.to(device)


def add_horizontal_cylinder_collider(model, center_xz, radius, y_min=0.0, y_max=1.0, surface="slip"):
    """
    Horizontal cylinder aligned with Y-axis.
    IMPORTANT: grid_x stores integer grid indices, not world coords.
    We convert radius and centers to index-space for correct comparison.
    """
    dx = model.dx
    cx_t    = torch.tensor(center_xz[0] / dx, device=model.device).float()
    cz_t    = torch.tensor(center_xz[1] / dx, device=model.device).float()
    r_t     = torch.tensor(radius / dx,        device=model.device).float()
    y_min_t = torch.tensor(y_min / dx,         device=model.device).float()
    y_max_t = torch.tensor(y_max / dx,         device=model.device).float()

    def collide(model, cx_t, cz_t, r_t, y_min_t, y_max_t, surface):
        pos = model.grid_x   # integer indices [N, 3]
        in_y = (pos[:, 1] >= y_min_t) & (pos[:, 1] <= y_max_t)
        ex = pos[:, 0].float() - cx_t
        ez = pos[:, 2].float() - cz_t
        dist = torch.sqrt(ex**2 + ez**2) - r_t
        inside = (dist < 0) & in_y
        if not inside.any():
            return
        norm = torch.sqrt(ex[inside]**2 + ez[inside]**2).clamp(min=1e-8)
        normal_3d = torch.zeros(inside.sum(), 3, device=model.device)
        normal_3d[:, 0] = ex[inside] / norm
        normal_3d[:, 2] = ez[inside] / norm
        mv  = model.grid_mv[inside]
        dot = (mv * normal_3d).sum(dim=1, keepdim=True)
        if surface == "sticky":
            model.grid_mv[inside] = 0.0
        elif surface == "slip":
            model.grid_mv[inside] = mv - torch.clamp(dot, max=0.0) * normal_3d
        elif surface == "collide":
            model.grid_mv[inside] = mv - 2.0 * torch.clamp(dot, max=0.0) * normal_3d

    model.post_grid_process.append(
        partial(collide, cx_t=cx_t, cz_t=cz_t, r_t=r_t,
                y_min_t=y_min_t, y_max_t=y_max_t, surface=surface))


def draw_horizontal_cylinder(ax, cx, cz, r, y_min, y_max, color='dimgray', alpha=0.5):
    theta = np.linspace(0, 2*np.pi, 30)
    for y in [y_min, y_max]:
        ax.plot(cx + r*np.cos(theta), np.full(30, y), cz + r*np.sin(theta),
                color=color, alpha=alpha, lw=1)
    for t in theta[::5]:
        ax.plot([cx+r*np.cos(t)]*2, [y_min, y_max], [cz+r*np.sin(t)]*2,
                color=color, alpha=alpha, lw=1)


def visualize(frames, export_path, c='blue', s=15, fps=24, cylinders=None):
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=18, azim=-55)

    def update(i):
        ax.cla()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Frame {i}')
        if cylinders:
            for cyl in cylinders:
                draw_horizontal_cylinder(ax, cyl['cx'], cyl['cz'], cyl['r'],
                                         cyl['y_min'], cyl['y_max'])
        ax.scatter(frames[i][:, 0], frames[i][:, 1], frames[i][:, 2],
                   s=s, c=c, depthshade=True)
        return []

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(export_path, writer='pillow', fps=fps)
    plt.close()
    print(f'  GIF saved → {export_path}')


def save_energy_plot(energy_log, export_path):
    frames_arr = [e['frame'] for e in energy_log]
    ke_arr     = [e['kinetic_energy'] for e in energy_log]
    plt.figure(figsize=(7, 4))
    plt.plot(frames_arr, ke_arr, label='Kinetic Energy')
    plt.xlabel('Frame'); plt.ylabel('Energy (J)')
    plt.title('Kinetic Energy over Time — Filter Drop Test')
    plt.legend(); plt.tight_layout()
    plt.savefig(export_path, dpi=120)
    plt.close()
    print(f'  Energy plot saved → {export_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    mat = cfg.material
    sim = cfg.sim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = cfg.tag + "_filter"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # GEOMETRY
    # Block: wide and thin so it spans all 3 cylinders
    #   center=[0.5, 0.5, 0.80], size=[0.40, 0.40, 0.14]
    #
    # 3 cylinders at X = 0.20, 0.50, 0.80, all at Z = 0.48
    #   radius = 0.04, spanning full Y
    #   gaps between cylinders: 0.26 wide (> block half-width 0.04)
    #   so material WILL fall between them
    # ----------------------------------------------------------------
    particles = get_cube(
        center=[0.5, 0.5, 0.80],
        size=[0.40, 0.40, 0.14],
        num=14, add_noise=True, device=device)
    n_particles = particles.shape[0]
    print(f'Particles: {n_particles}')

    solver = MPMSolver(particles, enable_train=False, device=device)
    set_boundary_conditions(solver, sim.boundary_conditions)

    RADIUS = 0.06
    CYL_Z  = 0.50
    cylinders = [
        {'cx': 0.20, 'cz': CYL_Z, 'r': RADIUS, 'y_min': 0.02, 'y_max': 0.98},
        {'cx': 0.50, 'cz': CYL_Z, 'r': RADIUS, 'y_min': 0.02, 'y_max': 0.98},
        {'cx': 0.80, 'cz': CYL_Z, 'r': RADIUS, 'y_min': 0.02, 'y_max': 0.98},
    ]
    for cyl in cylinders:
        add_horizontal_cylinder_collider(
            solver, [cyl['cx'], cyl['cz']], cyl['r'],
            cyl['y_min'], cyl['y_max'], surface="slip")

    elasticity = get_constitutive(mat.elasticity, device=device)
    plasticity = get_constitutive(mat.plasticity, device=device)

    x = particles
    v = torch.stack([torch.tensor(sim.initial_velocity, device=device)
                     for _ in range(n_particles)])
    C = torch.zeros((n_particles, 3, 3), device=device)
    F = torch.eye(3, device=device).unsqueeze(0).repeat(n_particles, 1, 1)

    frames     = []
    energy_log = []

    for frame in tqdm(range(sim.num_frames), desc='Simulating'):
        frames.append(x.cpu().numpy())
        ke = 0.5 * solver.p_mass * (v ** 2).sum(dim=1).sum().item()
        energy_log.append({'frame': frame, 'kinetic_energy': ke})

        for _ in range(sim.steps_per_frame):
            stress = elasticity(F)
            x, v, C, F = solver(x, v, C, F, stress)
            F = plasticity(F)

    # Save outputs
    gif_path    = os.path.join(cfg.output_dir, tag + ".gif")
    energy_path = os.path.join(cfg.output_dir, tag + "_energy.png")
    csv_path    = os.path.join(cfg.output_dir, tag + "_energy.csv")

    print('Rendering...')
    visualize(frames, gif_path, c=mat.color, cylinders=cylinders)
    save_energy_plot(energy_log, energy_path)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'kinetic_energy'])
        writer.writeheader(); writer.writerows(energy_log)
    print(f'  CSV saved → {csv_path}')
