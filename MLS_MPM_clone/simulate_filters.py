# run this one like: python simulate_filters.py --config examples/water.yaml
# From "A PyTorch Implementation of MLS-MPM (Moving Least Squares Material Point Method)"
# https://github.com/wgsxm/MPM-PyTorch
from typing import *

import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

from mpm_pytorch import MPMSolver, set_boundary_conditions, get_constitutive


def get_cube(
        center: List[float],
        size: List[float],
        num: int,
        add_noise: bool = False,
        device: torch.device = torch.device("cuda")
) -> Tensor:
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


def add_horizontal_cylinder_collider(
    model: MPMSolver,
    center_xz: List[float],
    radius: float,
    y_min: float = 0.0,
    y_max: float = 1.0,
    surface: str = "slip",
):
    """
    Horizontal cylinder aligned with Y-axis (like a log lying sideways).
    center_xz = [x_world, z_world] of the cylinder axis.
    radius, y_min, y_max all in world-space (0..1).

    KEY FIX: grid_x stores INTEGER grid indices, not world coords.
    We convert everything to index-space for correct comparison.
    """
    dx = model.dx   # e.g. 1/25 = 0.04

    cx_idx    = center_xz[0] / dx
    cz_idx    = center_xz[1] / dx
    r_idx     = radius / dx
    y_min_idx = y_min / dx
    y_max_idx = y_max / dx

    cx_t    = torch.tensor(cx_idx,    device=model.device).float()
    cz_t    = torch.tensor(cz_idx,    device=model.device).float()
    r_t     = torch.tensor(r_idx,     device=model.device).float()
    y_min_t = torch.tensor(y_min_idx, device=model.device).float()
    y_max_t = torch.tensor(y_max_idx, device=model.device).float()

    def collide(model, cx_t, cz_t, r_t, y_min_t, y_max_t, surface):
        pos = model.grid_x   # [N, 3] integer indices: [X_idx, Y_idx, Z_idx]

        # Restrict to Y extent of the cylinder
        in_y = (pos[:, 1] >= y_min_t) & (pos[:, 1] <= y_max_t)

        # Distance from cylinder axis in XZ index-space
        ex = pos[:, 0].float() - cx_t
        ez = pos[:, 2].float() - cz_t
        dist = torch.sqrt(ex ** 2 + ez ** 2) - r_t

        inside = (dist < 0) & in_y

        if not inside.any():
            return

        # Outward normal in XZ (world X and Z axes)
        norm = torch.sqrt(ex[inside] ** 2 + ez[inside] ** 2).clamp(min=1e-8)
        nx = ex[inside] / norm
        nz = ez[inside] / norm

        normal_3d = torch.zeros(inside.sum(), 3, device=model.device)
        normal_3d[:, 0] = nx
        normal_3d[:, 2] = nz

        mv = model.grid_mv[inside]
        dot = (mv * normal_3d).sum(dim=1, keepdim=True)

        if surface == "sticky":
            model.grid_mv[inside] = 0.0
        elif surface == "slip":
            # Only cancel velocity pointing INTO the surface
            dot_in = torch.clamp(dot, max=0.0)
            model.grid_mv[inside] = mv - dot_in * normal_3d
        elif surface == "collide":
            dot_in = torch.clamp(dot, max=0.0)
            model.grid_mv[inside] = mv - 2.0 * dot_in * normal_3d

    model.post_grid_process.append(
        partial(collide,
                cx_t=cx_t, cz_t=cz_t, r_t=r_t,
                y_min_t=y_min_t, y_max_t=y_max_t,
                surface=surface)
    )


def draw_horizontal_cylinder(ax, cx, cz, radius, y_min, y_max,
                              color='gray', alpha=0.5, n_theta=30):
    """Wireframe of a Y-axis-aligned cylinder in world space."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    for y in [y_min, y_max]:
        ax.plot(cx + radius * np.cos(theta),
                np.full(n_theta, y),
                cz + radius * np.sin(theta),
                color=color, alpha=alpha, linewidth=1)
    for t in theta[::5]:
        ax.plot([cx + radius * np.cos(t), cx + radius * np.cos(t)],
                [y_min, y_max],
                [cz + radius * np.sin(t), cz + radius * np.sin(t)],
                color=color, alpha=alpha, linewidth=1)


def visualize_frames(
    frames, export_path,
    center=[0.5, 0.5, 0.5], size=[1.0, 1.0, 1.0],
    c='blue', s=20, fps=30,
    horiz_cylinders=None,
):
    xlim = [center[0] - size[0]/2, center[0] + size[0]/2]
    ylim = [center[1] - size[1]/2, center[1] + size[1]/2]
    zlim = [center[2] - size[2]/2, center[2] + size[2]/2]

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-60)

    def update(frame):
        ax.cla()
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_xlabel('X');  ax.set_ylabel('Y');  ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame}')
        if horiz_cylinders:
            for cyl in horiz_cylinders:
                draw_horizontal_cylinder(ax, cyl['cx'], cyl['cz'], cyl['r'],
                                         cyl['y_min'], cyl['y_max'])
        ax.scatter(frames[frame][:, 0], frames[frame][:, 1],
                   frames[frame][:, 2], s=s, c=c, depthshade=True)
        return []

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(export_path, writer='pillow', fps=fps)
    plt.close()
    print(f'Saved to {export_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    print(f'Start simulation with config: {args.config}')

    cfg = OmegaConf.load(args.config)
    material_params = cfg.material
    sim_params      = cfg.sim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)
    export_path = os.path.join(cfg.output_dir, cfg.tag + "_filter.gif")

    # ------------------------------------------------------------------
    # SCENE: block falls from Z=0.78, hits 3 horizontal cylinders at Z=0.50
    #
    # Side view (X on horizontal, Z on vertical):
    #
    #   Z=0.86 ┌──────────────────────┐
    #          │  block (0.4 x 0.4 x  │
    #   Z=0.70 └──0.16)───────────────┘
    #
    #   Z=0.50    ○      ○      ○     <- horizontal cylinders (Y-axis)
    #            X=0.2  X=0.5  X=0.8
    #
    #   Z=0.02 ══════════════════════  <- floor
    # ------------------------------------------------------------------

    particles = get_cube(
        center=[0.5, 0.5, 0.78],
        size=[0.4, 0.4, 0.16],
        num=12,
        add_noise=True,
        device=device,
    )
    n_particles = particles.shape[0]

    mpm_solver = MPMSolver(particles, enable_train=False, device=device)
    set_boundary_conditions(mpm_solver, sim_params.boundary_conditions)

    RADIUS = 0.05
    CYL_Z  = 0.50
    Y_MIN  = 0.05
    Y_MAX  = 0.95

    cylinder_defs = [
        {'cx': 0.20, 'cz': CYL_Z, 'r': RADIUS, 'y_min': Y_MIN, 'y_max': Y_MAX},
        {'cx': 0.50, 'cz': CYL_Z, 'r': RADIUS, 'y_min': Y_MIN, 'y_max': Y_MAX},
        {'cx': 0.80, 'cz': CYL_Z, 'r': RADIUS, 'y_min': Y_MIN, 'y_max': Y_MAX},
    ]

    for cyl in cylinder_defs:
        add_horizontal_cylinder_collider(
            mpm_solver,
            center_xz=[cyl['cx'], cyl['cz']],
            radius=cyl['r'],
            y_min=cyl['y_min'],
            y_max=cyl['y_max'],
            surface="slip",
        )

    elasticity = get_constitutive(material_params.elasticity, device=device)
    plasticity = get_constitutive(material_params.plasticity, device=device)

    x = particles
    v = torch.stack([torch.tensor(sim_params.initial_velocity, device=device)
                     for _ in range(n_particles)])
    C = torch.zeros((n_particles, 3, 3), device=device)
    F = torch.eye(3, device=device).unsqueeze(0).repeat(n_particles, 1, 1)

    frames = []
    for frame in tqdm(range(sim_params.num_frames), desc='Simulating', leave=False):
        frames.append(x.cpu().numpy())
        for step in tqdm(range(sim_params.steps_per_frame), desc='Step', leave=False):
            stress = elasticity(F)
            x, v, C, F = mpm_solver(x, v, C, F, stress)
            F = plasticity(F)

    print(f'Rendering to {export_path}...')
    visualize_frames(
        frames,
        export_path=export_path,
        center=[0.5, 0.5, 0.5],
        size=[1.0, 1.0, 1.0],
        c=material_params.color,
        fps=30,
        horiz_cylinders=cylinder_defs,
    )