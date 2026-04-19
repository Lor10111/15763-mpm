"""
Test 4: Waterfall-to-bottleneck (fluid behavior)
A wide liquid block falls into a container that funnels into a narrow opening.

Run: python test4_waterfall_bottleneck.py --config examples/water.yaml

Metrics recorded:
  - Total particle count per frame (volume conservation proxy)
  - Mean Z velocity at 3 stages: free-fall, post-impact, bottleneck passage
  - Kinetic energy over time
"""
from typing import *
import argparse, os, csv
from omegaconf import OmegaConf
import numpy as np
import torch
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


def add_box_wall(model, axis, bound, side='low', surface='sticky'):
    """
    Flat wall perpendicular to `axis`.
    side='low'  → wall at x[axis] < bound  (floor / left wall)
    side='high' → wall at x[axis] > bound  (ceiling / right wall)
    Operates in INDEX space (grid_x stores integer indices).
    """
    dx = model.dx
    bound_idx = bound / dx

    normal_world = torch.zeros(3, device=model.device)
    if side == 'low':
        normal_world[axis] =  1.0   # normal points inward (+)
    else:
        normal_world[axis] = -1.0   # normal points inward (-)

    def collide(model, axis, bound_idx, side, surface, normal_world):
        pos = model.grid_x
        if side == 'low':
            mask = pos[:, axis].float() < bound_idx
        else:
            mask = pos[:, axis].float() > bound_idx
        if not mask.any():
            return
        if surface == 'sticky':
            model.grid_mv[mask] = 0.0
        elif surface == 'slip':
            mv  = model.grid_mv[mask]
            dot = (mv * normal_world).sum(dim=1, keepdim=True)
            model.grid_mv[mask] = mv - torch.clamp(dot, max=0.0) * normal_world

    model.post_grid_process.append(
        partial(collide, axis=axis, bound_idx=bound_idx,
                side=side, surface=surface, normal_world=normal_world))


def add_bottleneck_walls(model, neck_x_lo, neck_x_hi, z_top, surface='sticky'):
    """
    Two vertical walls that create a narrow bottleneck at z < z_top.
    Left wall:  x < neck_x_lo   for z < z_top
    Right wall: x > neck_x_hi   for z < z_top
    """
    dx = model.dx
    xlo_idx  = neck_x_lo / dx
    xhi_idx  = neck_x_hi / dx
    ztop_idx = z_top / dx

    def collide_neck(model, xlo_idx, xhi_idx, ztop_idx, surface):
        pos  = model.grid_x
        low_z = pos[:, 2].float() < ztop_idx
        # Left wall
        left  = (pos[:, 0].float() < xlo_idx) & low_z
        # Right wall
        right = (pos[:, 0].float() > xhi_idx) & low_z

        n_left  = torch.tensor([1., 0., 0.], device=model.device)
        n_right = torch.tensor([-1., 0., 0.], device=model.device)

        for mask, normal in [(left, n_left), (right, n_right)]:
            if mask.any():
                if surface == 'sticky':
                    model.grid_mv[mask] = 0.0
                elif surface == 'slip':
                    mv  = model.grid_mv[mask]
                    dot = (mv * normal).sum(dim=1, keepdim=True)
                    model.grid_mv[mask] = mv - torch.clamp(dot, max=0.0) * normal

    model.post_grid_process.append(
        partial(collide_neck, xlo_idx=xlo_idx, xhi_idx=xhi_idx,
                ztop_idx=ztop_idx, surface=surface))


def draw_bottleneck(ax, neck_x_lo, neck_x_hi, z_top, y_lo=0.1, y_hi=0.9):
    """Draw the container walls as lines."""
    # Left wall above neck
    ax.plot([neck_x_lo, neck_x_lo], [y_lo, y_hi], [z_top, z_top],
            color='brown', lw=2, alpha=0.7)
    ax.plot([neck_x_lo, neck_x_lo], [y_lo, y_hi], [0.05, z_top],
            color='brown', lw=2, alpha=0.7, linestyle='--')
    # Right wall above neck
    ax.plot([neck_x_hi, neck_x_hi], [y_lo, y_hi], [z_top, z_top],
            color='brown', lw=2, alpha=0.7)
    ax.plot([neck_x_hi, neck_x_hi], [y_lo, y_hi], [0.05, z_top],
            color='brown', lw=2, alpha=0.7, linestyle='--')


def visualize(frames, export_path, c='cyan', s=12, fps=24,
              neck_x_lo=0.38, neck_x_hi=0.62, z_top=0.35):
    fig = plt.figure(figsize=(6, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15, azim=-50)

    def update(i):
        ax.cla()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Frame {i}')
        draw_bottleneck(ax, neck_x_lo, neck_x_hi, z_top)
        ax.scatter(frames[i][:, 0], frames[i][:, 1], frames[i][:, 2],
                   s=s, c=c, depthshade=True)
        return []

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(export_path, writer='pillow', fps=fps)
    plt.close()
    print(f'  GIF → {export_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    mat = cfg.material
    sim = cfg.sim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)
    tag = cfg.tag + "_bottleneck"

    # ----------------------------------------------------------------
    # GEOMETRY
    #
    #  Z=0.92 ┌──────────────────────┐
    #         │   FLUID BLOCK        │  wide: 0.55 x 0.55 x 0.10
    #  Z=0.82 └──────────────────────┘
    #
    #  Z=0.35 |   left    |  neck  |  right  |   <- bottleneck walls
    #         |  blocked  | x:0.38 |  blocked|      narrow gap = 0.24
    #
    #  Z=0.02 ═════════════════════════════════  <- floor
    # ----------------------------------------------------------------
    NECK_X_LO = 0.38
    NECK_X_HI = 0.62
    Z_TOP     = 0.35   # walls only exist below this height

    particles = get_cube(
        center=[0.5, 0.5, 0.87],
        size=[0.55, 0.55, 0.10],
        num=14, add_noise=True, device=device)
    n_particles = particles.shape[0]
    print(f'Particles: {n_particles}')

    solver = MPMSolver(particles, enable_train=False, device=device)
    set_boundary_conditions(solver, sim.boundary_conditions)
    add_bottleneck_walls(solver, NECK_X_LO, NECK_X_HI, Z_TOP, surface='sticky')

    elasticity = get_constitutive(mat.elasticity, device=device)
    plasticity = get_constitutive(mat.plasticity, device=device)

    x = particles
    v = torch.stack([torch.tensor(sim.initial_velocity, device=device)
                     for _ in range(n_particles)])
    C = torch.zeros((n_particles, 3, 3), device=device)
    F = torch.eye(3, device=device).unsqueeze(0).repeat(n_particles, 1, 1)

    frames     = []
    energy_log = []
    # Stage markers: free-fall = frame 30, post-impact = frame 70, bottleneck = frame 120
    stage_frames = {30: 'free_fall', 70: 'post_impact', 120: 'bottleneck'}
    stage_data   = {}

    for frame in tqdm(range(sim.num_frames), desc='Simulating'):
        frames.append(x.cpu().numpy())
        ke = 0.5 * solver.p_mass * (v**2).sum(dim=1).sum().item()
        mean_vz = v[:, 2].mean().item()
        energy_log.append({'frame': frame, 'kinetic_energy': ke, 'mean_vz': mean_vz})
        if frame in stage_frames:
            stage_data[stage_frames[frame]] = {
                'mean_vz': mean_vz, 'ke': ke,
                'n_below_neck': int((x[:, 2] < Z_TOP).sum().item())
            }

        for _ in range(sim.steps_per_frame):
            stress = elasticity(F)
            x, v, C, F = solver(x, v, C, F, stress)
            F = plasticity(F)

    # Outputs
    gif_path    = os.path.join(cfg.output_dir, tag + ".gif")
    energy_path = os.path.join(cfg.output_dir, tag + "_energy.png")
    csv_path    = os.path.join(cfg.output_dir, tag + "_metrics.csv")

    print('Rendering...')
    visualize(frames, gif_path, c=mat.color,
              neck_x_lo=NECK_X_LO, neck_x_hi=NECK_X_HI, z_top=Z_TOP)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    frames_arr = [e['frame'] for e in energy_log]
    ax1.plot(frames_arr, [e['kinetic_energy'] for e in energy_log])
    ax1.set_ylabel('Kinetic Energy'); ax1.set_title('Waterfall-to-Bottleneck Metrics')
    ax2.plot(frames_arr, [e['mean_vz'] for e in energy_log], color='orange')
    ax2.set_ylabel('Mean Z-velocity'); ax2.set_xlabel('Frame')
    for f, label in stage_frames.items():
        ax1.axvline(f, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(f, color='gray', linestyle='--', alpha=0.5)
        ax2.text(f+1, ax2.get_ylim()[0], label, fontsize=7, rotation=90)
    plt.tight_layout(); plt.savefig(energy_path, dpi=120); plt.close()
    print(f'  Energy plot → {energy_path}')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'kinetic_energy', 'mean_vz'])
        writer.writeheader(); writer.writerows(energy_log)
    print(f'  CSV → {csv_path}')
    print(f'\nStage summary: {stage_data}')
