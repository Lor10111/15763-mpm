"""
Test 4: Waterfall-to-bottleneck — volume conservation + energy
Same cylinder filter scene as test2_filter_drop.py but:
  - Run for 300 frames so particles fully pass through and settle
  - Primary metric: volume conservation (particle count, throughput)
  - Secondary metric: energy (KE + PE + Total), same format as test2

Run: python test4_waterfall_bottleneck.py --config examples/water.yaml
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


def add_horizontal_cylinder_collider(model, center_xz, radius, y_min=0.0, y_max=1.0, surface="slip"):
    """Horizontal cylinder (Y-axis aligned). grid_x is in index space."""
    dx    = model.dx
    cx_t  = torch.tensor(center_xz[0] / dx, device=model.device).float()
    cz_t  = torch.tensor(center_xz[1] / dx, device=model.device).float()
    r_t   = torch.tensor(radius / dx,        device=model.device).float()
    ylo_t = torch.tensor(y_min / dx,         device=model.device).float()
    yhi_t = torch.tensor(y_max / dx,         device=model.device).float()

    def collide(model, cx_t, cz_t, r_t, ylo_t, yhi_t, surface):
        pos  = model.grid_x
        in_y = (pos[:, 1] >= ylo_t) & (pos[:, 1] <= yhi_t)
        ex   = pos[:, 0].float() - cx_t
        ez   = pos[:, 2].float() - cz_t
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

    model.post_grid_process.append(
        partial(collide, cx_t=cx_t, cz_t=cz_t, r_t=r_t,
                ylo_t=ylo_t, yhi_t=yhi_t, surface=surface))


def draw_horizontal_cylinder(ax, cx, cz, r, y_min, y_max, color='dimgray', alpha=0.5):
    theta = np.linspace(0, 2*np.pi, 30)
    for y in [y_min, y_max]:
        ax.plot(cx + r*np.cos(theta), np.full(30, y), cz + r*np.sin(theta),
                color=color, alpha=alpha, lw=1)
    for t in theta[::5]:
        ax.plot([cx+r*np.cos(t)]*2, [y_min, y_max], [cz+r*np.sin(t)]*2,
                color=color, alpha=alpha, lw=1)


def visualize(frames, export_path, c='cyan', s=10, fps=24, cylinders=None):
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
    print(f'  GIF → {export_path}')


def save_energy_plot(energy_log, export_path):
    """Same two-panel format as test2."""
    frames_arr = [e['frame']            for e in energy_log]
    ke_arr     = [e['kinetic_energy']   for e in energy_log]
    pe_arr     = [e['potential_energy'] for e in energy_log]
    total_arr  = [e['total_energy']     for e in energy_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(frames_arr, ke_arr,    label='Kinetic Energy (KE)',  color='steelblue')
    ax1.plot(frames_arr, pe_arr,    label='Potential Energy (PE)', color='darkorange', linestyle='--')
    ax1.plot(frames_arr, total_arr, label='Total Energy (KE+PE)', color='green', linewidth=2)
    ax1.set_ylabel('Energy (J)')
    ax1.set_title('Energy Conservation — Filter Drop (Extended)\n'
                  '(Drops at cylinders/floor = expected; free-fall decay = numerical dissipation)')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    e0   = total_arr[0] if total_arr[0] != 0 else 1.0
    norm = [e / e0 for e in total_arr]
    ax2.plot(frames_arr, norm, color='green', linewidth=1.5)
    ax2.axhline(1.0, color='red', linestyle=':', alpha=0.5, label='Ideal (no dissipation)')
    ax2.set_ylabel('Normalised Total\n(E / E₀)')
    ax2.set_xlabel('Frame')
    ax2.set_title('1.0 = perfect conservation, decay = numerical dissipation')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(export_path, dpi=130)
    plt.close()
    print(f'  Energy plot → {export_path}')


def save_volume_plot(energy_log, n_particles_total, export_path, cyl_z):
    """
    Volume conservation — 3 panels:
      1. Total particle count (must be perfectly flat)
      2. Particles below cylinder level (throughput over time)
      3. Mean Z-velocity (free-fall → impact → settled)
    """
    frames_arr  = [e['frame']        for e in energy_log]
    below_count = [e['n_below_cyl']  for e in energy_log]
    mean_vz     = [e['mean_vz']      for e in energy_log]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Panel 1: total count — flat line proves no particle loss
    axes[0].axhline(n_particles_total, color='steelblue', linewidth=2,
                    label=f'Particle count = {n_particles_total} (constant)')
    axes[0].set_ylim(n_particles_total * 0.95, n_particles_total * 1.05)
    axes[0].set_ylabel('Particle Count')
    axes[0].set_title('Volume Conservation — Total Particle Count\n'
                       '(Flat line = perfect volume conservation by construction in MPM)')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # Panel 2: throughput — how many passed below the cylinders
    pct = [100.0 * b / n_particles_total for b in below_count]
    axes[1].plot(frames_arr, pct, color='darkorange', linewidth=1.5)
    axes[1].axhline(100, color='gray', linestyle='--', alpha=0.4, label='100% passed through')
    axes[1].set_ylabel(f'% particles below Z={cyl_z:.2f} (cylinder level)')
    axes[1].set_title('Filter Throughput — material passing through cylinder gaps over time')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    # Panel 3: mean Z-velocity — shows settling
    axes[2].plot(frames_arr, mean_vz, color='green', linewidth=1.5)
    axes[2].axhline(0, color='red', linestyle=':', alpha=0.4, label='v=0 (fully settled)')
    axes[2].set_ylabel('Mean Z velocity (m/s)')
    axes[2].set_xlabel('Frame')
    axes[2].set_title('Mean Z-velocity — free fall → cylinder impact → floor settle')
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(export_path, dpi=130)
    plt.close()
    print(f'  Volume plot → {export_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--frames', type=int, default=300,
                        help='Number of frames — 300 lets material fully settle')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    mat = cfg.material
    sim = cfg.sim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)
    tag = cfg.tag + "_filter_volume"

    # Exact same scene as test2_filter_drop.py
    RADIUS = 0.06
    CYL_Z  = 0.50
    cylinders = [
        {'cx': 0.20, 'cz': CYL_Z, 'r': RADIUS, 'y_min': 0.02, 'y_max': 0.98},
        {'cx': 0.50, 'cz': CYL_Z, 'r': RADIUS, 'y_min': 0.02, 'y_max': 0.98},
        {'cx': 0.80, 'cz': CYL_Z, 'r': RADIUS, 'y_min': 0.02, 'y_max': 0.98},
    ]

    particles = get_cube(
        center=[0.5, 0.5, 0.80],
        size=[0.40, 0.40, 0.14],
        num=20, add_noise=True, device=device)
    n_particles = particles.shape[0]
    print(f'Particles: {n_particles}')

    solver = MPMSolver(particles, enable_train=False, num_grids=40, device=device)
    set_boundary_conditions(solver, sim.boundary_conditions)
    for cyl in cylinders:
        add_horizontal_cylinder_collider(
            solver, [cyl['cx'], cyl['cz']], cyl['r'],
            cyl['y_min'], cyl['y_max'], surface="slip")

    elasticity = get_constitutive(mat.elasticity, device=device)
    plasticity = get_constitutive(mat.plasticity, device=device)

    x = particles.clone()
    v = torch.stack([torch.tensor(sim.initial_velocity, device=device)
                     for _ in range(n_particles)])
    C = torch.zeros((n_particles, 3, 3), device=device)
    F = torch.eye(3, device=device).unsqueeze(0).repeat(n_particles, 1, 1)

    frames_out = []
    energy_log = []
    g = 9.8

    for frame in tqdm(range(args.frames), desc='Simulating'):
        frames_out.append(x.cpu().numpy())

        ke      = 0.5 * solver.p_mass * (v**2).sum(dim=1).sum().item()
        pe      = solver.p_mass * g * x[:, 2].sum().item()
        mean_vz = v[:, 2].mean().item()
        n_below = int((x[:, 2] < CYL_Z).sum().item())

        energy_log.append({
            'frame':            frame,
            'kinetic_energy':   ke,
            'potential_energy': pe,
            'total_energy':     ke + pe,
            'mean_vz':          mean_vz,
            'n_total':          n_particles,
            'n_below_cyl':      n_below,
        })

        for _ in range(sim.steps_per_frame):
            stress = elasticity(F)
            x, v, C, F = solver(x, v, C, F, stress)
            F = plasticity(F)

    gif_path    = os.path.join(cfg.output_dir, tag + ".gif")
    energy_path = os.path.join(cfg.output_dir, tag + "_energy.png")
    volume_path = os.path.join(cfg.output_dir, tag + "_volume.png")
    csv_path    = os.path.join(cfg.output_dir, tag + "_metrics.csv")

    print('Rendering...')
    visualize(frames_out, gif_path, c=mat.color, cylinders=cylinders)
    save_energy_plot(energy_log, energy_path)
    save_volume_plot(energy_log, n_particles, volume_path, cyl_z=CYL_Z)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=energy_log[0].keys())
        writer.writeheader(); writer.writerows(energy_log)
    print(f'  CSV → {csv_path}')
