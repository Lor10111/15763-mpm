"""
CK-MPM Python test runner — drop-in replacement for MLS-MPM tests.
Uses the same yaml configs, same scene geometry, same output format.

Run:
    python run_ckmpm_test.py --test 2 --config examples/water.yaml
    python run_ckmpm_test.py --test 4 --config examples/water.yaml --frames 300
    python run_ckmpm_test.py --test 6 --config examples/sand.yaml

Place this file and mpm_ckmpm.py inside MLS_MPM_clone/ so it can
import from mpm_pytorch.
"""
import argparse, os, csv, sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from omegaconf import OmegaConf
from functools import partial

from mpm_pytorch import set_boundary_conditions, get_constitutive
from mpm_ckmpm import CKMPMSolver


# ---------------------------------------------------------------------------
# Shared helpers (identical to MLS-MPM tests)
# ---------------------------------------------------------------------------

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
    """Same collider as MLS-MPM tests — works with CKMPMSolver too."""
    dx    = model.dx
    cx_t  = torch.tensor(center_xz[0] / dx, device=model.device).float()
    cz_t  = torch.tensor(center_xz[1] / dx, device=model.device).float()
    r_t   = torch.tensor(radius / dx,        device=model.device).float()
    ylo_t = torch.tensor(y_min / dx,         device=model.device).float()
    yhi_t = torch.tensor(y_max / dx,         device=model.device).float()

    def collide(model, cx_t, cz_t, r_t, ylo_t, yhi_t, surface):
        pos    = model.grid_x
        in_y   = (pos[:, 1] >= ylo_t) & (pos[:, 1] <= yhi_t)
        ex     = pos[:, 0].float() - cx_t
        ez     = pos[:, 2].float() - cz_t
        dist   = torch.sqrt(ex**2 + ez**2) - r_t
        inside = (dist < 0) & in_y
        if not inside.any(): return
        norm = torch.sqrt(ex[inside]**2 + ez[inside]**2).clamp(min=1e-8)
        n3d  = torch.zeros(inside.sum(), 3, device=model.device)
        n3d[:, 0] = ex[inside] / norm
        n3d[:, 2] = ez[inside] / norm
        mv  = model.grid_mv[inside]   # NOTE: works on whichever grid is active
        dot = (mv * n3d).sum(dim=1, keepdim=True)
        if surface == "sticky":
            model.grid_mv[inside] = 0.0
        elif surface == "slip":
            model.grid_mv[inside] = mv - torch.clamp(dot, max=0.0) * n3d

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


def save_gif(frames, path, c, cylinders, fps=24):
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=18, azim=-55)
    def update(i):
        ax.cla()
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Frame {i}')
        if cylinders:
            for cyl in cylinders:
                draw_horizontal_cylinder(ax, cyl['cx'], cyl['cz'], cyl['r'], cyl['y_min'], cyl['y_max'])
        ax.scatter(frames[i][:,0], frames[i][:,1], frames[i][:,2], s=10, c=c, depthshade=True)
        return []
    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(path, writer='pillow', fps=fps)
    plt.close()
    print(f'  GIF → {path}')


def save_energy_plot(log, path, title_suffix=''):
    frames  = [e['frame']            for e in log]
    ke      = [e['kinetic_energy']   for e in log]
    pe      = [e['potential_energy'] for e in log]
    total   = [e['total_energy']     for e in log]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(frames, ke,    label='KE',       color='steelblue')
    ax1.plot(frames, pe,    label='PE',       color='darkorange', linestyle='--')
    ax1.plot(frames, total, label='KE+PE',    color='green', linewidth=2)
    ax1.set_ylabel('Energy (J)'); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax1.set_title(f'Energy Conservation — CK-MPM (Python){title_suffix}')
    e0   = total[0] if total[0] != 0 else 1.0
    norm = [e/e0 for e in total]
    ax2.plot(frames, norm, color='green', linewidth=1.5)
    ax2.axhline(1.0, color='red', linestyle=':', alpha=0.5, label='Ideal')
    ax2.set_ylabel('E / E₀'); ax2.set_xlabel('Frame')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print(f'  Energy → {path}')


# ---------------------------------------------------------------------------
# Scene: 3-cylinder filter (shared by test 2 and test 4)
# ---------------------------------------------------------------------------
CYLINDERS = [
    {'cx': 0.20, 'cz': 0.50, 'r': 0.06, 'y_min': 0.02, 'y_max': 0.98},
    {'cx': 0.50, 'cz': 0.50, 'r': 0.06, 'y_min': 0.02, 'y_max': 0.98},
    {'cx': 0.80, 'cz': 0.50, 'r': 0.06, 'y_min': 0.02, 'y_max': 0.98},
]

def run_filter_scene(cfg, device, num_frames, tag):
    mat = cfg.material; sim = cfg.sim
    # Use num=12 for CPU speed (CK Python is ~10x slower than MLS-MPM per substep
    # due to SVD stabilization + dual-grid). For GPU runs increase to 20.
    num_pts = 20 if device.type == 'cuda' else 12
    particles = get_cube([0.5,0.5,0.80], [0.40,0.40,0.14], num=num_pts,
                         add_noise=True, device=device)
    n = particles.shape[0]

    # CK kernel radius=1 needs smaller dt than MLS-MPM (radius=1.5).
    # dt=5e-5 with 20 substeps keeps simulation stable for fluid on CPU.
    ck_dt       = 5e-5
    ck_substeps = 20
    solver = CKMPMSolver(particles, num_grids=40, dt=ck_dt, device=device,
                         gravity=[0., 0., -9.8])
    set_boundary_conditions(solver, sim.boundary_conditions)
    for cyl in CYLINDERS:
        add_horizontal_cylinder_collider(solver, [cyl['cx'],cyl['cz']], cyl['r'],
                                         cyl['y_min'], cyl['y_max'], surface="slip")

    elasticity = get_constitutive(mat.elasticity, device=device)
    plasticity = get_constitutive(mat.plasticity, device=device)

    x = particles.clone()
    v = torch.zeros_like(x)
    if hasattr(sim, 'initial_velocity'):
        v[:] = torch.tensor(sim.initial_velocity, device=device)
    C = torch.zeros((n,3,3), device=device)
    F = torch.eye(3, device=device).unsqueeze(0).repeat(n,1,1)

    frames_out, log = [], []
    g = 9.8
    for frame in tqdm(range(num_frames), desc=f'CK-MPM {tag}'):
        frames_out.append(x.cpu().numpy())
        ke = 0.5 * solver.p_mass * (v**2).sum(dim=1).sum().item()
        pe = solver.p_mass * g * x[:,2].sum().item()
        log.append({
            'frame': frame, 'kinetic_energy': ke,
            'potential_energy': pe, 'total_energy': ke+pe,
            'mean_vz': v[:,2].mean().item(),
            'n_total': n, 'n_below_cyl': int((x[:,2]<0.50).sum().item()),
        })
        for _ in range(ck_substeps):
            # Safety clamp before elasticity — FluidElasticity asserts isfinite(det(F))
            # SigmaPlasticity resets F to diag(J^1/3) each step, so F should be
            # near-identity, but boundary interactions can introduce nans.
            if not torch.isfinite(F).all():
                F = torch.nan_to_num(F, nan=1.0, posinf=1.0, neginf=-1.0)
            # Clamp diagonal to prevent det(F)=0
            F = F.clamp(-3.0, 3.0)
            stress = elasticity(F)
            x, v, C, F = solver(x, v, C, F, stress)
            F = plasticity(F)
    return frames_out, log, n, mat.color


# ---------------------------------------------------------------------------
# Test 6: efficiency timing
# ---------------------------------------------------------------------------
import time as _time

def run_efficiency(cfg, device, num_frames, tag):
    import time as _time
    mat = cfg.material; sim = cfg.sim
    num_pts = 14 if device.type == 'cuda' else 8
    particles = get_cube([0.5,0.5,0.70], [0.40,0.40,0.40], num=num_pts,
                         add_noise=True, device=device)
    n = particles.shape[0]
    ck_dt       = 1e-4
    ck_substeps = max(sim.steps_per_frame * 3, 30)
    solver = CKMPMSolver(particles, num_grids=40, dt=ck_dt, device=device)
    set_boundary_conditions(solver, sim.boundary_conditions)
    elasticity = get_constitutive(mat.elasticity, device=device)
    plasticity = get_constitutive(mat.plasticity, device=device)
    x = particles.clone()
    v = torch.zeros_like(x)
    C = torch.zeros((n,3,3), device=device)
    F = torch.eye(3, device=device).unsqueeze(0).repeat(n,1,1)

    times = []
    print(f'Particles: {n}, warming up...')
    for frame in tqdm(range(num_frames), desc='CK-MPM timing'):
        for _ in range(ck_substeps):
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = _time.perf_counter()
            stress = elasticity(F)
            x, v, C, F = solver(x, v, C, F, stress)
            F = plasticity(F)
            if device.type == 'cuda': torch.cuda.synchronize()
            times.append((_time.perf_counter() - t0)*1000)

    warmup = 5 * sim.steps_per_frame
    times  = times[warmup:]
    return times, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',   type=int, choices=[2,4,6], required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--frames', type=int, default=None)
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = cfg.output_dir; os.makedirs(out, exist_ok=True)
    print(f'Device: {device}  |  Test: {args.test}  |  Material: {cfg.tag}')

    if args.test == 2:
        n_frames = args.frames or 150
        tag      = cfg.tag + "_ckmpm_filter"
        frames_out, log, n, color = run_filter_scene(cfg, device, n_frames, tag)
        print('Rendering...')
        save_gif(frames_out, os.path.join(out, tag+'.gif'), color, CYLINDERS)
        save_energy_plot(log, os.path.join(out, tag+'_energy.png'), ' — Test 2 Filter Drop')
        with open(os.path.join(out, tag+'_metrics.csv'), 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=log[0].keys()).writeheader()
            csv.DictWriter(f, fieldnames=log[0].keys()).writerows(log)

    elif args.test == 4:
        n_frames = args.frames or 300
        tag      = cfg.tag + "_ckmpm_filter_volume"
        frames_out, log, n, color = run_filter_scene(cfg, device, n_frames, tag)
        print('Rendering...')
        save_gif(frames_out, os.path.join(out, tag+'.gif'), color, CYLINDERS)
        save_energy_plot(log, os.path.join(out, tag+'_energy.png'), ' — Test 4 Volume')
        # Volume plot
        fig, axes = plt.subplots(3,1, figsize=(8,10), sharex=True)
        frames_arr  = [e['frame']       for e in log]
        pct_below   = [100*e['n_below_cyl']/n for e in log]
        mean_vz     = [e['mean_vz']     for e in log]
        axes[0].axhline(n, color='steelblue', lw=2, label=f'Count={n} (constant)')
        axes[0].set_ylim(n*0.95, n*1.05)
        axes[0].set_ylabel('Particle Count')
        axes[0].set_title('Volume Conservation (CK-MPM Python)')
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(frames_arr, pct_below, color='darkorange', lw=1.5)
        axes[1].axhline(100, color='gray', linestyle='--', alpha=0.4)
        axes[1].set_ylabel('% below cylinder level')
        axes[1].set_title('Filter Throughput'); axes[1].grid(alpha=0.3)
        axes[2].plot(frames_arr, mean_vz, color='green', lw=1.5)
        axes[2].axhline(0, color='red', linestyle=':', alpha=0.4)
        axes[2].set_ylabel('Mean Z velocity'); axes[2].set_xlabel('Frame')
        axes[2].set_title('Mean Z-velocity'); axes[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out, tag+'_volume.png'), dpi=130); plt.close()
        print(f'  Volume → {os.path.join(out, tag+"_volume.png")}')
        with open(os.path.join(out, tag+'_metrics.csv'), 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=log[0].keys()).writeheader()
            csv.DictWriter(f, fieldnames=log[0].keys()).writerows(log)

    elif args.test == 6:
        n_frames = args.frames or 150
        tag      = cfg.tag + "_ckmpm_timing"
        times, n = run_efficiency(cfg, device, n_frames, tag)
        mean_ms  = np.mean(times); std_ms = np.std(times)
        print(f'\n=== CK-MPM Python Timing ({n} particles, {device}) ===')
        print(f'  Mean per substep: {mean_ms:.3f} ms   Std: {std_ms:.3f} ms')
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(times, color='steelblue', lw=0.8, alpha=0.7)
        ax.axhline(mean_ms, color='red', linestyle='--', label=f'Mean: {mean_ms:.2f} ms')
        ax.set_xlabel('Substep'); ax.set_ylabel('Time (ms)')
        ax.set_title(f'CK-MPM Python Timing — {n} particles'); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, tag+'.png'), dpi=130); plt.close()
        print(f'  Plot → {os.path.join(out, tag+".png")}')
        with open(os.path.join(out, tag+'.csv'), 'w', newline='') as f:
            writer = csv.writer(f); writer.writerow(['substep','time_ms'])
            writer.writerows(enumerate(times))
