"""
Test 6: Efficiency profiling — shared scene across MLS-MPM, CK-MPM, baseline
This script runs the MLS-MPM (MPM-PyTorch) version of the shared benchmark scene
and records per-component timing at each frame.

Run: python test6_efficiency_profiling.py --config examples/sand.yaml

Outputs:
  - timing_mlsmpm.csv   : per-frame breakdown of P2G, grid update, G2P, plasticity
  - timing_summary.png  : bar chart of average times per component
  - timing_detail.png   : per-frame line plot of total time

Use the same scene parameters (particle count, grid res, dt) when running
the CK-MPM and baseline versions so comparisons are fair.
"""
from typing import *
import argparse, os, csv, time
from omegaconf import OmegaConf
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def timed(fn, *args, device=None, **kwargs):
    """Run fn(*args, **kwargs) and return (result, elapsed_ms)."""
    if device and device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if device and device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000  # ms
    return result, elapsed


class TimedMPMSolver(MPMSolver):
    """
    Subclass that times each sub-step of the simulation:
      p2g, grid_update, post_grid (boundary conditions), g2p
    We override p2g2p to insert timers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_log = []

    def p2g2p(self, x, v, C, F, stress):
        dt       = self.dt
        vol      = self.vol
        p_mass   = self.p_mass
        dx       = self.dx
        inv_dx   = self.inv_dx
        n_grids  = self.num_grids
        clip_bound = self.clip_bound
        device   = self.device

        px   = x * inv_dx
        base = (px - 0.5).long()
        fx   = px - base.float()

        w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2]
        w = torch.stack(w, dim=-1)
        w_e    = torch.einsum('bi,bj,bk->bijk', w[:,0], w[:,1], w[:,2])
        weight = w_e.reshape(-1, 27)

        dw = [fx-1.5, -2.0*(fx-1.0), fx-0.5]
        dw = torch.stack(dw, dim=-1)
        dweight = [
            torch.einsum('pi,pj,pk->pijk', dw[:,0], w[:,1], w[:,2]),
            torch.einsum('pi,pj,pk->pijk', w[:,0], dw[:,1], w[:,2]),
            torch.einsum('pi,pj,pk->pijk', w[:,0], w[:,1], dw[:,2])
        ]
        dweight = inv_dx * torch.stack(dweight, dim=-1).reshape(-1, 27, 3)
        dpos    = (self.offset - fx.unsqueeze(1)) * dx
        index   = base.unsqueeze(1) + self.offset.unsqueeze(0).long()
        index   = (index[:,:,0]*n_grids*n_grids + index[:,:,1]*n_grids + index[:,:,2]).reshape(-1)
        index   = index.clamp(0, n_grids**3 - 1)

        self.grid_mv = torch.zeros_like(self.grid_mv)
        self.grid_m  = torch.zeros_like(self.grid_m)

        # ---- P2G ----
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        mv = (-dt * vol * torch.einsum('bij,bkj->bki', stress, dweight) +
              p_mass * weight.unsqueeze(2) *
              (v.unsqueeze(1) + torch.einsum('bij,bkj->bki', C, dpos))).reshape(-1, 3)
        m  = (weight * p_mass).reshape(-1)
        self.grid_mv = self.grid_mv.index_add(0, index, mv)
        self.grid_m  = self.grid_m.index_add(0, index, m)
        if device.type == 'cuda': torch.cuda.synchronize()
        t_p2g = (time.perf_counter() - t0) * 1000

        # ---- Grid update ----
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.grid_update()
        if device.type == 'cuda': torch.cuda.synchronize()
        t_grid = (time.perf_counter() - t0) * 1000

        # ---- Boundary conditions (post_grid_process) ----
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for op in self.post_grid_process:
            op(self)
        if device.type == 'cuda': torch.cuda.synchronize()
        t_bc = (time.perf_counter() - t0) * 1000

        # ---- G2P ----
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        v_g  = self.grid_mv.index_select(0, index).reshape(-1, 27, 3)
        C_g  = torch.einsum('bij,bik->bijk', v_g, dpos)
        nF_g = torch.einsum('bij,bik->bijk', v_g, dweight)
        v    = (weight.unsqueeze(2) * v_g).sum(dim=1)
        C    = (4.0*inv_dx*inv_dx * weight.unsqueeze(2).unsqueeze(3) * C_g).sum(dim=1)
        new_F = dt * nF_g.sum(dim=1)
        x    = x + v * dt
        x    = x.clamp(clip_bound, 1.0 - clip_bound)
        F    = F + torch.bmm(new_F, F)
        F    = F.clamp(-2.0, 2.0)
        self.time += dt
        if device.type == 'cuda': torch.cuda.synchronize()
        t_g2p = (time.perf_counter() - t0) * 1000

        self.timing_log.append({
            'p2g_ms': t_p2g, 'grid_update_ms': t_grid,
            'bc_ms': t_bc,   'g2p_ms': t_g2p,
            'total_ms': t_p2g + t_grid + t_bc + t_g2p
        })
        return x, v, C, F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--warmup', type=int, default=5,
                        help='Frames to skip before recording (GPU JIT warmup)')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    mat = cfg.material
    sim = cfg.sim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)
    tag = cfg.tag + "_timing"
    print(f'Device: {device}')

    # ----------------------------------------------------------------
    # SHARED BENCHMARK SCENE — use these EXACT parameters when running
    # CK-MPM and baseline so the comparison is fair.
    #
    # Particle count: ~3000  (14^3 grid * 2 with noise)
    # Grid: 25^3 (default MPMSolver)
    # dt: 3e-4 (default)
    # Material: whatever --config specifies
    # ----------------------------------------------------------------
    particles = get_cube(
        center=[0.5, 0.5, 0.7],
        size=[0.4, 0.4, 0.4],
        num=14, add_noise=True, device=device)
    n_particles = particles.shape[0]
    print(f'Particles: {n_particles}')

    solver = TimedMPMSolver(particles, enable_train=False, device=device)
    set_boundary_conditions(solver, sim.boundary_conditions)

    elasticity = get_constitutive(mat.elasticity, device=device)
    plasticity = get_constitutive(mat.plasticity, device=device)

    x = particles
    v = torch.stack([torch.tensor(sim.initial_velocity, device=device)
                     for _ in range(n_particles)])
    C = torch.zeros((n_particles, 3, 3), device=device)
    F = torch.eye(3, device=device).unsqueeze(0).repeat(n_particles, 1, 1)

    print(f'Running {sim.num_frames} frames ({args.warmup} warmup)...')
    for frame in tqdm(range(sim.num_frames), desc='Profiling'):
        for _ in range(sim.steps_per_frame):
            stress = elasticity(F)
            x, v, C, F = solver(x, v, C, F, stress)
            F = plasticity(F)

    # Discard warmup steps
    warmup_steps = args.warmup * sim.steps_per_frame
    log = solver.timing_log[warmup_steps:]

    # ---- Save CSV ----
    csv_path = os.path.join(cfg.output_dir, tag + ".csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader(); writer.writerows(log)
    print(f'  CSV → {csv_path}')

    # ---- Summary statistics ----
    keys = ['p2g_ms', 'grid_update_ms', 'bc_ms', 'g2p_ms', 'total_ms']
    means = {k: np.mean([r[k] for r in log]) for k in keys}
    stds  = {k: np.std( [r[k] for r in log]) for k in keys}

    print('\n=== Timing Summary (per substep, ms) ===')
    print(f'  Method: MLS-MPM (PyTorch)  |  Particles: {n_particles}  |  Device: {device}')
    print(f'  {"Component":<20} {"Mean (ms)":>10} {"Std (ms)":>10}')
    print('  ' + '-'*42)
    for k in keys:
        print(f'  {k:<20} {means[k]:>10.3f} {stds[k]:>10.3f}')

    # ---- Bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    components = ['p2g_ms', 'grid_update_ms', 'bc_ms', 'g2p_ms']
    labels     = ['P2G', 'Grid Update', 'Boundary\nConditions', 'G2P']
    mean_vals  = [means[k] for k in components]
    std_vals   = [stds[k]  for k in components]
    colors     = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    axes[0].bar(labels, mean_vals, yerr=std_vals, color=colors,
                capsize=5, edgecolor='black', linewidth=0.8)
    axes[0].set_ylabel('Time (ms per substep)')
    axes[0].set_title(f'MLS-MPM Component Timing\n({n_particles} particles, {device})')
    axes[0].set_ylim(0, max(mean_vals) * 1.4)
    for i, (m, s) in enumerate(zip(mean_vals, std_vals)):
        axes[0].text(i, m + s + max(mean_vals)*0.02, f'{m:.2f}', ha='center', fontsize=9)

    # Per-frame total time line plot
    total_per_step = [r['total_ms'] for r in log]
    axes[1].plot(total_per_step, color='steelblue', linewidth=0.8, alpha=0.7)
    axes[1].axhline(means['total_ms'], color='red', linestyle='--',
                    label=f'Mean: {means["total_ms"]:.2f} ms')
    axes[1].set_xlabel('Substep'); axes[1].set_ylabel('Total time (ms)')
    axes[1].set_title('Total Time per Substep')
    axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(cfg.output_dir, tag + "_summary.png")
    plt.savefig(fig_path, dpi=130); plt.close()
    print(f'  Summary plot → {fig_path}')
    print('\nDone. Share timing_mlsmpm.csv with Lorene for comparison against CK-MPM.')
