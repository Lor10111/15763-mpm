"""
Tasty Meal — Water variant — MLS-MPM (PyTorch / MLS-APIC, quadratic B-spline)
=============================================================================

Drop an ice cube into a cup of water. Three heterogeneous materials in
contact: rigid-like elastic cup (FixedCorotated), weakly compressible fluid
water (Tait EOS), rigid-like elastic ice (FixedCorotated).

Same physical scene as CKMPM tests/tasty_meal_water; this implementation runs
on a coarser grid (dx = 1/64) for tractable PyTorch runtime.

Geometry (1 m³ domain, dx = 1/64):
  • Cup wall   : annulus  r ∈ [0.060, 0.070],  y ∈ [0.062, 0.170]
  • Cup floor  : disk     r ≤  0.070,          y ∈ [0.050, 0.062]
  • Water      : cylinder r ≤  0.058,          y ∈ [0.064, 0.100]
  • Ice cube   : 0.04 m cube, centre (0.5, 0.30, 0.5), v0 = (0, −5, 0) m/s

Output:
  - Per-frame .bgeo (if `partio` Python module installed) for Houdini import,
    one file per material group.
  - Else falls back to per-frame .npz (loadable in any tool).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mpm_pytorch import MPMSolver

# ──────────────────────────────────────────────────────────────────────────────
# Optional bgeo writer
# ──────────────────────────────────────────────────────────────────────────────
try:
    import partio  # pip install partio
    HAS_PARTIO = True
except ImportError:
    HAS_PARTIO = False
    print("[tasty_meal_water] partio not found; will dump .npz instead.")
    print("                   install with: pip install partio")


def write_group(out_dir, name, frame_id, pos, vel=None):
    """Dump one material group at one frame (bgeo if partio available, else npz)."""
    if HAS_PARTIO:
        p = partio.create()
        pos_attr = p.addAttribute("position", partio.VECTOR, 3)
        if vel is not None:
            vel_attr = p.addAttribute("velocity", partio.VECTOR, 3)
        for i in range(pos.shape[0]):
            idx = p.addParticle()
            p.set(pos_attr, idx, tuple(float(c) for c in pos[i]))
            if vel is not None:
                p.set(vel_attr, idx, tuple(float(c) for c in vel[i]))
        out_path = os.path.join(out_dir, f"{name}_{frame_id:04d}.bgeo")
        partio.write(out_path, p)
    else:
        out_path = os.path.join(out_dir, f"{name}_{frame_id:04d}.npz")
        if vel is None:
            np.savez_compressed(out_path, position=pos)
        else:
            np.savez_compressed(out_path, position=pos, velocity=vel)


# ──────────────────────────────────────────────────────────────────────────────
# Shared constants (must match CKMPM tasty_meal_water and basic-mpm version)
# ──────────────────────────────────────────────────────────────────────────────
GRID_SIZE   = 64
DX          = 1.0 / GRID_SIZE
PPC         = 8
FPS         = 48
TOTAL_TIME  = 1.5
GRAVITY_Y   = -9.8

CFL         = 0.5
PARTICLE_VOL = DX ** 3 / PPC

# Cup geometry (m)
CUP_CX, CUP_CZ        = 0.5, 0.5
CUP_R_INNER           = 0.060
CUP_R_OUTER           = 0.070
CUP_Y_BOTTOM          = 0.050
CUP_Y_TOP             = 0.170
CUP_FLOOR_THK         = 0.012
WATER_FILL_Y          = 0.100
ICE_CX, ICE_CY, ICE_CZ = 0.5, 0.30, 0.5
ICE_HALF              = 0.020          # 0.04 m cube
ICE_V0                = (0.0, -5.0, 0.0)

# Materials
CUP_E,   CUP_NU,   CUP_RHO   = 1e7, 0.4, 2000.0
ICE_E,   ICE_NU,   ICE_RHO   = 1e7, 0.4,  900.0
WATER_K, WATER_GAMMA, WATER_RHO = 5e5, 7.15, 1000.0   # Tait artificial bulk (Pa)

# Time-step from stiffest material (ice/cup share E)
C_S_ICE = ((ICE_E * (1.0 - ICE_NU)) / ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU) * ICE_RHO)) ** 0.5
DT      = CFL * DX / C_S_ICE


# ──────────────────────────────────────────────────────────────────────────────
# Particle generators
# ──────────────────────────────────────────────────────────────────────────────
def _sub_cell_offsets():
    out = []
    for w in range(8):
        di = (w & 4) >> 2
        dj = (w & 2) >> 1
        dk =  w & 1
        out.append((0.25 + di * 0.5, 0.25 + dj * 0.5, 0.25 + dk * 0.5))
    return out

SUB = _sub_cell_offsets()

def make_cup():
    pos = []
    iMin = int((CUP_CX - CUP_R_OUTER) / DX) - 1
    iMax = int((CUP_CX + CUP_R_OUTER) / DX) + 2
    jMin = int(CUP_Y_BOTTOM / DX) - 1
    jMax = int(CUP_Y_TOP    / DX) + 2
    kMin = int((CUP_CZ - CUP_R_OUTER) / DX) - 1
    kMax = int((CUP_CZ + CUP_R_OUTER) / DX) + 2
    for i in range(iMin, iMax):
      for j in range(jMin, jMax):
        for k in range(kMin, kMax):
          for (ox, oy, oz) in SUB:
              px = (i + ox) * DX
              py = (j + oy) * DX
              pz = (k + oz) * DX
              dxr = px - CUP_CX
              dzr = pz - CUP_CZ
              r   = (dxr * dxr + dzr * dzr) ** 0.5
              in_wall  = (CUP_R_INNER <= r <= CUP_R_OUTER) and \
                         (CUP_Y_BOTTOM + CUP_FLOOR_THK <= py <= CUP_Y_TOP)
              in_floor = (r <= CUP_R_OUTER) and \
                         (CUP_Y_BOTTOM <= py < CUP_Y_BOTTOM + CUP_FLOOR_THK)
              if in_wall or in_floor:
                  pos.append((px, py, pz))
    return np.asarray(pos, dtype=np.float32)

def make_water():
    pos = []
    waterR    = CUP_R_INNER - 0.002
    waterYBot = CUP_Y_BOTTOM + CUP_FLOOR_THK + 0.002
    iMin = int((CUP_CX - waterR) / DX) - 1
    iMax = int((CUP_CX + waterR) / DX) + 2
    jMin = int(waterYBot / DX) - 1
    jMax = int(WATER_FILL_Y / DX) + 2
    kMin = int((CUP_CZ - waterR) / DX) - 1
    kMax = int((CUP_CZ + waterR) / DX) + 2
    for i in range(iMin, iMax):
      for j in range(jMin, jMax):
        for k in range(kMin, kMax):
          for (ox, oy, oz) in SUB:
              px = (i + ox) * DX
              py = (j + oy) * DX
              pz = (k + oz) * DX
              dxr = px - CUP_CX
              dzr = pz - CUP_CZ
              r   = (dxr * dxr + dzr * dzr) ** 0.5
              if r <= waterR and waterYBot <= py <= WATER_FILL_Y:
                  pos.append((px, py, pz))
    return np.asarray(pos, dtype=np.float32)

def make_ice():
    pos = []
    iMin = int((ICE_CX - ICE_HALF) / DX) - 1
    iMax = int((ICE_CX + ICE_HALF) / DX) + 2
    jMin = int((ICE_CY - ICE_HALF) / DX) - 1
    jMax = int((ICE_CY + ICE_HALF) / DX) + 2
    kMin = int((ICE_CZ - ICE_HALF) / DX) - 1
    kMax = int((ICE_CZ + ICE_HALF) / DX) + 2
    for i in range(iMin, iMax):
      for j in range(jMin, jMax):
        for k in range(kMin, kMax):
          for (ox, oy, oz) in SUB:
              px = (i + ox) * DX
              py = (j + oy) * DX
              pz = (k + oz) * DX
              if abs(px - ICE_CX) <= ICE_HALF and \
                 abs(py - ICE_CY) <= ICE_HALF and \
                 abs(pz - ICE_CZ) <= ICE_HALF:
                  pos.append((px, py, pz))
    return np.asarray(pos, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Heterogeneous solver: per-particle mass/volume + per-particle stress
# ──────────────────────────────────────────────────────────────────────────────
class HeteroMPMSolver(MPMSolver):
    """MPMSolver that supports per-particle mass and volume."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mass_per_p = None  # (N,) tensor
        self.vol_per_p  = None  # (N,) tensor

    def p2g2p(self, x, v, C, F, stress):
        dt        = self.dt
        dx        = self.dx
        inv_dx    = self.inv_dx
        n_grids   = self.num_grids
        clip_bound = self.clip_bound

        vol_p3  = self.vol_per_p.view(-1, 1, 1)   # (N,1,1) — for momentum tensor mul
        mass_p3 = self.mass_per_p.view(-1, 1, 1)  # (N,1,1)
        mass_p1 = self.mass_per_p.view(-1, 1)     # (N,1)   — for grid_m

        px = x * inv_dx
        base = (px - 0.5).long()
        fx   = px - base.float()

        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ]
        w = torch.stack(w, dim=-1)
        w_e = torch.einsum('bi, bj, bk -> bijk', w[:, 0], w[:, 1], w[:, 2])
        weight = w_e.reshape(-1, 27)

        dw = [fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5]
        dw = torch.stack(dw, dim=-1)
        dweight = [
            torch.einsum('pi,pj,pk->pijk', dw[:, 0], w[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], dw[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], w[:, 1], dw[:, 2]),
        ]
        dweight = inv_dx * torch.stack(dweight, dim=-1).reshape(-1, 27, 3)

        dpos  = (self.offset - fx.unsqueeze(1)) * dx
        index = base.unsqueeze(1) + self.offset.unsqueeze(0).long()
        index = (index[:, :, 0] * n_grids * n_grids
                 + index[:, :, 1] * n_grids
                 + index[:, :, 2]).reshape(-1)
        index = index.clamp(0, n_grids ** 3 - 1)

        self.grid_mv = torch.zeros_like(self.grid_mv)
        self.grid_m  = torch.zeros_like(self.grid_m)

        for op in self.pre_particle_process:
            op(self, x, v)

        # P2G with per-particle mass / volume
        mv = -dt * vol_p3 * torch.einsum('bij, bkj -> bki', stress, dweight) \
             + mass_p3 * weight.unsqueeze(2) \
                       * (v.unsqueeze(1) + torch.einsum('bij, bkj -> bki', C, dpos))
        mv = mv.reshape(-1, 3)

        m = (weight * mass_p1).reshape(-1)

        self.grid_mv = self.grid_mv.index_add(dim=0, index=index, source=mv)
        self.grid_m  = self.grid_m.index_add(dim=0, index=index, source=m)

        self.grid_update()

        for op in self.post_grid_process:
            op(self)

        v_grid = self.grid_mv.index_select(dim=0, index=index).reshape(-1, 27, 3)
        C_new  = torch.einsum('bij, bik -> bijk', v_grid, dpos)
        new_F  = torch.einsum('bij, bik -> bijk', v_grid, dweight)

        v = (weight.unsqueeze(2) * v_grid).sum(dim=1)
        C = (4.0 * inv_dx * inv_dx * weight.unsqueeze(2).unsqueeze(3) * C_new).sum(dim=1)
        new_F = dt * new_F.sum(dim=1)

        x = x + v * dt
        x = x.clamp(clip_bound, 1.0 - clip_bound)
        F = F + torch.bmm(new_F, F)
        F = F.clamp(-2.0, 2.0)
        self.time += dt
        return x, v, C, F


# ──────────────────────────────────────────────────────────────────────────────
# Per-material PK1 stress kernels
# ──────────────────────────────────────────────────────────────────────────────
def lame(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return float(la), float(mu)


def corotated_pk1(F, lam, mu):
    """P = 2μ(F − R) + λ(J−1) J F^{−T}"""
    U, sig, Vh = torch.linalg.svd(F, full_matrices=False)
    R   = torch.matmul(U, Vh)
    J   = torch.linalg.det(F).view(-1, 1, 1).clamp(min=1e-6)
    Finv_T = torch.linalg.inv(F).transpose(1, 2)
    return 2.0 * mu * (F - R) + lam * (J - 1.0) * J * Finv_T


def fluid_pk1_tait(F, K, gamma):
    """Weakly compressible Tait fluid:
         σ = −K (J^{-γ} − 1) I        (Cauchy stress)
         P = J σ F^{-T} = K (1 − J^{-γ}) J F^{-T}
    """
    J     = torch.linalg.det(F).view(-1, 1, 1).clamp(min=0.05)
    Finv_T = torch.linalg.inv(F).transpose(1, 2)
    return K * (1.0 - J.pow(-gamma)) * J * Finv_T


# ──────────────────────────────────────────────────────────────────────────────
# Boundary: cup-floor anchor + reflective domain walls
# ──────────────────────────────────────────────────────────────────────────────
def make_cup_floor_anchor_op(grid_size, dx):
    """Returns a post_grid_process op that zeroes velocity at cup-floor cells."""
    coords = torch.arange(grid_size)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing='ij')
    px = gx.float() * dx
    py = gy.float() * dx
    pz = gz.float() * dx
    dxr = px - CUP_CX
    dzr = pz - CUP_CZ
    r2  = dxr * dxr + dzr * dzr
    anchor_R = CUP_R_OUTER + 0.005
    in_anchor = ((py >= CUP_Y_BOTTOM - 0.001) &
                 (py <  CUP_Y_BOTTOM + CUP_FLOOR_THK) &
                 (r2 <= anchor_R * anchor_R)).reshape(-1)

    def op(model):
        idx = in_anchor.to(model.device)
        model.grid_mv[idx] = 0.0
    return op


def make_domain_wall_op(grid_size):
    """Reflective walls (4-cell margin)."""
    coords = torch.arange(grid_size)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing='ij')
    near_lo = (gx < 4) | (gy < 4) | (gz < 4)
    near_hi = (gx >= grid_size - 4) | (gy >= grid_size - 4) | (gz >= grid_size - 4)
    on_wall = (near_lo | near_hi).reshape(-1)

    # one-sided reflect normal per cell
    nx = torch.where(gx < 4, 1.0, torch.where(gx >= grid_size - 4, -1.0, 0.0))
    ny = torch.where(gy < 4, 1.0, torch.where(gy >= grid_size - 4, -1.0, 0.0))
    nz = torch.where(gz < 4, 1.0, torch.where(gz >= grid_size - 4, -1.0, 0.0))
    n  = torch.stack([nx, ny, nz], dim=-1).reshape(-1, 3)
    n_norm = n.norm(dim=1, keepdim=True).clamp(min=1e-6)
    n  = n / n_norm

    def op(model):
        sel = on_wall.to(model.device)
        nrm = n.to(model.device)[sel]
        v   = model.grid_mv[sel]
        mag = (v * nrm).sum(dim=1, keepdim=True)
        # only zero the inward (negative) normal component
        mag = torch.clamp(mag, max=0.0)
        model.grid_mv[sel] = v - mag * nrm
    return op


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    out_dir = os.path.join(os.path.dirname(__file__), "output", "tasty_meal_water_mlsmpm")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tasty_meal_water/mlsmpm]  device={device}  dx={DX:.4e}  dt={DT:.4e}")
    print(f"[tasty_meal_water/mlsmpm]  total_time={TOTAL_TIME}s  fps={FPS}  steps={int(TOTAL_TIME/DT)}")

    # ── Build particle clouds ───────────────────────────────────────────────
    cup_pos   = make_cup()
    water_pos = make_water()
    ice_pos   = make_ice()

    n_cup, n_water, n_ice = len(cup_pos), len(water_pos), len(ice_pos)
    print(f"[tasty_meal_water/mlsmpm]  cup={n_cup}  water={n_water}  ice={n_ice}  total={n_cup+n_water+n_ice}")

    all_pos = np.concatenate([cup_pos, water_pos, ice_pos], axis=0)
    all_vel = np.zeros_like(all_pos)
    all_vel[n_cup + n_water:] = ICE_V0   # initial velocity for ice particles only

    # ── Per-particle mass & volume ──────────────────────────────────────────
    mass_arr = np.empty(len(all_pos), dtype=np.float32)
    vol_arr  = np.full (len(all_pos), PARTICLE_VOL, dtype=np.float32)
    mass_arr[             :n_cup           ] = PARTICLE_VOL * CUP_RHO
    mass_arr[n_cup        :n_cup + n_water ] = PARTICLE_VOL * WATER_RHO
    mass_arr[n_cup+n_water:                ] = PARTICLE_VOL * ICE_RHO

    # ── Solver setup ────────────────────────────────────────────────────────
    x_t = torch.tensor(all_pos, device=device)
    v_t = torch.tensor(all_vel, device=device)
    N   = x_t.shape[0]
    C_t = torch.zeros(N, 3, 3, device=device)
    F_t = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1).clone()

    solver = HeteroMPMSolver(
        init_pos = x_t,
        rho      = WATER_RHO,
        num_grids= GRID_SIZE,
        dt       = DT,
        gravity  = [0.0, GRAVITY_Y, 0.0],
        clip_bound = 0.0,
        damping  = 1.0,
        device   = device,
    )
    solver.mass_per_p = torch.tensor(mass_arr, device=device)
    solver.vol_per_p  = torch.tensor(vol_arr,  device=device)
    solver.pre_particle_process = []
    solver.post_grid_process    = [
        make_cup_floor_anchor_op(GRID_SIZE, DX),
        make_domain_wall_op(GRID_SIZE),
    ]

    # ── Per-material Lamé parameters ────────────────────────────────────────
    lam_cup, mu_cup = lame(CUP_E, CUP_NU)
    lam_ice, mu_ice = lame(ICE_E, ICE_NU)

    def compute_stress(F):
        s = torch.empty_like(F)
        s[:n_cup]                    = corotated_pk1(F[:n_cup],                 lam_cup, mu_cup)
        s[n_cup:n_cup+n_water]       = fluid_pk1_tait(F[n_cup:n_cup+n_water],   WATER_K, WATER_GAMMA)
        s[n_cup+n_water:]            = corotated_pk1(F[n_cup+n_water:],         lam_ice, mu_ice)
        return s

    # ── Run + dump per frame ────────────────────────────────────────────────
    target_dt   = 1.0 / FPS
    next_t      = 0.0
    sim_t       = 0.0
    frame_id    = 0

    def dump():
        pos_np = x_t.detach().cpu().numpy()
        vel_np = v_t.detach().cpu().numpy()
        write_group(out_dir, "cup",   frame_id, pos_np[:n_cup],            vel_np[:n_cup])
        write_group(out_dir, "water", frame_id, pos_np[n_cup:n_cup+n_water], vel_np[n_cup:n_cup+n_water])
        write_group(out_dir, "ice",   frame_id, pos_np[n_cup+n_water:],    vel_np[n_cup+n_water:])

    dump()
    next_t   += target_dt
    frame_id += 1

    n_steps_total = int(TOTAL_TIME / DT)
    pbar = tqdm(total=n_steps_total, desc="[mlsmpm] sim")
    while sim_t < TOTAL_TIME - 1e-12:
        stress = compute_stress(F_t)
        x_t, v_t, C_t, F_t = solver(x_t, v_t, C_t, F_t, stress)
        sim_t += DT
        pbar.update(1)
        if sim_t + 1e-12 >= next_t:
            dump()
            next_t   += target_dt
            frame_id += 1
    pbar.close()

    print(f"[tasty_meal_water/mlsmpm]  done. {frame_id} frames -> {out_dir}")


if __name__ == "__main__":
    main()
