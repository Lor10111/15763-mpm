"""
2D Ice Drop Benchmark — MLS-MPM (PyTorch, APIC, quadratic B-spline)
====================================================================

Drop an ice block into a pool of water / honey / snow in 2D and compare
solver behaviour across MPM kernels.

Run with:
    python test_ice_drop_2d.py water
    python test_ice_drop_2d.py honey
    python test_ice_drop_2d.py snow

Physics
-------
Transfer  : MLS-APIC  (affine velocity field stored in C per particle)
Kernel    : quadratic B-spline  (3×3 = 9 neighbours per particle)
Stress    : Kirchhoff stress  tau = P @ F^T  is passed to P2G, so the force
            contribution is exactly  -V0 * tau * grad_w.

Materials
---------
  water  : Tait weakly-compressible fluid  (isotropic F reset each step)
  honey  : same Tait EOS + per-step grid velocity damping (effective high viscosity)
  snow   : corotated elasticity + SVD-clamp plasticity + exponential hardening
           (Stomakhin et al. 2013)
  ice    : corotated elasticity (same params in all three sub-experiments)

Scene (identical geometry in all three):
  Domain     : [0, 1] × [0, 1]
  Container  : rigid walls at x = 0.25 / 0.75, floor at y = 0.10
  Medium pool: x ∈ [0.28, 0.72],  y ∈ [0.10, 0.32]
  Ice block  : 0.08 m square, centre (0.5, 0.62), v0 = (0, -1.2) m/s

Output  (per medium):
  output/ice_into_{medium}_2d_mlsmpm.gif
  output/ice_into_{medium}_2d_mlsmpm_com.png
  output/ice_into_{medium}_2d_mlsmpm_metrics.npz
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# ── Medium selector ───────────────────────────────────────────────────────────
VALID_MEDIA = ("water", "honey", "snow")
MEDIUM = sys.argv[1].lower() if len(sys.argv) > 1 else "water"
if MEDIUM not in VALID_MEDIA:
    raise ValueError(f"MEDIUM must be one of {VALID_MEDIA}, got '{MEDIUM}'")

print(f"[ice2d/{MEDIUM}/mlsmpm]  medium={MEDIUM}")

# ── Simulation constants ──────────────────────────────────────────────────────
GRID_SIZE  = 256
DX         = 1.0 / GRID_SIZE
INV_DX     = float(GRID_SIZE)
PPC        = 4            # 2×2 sub-cell
FPS        = 48
TOTAL_TIME = 2.5
GRAVITY_Y  = -9.8
CFL        = 0.5  # match CKMPM tasty_meal_water (kCfl_ = 0.5)

# ── Geometry ──────────────────────────────────────────────────────────────────
BOX_X0, BOX_X1    = 0.25, 0.75
BOX_Y0             = 0.10
MEDIUM_X0, MEDIUM_X1 = 0.28, 0.72
MEDIUM_Y0, MEDIUM_Y1 = 0.10, 0.32
ICE_CX, ICE_CY     = 0.50, 0.62
ICE_HALF            = 0.04
ICE_V0              = (0.0, -1.2)

# ── Material IDs ──────────────────────────────────────────────────────────────
MAT_MEDIUM = 0
MAT_ICE    = 1

# ── Ice (same across all three sub-experiments) — match CKMPM kIce* ──────────
# (the previous E=2e5 was ~50× too soft; ice would itself collapse on impact
#  and inject huge grad_v into the medium, breaking Tait stability.)
ICE_RHO = 900.0
ICE_E   = 1e7
ICE_NU  = 0.40
ICE_MU  = ICE_E / (2.0 * (1.0 + ICE_NU))
ICE_LAM = ICE_E * ICE_NU / ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU))

C_S_ICE     = ((ICE_E * (1.0 - ICE_NU)) /
               ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU) * ICE_RHO)) ** 0.5
DT          = CFL * DX / C_S_ICE
PARTICLE_VOL = DX ** 2 / PPC

# ── Medium-specific parameters — Tait values match CKMPM kWater* ────────────
if MEDIUM == "water":
    MEDIUM_RHO        = 1000.0
    WATER_K           = 5e4
    WATER_GAMMA       = 7.15
    WATER_VISCO       = 0.1
    HONEY_VIS_FACTOR  = 1.0    # no damping
elif MEDIUM == "honey":
    MEDIUM_RHO        = 1400.0
    WATER_K           = 5e4
    WATER_GAMMA       = 7.15
    WATER_VISCO       = 4.0
    # Per-step damping factor applied ONLY to honey-dominant grid cells
    # (gating implemented in apply_boundary). Decay τ ≈ dt/(1−factor).
    # 0.99 → τ ≈ 1.3 ms with the new dt — looks like cold honey.
    HONEY_VIS_FACTOR  = 0.99
else:  # snow
    MEDIUM_RHO    = 400.0
    SNOW_E        = 1.4e5
    SNOW_NU       = 0.2
    SNOW_MU0      = SNOW_E / (2.0 * (1.0 + SNOW_NU))
    SNOW_LAM0     = SNOW_E * SNOW_NU / ((1.0 + SNOW_NU) * (1.0 - 2.0 * SNOW_NU))
    SNOW_THETA_C  = 0.025
    SNOW_THETA_S  = 0.0075
    SNOW_XI       = 10.0

print(f"[ice2d/{MEDIUM}/mlsmpm]  dx={DX:.4e}  dt={DT:.4e}  "
      f"steps_total={int(TOTAL_TIME/DT)}")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output",
                       f"ice_into_{MEDIUM}_2d_mlsmpm")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Particle generation ───────────────────────────────────────────────────────
def sub_offsets():
    return [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]

SUB = sub_offsets()

def make_rect(x0, x1, y0, y1):
    pos = []
    i0 = int(x0 / DX) - 1;  i1 = int(x1 / DX) + 2
    j0 = int(y0 / DX) - 1;  j1 = int(y1 / DX) + 2
    for i in range(i0, i1):
        for j in range(j0, j1):
            for ox, oy in SUB:
                px = (i + ox) * DX
                py = (j + oy) * DX
                if x0 <= px <= x1 and y0 <= py <= y1:
                    pos.append((px, py))
    return np.asarray(pos, dtype=np.float32)


medium_pos = make_rect(MEDIUM_X0, MEDIUM_X1, MEDIUM_Y0, MEDIUM_Y1)
ice_pos    = make_rect(ICE_CX - ICE_HALF, ICE_CX + ICE_HALF,
                       ICE_CY - ICE_HALF, ICE_CY + ICE_HALF)

n_medium = len(medium_pos)
n_ice    = len(ice_pos)
N        = n_medium + n_ice
print(f"[ice2d/{MEDIUM}/mlsmpm]  medium={n_medium}  ice={n_ice}  total={N}")

all_pos = np.concatenate([medium_pos, ice_pos], axis=0)
all_vel = np.zeros((N, 2), dtype=np.float32)
all_vel[n_medium:, 0] = ICE_V0[0]
all_vel[n_medium:, 1] = ICE_V0[1]

mat_id_np           = np.empty(N, dtype=np.int32)
mat_id_np[:n_medium] = MAT_MEDIUM
mat_id_np[n_medium:] = MAT_ICE

mass_np             = np.empty(N, dtype=np.float32)
mass_np[:n_medium]   = PARTICLE_VOL * MEDIUM_RHO
mass_np[n_medium:]   = PARTICLE_VOL * ICE_RHO


# ── Device setup ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ice2d/{MEDIUM}/mlsmpm]  device={device}")

x_t      = torch.tensor(all_pos, device=device)       # (N,2)
v_t      = torch.tensor(all_vel, device=device)       # (N,2)
C_t      = torch.zeros(N, 2, 2, device=device)        # (N,2,2) APIC affine matrix
F_t      = torch.eye(2, device=device).unsqueeze(0).expand(N, -1, -1).clone()  # (N,2,2)
C_fluid_t = torch.zeros(N, 2, 2, device=device)       # (N,2,2) fluid velocity gradient cache
J_t      = torch.ones(N, device=device)               # (N,) fluid/snow/ice volume ratio cache
mat_id_t = torch.tensor(mat_id_np, device=device)     # (N,)
mass_t   = torch.tensor(mass_np,   device=device)     # (N,)
vol_t    = torch.full((N,), PARTICLE_VOL, device=device)  # (N,)

# Jp (plastic volume ratio) — only used for snow, but always allocated
Jp_t = torch.ones(N, device=device)                   # (N,)

# 3×3 = 9 neighbour offsets in 2D
offset_2d = torch.tensor([[i, j] for i in range(3) for j in range(3)],
                          device=device, dtype=torch.float32)  # (9,2)

grid_mv = torch.zeros(GRID_SIZE * GRID_SIZE, 2, device=device)
grid_m  = torch.zeros(GRID_SIZE * GRID_SIZE,    device=device)


# ── Constitutive models (batch, PyTorch) ──────────────────────────────────────
def lame(E, nu):
    mu  = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return float(lam), float(mu)


def corotated_kirchhoff(F, lam, mu):
    """Kirchhoff stress  tau = P @ F^T  for fixed-corotated model.
    tau = 2*mu*(F - R)*F^T + lam*(J-1)*J*I
    """
    U, sig, Vh = torch.linalg.svd(F, full_matrices=False)
    R  = U @ Vh
    J  = torch.linalg.det(F).view(-1, 1, 1).clamp(min=1e-6)
    return (2.0 * mu * (F - R) @ F.transpose(1, 2)
            + lam * (J - 1.0) * J * torch.eye(2, device=F.device).unsqueeze(0))


def fluid_kirchhoff_tait(F, K, gamma):
    raise NotImplementedError("Use fluid_kirchhoff_tait_jc for ice-drop fluid media.")


def fluid_kirchhoff_tait_jc(J, C, K, gamma, viscosity):
    """CKMPM-style fluid Kirchhoff stress using J and local velocity gradient.

    Matches mpm_material.cuh:466 (ComputeStress<kFluid>) and
    mpm_algorithm.cuh:541 (UpdateForce<kFluid>):
        pressure = K * (J^{-gamma} - 1)
        tau      = J * ((C + C^T) * viscosity  -  pressure * I)

    The clamp here is only a NaN guard — CKMPM does not clamp at all.
    The previous [0.2, 5.0] clamp was the source of the "water gets
    crushed to nothing" bug because at J=0.2, K·(0.2^-7.15 − 1) ≈ 4 GPa.
    """
    Jc = J.clamp(0.05, 20.0).view(-1, 1, 1)
    pressure = K * (Jc.pow(-gamma) - 1.0)
    viscous = viscosity * (C + C.transpose(1, 2))
    eye2 = torch.eye(2, device=J.device).unsqueeze(0)
    return Jc * (viscous - pressure * eye2)


def snow_kirchhoff(F_e, Jp_p, mu0, lam0, xi):
    """Kirchhoff stress for snow (hardened corotated)."""
    hardening = torch.exp(xi * (1.0 - Jp_p)).view(-1, 1, 1)
    mu  = mu0  * hardening
    lam = lam0 * hardening
    U, sig, Vh = torch.linalg.svd(F_e, full_matrices=False)
    R  = U @ Vh
    J  = torch.linalg.det(F_e).view(-1, 1, 1).clamp(min=1e-6)
    return (2.0 * mu * (F_e - R) @ F_e.transpose(1, 2)
            + lam * (J - 1.0) * J * torch.eye(2, device=F_e.device).unsqueeze(0))


def compute_kirchhoff_stress(F, C_fluid, J, Jp):
    """Compute Kirchhoff stress tau for all particles."""
    tau = torch.empty_like(F)
    ice_mask    = (mat_id_t == MAT_ICE)
    medium_mask = ~ice_mask
    if medium_mask.any():
        if MEDIUM in ("water", "honey"):
            tau[medium_mask] = fluid_kirchhoff_tait_jc(
                J[medium_mask], C_fluid[medium_mask], WATER_K, WATER_GAMMA, WATER_VISCO)
        else:  # snow
            tau[medium_mask] = snow_kirchhoff(
                F[medium_mask], Jp[medium_mask],
                SNOW_MU0, SNOW_LAM0, SNOW_XI)
    if ice_mask.any():
        tau[ice_mask] = corotated_kirchhoff(F[ice_mask], ICE_LAM, ICE_MU)
    return tau


# ── Snow plasticity update (return mapping) ───────────────────────────────────
def snow_plasticity_update(F_new, Jp):
    """Apply SVD clamp to snow particles and update Jp."""
    snow_mask = (mat_id_t == MAT_MEDIUM)  # MAT_MEDIUM == snow in snow experiment
    if not snow_mask.any():
        return F_new, Jp
    F_snow = F_new[snow_mask]             # (M,2,2)
    Jp_snow = Jp[snow_mask]               # (M,)
    U, sig, Vh = torch.linalg.svd(F_snow, full_matrices=False)  # sig: (M,2)
    sig_old = sig.clone()
    sig = sig.clamp(min=1.0 - SNOW_THETA_C, max=1.0 + SNOW_THETA_S)
    # Update Jp: account for plastic deformation on each axis
    Jp_snow = Jp_snow * (sig_old.prod(dim=1) / sig.prod(dim=1))
    Jp_snow = Jp_snow.clamp(0.6, 20.0)
    # Reconstruct elastic F from clamped singular values
    F_new_snow = U @ torch.diag_embed(sig) @ Vh
    F_new = F_new.clone()
    F_new[snow_mask]  = F_new_snow
    Jp = Jp.clone()
    Jp[snow_mask]     = Jp_snow
    return F_new, Jp


# ── Fluid F reset (isotropic, preserves det) ─────────────────────────────────
def fluid_F_reset(F_new, J):
    """For fluid particles, reset F to sqrtJ * I so det(F) = J."""
    fluid_mask = (mat_id_t == MAT_MEDIUM)
    if not fluid_mask.any():
        return F_new
    # Wide [0.05, 20] guard, matches the Tait stress clamp.
    sqrtJ = J[fluid_mask].clamp(0.05, 20.0).sqrt().view(-1, 1, 1)
    eye2  = torch.eye(2, device=device).unsqueeze(0)
    F_new = F_new.clone()
    F_new[fluid_mask] = sqrtJ * eye2
    return F_new


# ── Grid boundary conditions ──────────────────────────────────────────────────
def apply_boundary(grid_mv_flat, grid_m_medium_flat=None, grid_m_flat=None):
    """Apply container walls, domain guard, and (for honey) per-cell damping.

    The BC band must cover the 3-cell quadratic stencil that a particle
    pinned at advect()'s 1.5*DX position clamp reads — otherwise particles
    end up pinned in position while their interpolated grid velocity keeps
    accumulating gravity (a fake "free-fall after rest" signature).
    """
    gv = grid_mv_flat.reshape(GRID_SIZE, GRID_SIZE, 2)

    # container walls — band widened from 1 cell to 3 cells (covers
    # advect()'s 1.5*DX clamp + the quadratic stencil reach above it)
    xl = int(BOX_X0 / DX) + 2
    xr = int(BOX_X1 / DX) - 2
    yb = int(BOX_Y0 / DX) + 2
    gv[:xl, :, 0] = gv[:xl, :, 0].clamp(min=0.0)
    gv[xr:, :, 0] = gv[xr:, :, 0].clamp(max=0.0)
    gv[:, :yb, 1] = gv[:, :yb, 1].clamp(min=0.0)

    # outer domain guard (zero velocity)
    gv[:3, :, :] = 0.0
    gv[-3:, :, :] = 0.0
    gv[:, :3, :] = 0.0
    gv[:, -3:, :] = 0.0

    # honey: damp ONLY honey-dominant cells (was: damped all cells, which
    # killed the ice's freefall through air before it ever hit the honey
    # surface and made it look like the simulation froze).
    if MEDIUM == "honey" and grid_m_medium_flat is not None:
        gm        = grid_m_flat.reshape(GRID_SIZE, GRID_SIZE)
        gm_honey  = grid_m_medium_flat.reshape(GRID_SIZE, GRID_SIZE)
        honey_cell = (gm_honey > 0.5 * gm) & (gm > 0.0)
        gv[honey_cell] *= HONEY_VIS_FACTOR

    return gv.reshape(GRID_SIZE * GRID_SIZE, 2)


# ── One simulation step (P2G → grid update → G2P) ────────────────────────────
def step(x, v, C, F, C_fluid, J, Jp):
    global grid_mv, grid_m

    # ── Compute Kirchhoff stress ──────────────────────────────────────────────
    tau = compute_kirchhoff_stress(F, C_fluid, J, Jp)   # (N,2,2)

    # ── P2G ──────────────────────────────────────────────────────────────────
    px   = x * INV_DX                        # (N,2)
    base = (px - 0.5).long()                 # (N,2)
    fx   = px - base.float()                 # (N,2)

    # Quadratic B-spline weights and gradients
    w  = [0.5 * (1.5 - fx) ** 2,
           0.75 - (fx - 1.0) ** 2,
           0.5 * (fx - 0.5) ** 2]            # each (N,2)
    dw = [fx - 1.5,
           -2.0 * (fx - 1.0),
           fx - 0.5]                         # each (N,2)
    w  = torch.stack(w,  dim=-1)            # (N,2,3)
    dw = torch.stack(dw, dim=-1)            # (N,2,3)

    # 2D outer product of 1D weights → (N,3,3) → reshape to (N,9)
    w_2d     = torch.einsum('bi,bj->bij', w[:, 0], w[:, 1]).reshape(N, 9)
    # Gradient weight components: each (N,3,3) → (N,9)
    dw_x     = (torch.einsum('bi,bj->bij', dw[:, 0], w[:, 1]) * INV_DX).reshape(N, 9)
    dw_y     = (torch.einsum('bi,bj->bij', w[:, 0], dw[:, 1]) * INV_DX).reshape(N, 9)
    grad_w   = torch.stack([dw_x, dw_y], dim=-1)   # (N,9,2)

    # Neighbour positions and dpos
    dpos = (offset_2d.unsqueeze(0) - fx.unsqueeze(1)) * DX   # (N,9,2)

    # Neighbour grid indices
    idx_2d = (base.unsqueeze(1).long()
               + offset_2d.unsqueeze(0).long())               # (N,9,2)
    idx    = (idx_2d[:, :, 0] * GRID_SIZE + idx_2d[:, :, 1]).reshape(-1)
    idx    = idx.clamp(0, GRID_SIZE * GRID_SIZE - 1)

    # MLS-APIC: affine velocity  v_p + C_p @ dpos
    # C: (N,2,2), dpos: (N,9,2)
    # want result[b,j,i] = sum_k C[b,i,k] * dpos[b,j,k]  →  (N,9,2)
    v_affine = v.unsqueeze(1) + torch.einsum('bik,bjk->bji', C, dpos)  # (N,9,2)

    # Stress contribution: -dt * V * tau * grad_w  (Kirchhoff stress → correct)
    stress_contrib = -DT * vol_t.view(-1, 1, 1) * torch.einsum(
        'bij,bkj->bki', tau, grad_w)                          # (N,9,2)

    mv_contrib = (mass_t.view(-1, 1, 1) * w_2d.unsqueeze(2)
                  * v_affine + stress_contrib)                 # (N,9,2)
    m_contrib  = (mass_t.unsqueeze(1) * w_2d).reshape(-1)     # (N*9,)

    grid_mv = torch.zeros(GRID_SIZE * GRID_SIZE, 2, device=device)
    grid_m  = torch.zeros(GRID_SIZE * GRID_SIZE,    device=device)
    grid_mv.index_add_(0, idx, mv_contrib.reshape(-1, 2))
    grid_m.index_add_(0, idx, m_contrib)

    # Per-material grid mass (only needed for honey-cell detection in BC)
    grid_m_medium = None
    if MEDIUM == "honey":
        is_medium = (mat_id_t == MAT_MEDIUM).float().unsqueeze(1)   # (N,1)
        m_medium_contrib = (mass_t.unsqueeze(1) * w_2d * is_medium).reshape(-1)
        grid_m_medium = torch.zeros(GRID_SIZE * GRID_SIZE, device=device)
        grid_m_medium.index_add_(0, idx, m_medium_contrib)

    # ── Grid update ───────────────────────────────────────────────────────────
    sel = grid_m > 1e-15
    grid_mv[sel] = grid_mv[sel] / grid_m[sel].unsqueeze(1)
    # gravity
    grid_mv[:, 1] += DT * GRAVITY_Y
    # boundary conditions (walls + per-cell honey damping)
    grid_mv = apply_boundary(grid_mv, grid_m_medium, grid_m)

    # ── G2P ──────────────────────────────────────────────────────────────────
    v_grid = grid_mv[idx].reshape(N, 9, 2)                    # (N,9,2)

    v_new = (w_2d.unsqueeze(2) * v_grid).sum(dim=1)           # (N,2)
    # APIC affine matrix C = (4/dx^2) * sum_i w_i * v_i * dpos_i^T
    C_new = (4.0 * INV_DX ** 2
             * (w_2d.unsqueeze(2).unsqueeze(3)
                * torch.einsum('bni,bnj->bnij', v_grid, dpos)).sum(dim=1))  # (N,2,2)

    # F update: F_new = (I + dt * sum_i v_i * grad_w_i^T) @ F_old
    #   = (I + dt * grad_v) @ F_old
    grad_v = (torch.einsum('bni,bnj->bnij', v_grid, grad_w)).sum(dim=1)  # (N,2,2)
    F_new  = (torch.eye(2, device=device).unsqueeze(0)
               + DT * grad_v) @ F                              # (N,2,2)

    # Clamp F (prevent numerical explosion)
    F_new = F_new.clamp(-2.0, 2.0)

    # ── Material-specific F update ────────────────────────────────────────────
    if MEDIUM in ("water", "honey"):
        J_new = J.clone()
        fluid_mask = (mat_id_t == MAT_MEDIUM)
        # CKMPM-style J += trace(C)*dt*J with only a wide NaN guard.
        J_new[fluid_mask] = (J[fluid_mask]
                             + DT * torch.diagonal(grad_v[fluid_mask], dim1=1, dim2=2).sum(dim=1)
                             * J[fluid_mask]).clamp(0.05, 20.0)
        C_fluid_new = C_fluid.clone()
        C_fluid_new[fluid_mask] = grad_v[fluid_mask]
        F_new = fluid_F_reset(F_new, J_new)
    elif MEDIUM == "snow":
        F_new, Jp = snow_plasticity_update(F_new, Jp)
        J_new = J
        C_fluid_new = C_fluid
    else:
        J_new = J
        C_fluid_new = C_fluid

    # ── Advect ────────────────────────────────────────────────────────────────
    x_new = x + DT * v_new
    # Clamp inside container
    x_new[:, 0] = x_new[:, 0].clamp(BOX_X0 + 1.5 * DX, BOX_X1 - 1.5 * DX)
    x_new[:, 1] = x_new[:, 1].clamp(BOX_Y0 + 1.5 * DX, 1.0 - 2.0 * DX)

    return x_new, v_new, C_new, F_new, C_fluid_new, J_new, Jp


# ── Run ───────────────────────────────────────────────────────────────────────
def run_simulation():
    x, v, C, F, C_fluid, J, Jp = x_t, v_t, C_t, F_t, C_fluid_t, J_t, Jp_t

    frames_medium = []
    frames_ice    = []
    ice_com_y     = []
    ice_vy_list   = []
    crater_depth  = []
    initial_top_y = MEDIUM_Y1

    target_frame_dt = 1.0 / FPS
    next_frame_time = 0.0
    sim_time        = 0.0

    def record():
        pos = x.detach().cpu().numpy()
        vel = v.detach().cpu().numpy()
        frames_medium.append(pos[:n_medium].copy())
        frames_ice.append(pos[n_medium:].copy())
        ice_com_y.append(float(pos[n_medium:, 1].mean()))
        ice_vy_list.append(float(vel[n_medium:, 1].mean()))
        # crater depth for snow
        if MEDIUM == "snow":
            col_mask = ((pos[:n_medium, 0] >= ICE_CX - ICE_HALF - 0.02) &
                        (pos[:n_medium, 0] <= ICE_CX + ICE_HALF + 0.02))
            if col_mask.any():
                d = float(max(0.0, initial_top_y - pos[:n_medium][col_mask, 1].min()))
            else:
                d = 0.0
            crater_depth.append(d)

    record()
    next_frame_time += target_frame_dt

    n_steps_total = int(TOTAL_TIME / DT)
    pbar = tqdm(total=n_steps_total, desc=f"[mlsmpm/{MEDIUM}]")
    while sim_time < TOTAL_TIME - 1e-12:
        x, v, C, F, C_fluid, J, Jp = step(x, v, C, F, C_fluid, J, Jp)
        sim_time += DT
        pbar.update(1)
        if sim_time + 1e-12 >= next_frame_time:
            record()
            next_frame_time += target_frame_dt
    pbar.close()

    extra = {"crater_depth": np.array(crater_depth)} if MEDIUM == "snow" else {}
    return (frames_medium, frames_ice,
            np.array(ice_com_y), np.array(ice_vy_list), extra)


# ── Visualise ─────────────────────────────────────────────────────────────────
MEDIUM_COLOURS = {"water": "#2e86de", "honey": "#f6c90e", "snow": "#b2bec3"}

def render_gif(frames_medium, frames_ice):
    out_path = os.path.join(OUT_DIR, f"ice_into_{MEDIUM}_2d_mlsmpm.gif")
    fig, ax  = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.02, 0.90)
    ax.set_aspect("equal")
    ax.set_title(f"2D Ice Into {MEDIUM.capitalize()} — MLS-MPM")
    ax.plot([BOX_X0, BOX_X0], [BOX_Y0, 0.85], "k-", lw=2)
    ax.plot([BOX_X1, BOX_X1], [BOX_Y0, 0.85], "k-", lw=2)
    ax.plot([BOX_X0, BOX_X1], [BOX_Y0, BOX_Y0], "k-", lw=2)
    scat_m = ax.scatter([], [], s=3, c=MEDIUM_COLOURS[MEDIUM],
                        alpha=0.8, label=MEDIUM.capitalize())
    scat_i = ax.scatter([], [], s=6, c="#dfe6e9",
                        edgecolors="#34495e", linewidths=0.2, label="Ice")
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def update(fid):
        scat_m.set_offsets(frames_medium[fid])
        scat_i.set_offsets(frames_ice[fid])
        txt.set_text(f"t = {fid / FPS:.2f} s")
        return scat_m, scat_i, txt

    ani = animation.FuncAnimation(fig, update, frames=len(frames_medium),
                                  interval=1000 // FPS, blit=False)
    ani.save(out_path, writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"[mlsmpm/{MEDIUM}]  GIF → {out_path}")


def plot_metrics(ice_com_y, ice_vy, extra):
    times     = np.arange(len(ice_com_y)) / FPS
    n_panels  = 3 if MEDIUM == "snow" else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 3 * n_panels), sharex=True)

    axes[0].plot(times, ice_com_y, color="#c0392b", lw=1.8)
    axes[0].set_ylabel("Ice COM y (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"2D Ice Into {MEDIUM.capitalize()} — MLS-MPM Metrics")

    axes[1].plot(times, ice_vy, color="#2980b9", lw=1.8)
    axes[1].set_ylabel("Ice vy (m/s)")
    axes[1].grid(True, alpha=0.3)

    if MEDIUM == "snow":
        axes[2].plot(times, extra["crater_depth"], color="#8e44ad", lw=1.8)
        axes[2].set_ylabel("Crater depth (m)")
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    png_path = os.path.join(OUT_DIR, f"ice_into_{MEDIUM}_2d_mlsmpm_com.png")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    npz_path = os.path.join(OUT_DIR, f"ice_into_{MEDIUM}_2d_mlsmpm_metrics.npz")
    np.savez(npz_path, times=times, ice_com_y=ice_com_y, ice_vy=ice_vy, **extra)
    print(f"[mlsmpm/{MEDIUM}]  metrics → {npz_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    frames_medium, frames_ice, ice_com_y, ice_vy, extra = run_simulation()
    print(f"[mlsmpm/{MEDIUM}]  frames={len(frames_medium)}  "
          f"final_com_y={ice_com_y[-1]:.4f}  "
          f"final_ice_vy={ice_vy[-1]:.4f}")
    render_gif(frames_medium, frames_ice)
    plot_metrics(ice_com_y, ice_vy, extra)
