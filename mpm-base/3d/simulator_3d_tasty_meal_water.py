"""
Tasty Meal — Water variant — Basic MPM 3D (Taichi, quadratic B-spline, no APIC)
================================================================================

Drop an ice cube into a cup of water. Three heterogeneous materials in
contact: rigid-like elastic cup (FixedCorotated), weakly compressible fluid
water (Tait EOS), rigid-like elastic ice (FixedCorotated).

Same physical scene as CKMPM tests/tasty_meal_water; this implementation runs
on a coarser grid (dx = 1/64) for tractable Taichi runtime on CPU.

Geometry (1 m³ domain, dx = 1/64):
  • Cup wall   : annulus  r ∈ [0.060, 0.070],  y ∈ [0.062, 0.170]
  • Cup floor  : disk     r ≤  0.070,          y ∈ [0.050, 0.062]
  • Water      : cylinder r ≤  0.058,          y ∈ [0.064, 0.100]
  • Ice cube   : 0.04 m cube, centre (0.5, 0.30, 0.5), v0 = (0, −5, 0) m/s

Output:
  - Per-frame .bgeo (if `partio` Python module installed) for Houdini import.
  - Else falls back to per-frame .npz.
"""

import os
import math
import numpy as np
import taichi as ti
from tqdm import tqdm

ti.init(arch=ti.cpu)   # change to ti.gpu / ti.metal for faster runtime

# ──────────────────────────────────────────────────────────────────────────────
# Optional bgeo writer
# ──────────────────────────────────────────────────────────────────────────────
try:
    import partio
    HAS_PARTIO = True
except ImportError:
    HAS_PARTIO = False
    print("[tasty_meal_water] partio not found; will dump .npz instead.")
    print("                   install with: pip install partio")


def write_group(out_dir, name, frame_id, pos, vel=None):
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
        partio.write(os.path.join(out_dir, f"{name}_{frame_id:04d}.bgeo"), p)
    else:
        out = os.path.join(out_dir, f"{name}_{frame_id:04d}.npz")
        if vel is None:
            np.savez_compressed(out, position=pos)
        else:
            np.savez_compressed(out, position=pos, velocity=vel)


# ──────────────────────────────────────────────────────────────────────────────
# Shared constants (must match CKMPM tasty_meal_water and MLS-MPM version)
# ──────────────────────────────────────────────────────────────────────────────
GRID_SIZE   = 64
DX          = 1.0 / GRID_SIZE
PPC         = 8
FPS         = 48
TOTAL_TIME  = 1.5
GRAVITY_Y   = -9.8

CFL          = 0.5
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
ICE_HALF              = 0.020
ICE_V0                = (0.0, -5.0, 0.0)

# Materials
CUP_E,   CUP_NU,   CUP_RHO   = 1e7, 0.4, 2000.0
ICE_E,   ICE_NU,   ICE_RHO   = 1e7, 0.4,  900.0
WATER_K, WATER_GAMMA, WATER_RHO = 5e4, 7.15, 1000.0

cup_lam = CUP_E * CUP_NU / ((1.0 + CUP_NU) * (1.0 - 2.0 * CUP_NU))
cup_mu  = CUP_E / (2.0 * (1.0 + CUP_NU))
ice_lam = ICE_E * ICE_NU / ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU))
ice_mu  = ICE_E / (2.0 * (1.0 + ICE_NU))

# Time step from stiffest material
c_s_ice = ((ICE_E * (1.0 - ICE_NU)) / ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU) * ICE_RHO)) ** 0.5
DT      = CFL * DX / c_s_ice

# Material IDs
MAT_CUP, MAT_WATER, MAT_ICE = 0, 1, 2


# ──────────────────────────────────────────────────────────────────────────────
# Particle generation (CPU side)
# ──────────────────────────────────────────────────────────────────────────────
def _sub():
    out = []
    for w in range(8):
        di = (w & 4) >> 2
        dj = (w & 2) >> 1
        dk =  w & 1
        out.append((0.25 + di * 0.5, 0.25 + dj * 0.5, 0.25 + dk * 0.5))
    return out
SUB = _sub()

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
          for ox, oy, oz in SUB:
              px = (i + ox) * DX
              py = (j + oy) * DX
              pz = (k + oz) * DX
              dxr = px - CUP_CX; dzr = pz - CUP_CZ
              r   = math.sqrt(dxr * dxr + dzr * dzr)
              in_wall  = (CUP_R_INNER <= r <= CUP_R_OUTER) and \
                         (CUP_Y_BOTTOM + CUP_FLOOR_THK <= py <= CUP_Y_TOP)
              in_floor = (r <= CUP_R_OUTER) and \
                         (CUP_Y_BOTTOM <= py <  CUP_Y_BOTTOM + CUP_FLOOR_THK)
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
          for ox, oy, oz in SUB:
              px = (i + ox) * DX
              py = (j + oy) * DX
              pz = (k + oz) * DX
              dxr = px - CUP_CX; dzr = pz - CUP_CZ
              r   = math.sqrt(dxr * dxr + dzr * dzr)
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
          for ox, oy, oz in SUB:
              px = (i + ox) * DX
              py = (j + oy) * DX
              pz = (k + oz) * DX
              if abs(px - ICE_CX) <= ICE_HALF and \
                 abs(py - ICE_CY) <= ICE_HALF and \
                 abs(pz - ICE_CZ) <= ICE_HALF:
                  pos.append((px, py, pz))
    return np.asarray(pos, dtype=np.float32)


cup_pos   = make_cup()
water_pos = make_water()
ice_pos   = make_ice()
n_cup, n_water, n_ice = len(cup_pos), len(water_pos), len(ice_pos)
N = n_cup + n_water + n_ice
print(f"[tasty_meal_water/basic3d]  cup={n_cup}  water={n_water}  ice={n_ice}  total={N}")
print(f"[tasty_meal_water/basic3d]  dx={DX:.4e}  dt={DT:.4e}  steps={int(TOTAL_TIME/DT)}")

all_pos = np.concatenate([cup_pos, water_pos, ice_pos], axis=0).astype(np.float32)
all_vel = np.zeros_like(all_pos)
all_vel[n_cup + n_water:] = ICE_V0

mat_id_arr = np.empty(N, dtype=np.int32)
mat_id_arr[:n_cup]                    = MAT_CUP
mat_id_arr[n_cup:n_cup+n_water]       = MAT_WATER
mat_id_arr[n_cup+n_water:]            = MAT_ICE

mass_arr = np.empty(N, dtype=np.float32)
mass_arr[:n_cup]                    = PARTICLE_VOL * CUP_RHO
mass_arr[n_cup:n_cup+n_water]       = PARTICLE_VOL * WATER_RHO
mass_arr[n_cup+n_water:]            = PARTICLE_VOL * ICE_RHO


# ──────────────────────────────────────────────────────────────────────────────
# Taichi fields
# ──────────────────────────────────────────────────────────────────────────────
x       = ti.Vector.field(3, float, N)
v       = ti.Vector.field(3, float, N)
mat_id  = ti.field(ti.i32, N)
m_p     = ti.field(float, N)
vol_p   = ti.field(float, N)
F_f     = ti.Matrix.field(3, 3, float, N)

grid_m  = ti.field(float, (GRID_SIZE, GRID_SIZE, GRID_SIZE))
grid_v  = ti.Vector.field(3, float, (GRID_SIZE, GRID_SIZE, GRID_SIZE))

x.from_numpy(all_pos)
v.from_numpy(all_vel)
mat_id.from_numpy(mat_id_arr)
m_p.from_numpy(mass_arr)
vol_p.fill(PARTICLE_VOL)
F_f.from_numpy(np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)))


# ──────────────────────────────────────────────────────────────────────────────
# Per-material PK1 stress
# ──────────────────────────────────────────────────────────────────────────────
@ti.func
def corotated_pk1(Fp, lam, mu):
    U, sig, V = ti.svd(Fp)
    R   = U @ V.transpose()
    J   = Fp.determinant()
    Fit = Fp.inverse().transpose()
    return 2.0 * mu * (Fp - R) + lam * (J - 1.0) * J * Fit

@ti.func
def fluid_pk1_tait(Fp, K, gamma):
    J     = Fp.determinant()
    if J < 0.05:
        J = 0.05
    Fit   = Fp.inverse().transpose()
    # σ = -K (J^{-γ} - 1) I  →  P = J σ F^{-T} = K (1 - J^{-γ}) J F^{-T}
    return K * (1.0 - ti.pow(J, -gamma)) * J * Fit

@ti.func
def particle_pk1(p):
    Fp = F_f[p]
    P  = ti.Matrix.zero(float, 3, 3)
    if mat_id[p] == MAT_CUP:
        P = corotated_pk1(Fp, cup_lam, cup_mu)
    elif mat_id[p] == MAT_ICE:
        P = corotated_pk1(Fp, ice_lam, ice_mu)
    else:                       # MAT_WATER
        P = fluid_pk1_tait(Fp, WATER_K, WATER_GAMMA)
    return P


# ──────────────────────────────────────────────────────────────────────────────
# Time step kernels
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def reset_grid():
    for I in ti.grouped(grid_m):
        grid_m[I] = 0.0
        grid_v[I] = ti.Vector.zero(float, 3)

@ti.kernel
def p2g():
    for p in range(N):
        base = (x[p] / DX - 0.5).cast(int)
        fx   = x[p] / DX - base.cast(float)
        w    = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        dw   = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]
        P    = particle_pk1(p)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    weight = w[i][0] * w[j][1] * w[k][2]
                    grad_w = ti.Vector([
                        dw[i][0] / DX * w[j][1]        * w[k][2],
                        w[i][0]        * dw[j][1] / DX * w[k][2],
                        w[i][0]        * w[j][1]        * dw[k][2] / DX,
                    ])
                    node = base + ti.Vector([i, j, k])
                    grid_m[node[0], node[1], node[2]] += weight * m_p[p]
                    grid_v[node[0], node[1], node[2]] += weight * m_p[p] * v[p]
                    fi = -vol_p[p] * P @ F_f[p].transpose() @ grad_w
                    grid_v[node[0], node[1], node[2]] += DT * fi

@ti.kernel
def update_grid():
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0.0:
            grid_v[i, j, k] /= grid_m[i, j, k]
            grid_v[i, j, k][1] += DT * GRAVITY_Y

            # 1. Cup-floor anchor — rigid Dirichlet on the cup base disk
            px = i * DX; py = j * DX; pz = k * DX
            dxr = px - CUP_CX; dzr = pz - CUP_CZ
            r2  = dxr * dxr + dzr * dzr
            anchor_R = CUP_R_OUTER + 0.005
            in_anchor = (py >= CUP_Y_BOTTOM - 0.001) and \
                        (py <  CUP_Y_BOTTOM + CUP_FLOOR_THK) and \
                        (r2 <= anchor_R * anchor_R)
            if in_anchor:
                grid_v[i, j, k] = ti.Vector.zero(float, 3)

            # 2. Reflective domain walls (4-cell margin)
            if (i < 4 and grid_v[i, j, k][0] < 0.0) or \
               (i >= GRID_SIZE - 4 and grid_v[i, j, k][0] > 0.0):
                grid_v[i, j, k][0] = 0.0
            if (j < 4 and grid_v[i, j, k][1] < 0.0) or \
               (j >= GRID_SIZE - 4 and grid_v[i, j, k][1] > 0.0):
                grid_v[i, j, k][1] = 0.0
            if (k < 4 and grid_v[i, j, k][2] < 0.0) or \
               (k >= GRID_SIZE - 4 and grid_v[i, j, k][2] > 0.0):
                grid_v[i, j, k][2] = 0.0

@ti.kernel
def g2p():
    for p in range(N):
        base = (x[p] / DX - 0.5).cast(int)
        fx   = x[p] / DX - base.cast(float)
        w    = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        dw   = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]

        new_v     = ti.Vector.zero(float, 3)
        v_grad_wT = ti.Matrix.zero(float, 3, 3)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    weight = w[i][0] * w[j][1] * w[k][2]
                    grad_w = ti.Vector([
                        dw[i][0] / DX * w[j][1]        * w[k][2],
                        w[i][0]        * dw[j][1] / DX * w[k][2],
                        w[i][0]        * w[j][1]        * dw[k][2] / DX,
                    ])
                    node = base + ti.Vector([i, j, k])
                    gv = grid_v[node[0], node[1], node[2]]
                    new_v     += weight * gv
                    v_grad_wT += gv.outer_product(grad_w)
        v[p] = new_v
        F_f[p] = (ti.Matrix.identity(float, 3) + DT * v_grad_wT) @ F_f[p]

@ti.kernel
def advect():
    for p in range(N):
        x[p] += DT * v[p]

def step():
    reset_grid()
    p2g()
    update_grid()
    g2p()
    advect()


# ──────────────────────────────────────────────────────────────────────────────
# Run + dump
# ──────────────────────────────────────────────────────────────────────────────
def main():
    out_dir = os.path.join(os.path.dirname(__file__), "output", "tasty_meal_water_basic3d")
    os.makedirs(out_dir, exist_ok=True)

    target_dt = 1.0 / FPS
    next_t    = 0.0
    sim_t     = 0.0
    frame_id  = 0

    def dump():
        pos_np = x.to_numpy()
        vel_np = v.to_numpy()
        write_group(out_dir, "cup",   frame_id, pos_np[:n_cup],            vel_np[:n_cup])
        write_group(out_dir, "water", frame_id, pos_np[n_cup:n_cup+n_water], vel_np[n_cup:n_cup+n_water])
        write_group(out_dir, "ice",   frame_id, pos_np[n_cup+n_water:],    vel_np[n_cup+n_water:])

    dump()
    next_t   += target_dt
    frame_id += 1

    n_steps_total = int(TOTAL_TIME / DT)
    pbar = tqdm(total=n_steps_total, desc="[basic3d] sim")
    while sim_t < TOTAL_TIME - 1e-12:
        step()
        sim_t += DT
        pbar.update(1)
        if sim_t + 1e-12 >= next_t:
            dump()
            next_t   += target_dt
            frame_id += 1
    pbar.close()

    print(f"[tasty_meal_water/basic3d]  done. {frame_id} frames -> {out_dir}")


if __name__ == "__main__":
    main()
