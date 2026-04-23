"""
Filter Drop — JELLY version using CKMPM (Taichi, 3D)

A soft elastic block falls under gravity onto 3 horizontal cylinders.
Implemented with the **CKMPM** scheme: compact sine-based kernel +
dual offset grid + fused-style P2G/G2P (here split for clarity).

Scene mirrors test2_filter_drop.py (MLS-MPM) so MLS vs CKMPM is comparable.
Note: CKMPM here uses Y as the vertical axis (gravity = -Y), matching the
CKMPM C++ test_filter_drop.cuh convention.

Run:
    python test2_filter_drop_jelly_ckmpm.py
    python test2_filter_drop_jelly_ckmpm.py --total_time 3.0 --fps 48

Outputs:
    output_ckmpm_jelly_filter/jelly_<NNNN>.ply
"""
import os
import argparse
import numpy as np
import taichi as ti

# ----------------------------------------------------------------------
# Taichi init  (use cuda if you have it, else fall back)
# ----------------------------------------------------------------------
try:
    ti.init(arch=ti.cuda, default_fp=ti.f32, device_memory_GB=8,
            kernel_profiler=True, fast_math=False)
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32, kernel_profiler=True)

# ----------------------------------------------------------------------
# Scene constants
# ----------------------------------------------------------------------
n_grid = 256
n_max_particles = 4_000_000   # PPC=8 fills the block with ~3.2M particles
dx = 1.0 / n_grid

# --- JELLY material ---
JELLY_E   = 5e4       # soft jelly (rubber would be ~2e6)
JELLY_NU  = 0.4
JELLY_RHO = 1000.0

mu   = JELLY_E / (2 * (1 + JELLY_NU))
lamb = JELLY_E * JELLY_NU / ((1 + JELLY_NU) * (1 - 2 * JELLY_NU))

# PPC is chosen at runtime via --ppc, then particle_volume and particle_mass are
# (re)assigned in main() *before* any kernel runs. Each particle represents
# (dx^3 / ppc) of volume, so total mass is independent of ppc.
particle_volume = dx * dx * dx / 8.0     # placeholder, overwritten in main()
particle_mass   = particle_volume * JELLY_RHO

GRAVITY = -9.8
CFL     = 0.5

# --- Geometry (mirrors MLS filter_drop, but Y is vertical here) ---
BLOCK_CENTER = (0.5, 0.80, 0.5)   # (x, y, z) — Y is vertical
BLOCK_HALF   = (0.20, 0.07, 0.20) # half-size: thin in Y, wide in X/Z

CYL_RADIUS = 0.06
CYL_Y      = 0.50                 # height of cylinder axes
CYL_X      = (0.20, 0.50, 0.80)   # 3 cylinders, axis along Z
FLOOR_Y    = 0.02

# --- Time ---
twopi      = 2 * np.pi
mpm_float  = ti.f32

def compute_sound_cfl_dt(cfl, dx, E, nu, rho):
    return cfl * dx / np.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho))

# CKMPM needs the *0.1 safety factor on the compact kernel
default_dt = compute_sound_cfl_dt(CFL, dx, JELLY_E, JELLY_NU, JELLY_RHO) * 0.1
print(f"default_dt = {default_dt:.3e}")

# ----------------------------------------------------------------------
# CKMPM dual grid:  last axis size = 2  → two interleaved offset grids
# ----------------------------------------------------------------------
axes_4d = ti.ijkl
parent_prefix = (1,)   # parent block dim along the dual axis
dim_prefix    = (2,)

mass_grid = ti.field(dtype=mpm_float)
mass_root = ti.root.pointer(axes_4d, (16, 16, 16) + parent_prefix)
mass_ptr  = mass_root.pointer(axes_4d, (8, 8, 8) + parent_prefix)
mass_dn   = mass_ptr.dense (axes_4d, (4, 4, 4) + dim_prefix)
mass_dn.place(mass_grid)

momentum_grid = ti.Vector.field(n=3, dtype=mpm_float)
mom_root = ti.root.pointer(axes_4d, (16, 16, 16) + parent_prefix)
mom_ptr  = mom_root.pointer(axes_4d, (8, 8, 8) + parent_prefix)
mom_dn   = mom_ptr.dense (axes_4d, (4, 4, 4) + dim_prefix)
mom_dn.place(momentum_grid)

# ----------------------------------------------------------------------
# Particle fields
# ----------------------------------------------------------------------
particle_position = ti.Vector.field(3, dtype=mpm_float, shape=(n_max_particles,))
particle_velocity = ti.Vector.field(3, dtype=mpm_float, shape=(n_max_particles,))
particle_C        = ti.Matrix.field(3, 3, dtype=mpm_float, shape=(n_max_particles,))
particle_F        = ti.Matrix.field(3, 3, dtype=mpm_float, shape=(n_max_particles,))

# ----------------------------------------------------------------------
# Compact kernel (CKMPM)
# ----------------------------------------------------------------------
@ti.func
def compact_kernel_stencil(dw):
    s = ti.Matrix.zero(mpm_float, 3, 2)
    for i in ti.static(range(3)):
        s[i, 0] = 1 - dw[i] + ti.sin(twopi * dw[i]) / twopi
        s[i, 1] = 1 - s[i, 0]
    return s

@ti.func
def compact_kernel_gradient(cell, stencil, dw):
    kx = stencil[0, cell[0]]
    ky = stencil[1, cell[1]]
    kz = stencil[2, cell[2]]
    grad = ti.math.sign(dw) * (ti.cos(twopi * ti.abs(dw)) - 1)
    return n_grid * ti.Vector([ky * kz, kx * kz, kx * ky]) * grad

# ----------------------------------------------------------------------
# Fixed-corotated stress  (PF = -V * (2μ(F-R)Fᵀ + λ(J-1)J·I))
# ----------------------------------------------------------------------
@ti.func
def compute_fixed_corotated(F):
    U, S, V = ti.svd(F)
    R = U @ V.transpose()
    J = S[0, 0] * S[1, 1] * S[2, 2]
    return -particle_volume * (2 * mu * (F - R) @ F.transpose()
                               + lamb * (J - 1) * J * ti.Matrix.identity(mpm_float, 3))

# ----------------------------------------------------------------------
# CFL dt  (clamp by max particle velocity)
# ----------------------------------------------------------------------
@ti.kernel
def compute_cfl_dt(particle_count: ti.i32) -> ti.f32:
    md = default_dt
    for p in range(particle_count):
        ti.atomic_min(md, 0.5 * dx / (particle_velocity[p].norm() + 1e-8))
    return md

# ----------------------------------------------------------------------
# Substep — CKMPM only
# ----------------------------------------------------------------------
@ti.kernel
def substep(dt: ti.f32, particle_count: ti.i32) -> ti.f32:
    # --- clear grid ---
    for I in ti.grouped(mass_grid):
        mass_grid[I] = 0
        momentum_grid[I] = ti.Vector.zero(mpm_float, 3)

    # --- P2G ---
    ti.loop_config(block_dim=128)
    for p in range(particle_count):
        gp = particle_position[p]
        v  = particle_velocity[p]
        PF = compute_fixed_corotated(particle_F[p])

        for w in ti.static(range(2)):
            sign = ti.static(-1 if w == 0 else 1)
            cell_index = ti.cast(gp * n_grid - sign * 0.25, ti.i32)
            offset     = gp * n_grid - (cell_index + 0.25 * sign)
            stencil    = compact_kernel_stencil(offset)

            for I in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                cell = ti.Vector([cell_index[0] + I[0],
                                  cell_index[1] + I[1],
                                  cell_index[2] + I[2], w])
                weight = stencil[0, I[0]] * stencil[1, I[1]] * stencil[2, I[2]]
                mass_grid[cell]     += weight * particle_mass
                momentum_grid[cell] += (weight * particle_mass * v
                                        + dt * PF @ compact_kernel_gradient(I, stencil, offset - I))

    # --- Grid update + colliders ---
    max_v = 0.0
    ti.loop_config(block_dim=128)
    for I in ti.grouped(mass_grid):
        m = mass_grid[I]
        vel = ti.Vector.zero(mpm_float, 3)
        if m > particle_mass * 1e-8:
            vel = momentum_grid[I] / m
        vel[1] += dt * GRAVITY

        # World-space position of this dual-grid node (approximate)
        # node ≈ (i + 0.25 + 0.5*w) * dx for each spatial axis
        wx = (I[0] + 0.25 + 0.5 * I[3]) * dx
        wy = (I[1] + 0.25 + 0.5 * I[3]) * dx
        wz = (I[2] + 0.25 + 0.5 * I[3]) * dx

        # ---- Floor (sticky) ----
        if wy < FLOOR_Y:
            vel = ti.Vector.zero(mpm_float, 3)

        # ---- 3 cylinders (axis = Z), slip ----
        # cylinder 1
        ex = wx - 0.20
        ey = wy - CYL_Y
        d2 = ex * ex + ey * ey
        if d2 < CYL_RADIUS * CYL_RADIUS and d2 > 1e-12:
            d = ti.sqrt(d2)
            nx, ny = ex / d, ey / d
            dot = vel[0] * nx + vel[1] * ny
            if dot < 0:
                vel[0] -= dot * nx
                vel[1] -= dot * ny

        # cylinder 2
        ex = wx - 0.50
        ey = wy - CYL_Y
        d2 = ex * ex + ey * ey
        if d2 < CYL_RADIUS * CYL_RADIUS and d2 > 1e-12:
            d = ti.sqrt(d2)
            nx, ny = ex / d, ey / d
            dot = vel[0] * nx + vel[1] * ny
            if dot < 0:
                vel[0] -= dot * nx
                vel[1] -= dot * ny

        # cylinder 3
        ex = wx - 0.80
        ey = wy - CYL_Y
        d2 = ex * ex + ey * ey
        if d2 < CYL_RADIUS * CYL_RADIUS and d2 > 1e-12:
            d = ti.sqrt(d2)
            nx, ny = ex / d, ey / d
            dot = vel[0] * nx + vel[1] * ny
            if dot < 0:
                vel[0] -= dot * nx
                vel[1] -= dot * ny

        # ---- Domain box (kill anything escaping the active region) ----
        oob = (I[0] < 6) or (I[0] >= n_grid - 6) or \
              (I[1] < 6) or (I[1] >= n_grid - 6) or \
              (I[2] < 6) or (I[2] >= n_grid - 6)
        if oob:
            vel = ti.Vector.zero(mpm_float, 3)

        momentum_grid[I] = vel
        ti.atomic_max(max_v, vel.norm())

    # --- G2P ---
    ti.loop_config(block_dim=128)
    for p in range(particle_count):
        gp = particle_position[p]
        v  = ti.Vector.zero(mpm_float, 3)
        cv = ti.Matrix.zero(mpm_float, 3, 3)

        for w in ti.static(range(2)):
            sign = ti.static(-1 if w == 0 else 1)
            cell_index = ti.cast(gp * n_grid - sign * 0.25, ti.i32)
            offset     = gp * n_grid - (cell_index + 0.25 * sign)
            stencil    = compact_kernel_stencil(offset)

            for I in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                cell = ti.Vector([cell_index[0] + I[0],
                                  cell_index[1] + I[1],
                                  cell_index[2] + I[2], w])
                weight = stencil[0, I[0]] * stencil[1, I[1]] * stencil[2, I[2]]
                gv = momentum_grid[cell]
                v  += weight * gv
                cv += gv.outer_product(compact_kernel_gradient(I, stencil, offset - I))

        v *= 0.5    # average over the 2 dual grids
        particle_position[p] += dt * v
        particle_velocity[p]  = v
        particle_F[p] = (ti.Matrix.identity(mpm_float, 3) + 0.5 * dt * cv) @ particle_F[p]

    return max_v

# ----------------------------------------------------------------------
# Init: lay out a jelly block with PPC=8, exactly like CKMPM C++ test
# ----------------------------------------------------------------------
def make_jelly_particles(ppc: int = 8):
    """Lay out a jelly block with `ppc` particles per cell (1, 2, 4, or 8).

    Uses the same 8-corner placement scheme as the CKMPM C++ test,
    but only emits the first `ppc` of the 8 sub-positions per cell.
    """
    assert ppc in (1, 2, 4, 8), "ppc must be 1, 2, 4, or 8"
    cx, cy, cz = BLOCK_CENTER
    hx, hy, hz = BLOCK_HALF

    ix0, ix1 = int((cx - hx) * n_grid), int((cx + hx) * n_grid)
    iy0, iy1 = int((cy - hy) * n_grid), int((cy + hy) * n_grid)
    iz0, iz1 = int((cz - hz) * n_grid), int((cz + hz) * n_grid)

    # Pick `ppc` of the 8 corner offsets, spread out as much as possible
    all_corners = [(0,0,0),(1,1,1),(1,0,0),(0,1,1),
                   (0,1,0),(1,0,1),(0,0,1),(1,1,0)]
    corners = all_corners[:ppc]

    pts = []
    for i in range(ix0, ix1 + 1):
        for j in range(iy0, iy1 + 1):
            for k in range(iz0, iz1 + 1):
                for di, dj, dk in corners:
                    pts.append((
                        (i + 0.25 + di * 0.5) * dx,
                        (j + 0.25 + dj * 0.5) * dx,
                        (k + 0.25 + dk * 0.5) * dx,
                    ))
    return np.array(pts, dtype=np.float32)

@ti.kernel
def init_particles(n: ti.i32, pos: ti.types.ndarray()):
    for i in range(n):
        particle_position[i] = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])
        particle_velocity[i] = ti.Vector.zero(mpm_float, 3)
        particle_C[i]        = ti.Matrix.zero(mpm_float, 3, 3)
        particle_F[i]        = ti.Matrix.identity(mpm_float, 3)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total_time', type=float, default=3.0)
    ap.add_argument('--fps', type=int, default=48)
    ap.add_argument('--ppc', type=int, default=1,
                    help='particles per cell: 1 (CPU friendly), 2, 4, or 8 (GPU)')
    ap.add_argument('--output_dir', type=str,
                    default='./output_ckmpm_jelly_filter')
    args = ap.parse_args()

    # Recompute particle volume/mass for the chosen ppc so total mass stays right
    global particle_volume, particle_mass
    particle_volume = dx * dx * dx / args.ppc
    particle_mass   = particle_volume * JELLY_RHO

    os.makedirs(args.output_dir, exist_ok=True)

    pts = make_jelly_particles(ppc=args.ppc)
    n   = pts.shape[0]
    assert n <= n_max_particles, f"too many particles: {n} > {n_max_particles}"
    print(f"jelly particles: {n}  (ppc={args.ppc})")
    print(f"particle_mass = {particle_mass:.3e}")

    init_particles(n, pts)

    writer = ti.tools.PLYWriter(num_vertices=n)
    n_frames = int(args.total_time * args.fps + 0.5)
    max_v = 0.0

    for frame in range(n_frames):
        time_remain = 1.0 / args.fps
        substep_idx = 0
        while time_remain > 0:
            mass_root.deactivate_all()
            mom_root.deactivate_all()

            dt = min(default_dt, compute_cfl_dt(n))
            dt = min(dt, time_remain)
            if max_v > 0.0:
                dt = min(dt, CFL * dx / max_v)

            max_v = substep(dt, n)
            time_remain -= dt
            substep_idx += 1

        # export PLY
        export = particle_position.to_numpy()[:n]
        writer.add_vertex_pos(export[:, 0], export[:, 1], export[:, 2])
        writer.export_frame(frame, os.path.join(args.output_dir, "jelly"))

        print(f"frame {frame+1}/{n_frames}  substeps={substep_idx}  "
              f"y∈[{export[:,1].min():.3f}, {export[:,1].max():.3f}]")

    ti.profiler.print_kernel_profiler_info()
    print(f"\nDone. PLY frames in {args.output_dir}/")

if __name__ == "__main__":
    main()
