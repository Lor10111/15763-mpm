# Colliding Cubes – Basic MPM 3D  (Taichi)
# =========================================
# Two identical elastic cubes launched toward each other along the x-axis.
# Real-time 3D visualisation via Taichi GGUI (right-click drag to orbit).
#
# Shared parameters (identical across CKMPM / MLS-MPM / Basic-MPM-3D):
#   E          = 1e6 Pa   (Fixed Corotated)
#   nu         = 0.4
#   rho        = 1000 kg/m³
#   v0         = ±(0.1, 0, 0) m/s
#   ppc        = 8  (2×2×2 sub-cell)
#   CFL        = 0.5
#   fps        = 48
#   total_time = 3.0 s
#
# Grid resolution for this implementation:
#   grid_size = 64   →   dx = 1/64 m
#
# Cube geometry (cell-loop identical to CKMPM):
#   Loop i,j,k ∈ [−4,+4], 2×2×2 sub-cell  →  9³×8 = 5832 particles/cube
#   Cube 1 centre: (16, 32, 32) cells = (0.25, 0.50, 0.50) m   vel = (+0.1, 0, 0)
#   Cube 2 centre: (48, 32, 32) cells = (0.75, 0.50, 0.50) m   vel = (−0.1, 0, 0)
#
# Usage:
#   python simulator_3d.py

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)   # swap to ti.gpu / ti.metal for faster execution

# ── shared parameters ─────────────────────────────────────────────────────────
# ANCHOR: property_def
grid_size = 64
dx        = 1.0 / grid_size
E, nu     = 1e6, 0.4
density   = 1000.0
mu        = E / (2.0 * (1.0 + nu))
lam       = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
c_s       = ((E * (1.0 - nu)) / ((1.0 + nu) * (1.0 - 2.0 * nu) * density)) ** 0.5
CFL       = 0.5
dt        = CFL * dx / c_s
ppc       = 8
fps       = 48
total_time = 3.0

particle_vol  = dx**3 / ppc
particle_mass = particle_vol * density
# ANCHOR_END: property_def

print(f"[basic3d]  dx={dx:.4e}  c_s={c_s:.3f}  dt={dt:.4e}")
print(f"           mu={mu:.3e}  lam={lam:.3e}")
print(f"           particle_vol={particle_vol:.3e}  particle_mass={particle_mass:.3e}")

# ── particle setup ────────────────────────────────────────────────────────────
# ANCHOR: setting
# Cell-loop identical to CKMPM test_colliding_cubes.cuh SetupModel().
def make_cube(cx, cy, cz, vx):
    """Return (N,3) positions and (N,3) velocities for one cube."""
    pos, vel = [], []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for w in range(8):
                    di = (w & 4) >> 2
                    dj = (w & 2) >> 1
                    dk =  w & 1
                    pos.append([(cx + i + 0.25 + di * 0.5) * dx,
                                 (cy + j + 0.25 + dj * 0.5) * dx,
                                 (cz + k + 0.25 + dk * 0.5) * dx])
                    vel.append([vx, 0.0, 0.0])
    return (np.array(pos, dtype=np.float32),
            np.array(vel, dtype=np.float32))

pos0, vel0 = make_cube(16, 32, 32, +0.1)   # Cube 1 →
pos1, vel1 = make_cube(48, 32, 32, -0.1)   # Cube 2 ←
n0 = len(pos0)   # 5832
n1 = len(pos1)   # 5832

all_pos = np.concatenate([pos0, pos1], axis=0)
all_vel = np.concatenate([vel0, vel1], axis=0)
# ANCHOR_END: setting

N_particles = len(all_pos)
print(f"[basic3d]  N={N_particles}  (n0={n0}, n1={n1})")

# ── taichi fields ─────────────────────────────────────────────────────────────
# ANCHOR: data_def
x       = ti.Vector.field(3, float, N_particles)
v       = ti.Vector.field(3, float, N_particles)
vol_f   = ti.field(float, N_particles)
m_f     = ti.field(float, N_particles)
F_f     = ti.Matrix.field(3, 3, float, N_particles)

grid_m  = ti.field(float, (grid_size, grid_size, grid_size))
grid_v  = ti.Vector.field(3, float, (grid_size, grid_size, grid_size))

# Per-particle colour for GGUI: cube 1 = red, cube 2 = blue
color_f = ti.Vector.field(3, float, N_particles)

x.from_numpy(all_pos)
v.from_numpy(all_vel)
vol_f.fill(particle_vol)
m_f.fill(particle_mass)
F_f.from_numpy(np.tile(np.eye(3, dtype=np.float32), (N_particles, 1, 1)))

@ti.kernel
def init_colors(n_first: int):
    for p in range(N_particles):
        if p < n_first:
            color_f[p] = ti.Vector([0.9, 0.2, 0.2])   # red  – cube 1
        else:
            color_f[p] = ti.Vector([0.2, 0.4, 0.9])   # blue – cube 2

init_colors(n0)
# ANCHOR_END: data_def

# ── constitutive model: Fixed Corotated ───────────────────────────────────────
# ANCHOR: fixed_corotated
# P = 2μ(F − R) + λ(J−1)J F^{−T}
# Matches CKMPM kFixedCorotated and MLS-MPM CorotatedElasticity.
@ti.func
def fixed_corotated_pk1(Fp):
    U, sig, V = ti.svd(Fp)
    R   = U @ V.transpose()
    J   = Fp.determinant()
    Fit = Fp.inverse().transpose()   # F^{−T}
    return 2.0 * mu * (Fp - R) + lam * (J - 1.0) * J * Fit
# ANCHOR_END: fixed_corotated

# ── grid reset ────────────────────────────────────────────────────────────────
# ANCHOR: reset_grid
@ti.kernel
def reset_grid():
    for i, j, k in grid_m:
        grid_m[i, j, k] = 0.0
        grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
# ANCHOR_END: reset_grid

# ── P2G ──────────────────────────────────────────────────────────────────────
# ANCHOR: p2g
@ti.kernel
def p2g():
    for p in range(N_particles):
        base = (x[p] / dx - 0.5).cast(int)
        fx   = x[p] / dx - base.cast(float)
        w    = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        dw   = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]
        P    = fixed_corotated_pk1(F_f[p])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    weight = w[i][0] * w[j][1] * w[k][2]
                    grad_w = ti.Vector([
                        dw[i][0] / dx * w[j][1]        * w[k][2],
                        w[i][0]        * dw[j][1] / dx * w[k][2],
                        w[i][0]        * w[j][1]        * dw[k][2] / dx,
                    ])
                    node = base + ti.Vector([i, j, k])
                    grid_m[node[0], node[1], node[2]] += weight * m_f[p]
                    grid_v[node[0], node[1], node[2]] += weight * m_f[p] * v[p]
                    fi = -vol_f[p] * P @ F_f[p].transpose() @ grad_w
                    grid_v[node[0], node[1], node[2]] += dt * fi
# ANCHOR_END: p2g

# ── grid update ───────────────────────────────────────────────────────────────
# ANCHOR: grid_update
@ti.kernel
def update_grid():
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0.0:
            grid_v[i, j, k] /= grid_m[i, j, k]
            # Dirichlet BC: zero velocity at 3-cell boundary guard
            if (i <= 3 or i >= grid_size - 3 or
                j <= 3 or j >= grid_size - 3 or
                k <= 3 or k >= grid_size - 3):
                grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
# ANCHOR_END: grid_update

# ── G2P ──────────────────────────────────────────────────────────────────────
# ANCHOR: g2p
@ti.kernel
def g2p():
    for p in range(N_particles):
        base = (x[p] / dx - 0.5).cast(int)
        fx   = x[p] / dx - base.cast(float)
        w    = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        dw   = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]
        new_v     = ti.Vector.zero(float, 3)
        v_grad_wT = ti.Matrix.zero(float, 3, 3)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    weight = w[i][0] * w[j][1] * w[k][2]
                    grad_w = ti.Vector([
                        dw[i][0] / dx * w[j][1]        * w[k][2],
                        w[i][0]        * dw[j][1] / dx * w[k][2],
                        w[i][0]        * w[j][1]        * dw[k][2] / dx,
                    ])
                    node = base + ti.Vector([i, j, k])
                    gv = grid_v[node[0], node[1], node[2]]
                    new_v     += weight * gv
                    v_grad_wT += gv.outer_product(grad_w)
        v[p] = new_v
        F_f[p] = (ti.Matrix.identity(float, 3) + dt * v_grad_wT) @ F_f[p]
# ANCHOR_END: g2p

# ── advect ────────────────────────────────────────────────────────────────────
# ANCHOR: particle_update
@ti.kernel
def advect():
    for p in range(N_particles):
        x[p] += dt * v[p]
# ANCHOR_END: particle_update

# ── time step ─────────────────────────────────────────────────────────────────
# ANCHOR: time_step
def step():
    reset_grid()
    p2g()
    update_grid()
    g2p()
    advect()
# ANCHOR_END: time_step

# ── GGUI 3D visualisation ─────────────────────────────────────────────────────
steps_per_frame = int(1.0 / (24 * dt))
total_steps     = int(round(total_time * fps)) * steps_per_frame

window = ti.ui.Window("Colliding Cubes – Basic MPM 3D", (900, 700))
canvas = window.get_canvas()
scene  = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.5, 0.7, 2.0)
camera.lookat(0.5, 0.5, 0.5)
camera.up(0.0, 1.0, 0.0)

sim_time   = 0.0
step_count = 0

while window.running and sim_time < total_time + 1e-6:
    # advance one output frame
    for _ in range(steps_per_frame):
        step()
        step_count += 1
    sim_time = step_count * dt

    # render
    camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.55, 0.55, 0.55))
    scene.point_light(pos=(0.5, 2.0, 2.0), color=(1.0, 1.0, 1.0))
    scene.particles(x, per_vertex_color=color_f, radius=dx * 0.45)
    canvas.scene(scene)

    # on-screen time label (shown in window title)
    window.get_canvas().set_background_color((0.08, 0.08, 0.08))
    window.show()
