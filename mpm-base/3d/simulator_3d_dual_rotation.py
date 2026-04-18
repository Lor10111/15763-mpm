# Dual Rotation – Basic MPM 3D  (Taichi / PIC)
# ===============================================
# Outputs: output/dual_rotation_basic3d.gif   – 3-D particle animation
#          output/dual_rotation_Lz_basic3d.png – angular momentum vs time
# Two identical elastic cubes given an initial pure rotation about the z-axis.
#   Cube 1: ω = +OMEGA rad/s (counterclockwise when viewed from +z)
#   Cube 2: ω = −OMEGA rad/s (clockwise)
# Total initial angular momentum L_z ≈ 0 → should be conserved.
#
# Shared parameters (identical across CKMPM / MLS-MPM / Basic-MPM-3D):
#   E          = 1e6 Pa   (Fixed Corotated)
#   nu         = 0.4
#   rho        = 1000 kg/m³
#   omega      = ±2.0 rad/s
#   ppc        = 8  (2×2×2 sub-cell sampling)
#   CFL        = 0.5
#   fps        = 48
#   total_time = 5.0 s
#   gravity    = 0.0
#
# Grid resolution for this implementation:
#   grid_size = 64   →   dx = 1/64 m
#
# Cube geometry – fractional positions match CKMPM dual_rotation:
#   Cube 1 centre: (20, 32, 32) cells = (0.3125, 0.5, 0.5) m   ω = +2.0
#   Cube 2 centre: (44, 32, 32) cells = (0.6875, 0.5, 0.5) m   ω = −2.0
#
# Usage:
#   python simulator_3d_dual_rotation.py
#   (right-click drag in GGUI window to orbit the camera)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import taichi as ti

ti.init(arch=ti.cpu)   # swap to ti.gpu / ti.metal for faster execution

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUT_GIF = os.path.join(OUT_DIR, "dual_rotation_basic3d.gif")
OUT_LZ  = os.path.join(OUT_DIR, "dual_rotation_Lz_basic3d.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared parameters ─────────────────────────────────────────────────────────
grid_size  = 64
dx         = 1.0 / grid_size
E, nu      = 1e6, 0.4
density    = 1000.0
mu         = E / (2.0 * (1.0 + nu))
lam        = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
c_s        = ((E * (1.0 - nu)) / ((1.0 + nu) * (1.0 - 2.0 * nu) * density)) ** 0.5
CFL        = 0.5
dt         = CFL * dx / c_s
ppc        = 8
fps        = 48
total_time = 5.0
OMEGA      = 2.0       # rad/s

particle_vol  = dx**3 / ppc
particle_mass = particle_vol * density

print(f"[basic3d-dual-rot]  dx={dx:.4e}  c_s={c_s:.4f}  dt={dt:.4e}")
print(f"[basic3d-dual-rot]  mu={mu:.4e}  lam={lam:.4e}")
print(f"[basic3d-dual-rot]  particle_vol={particle_vol:.4e}  particle_mass={particle_mass:.4e}")


# ── Particle initialisation (Python) ──────────────────────────────────────────
def make_rotating_cube(cx_cell, cy_cell, cz_cell, omega_z):
    """
    9×9×9 grid of cells, 2×2×2 sub-cell sampling → 5832 particles per cube.
    Velocity field: v = ω × r,  ω = (0, 0, omega_z)
        vx = −omega_z * ry,   vy = +omega_z * rx,   vz = 0
    """
    pos, vel = [], []
    cx = cx_cell * dx
    cy = cy_cell * dx
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for w in range(8):
                    di = (w & 4) >> 2
                    dj = (w & 2) >> 1
                    dk =  w & 1
                    px = (cx_cell + i - 0.25 + di * 0.5) * dx
                    py = (cy_cell + j - 0.25 + dj * 0.5) * dx
                    pz = (cz_cell + k - 0.25 + dk * 0.5) * dx
                    rx = px - cx
                    ry = py - cy
                    vx = -omega_z * ry
                    vy = +omega_z * rx
                    pos.append([px, py, pz])
                    vel.append([vx, vy, 0.0])
    return (np.array(pos, dtype=np.float32),
            np.array(vel, dtype=np.float32))


# Cube centres: (0.3125, 0.5, 0.5) m and (0.6875, 0.5, 0.5) m
pos0, vel0 = make_rotating_cube(20, 32, 32, +OMEGA)
pos1, vel1 = make_rotating_cube(44, 32, 32, -OMEGA)
n0 = len(pos0)   # 5832

# Initial angular momentum check (about domain centre 0.5, 0.5, 0.5)
domain_center = np.array([0.5, 0.5, 0.5])
r0 = pos0 - domain_center;  r1 = pos1 - domain_center
Lz0 = particle_mass * float(np.sum(r0[:, 0] * vel0[:, 1] - r0[:, 1] * vel0[:, 0]))
Lz1 = particle_mass * float(np.sum(r1[:, 0] * vel1[:, 1] - r1[:, 1] * vel1[:, 0]))
print(f"[basic3d-dual-rot]  Cube0 init L_z = {Lz0:.6e}")
print(f"[basic3d-dual-rot]  Cube1 init L_z = {Lz1:.6e}")
print(f"[basic3d-dual-rot]  Total init L_z = {Lz0 + Lz1:.6e}  (should be ~0)")

all_pos = np.concatenate([pos0, pos1], axis=0)
all_vel = np.concatenate([vel0, vel1], axis=0)
N = len(all_pos)
print(f"[basic3d-dual-rot]  Total particles: {N}")


# ── Taichi fields ─────────────────────────────────────────────────────────────
x      = ti.Vector.field(3, float, N)
v      = ti.Vector.field(3, float, N)
F_f    = ti.Matrix.field(3, 3, float, N)
vol_f  = ti.field(float, N)
m_f    = ti.field(float, N)

grid_m = ti.field(float, (grid_size, grid_size, grid_size))
grid_v = ti.Vector.field(3, float, (grid_size, grid_size, grid_size))

color_f = ti.Vector.field(3, float, N)   # per-particle colour for GGUI

x.from_numpy(all_pos)
v.from_numpy(all_vel)
vol_f.fill(particle_vol)
m_f.fill(particle_mass)
F_f.from_numpy(np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)))


@ti.kernel
def init_colors(n_first: int):
    for p in range(N):
        if p < n_first:
            color_f[p] = ti.Vector([0.9, 0.2, 0.2])   # red  – cube 1 (+ω)
        else:
            color_f[p] = ti.Vector([0.2, 0.4, 0.9])   # blue – cube 2 (−ω)

init_colors(n0)


# ── Fixed Corotated: P = 2μ(F−R) + λ(J−1)J F^{−T} ───────────────────────────
@ti.func
def fixed_corotated_pk1(Fp):
    U, sig, V = ti.svd(Fp)
    R   = U @ V.transpose()
    J   = Fp.determinant()
    Fit = Fp.inverse().transpose()
    return 2.0 * mu * (Fp - R) + lam * (J - 1.0) * J * Fit


# ── Grid reset ────────────────────────────────────────────────────────────────
@ti.kernel
def reset_grid():
    for i, j, k in grid_m:
        grid_m[i, j, k] = 0.0
        grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])


# ── P2G ──────────────────────────────────────────────────────────────────────
@ti.kernel
def p2g():
    for p in range(N):
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
                        dw[i][0] / dx * w[j][1]         * w[k][2],
                        w[i][0]        * dw[j][1] / dx  * w[k][2],
                        w[i][0]        * w[j][1]         * dw[k][2] / dx,
                    ])
                    node = base + ti.Vector([i, j, k])
                    grid_m[node[0], node[1], node[2]] += weight * m_f[p]
                    grid_v[node[0], node[1], node[2]] += weight * m_f[p] * v[p]
                    fi = -vol_f[p] * P @ F_f[p].transpose() @ grad_w
                    grid_v[node[0], node[1], node[2]] += dt * fi


# ── Grid update (no gravity) ──────────────────────────────────────────────────
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


# ── G2P ──────────────────────────────────────────────────────────────────────
@ti.kernel
def g2p():
    for p in range(N):
        base = (x[p] / dx - 0.5).cast(int)
        fx   = x[p] / dx - base.cast(float)
        w    = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        dw   = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]
        new_v  = ti.Vector.zero(float, 3)
        grad_v = ti.Matrix.zero(float, 3, 3)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    weight = w[i][0] * w[j][1] * w[k][2]
                    grad_w = ti.Vector([
                        dw[i][0] / dx * w[j][1]         * w[k][2],
                        w[i][0]        * dw[j][1] / dx  * w[k][2],
                        w[i][0]        * w[j][1]         * dw[k][2] / dx,
                    ])
                    node = base + ti.Vector([i, j, k])
                    gv    = grid_v[node[0], node[1], node[2]]
                    new_v  += weight * gv
                    grad_v += gv.outer_product(grad_w)
        v[p]   = new_v
        F_f[p] = (ti.Matrix.identity(float, 3) + dt * grad_v) @ F_f[p]


# ── Advect ────────────────────────────────────────────────────────────────────
@ti.kernel
def advect():
    for p in range(N):
        x[p] += dt * v[p]


# ── Angular momentum L_z about domain centre (0.5, 0.5, 0.5) ─────────────────
@ti.kernel
def compute_Lz() -> float:
    result = 0.0
    for p in range(N):
        rx = x[p][0] - 0.5
        ry = x[p][1] - 0.5
        result += m_f[p] * (rx * v[p][1] - ry * v[p][0])
    return result


# ── Single time step ──────────────────────────────────────────────────────────
def step():
    reset_grid()
    p2g()
    update_grid()
    g2p()
    advect()


# ── Simulation loop – record frames ──────────────────────────────────────────
steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
total_frames    = int(round(total_time * fps))

print(f"[basic3d-dual-rot]  steps_per_frame={steps_per_frame}  total_frames={total_frames}")

frames_0   = []   # cube 1 positions per output frame
frames_1   = []   # cube 2 positions per output frame
Lz_history = []   # total L_z per output frame

def record():
    pts = x.to_numpy()          # shape (N, 3)
    vels = v.to_numpy()
    frames_0.append(pts[:n0].copy())
    frames_1.append(pts[n0:].copy())
    r   = pts - domain_center
    Lz  = float(particle_mass * np.sum(r[:, 0] * vels[:, 1] - r[:, 1] * vels[:, 0]))
    Lz_history.append(Lz)

record()   # frame 0 = initial state

for frame in range(1, total_frames + 1):
    for _ in range(steps_per_frame):
        step()
    record()
    Lz = Lz_history[-1]
    if frame % 12 == 0:
        print(f"  Frame {frame:4d}/{total_frames}  "
              f"t={frame / fps:.3f}s  L_z={Lz:.4e}")

print(f"[basic3d-dual-rot]  simulation done, rendering …")


# ── 3-D GIF ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 7.0))
ax  = fig.add_subplot(111, projection="3d")
ax.set_xlim(0.25, 0.75); ax.set_ylim(0.25, 0.75); ax.set_zlim(0.25, 0.75)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.set_title("Dual Rotation – Basic MPM 3D")
ax.set_box_aspect((3, 3, 3))
ax.view_init(elev=20.0, azim=-65.0)

scat0 = ax.scatter([], [], [], s=1, c="#e74c3c", depthshade=False, label="Cube 1 (+ω)")
scat1 = ax.scatter([], [], [], s=1, c="#2e86de", depthshade=False, label="Cube 2 (−ω)")
time_txt = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)
ax.legend(loc="upper right", markerscale=5)

def update(frame_id):
    p0 = frames_0[frame_id]; p1 = frames_1[frame_id]
    scat0._offsets3d = (p0[:, 0], p0[:, 1], p0[:, 2])
    scat1._offsets3d = (p1[:, 0], p1[:, 1], p1[:, 2])
    time_txt.set_text(f"t = {frame_id / fps:.2f} s")
    return scat0, scat1, time_txt

ani = animation.FuncAnimation(fig, update, frames=len(frames_0),
                               interval=1000 // fps, blit=False)
ani.save(OUT_GIF, writer="pillow", fps=fps, dpi=100)
plt.close(fig)
print(f"[basic3d-dual-rot]  GIF saved → {OUT_GIF}")


# ── Angular momentum plot ─────────────────────────────────────────────────────
times  = np.arange(len(Lz_history)) / fps
Lz_arr = np.array(Lz_history)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(times, Lz_arr, lw=1.5, color="#e67e22", label="L_z (total)")
ax2.axhline(0, color="k", ls="--", lw=0.8, label="L_z = 0")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("L_z  (kg·m²/s)")
ax2.set_title("Total Angular Momentum Conservation – Basic MPM 3D, Dual Rotation")
ax2.legend(); ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(OUT_LZ, dpi=120)
plt.close(fig2)
print(f"[basic3d-dual-rot]  L_z plot saved → {OUT_LZ}")
print(f"[basic3d-dual-rot]  L_z: mean={Lz_arr.mean():.4e}  std={Lz_arr.std():.4e}  "
      f"max|L_z|={np.abs(Lz_arr).max():.4e}")