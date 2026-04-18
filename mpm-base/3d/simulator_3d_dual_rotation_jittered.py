# Dual Rotation – Basic MPM 3D  (Taichi / PIC)  ── Jittered sampling variant
# =============================================================================
# IDENTICAL to simulator_3d_dual_rotation.py in every way EXCEPT:
#   particle initialisation uses per-cell random jitter (seed=42) instead of
#   the regular ±0.25 sub-cell grid.
#
# Why: the regular 2×2×2 ±0.25 layout is perfectly anti-symmetric around
# every grid node, so PIC P2G cancels every velocity to zero on the first
# step.  Random jitter breaks that symmetry → cubes actually rotate and
# angular momentum decays visibly (still not conserved – that is the point).
#
# Outputs:
#   output/dual_rotation_jittered_basic3d.gif
#   output/dual_rotation_Lz_jittered_basic3d.png
#   output/dual_rotation_Lz_jittered_basic3d.npz

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import taichi as ti

ti.init(arch=ti.cpu)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUT_GIF = os.path.join(OUT_DIR, "dual_rotation_jittered_basic3d.gif")
OUT_LZ  = os.path.join(OUT_DIR, "dual_rotation_Lz_jittered_basic3d.png")
OUT_NPZ = os.path.join(OUT_DIR, "dual_rotation_Lz_jittered_basic3d.npz")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared parameters (same physical scene as the 64^3 reference) ─────────────
grid_size  = 128
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
OMEGA      = 20.0

particle_vol  = dx**3 / ppc
particle_mass = particle_vol * density

# Keep the physical cube geometry identical to the 64^3 setup.
# In the reference scene each cube spans 9 cells at dx=1/64, i.e. width 9/64 m.
# At dx=1/128 that same physical width corresponds to 18 cells, so we sample
# over 18 cells per axis with 8 particles per cell.
center0_phys = np.array([20.0 / 64.0, 32.0 / 64.0, 32.0 / 64.0], dtype=np.float32)
center1_phys = np.array([44.0 / 64.0, 32.0 / 64.0, 32.0 / 64.0], dtype=np.float32)
cube_width_phys = 9.0 / 64.0
cells_per_axis = int(round(cube_width_phys / dx))
if cells_per_axis < 1:
    raise ValueError("cells_per_axis must be positive")
cell_offsets = np.arange(cells_per_axis, dtype=np.float32) - 0.5 * cells_per_axis
target_particles_per_cube = cells_per_axis ** 3 * ppc

print(f"[jittered-dual-rot]  dx={dx:.4e}  c_s={c_s:.4f}  dt={dt:.4e}")
print(f"[jittered-dual-rot]  cells_per_axis={cells_per_axis}  particles/cube={target_particles_per_cube}")

# ── Particle initialisation – JITTERED (only difference from regular file) ────
def make_rotating_cube(center_phys, omega_z, rng):
    """
    Uniform jittered particles in a cube with the same physical size as the
    64^3 reference setup.  At 128^3 this becomes 18×18×18 cells, 8 particles
    per cell → 46656 total.
    Each particle placed uniformly at random within its cell [0.05, 0.95]*dx
    to avoid hitting cell boundaries.  Seed fixed for reproducibility.
    Velocity: v = ω × r,  vx = −ω*ry,  vy = +ω*rx,  vz = 0
    """
    pos, vel = [], []
    cx, cy, cz = [float(c) for c in center_phys]
    center_cell = center_phys / dx
    for i in cell_offsets:
        for j in cell_offsets:
            for k in cell_offsets:
                for _ in range(ppc):
                    # random position inside cell, 5% margin from boundaries
                    px = (center_cell[0] + i + rng.uniform(0.05, 0.95)) * dx
                    py = (center_cell[1] + j + rng.uniform(0.05, 0.95)) * dx
                    pz = (center_cell[2] + k + rng.uniform(0.05, 0.95)) * dx
                    rx = px - cx
                    ry = py - cy
                    vx = -omega_z * ry
                    vy = +omega_z * rx
                    pos.append([px, py, pz])
                    vel.append([vx, vy, 0.0])
    pos_arr = np.array(pos, dtype=np.float32)
    vel_arr = np.array(vel, dtype=np.float32)
    # Remove net linear momentum caused by asymmetric jitter positions:
    # jitter breaks Σ(ry)=0, giving a nonzero Σ(vx) that drifts the cube.
    vel_arr -= vel_arr.mean(axis=0)
    return pos_arr, vel_arr


rng = np.random.default_rng(42)
pos0, vel0 = make_rotating_cube(center0_phys, +OMEGA, rng)
pos1, vel1 = make_rotating_cube(center1_phys, -OMEGA, rng)
n0 = len(pos0)

domain_center = np.array([0.5, 0.5, 0.5])
r0 = pos0 - domain_center;  r1 = pos1 - domain_center
Lz0_init = particle_mass * float(np.sum(r0[:, 0] * vel0[:, 1] - r0[:, 1] * vel0[:, 0]))
Lz1_init = particle_mass * float(np.sum(r1[:, 0] * vel1[:, 1] - r1[:, 1] * vel1[:, 0]))
print(f"[jittered-dual-rot]  Cube0 init L_z = {Lz0_init:.6e}")
print(f"[jittered-dual-rot]  Cube1 init L_z = {Lz1_init:.6e}")
print(f"[jittered-dual-rot]  Total init L_z = {Lz0_init + Lz1_init:.6e}  (should be ~0)")

all_pos = np.concatenate([pos0, pos1], axis=0)
all_vel = np.concatenate([vel0, vel1], axis=0)
N = len(all_pos)
print(f"[jittered-dual-rot]  Total particles: {N}")


# ── Taichi fields (identical to original) ─────────────────────────────────────
x      = ti.Vector.field(3, float, N)
v      = ti.Vector.field(3, float, N)
F_f    = ti.Matrix.field(3, 3, float, N)
vol_f  = ti.field(float, N)
m_f    = ti.field(float, N)

grid_m = ti.field(float, (grid_size, grid_size, grid_size))
grid_v = ti.Vector.field(3, float, (grid_size, grid_size, grid_size))

x.from_numpy(all_pos)
v.from_numpy(all_vel)
vol_f.fill(particle_vol)
m_f.fill(particle_mass)
F_f.from_numpy(np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)))


# ── Fixed Corotated PK1 (unchanged) ──────────────────────────────────────────
@ti.func
def fixed_corotated_pk1(Fp):
    U, sig, V = ti.svd(Fp)
    R   = U @ V.transpose()
    J   = Fp.determinant()
    Fit = Fp.inverse().transpose()
    return 2.0 * mu * (Fp - R) + lam * (J - 1.0) * J * Fit


@ti.kernel
def reset_grid():
    for i, j, k in grid_m:
        grid_m[i, j, k] = 0.0
        grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])


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
                        dw[i][0] / dx * w[j][1]        * w[k][2],
                        w[i][0]       * dw[j][1] / dx  * w[k][2],
                        w[i][0]       * w[j][1]        * dw[k][2] / dx,
                    ])
                    node = base + ti.Vector([i, j, k])
                    grid_m[node[0], node[1], node[2]] += weight * m_f[p]
                    grid_v[node[0], node[1], node[2]] += weight * m_f[p] * v[p]
                    fi = -vol_f[p] * P @ F_f[p].transpose() @ grad_w
                    grid_v[node[0], node[1], node[2]] += dt * fi


@ti.kernel
def update_grid():
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0.0:
            grid_v[i, j, k] /= grid_m[i, j, k]
            if (i <= 3 or i >= grid_size - 3 or
                j <= 3 or j >= grid_size - 3 or
                k <= 3 or k >= grid_size - 3):
                grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])


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
                        dw[i][0] / dx * w[j][1]        * w[k][2],
                        w[i][0]       * dw[j][1] / dx  * w[k][2],
                        w[i][0]       * w[j][1]        * dw[k][2] / dx,
                    ])
                    node = base + ti.Vector([i, j, k])
                    gv     = grid_v[node[0], node[1], node[2]]
                    new_v  += weight * gv
                    grad_v += gv.outer_product(grad_w)
        v[p]   = new_v
        F_f[p] = (ti.Matrix.identity(float, 3) + dt * grad_v) @ F_f[p]


@ti.kernel
def advect():
    for p in range(N):
        x[p] += dt * v[p]


def step():
    reset_grid()
    p2g()
    update_grid()
    g2p()
    advect()


# ── Simulation loop ───────────────────────────────────────────────────────────
steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
total_frames    = int(round(total_time * fps))
print(f"[jittered-dual-rot]  steps_per_frame={steps_per_frame}  total_frames={total_frames}")

frames_0    = []
frames_1    = []
Lz0_history = []
Lz1_history = []

def record():
    pts  = x.to_numpy()
    vels = v.to_numpy()
    frames_0.append(pts[:n0].copy())
    frames_1.append(pts[n0:].copy())
    r0  = pts[:n0] - domain_center
    r1  = pts[n0:] - domain_center
    Lz0 = float(particle_mass * np.sum(r0[:, 0] * vels[:n0, 1] - r0[:, 1] * vels[:n0, 0]))
    Lz1 = float(particle_mass * np.sum(r1[:, 0] * vels[n0:, 1] - r1[:, 1] * vels[n0:, 0]))
    Lz0_history.append(Lz0)
    Lz1_history.append(Lz1)

record()

for frame in range(1, total_frames + 1):
    for _ in range(steps_per_frame):
        step()
    record()
    if frame % 12 == 0:
        Lz_tot = Lz0_history[-1] + Lz1_history[-1]
        print(f"  Frame {frame:4d}/{total_frames}  "
              f"t={frame / fps:.3f}s  "
              f"Lz0={Lz0_history[-1]:.4e}  Lz1={Lz1_history[-1]:.4e}  "
              f"Lz_tot={Lz_tot:.4e}")

print(f"[jittered-dual-rot]  simulation done, rendering …")


# ── 3-D GIF ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 7.0))
ax  = fig.add_subplot(111, projection="3d")
ax.set_xlim(0.25, 0.75); ax.set_ylim(0.25, 0.75); ax.set_zlim(0.25, 0.75)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.set_title("Dual Rotation – Basic MPM 3D (Jittered Sampling)")
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
print(f"[jittered-dual-rot]  GIF saved → {OUT_GIF}")


# ── Angular momentum plot ─────────────────────────────────────────────────────
times   = np.arange(len(Lz0_history)) / fps
Lz0_arr = np.array(Lz0_history)
Lz1_arr = np.array(Lz1_history)
Lz_tot  = Lz0_arr + Lz1_arr

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(times, Lz0_arr, lw=1.5, color="#e74c3c", label="Cube 0 (+ω)")
ax2.plot(times, Lz1_arr, lw=1.5, color="#f39c12", label="Cube 1 (−ω)")
ax2.plot(times, Lz_tot,  lw=1.5, color="#2980b9", label="Total")
ax2.axhline(0, color="k", ls="--", lw=0.8)
drift = float(np.max(np.abs(Lz_tot)))
ax2.text(0.97, 0.05, f"|ΔL_total|_max = {drift:.2e} kg·m²/s",
         transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=8, color="#2980b9",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2980b9", alpha=0.7))
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("L_z  (kg·m²/s)")
ax2.set_title("Angular Momentum Conservation – Basic MPM 3D (Jittered), Dual Rotation")
ax2.legend(); ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(OUT_LZ, dpi=120)
plt.close(fig2)
print(f"[jittered-dual-rot]  L_z plot saved → {OUT_LZ}")

np.savez(OUT_NPZ, times=times, Lz0=Lz0_arr, Lz1=Lz1_arr, Lz_total=Lz_tot)
print(f"[jittered-dual-rot]  L_z data saved → {OUT_NPZ}")
print(f"[jittered-dual-rot]  Lz0: init={Lz0_arr[0]:.4e}  final={Lz0_arr[-1]:.4e}  "
      f"decay={100*(1-abs(Lz0_arr[-1]/Lz0_arr[0])):.1f}%")
