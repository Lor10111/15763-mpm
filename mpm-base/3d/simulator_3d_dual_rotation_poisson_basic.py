# Dual Rotation – Basic MPM 3D  (Taichi / PIC, Poisson sampling)
# ==============================================================
# Same physics and transfer scheme as simulator_3d_dual_rotation.py, but the
# particle layout inside each cube is generated with Poisson-disk sampling
# instead of the symmetric 2x2x2 sub-cell pattern.

import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "output")
OUT_GIF = os.path.join(OUT_DIR, "dual_rotation_basic3d_poisson.gif")
OUT_LZ = os.path.join(OUT_DIR, "dual_rotation_Lz_basic3d_poisson.png")
OUT_NPZ = os.path.join(OUT_DIR, "dual_rotation_Lz_basic3d_poisson.npz")
os.makedirs(OUT_DIR, exist_ok=True)

# Shared physical parameters
grid_size = 64
dx = 1.0 / grid_size
E, nu = 1e6, 0.4
density = 1000.0
mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
c_s = ((E * (1.0 - nu)) / ((1.0 + nu) * (1.0 - 2.0 * nu) * density)) ** 0.5
CFL = 0.5
dt = CFL * dx / c_s
ppc = 8
fps = 48
total_time = 5.0
OMEGA = 2.0
SEED = 20260418

particle_vol = dx ** 3 / ppc
particle_mass = particle_vol * density

print(f"[basic3d-dual-rot-poisson]  dx={dx:.4e}  c_s={c_s:.4f}  dt={dt:.4e}")
print(f"[basic3d-dual-rot-poisson]  mu={mu:.4e}  lam={lam:.4e}")
print(f"[basic3d-dual-rot-poisson]  particle_vol={particle_vol:.4e}  particle_mass={particle_mass:.4e}")


def bridson_poisson_box(min_corner, max_corner, min_dist, rng, k=30):
    dims = np.asarray(max_corner, dtype=np.float64) - np.asarray(min_corner, dtype=np.float64)
    cell_size = min_dist / math.sqrt(3.0)
    grid_shape = tuple(int(math.ceil(d / cell_size)) for d in dims)
    grid = -np.ones(grid_shape, dtype=np.int32)

    samples = []
    active = []

    def grid_coords(pt):
        rel = (pt - min_corner) / cell_size
        return tuple(np.floor(rel).astype(np.int32))

    def in_bounds(pt):
        return np.all(pt >= min_corner) and np.all(pt <= max_corner)

    def far_enough(pt):
        gi = grid_coords(pt)
        for ix in range(max(gi[0] - 2, 0), min(gi[0] + 3, grid_shape[0])):
            for iy in range(max(gi[1] - 2, 0), min(gi[1] + 3, grid_shape[1])):
                for iz in range(max(gi[2] - 2, 0), min(gi[2] + 3, grid_shape[2])):
                    idx = grid[ix, iy, iz]
                    if idx >= 0:
                        if np.linalg.norm(samples[idx] - pt) < min_dist:
                            return False
        return True

    first = min_corner + rng.random(3) * dims
    samples.append(first)
    active.append(0)
    grid[grid_coords(first)] = 0

    while active:
        active_i = rng.randrange(len(active))
        base_idx = active[active_i]
        base_pt = samples[base_idx]
        accepted = False

        for _ in range(k):
            direction = rng.normalvariate(0.0, 1.0), rng.normalvariate(0.0, 1.0), rng.normalvariate(0.0, 1.0)
            direction = np.asarray(direction, dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction /= norm
            radius = min_dist * (1.0 + rng.random())
            candidate = base_pt + direction * radius
            if not in_bounds(candidate):
                continue
            if not far_enough(candidate):
                continue

            samples.append(candidate)
            new_idx = len(samples) - 1
            active.append(new_idx)
            grid[grid_coords(candidate)] = new_idx
            accepted = True
            break

        if not accepted:
            active.pop(active_i)

    return np.asarray(samples, dtype=np.float32)


def poisson_cube_points(center, target_count, rng):
    half_extent = 4.25 * dx
    min_corner = np.asarray(center, dtype=np.float64) - half_extent
    max_corner = np.asarray(center, dtype=np.float64) + half_extent
    cube_volume = float((2.0 * half_extent) ** 3)

    # Start from the volume-per-particle estimate and relax slightly until we
    # have enough points, then subsample to keep mass comparable to the regular grid case.
    base_spacing = (cube_volume / target_count) ** (1.0 / 3.0)
    spacing_scale = 0.92
    points = None
    for _ in range(8):
        min_dist = spacing_scale * base_spacing
        points = bridson_poisson_box(min_corner, max_corner, min_dist, rng)
        if len(points) >= target_count:
            break
        spacing_scale *= 0.92

    if points is None or len(points) < target_count:
        raise RuntimeError(
            f"Poisson sampler generated only {0 if points is None else len(points)} "
            f"points, fewer than requested {target_count}."
        )

    if len(points) > target_count:
        choose = np.asarray(rng.sample(range(len(points)), target_count), dtype=np.int32)
        points = points[choose]

    return points


def make_rotating_poisson_cube(cx_cell, cy_cell, cz_cell, omega_z, target_count, rng):
    center = np.array([cx_cell * dx, cy_cell * dx, cz_cell * dx], dtype=np.float32)
    pos = poisson_cube_points(center, target_count, rng)
    rel = pos - center
    vel = np.zeros_like(pos)
    vel[:, 0] = -omega_z * rel[:, 1]
    vel[:, 1] = +omega_z * rel[:, 0]
    return pos.astype(np.float32), vel.astype(np.float32)


target_particles_per_cube = 9 * 9 * 9 * 8
rng = random.Random(SEED)

pos0, vel0 = make_rotating_poisson_cube(20, 32, 32, +OMEGA, target_particles_per_cube, rng)
pos1, vel1 = make_rotating_poisson_cube(44, 32, 32, -OMEGA, target_particles_per_cube, rng)
n0 = len(pos0)

domain_center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
r0 = pos0 - domain_center
r1 = pos1 - domain_center
Lz0 = particle_mass * float(np.sum(r0[:, 0] * vel0[:, 1] - r0[:, 1] * vel0[:, 0]))
Lz1 = particle_mass * float(np.sum(r1[:, 0] * vel1[:, 1] - r1[:, 1] * vel1[:, 0]))
print(f"[basic3d-dual-rot-poisson]  Cube0 init L_z = {Lz0:.6e}")
print(f"[basic3d-dual-rot-poisson]  Cube1 init L_z = {Lz1:.6e}")
print(f"[basic3d-dual-rot-poisson]  Total init L_z = {Lz0 + Lz1:.6e}  (should be ~0)")

all_pos = np.concatenate([pos0, pos1], axis=0)
all_vel = np.concatenate([vel0, vel1], axis=0)
N = len(all_pos)
print(f"[basic3d-dual-rot-poisson]  Total particles: {N}")

x = ti.Vector.field(3, float, N)
v = ti.Vector.field(3, float, N)
F_f = ti.Matrix.field(3, 3, float, N)
vol_f = ti.field(float, N)
m_f = ti.field(float, N)

grid_m = ti.field(float, (grid_size, grid_size, grid_size))
grid_v = ti.Vector.field(3, float, (grid_size, grid_size, grid_size))

color_f = ti.Vector.field(3, float, N)

x.from_numpy(all_pos)
v.from_numpy(all_vel)
vol_f.fill(particle_vol)
m_f.fill(particle_mass)
F_f.from_numpy(np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)))


@ti.kernel
def init_colors(n_first: int):
    for p in range(N):
        if p < n_first:
            color_f[p] = ti.Vector([0.9, 0.2, 0.2])
        else:
            color_f[p] = ti.Vector([0.2, 0.4, 0.9])


init_colors(n0)


@ti.func
def fixed_corotated_pk1(Fp):
    U, sig, V = ti.svd(Fp)
    R = U @ V.transpose()
    J = Fp.determinant()
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
        fx = x[p] / dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]
        P = fixed_corotated_pk1(F_f[p])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    weight = w[i][0] * w[j][1] * w[k][2]
                    grad_w = ti.Vector([
                        dw[i][0] / dx * w[j][1] * w[k][2],
                        w[i][0] * dw[j][1] / dx * w[k][2],
                        w[i][0] * w[j][1] * dw[k][2] / dx,
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
            if (
                i <= 3 or i >= grid_size - 3 or
                j <= 3 or j >= grid_size - 3 or
                k <= 3 or k >= grid_size - 3
            ):
                grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def g2p():
    for p in range(N):
        base = (x[p] / dx - 0.5).cast(int)
        fx = x[p] / dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]
        new_v = ti.Vector.zero(float, 3)
        grad_v = ti.Matrix.zero(float, 3, 3)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    weight = w[i][0] * w[j][1] * w[k][2]
                    grad_w = ti.Vector([
                        dw[i][0] / dx * w[j][1] * w[k][2],
                        w[i][0] * dw[j][1] / dx * w[k][2],
                        w[i][0] * w[j][1] * dw[k][2] / dx,
                    ])
                    node = base + ti.Vector([i, j, k])
                    gv = grid_v[node[0], node[1], node[2]]
                    new_v += weight * gv
                    grad_v += gv.outer_product(grad_w)
        v[p] = new_v
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


steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
total_frames = int(round(total_time * fps))
print(f"[basic3d-dual-rot-poisson]  steps_per_frame={steps_per_frame}  total_frames={total_frames}")

frames_0 = []
frames_1 = []
Lz0_history = []
Lz1_history = []


def record():
    pts = x.to_numpy()
    vels = v.to_numpy()
    frames_0.append(pts[:n0].copy())
    frames_1.append(pts[n0:].copy())
    r0_now = pts[:n0] - domain_center
    r1_now = pts[n0:] - domain_center
    L0 = float(particle_mass * np.sum(r0_now[:, 0] * vels[:n0, 1] - r0_now[:, 1] * vels[:n0, 0]))
    L1 = float(particle_mass * np.sum(r1_now[:, 0] * vels[n0:, 1] - r1_now[:, 1] * vels[n0:, 0]))
    Lz0_history.append(L0)
    Lz1_history.append(L1)


record()

for frame in range(1, total_frames + 1):
    for _ in range(steps_per_frame):
        step()
    record()
    if frame % 12 == 0:
        Lz_tot = Lz0_history[-1] + Lz1_history[-1]
        print(
            f"  Frame {frame:4d}/{total_frames}  "
            f"t={frame / fps:.3f}s  "
            f"Lz0={Lz0_history[-1]:.4e}  Lz1={Lz1_history[-1]:.4e}  "
            f"Lz_tot={Lz_tot:.4e}"
        )

print(f"[basic3d-dual-rot-poisson]  simulation done, rendering …")

fig = plt.figure(figsize=(7.5, 7.0))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(0.25, 0.75)
ax.set_ylim(0.25, 0.75)
ax.set_zlim(0.25, 0.75)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("Dual Rotation – Basic MPM 3D (Poisson sampling)")
ax.set_box_aspect((3, 3, 3))
ax.view_init(elev=20.0, azim=-65.0)

scat0 = ax.scatter([], [], [], s=1, c="#e74c3c", depthshade=False, label="Cube 1 (+ω)")
scat1 = ax.scatter([], [], [], s=1, c="#2e86de", depthshade=False, label="Cube 2 (−ω)")
time_txt = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)
ax.legend(loc="upper right", markerscale=5)


def update(frame_id):
    p0 = frames_0[frame_id]
    p1 = frames_1[frame_id]
    scat0._offsets3d = (p0[:, 0], p0[:, 1], p0[:, 2])
    scat1._offsets3d = (p1[:, 0], p1[:, 1], p1[:, 2])
    time_txt.set_text(f"t = {frame_id / fps:.2f} s")
    return scat0, scat1, time_txt


ani = animation.FuncAnimation(fig, update, frames=len(frames_0), interval=1000 // fps, blit=False)
ani.save(OUT_GIF, writer="pillow", fps=fps, dpi=100)
plt.close(fig)
print(f"[basic3d-dual-rot-poisson]  GIF saved → {OUT_GIF}")

times = np.arange(len(Lz0_history)) / fps
Lz0_arr = np.array(Lz0_history)
Lz1_arr = np.array(Lz1_history)
Lz_tot = Lz0_arr + Lz1_arr

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(times, Lz0_arr, lw=1.5, color="#e74c3c", label="Cube 0 (+ω)")
ax2.plot(times, Lz1_arr, lw=1.5, color="#f39c12", label="Cube 1 (−ω)")
ax2.plot(times, Lz_tot, lw=1.5, color="#2980b9", label="Total")
ax2.axhline(0, color="k", ls="--", lw=0.8)
drift = float(np.max(np.abs(Lz_tot)))
ax2.text(
    0.97,
    0.05,
    f"|ΔL_total|_max = {drift:.2e} kg·m²/s",
    transform=ax2.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    color="#2980b9",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2980b9", alpha=0.7),
)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("L_z  (kg·m²/s)")
ax2.set_title("Angular Momentum Conservation – Basic MPM 3D, Poisson sampling")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(OUT_LZ, dpi=120)
plt.close(fig2)
print(f"[basic3d-dual-rot-poisson]  L_z plot saved → {OUT_LZ}")

np.savez(OUT_NPZ, times=times, Lz0=Lz0_arr, Lz1=Lz1_arr, Lz_total=Lz_tot)
print(f"[basic3d-dual-rot-poisson]  L_z data saved → {OUT_NPZ}")
print(
    f"[basic3d-dual-rot-poisson]  Lz0: init={Lz0_arr[0]:.4e}  final={Lz0_arr[-1]:.4e}  "
    f"decay={100 * (1 - abs(Lz0_arr[-1] / Lz0_arr[0])):.1f}%"
)
