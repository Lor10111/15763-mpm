"""
2D Ice Drop Benchmark — Water variant — Basic MPM (Taichi / PIC)
================================================================

Benchmark purpose:
  Drop the same elastic ice block into a rectangular pool of water and compare
  solver behavior across kernels / transfer schemes. This 2D version is meant
  to be a clean benchmark, not a visually rich "cup" scene.

Scene:
  - Domain      : [0, 1] x [0, 1]
  - Container   : rigid box, x in [0.25, 0.75], floor at y = 0.10
  - Water pool  : x in [0.28, 0.72], y in [0.10, 0.32]
  - Ice block   : square, side 0.08 m, center (0.50, 0.62), v0 = (0, -1.2) m/s

Output:
  - output/ice_into_water_2d_basic.gif
  - output/ice_into_water_2d_basic_com.png
  - output/ice_into_water_2d_basic_metrics.npz
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import taichi as ti

ti.init(arch=ti.gpu)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUT_GIF = os.path.join(OUT_DIR, "ice_into_water_2d_basic.gif")
OUT_COM = os.path.join(OUT_DIR, "ice_into_water_2d_basic_com.png")
OUT_NPZ = os.path.join(OUT_DIR, "ice_into_water_2d_basic_metrics.npz")
os.makedirs(OUT_DIR, exist_ok=True)

# Simulation
GRID_SIZE = 256
DX = 1.0 / GRID_SIZE
PPC = 16  # 4x4 stratified sub-cell (was 4 = 2x2)
FPS = 48
TOTAL_TIME = 2.5
GRAVITY_Y = -9.8
CFL = 0.5  # match CKMPM tasty_meal_water (kCfl_ = 0.5)

# Geometry — tuned to a surface-wave / bobbing regime so PIC's numerical
# dissipation is less visually dominant (see notes in companion MLS-MPM
# version). The original "heavy impact" params (0.04 / 0.62 / v=-1.2 /
# water to 0.32) excited exactly the modes PIC smears worst.
BOX_X0, BOX_X1 = 0.20, 0.80
BOX_Y0 = 0.10
WATER_X0, WATER_X1 = 0.23, 0.77
WATER_Y0, WATER_Y1 = 0.10, 0.38
ICE_CX, ICE_CY = 0.50, 0.44
ICE_HALF = 0.03
ICE_V0 = (0.0, 0.0)

# Materials
MAT_WATER = 0
MAT_ICE = 1

# Water: weakly compressible Tait EOS — match CKMPM kWater* exactly
WATER_RHO = 1000.0
WATER_K = 5e4      # bulk modulus  (CKMPM kWaterBulk_)
WATER_GAMMA = 7.15 # Tait exponent (CKMPM kWaterGamma_)
WATER_VISCO = 0.0  # isolate PIC/APIC transfer difference from explicit fluid viscosity

# Ice: stiff Fixed Corotated — match CKMPM kIce* exactly
# (the previous E=2e5 was ~50× too soft; ice would itself collapse on impact
#  and inject huge grad_v into the water, breaking Tait stability.)
ICE_RHO = 900.0
ICE_E = 1e7
ICE_NU = 0.40
ICE_MU = ICE_E / (2.0 * (1.0 + ICE_NU))
ICE_LAM = ICE_E * ICE_NU / ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU))

# dt picked from the stiffest material (ice). With CFL=0.5 and the new ice
# modulus, dt is well under both the ice and the water Tait CFL bounds, so
# the J update for water no longer needs a tight clamp.
C_S_ICE = ((ICE_E * (1.0 - ICE_NU)) / ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU) * ICE_RHO)) ** 0.5
DT = CFL * DX / C_S_ICE
PARTICLE_VOL = DX ** 2 / PPC

print(f"[ice2d-water/basic] dx={DX:.4e} dt={DT:.4e} grid={GRID_SIZE}")


SUB_N = 4  # 4x4 stratified grid -> 16 particles per cell
SUB = [((i + 0.5) / SUB_N, (j + 0.5) / SUB_N)
       for i in range(SUB_N) for j in range(SUB_N)]


def make_rect(x0, x1, y0, y1, seed=0):
    # Stratified 4x4 sub-cell sampling with per-particle jitter inside its
    # sub-cell (half-width = 0.5 / SUB_N). Breaks the lattice-aligned
    # correlation between ice and water particles at the interface.
    rng = np.random.default_rng(seed)
    jitter_half = 0.5 / SUB_N
    pos = []
    i0 = int(x0 / DX) - 1
    i1 = int(x1 / DX) + 2
    j0 = int(y0 / DX) - 1
    j1 = int(y1 / DX) + 2
    for i in range(i0, i1):
        for j in range(j0, j1):
            for ox, oy in SUB:
                jx = rng.uniform(-jitter_half, jitter_half)
                jy = rng.uniform(-jitter_half, jitter_half)
                px = (i + ox + jx) * DX
                py = (j + oy + jy) * DX
                if x0 <= px <= x1 and y0 <= py <= y1:
                    pos.append((px, py))
    return np.asarray(pos, dtype=np.float32)


water_pos = make_rect(WATER_X0, WATER_X1, WATER_Y0, WATER_Y1, seed=1)
ice_pos = make_rect(ICE_CX - ICE_HALF, ICE_CX + ICE_HALF, ICE_CY - ICE_HALF, ICE_CY + ICE_HALF, seed=2)

n_water = len(water_pos)
n_ice = len(ice_pos)
N = n_water + n_ice
print(f"[ice2d-water/basic] water={n_water} ice={n_ice} total={N}")

all_pos = np.concatenate([water_pos, ice_pos], axis=0)
all_vel = np.zeros((N, 2), dtype=np.float32)
all_vel[n_water:, 0] = ICE_V0[0]
all_vel[n_water:, 1] = ICE_V0[1]

mat_id_np = np.empty(N, dtype=np.int32)
mat_id_np[:n_water] = MAT_WATER
mat_id_np[n_water:] = MAT_ICE

mass_np = np.empty(N, dtype=np.float32)
mass_np[:n_water] = PARTICLE_VOL * WATER_RHO
mass_np[n_water:] = PARTICLE_VOL * ICE_RHO

x = ti.Vector.field(2, float, N)
v = ti.Vector.field(2, float, N)
F_f = ti.Matrix.field(2, 2, float, N)
C_f = ti.Matrix.field(2, 2, float, N)
J_f = ti.field(float, N)
mat_id = ti.field(ti.i32, N)
m_p = ti.field(float, N)
vol_p = ti.field(float, N)

grid_m = ti.field(float, (GRID_SIZE, GRID_SIZE))
grid_v = ti.Vector.field(2, float, (GRID_SIZE, GRID_SIZE))

x.from_numpy(all_pos)
v.from_numpy(all_vel)
F_f.from_numpy(np.tile(np.eye(2, dtype=np.float32), (N, 1, 1)))
C_f.fill(0.0)
J_f.fill(1.0)
mat_id.from_numpy(mat_id_np)
m_p.from_numpy(mass_np)
vol_p.fill(PARTICLE_VOL)


@ti.func
def corotated_pk1(Fp, lam, mu):
    U, sig, V = ti.svd(Fp)
    R = U @ V.transpose()
    J = Fp.determinant()
    Fit = Fp.inverse().transpose()
    return 2.0 * mu * (Fp - R) + lam * (J - 1.0) * J * Fit


@ti.func
def fluid_pk1_tait(J, C, K, gamma, viscosity):
    # Match CKMPM ComputeStress<kFluid> (mpm_material.cuh:466-477) and
    # UpdateForce<kFluid>           (mpm_algorithm.cuh:541-547):
    #   pressure = K * (J^-gamma - 1)
    #   PF       = (C + C^T) * viscosity      (deviatoric viscous part)
    #   PF      -= pressure * I               (isotropic pressure)
    #   PF      *= J                          (Kirchhoff form -> P @ F^T)
    # No J clamp on the EOS — the dt is sized so that |dJ| per step is
    # small. A loose safety clamp is applied in g2p() only to guard
    # against NaNs if a particle ever leaves the support.
    pressure = K * (ti.pow(J, -gamma) - 1.0)
    P = (C + C.transpose()) * viscosity
    P[0, 0] -= pressure
    P[1, 1] -= pressure
    return J * P


@ti.func
def particle_pf(p):
    PF = fluid_pk1_tait(J_f[p], C_f[p], WATER_K, WATER_GAMMA, WATER_VISCO)
    if mat_id[p] == MAT_ICE:
        PF = corotated_pk1(F_f[p], ICE_LAM, ICE_MU) @ F_f[p].transpose()
    return PF


@ti.func
def fluid_F_reset_from_J(J):
    sqrtJ = ti.sqrt(ti.max(0.05, ti.min(20.0, J)))
    I = ti.Matrix.identity(float, 2)
    return sqrtJ * I


@ti.kernel
def reset_grid():
    for i, j in grid_m:
        grid_m[i, j] = 0.0
        grid_v[i, j] = ti.Vector([0.0, 0.0])


@ti.kernel
def p2g():
    for p in range(N):
        base = (x[p] / DX - 0.5).cast(int)
        fx = x[p] / DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]
        PF = particle_pf(p)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                weight = w[i][0] * w[j][1]
                grad_w = ti.Vector([
                    dw[i][0] / DX * w[j][1],
                    w[i][0] * dw[j][1] / DX,
                ])
                node = base + ti.Vector([i, j])
                if 0 <= node[0] < GRID_SIZE and 0 <= node[1] < GRID_SIZE:
                    grid_m[node[0], node[1]] += weight * m_p[p]
                    grid_v[node[0], node[1]] += weight * m_p[p] * v[p]
                    fi = -vol_p[p] * PF @ grad_w
                    grid_v[node[0], node[1]] += DT * fi


@ti.kernel
def update_grid():
    for i, j in grid_m:
        if grid_m[i, j] > 0.0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y += DT * GRAVITY_Y

            # container side walls — must cover the stencil that the
            # particle at the position-clamp boundary reads (advect clamps
            # to BOX_Y0 + 1.5*DX; the 3-cell quadratic stencil reaches up
            # to ~2.5*DX above that). Otherwise particles get pinned at
            # the position clamp while their interpolated grid velocity
            # keeps accumulating gravity → fake "free fall after rest".
            x_node = i * DX
            y_node = j * DX
            if x_node < BOX_X0 + 2.0 * DX and grid_v[i, j].x < 0.0:
                grid_v[i, j].x = 0.0
            if x_node > BOX_X1 - 2.0 * DX and grid_v[i, j].x > 0.0:
                grid_v[i, j].x = 0.0
            if y_node < BOX_Y0 + 2.0 * DX and grid_v[i, j].y < 0.0:
                grid_v[i, j].y = 0.0

            # outer domain guard
            if i <= 2 or i >= GRID_SIZE - 3 or j <= 2 or j >= GRID_SIZE - 3:
                grid_v[i, j] = ti.Vector([0.0, 0.0])


@ti.kernel
def g2p():
    for p in range(N):
        base = (x[p] / DX - 0.5).cast(int)
        fx = x[p] / DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]
        new_v = ti.Vector.zero(float, 2)
        grad_v = ti.Matrix.zero(float, 2, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                weight = w[i][0] * w[j][1]
                grad_w = ti.Vector([
                    dw[i][0] / DX * w[j][1],
                    w[i][0] * dw[j][1] / DX,
                ])
                node = base + ti.Vector([i, j])
                if 0 <= node[0] < GRID_SIZE and 0 <= node[1] < GRID_SIZE:
                    gv = grid_v[node[0], node[1]]
                    new_v += weight * gv
                    grad_v += gv.outer_product(grad_w)
        v[p] = new_v
        if mat_id[p] == MAT_WATER:
            # Identical to CKMPM mpm_algorithm.cuh:541
            #   J += trace(C) * dt * J
            # (no clamp). Only a very loose [0.05, 20] safety clamp is
            # applied so that a single bad particle cannot poison the
            # whole frame with a NaN.
            C_f[p] = grad_v
            J_new = J_f[p] + DT * grad_v.trace() * J_f[p]
            J_f[p] = ti.max(0.05, ti.min(20.0, J_new))
            F_f[p] = fluid_F_reset_from_J(J_f[p])
        else:
            F_f[p] = (ti.Matrix.identity(float, 2) + DT * grad_v) @ F_f[p]
            for a, b in ti.static(ti.ndrange(2, 2)):
                F_f[p][a, b] = ti.max(-2.0, ti.min(2.0, F_f[p][a, b]))


@ti.kernel
def advect():
    for p in range(N):
        x[p] += DT * v[p]
        x[p].x = ti.min(ti.max(x[p].x, BOX_X0 + 1.5 * DX), BOX_X1 - 1.5 * DX)
        x[p].y = ti.min(ti.max(x[p].y, BOX_Y0 + 1.5 * DX), 1.0 - 2.0 * DX)


def step():
    reset_grid()
    p2g()
    update_grid()
    g2p()
    advect()


def run_simulation():
    frames_water = []
    frames_ice = []
    ice_com_y = []
    ice_vy = []

    target_frame_dt = 1.0 / FPS
    next_frame_time = 0.0
    sim_time = 0.0

    def record():
        pos = x.to_numpy()
        vel = v.to_numpy()
        frames_water.append(pos[:n_water].copy())
        frames_ice.append(pos[n_water:].copy())
        ice_com_y.append(float(pos[n_water:, 1].mean()))
        ice_vy.append(float(vel[n_water:, 1].mean()))

    record()
    next_frame_time += target_frame_dt

    while sim_time < TOTAL_TIME - 1e-12:
        step()
        sim_time += DT
        if sim_time + 1e-12 >= next_frame_time:
            record()
            next_frame_time += target_frame_dt

    return frames_water, frames_ice, np.array(ice_com_y), np.array(ice_vy)


def render_gif(frames_water, frames_ice):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.02, 0.90)
    ax.set_aspect("equal")
    ax.set_title("2D Ice Into Water - Basic MPM")
    ax.plot([BOX_X0, BOX_X0], [BOX_Y0, 0.85], color="black", lw=2)
    ax.plot([BOX_X1, BOX_X1], [BOX_Y0, 0.85], color="black", lw=2)
    ax.plot([BOX_X0, BOX_X1], [BOX_Y0, BOX_Y0], color="black", lw=2)

    scat_w = ax.scatter([], [], s=3, c="#2e86de", alpha=0.7, label="Water")
    scat_i = ax.scatter([], [], s=6, c="#dfe6e9", edgecolors="#34495e", linewidths=0.2, label="Ice")
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def update(fid):
        pw = frames_water[fid]
        pi = frames_ice[fid]
        scat_w.set_offsets(pw)
        scat_i.set_offsets(pi)
        txt.set_text(f"t = {fid / FPS:.2f} s")
        return scat_w, scat_i, txt

    ani = animation.FuncAnimation(fig, update, frames=len(frames_water), interval=1000 // FPS, blit=False)
    ani.save(OUT_GIF, writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)


def plot_metrics(ice_com_y, ice_vy):
    times = np.arange(len(ice_com_y)) / FPS
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(times, ice_com_y, color="#c0392b", lw=1.8)
    axes[0].set_ylabel("Ice COM y (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("2D Ice Into Water - Ice Metrics")
    axes[1].plot(times, ice_vy, color="#2980b9", lw=1.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Ice vy (m/s)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_COM, dpi=140)
    plt.close(fig)
    np.savez(OUT_NPZ, times=times, ice_com_y=ice_com_y, ice_vy=ice_vy)


if __name__ == "__main__":
    frames_water, frames_ice, ice_com_y, ice_vy = run_simulation()
    print(f"[ice2d-water/basic] frames={len(frames_water)} final_com_y={ice_com_y[-1]:.4f}")
    render_gif(frames_water, frames_ice)
    plot_metrics(ice_com_y, ice_vy)
    print(f"[ice2d-water/basic] GIF → {OUT_GIF}")
    print(f"[ice2d-water/basic] metrics → {OUT_NPZ}")
