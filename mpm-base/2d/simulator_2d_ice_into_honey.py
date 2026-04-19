"""
2D Ice Drop Benchmark — Honey variant — Basic MPM (Taichi / PIC)
================================================================

Drop the same ice block into a honey pool and compare solver behaviour
across MPM kernels.  Honey is modelled as a weakly-compressible fluid
(Tait EOS, identical bulk to water) with strong grid-level viscous
damping that approximates a high-viscosity Newtonian fluid.

Physics note
------------
  A simple per-step grid damping is equivalent to an implicit first-order
  drag: v_new = v_grid * (1 - alpha).  The effective kinematic viscosity
  scales as nu_eff ~ alpha * dx^2 / (2 * dt).  With alpha=0.08,
  dx=1/160, dt~9e-5 s  =>  nu_eff ~ 13 m²/s, far larger than water's
  ~1e-6 m²/s and in the range of thick fluids.

Geometry (identical to water variant):
  - Domain      : [0, 1] x [0, 1]
  - Container   : rigid box, x in [0.25, 0.75], floor at y = 0.10
  - Honey pool  : x in [0.28, 0.72], y in [0.10, 0.32]
  - Ice block   : square, side 0.08 m, centre (0.50, 0.62),
                  v0 = (0, -1.2) m/s

Output:
  - output/ice_into_honey_2d_basic.gif
  - output/ice_into_honey_2d_basic_com.png
  - output/ice_into_honey_2d_basic_metrics.npz
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import taichi as ti

ti.init(arch=ti.cpu)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUT_GIF = os.path.join(OUT_DIR, "ice_into_honey_2d_basic.gif")
OUT_COM = os.path.join(OUT_DIR, "ice_into_honey_2d_basic_com.png")
OUT_NPZ = os.path.join(OUT_DIR, "ice_into_honey_2d_basic_metrics.npz")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Simulation constants ──────────────────────────────────────────────────────
GRID_SIZE = 160
DX        = 1.0 / GRID_SIZE
PPC       = 4          # 2×2 sub-cell
FPS       = 48
TOTAL_TIME = 2.5
GRAVITY_Y = -9.8
CFL       = 0.5  # match CKMPM tasty_meal_water (kCfl_ = 0.5)

# ── Geometry ──────────────────────────────────────────────────────────────────
BOX_X0, BOX_X1 = 0.25, 0.75
BOX_Y0          = 0.10
HONEY_X0, HONEY_X1 = 0.28, 0.72
HONEY_Y0, HONEY_Y1 = 0.10, 0.32
ICE_CX, ICE_CY  = 0.50, 0.62
ICE_HALF         = 0.04
ICE_V0           = (0.0, -1.2)

# ── Material IDs ──────────────────────────────────────────────────────────────
MAT_HONEY = 0
MAT_ICE   = 1

# ── Honey (viscous fluid): Tait EOS + grid damping ───────────────────────────
HONEY_RHO   = 1400.0   # kg/m³  (real honey ≈ 1400)
HONEY_K     = 5e4      # Tait bulk modulus (Pa)  — match CKMPM kWaterBulk_
HONEY_GAMMA = 7.15     # Tait exponent          — match CKMPM kWaterGamma_
# Per-step velocity damping factor applied ONLY to honey-dominant grid cells.
# Decay time-constant τ ≈ dt / (1 − factor).  With dt ≈ 2e-5 s:
#   factor = 0.99   →  τ ≈ 2 ms       — looks like cold honey, very thick
#   factor = 0.998  →  τ ≈ 10 ms      — warm honey / syrup
#   factor = 0.9998 →  τ ≈ 0.1 s      — barely viscous
HONEY_VIS_FACTOR = 0.99

# ── Ice (stiff elastic) — match CKMPM kIce* ──────────────────────────────────
# (the previous E=2e5 was ~50× too soft; ice would itself collapse on impact
#  and inject huge grad_v into the fluid, breaking Tait stability.)
ICE_RHO = 900.0
ICE_E   = 1e7
ICE_NU  = 0.40
ICE_MU  = ICE_E / (2.0 * (1.0 + ICE_NU))
ICE_LAM = ICE_E * ICE_NU / ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU))

C_S_ICE    = ((ICE_E * (1.0 - ICE_NU)) /
              ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU) * ICE_RHO)) ** 0.5
DT         = CFL * DX / C_S_ICE
PARTICLE_VOL = DX ** 2 / PPC

print(f"[ice2d-honey/basic] dx={DX:.4e}  dt={DT:.4e}  grid={GRID_SIZE}")
print(f"[ice2d-honey/basic] HONEY_VIS_FACTOR={HONEY_VIS_FACTOR}  "
      f"nu_eff~{(1-HONEY_VIS_FACTOR)*DX**2/(2*DT):.2f} m²/s")


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


honey_pos = make_rect(HONEY_X0, HONEY_X1, HONEY_Y0, HONEY_Y1)
ice_pos   = make_rect(ICE_CX - ICE_HALF, ICE_CX + ICE_HALF,
                      ICE_CY - ICE_HALF, ICE_CY + ICE_HALF)

n_honey = len(honey_pos)
n_ice   = len(ice_pos)
N       = n_honey + n_ice
print(f"[ice2d-honey/basic] honey={n_honey}  ice={n_ice}  total={N}")

all_pos = np.concatenate([honey_pos, ice_pos], axis=0)
all_vel = np.zeros((N, 2), dtype=np.float32)
all_vel[n_honey:, 0] = ICE_V0[0]
all_vel[n_honey:, 1] = ICE_V0[1]

mat_id_np      = np.empty(N, dtype=np.int32)
mat_id_np[:n_honey] = MAT_HONEY
mat_id_np[n_honey:] = MAT_ICE

mass_np        = np.empty(N, dtype=np.float32)
mass_np[:n_honey] = PARTICLE_VOL * HONEY_RHO
mass_np[n_honey:] = PARTICLE_VOL * ICE_RHO


# ── Taichi fields ─────────────────────────────────────────────────────────────
x      = ti.Vector.field(2, float, N)
v      = ti.Vector.field(2, float, N)
F_f    = ti.Matrix.field(2, 2, float, N)
mat_id = ti.field(ti.i32, N)
m_p    = ti.field(float, N)
vol_p  = ti.field(float, N)

grid_m       = ti.field(float, (GRID_SIZE, GRID_SIZE))
grid_m_honey = ti.field(float, (GRID_SIZE, GRID_SIZE))  # for per-cell honey detection
grid_v       = ti.Vector.field(2, float, (GRID_SIZE, GRID_SIZE))

x.from_numpy(all_pos)
v.from_numpy(all_vel)
F_f.from_numpy(np.tile(np.eye(2, dtype=np.float32), (N, 1, 1)))
mat_id.from_numpy(mat_id_np)
m_p.from_numpy(mass_np)
vol_p.fill(PARTICLE_VOL)


# ── Constitutive models ───────────────────────────────────────────────────────
@ti.func
def corotated_pk1(Fp, lam, mu):
    U, sig, V = ti.svd(Fp)
    R   = U @ V.transpose()
    J   = Fp.determinant()
    Fit = Fp.inverse().transpose()
    return 2.0 * mu * (Fp - R) + lam * (J - 1.0) * J * Fit


@ti.func
def fluid_pk1_tait(Fp, K, gamma):
    J = Fp.determinant()
    if J < 0.05:   # prevent near-singular / over-compressed
        J = 0.05
    if J > 5.0:    # prevent explosive expansion
        J = 5.0
    Fit = Fp.inverse().transpose()
    return K * (1.0 - ti.pow(J, -gamma)) * J * Fit


@ti.func
def particle_pk1(p):
    Fp = F_f[p]
    P = fluid_pk1_tait(Fp, HONEY_K, HONEY_GAMMA)
    if mat_id[p] == MAT_ICE:
        P = corotated_pk1(Fp, ICE_LAM, ICE_MU)
    return P


# ── Simulation kernels ────────────────────────────────────────────────────────
@ti.kernel
def reset_grid():
    for i, j in grid_m:
        grid_m[i, j]       = 0.0
        grid_m_honey[i, j] = 0.0
        grid_v[i, j]       = ti.Vector([0.0, 0.0])


@ti.kernel
def p2g():
    for p in range(N):
        base = (x[p] / DX - 0.5).cast(int)
        fx   = x[p] / DX - base.cast(float)
        w    = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw   = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]
        P    = particle_pk1(p)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                weight = w[i][0] * w[j][1]
                grad_w = ti.Vector([
                    dw[i][0] / DX * w[j][1],
                    w[i][0]  * dw[j][1] / DX,
                ])
                node = base + ti.Vector([i, j])
                if 0 <= node[0] < GRID_SIZE and 0 <= node[1] < GRID_SIZE:
                    mw = weight * m_p[p]
                    grid_m[node[0], node[1]] += mw
                    if mat_id[p] == MAT_HONEY:
                        grid_m_honey[node[0], node[1]] += mw
                    grid_v[node[0], node[1]] += mw * v[p]
                    fi = -vol_p[p] * P @ F_f[p].transpose() @ grad_w
                    grid_v[node[0], node[1]] += DT * fi


@ti.kernel
def update_grid():
    for i, j in grid_m:
        if grid_m[i, j] > 0.0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y += DT * GRAVITY_Y

            # ── Viscous damping — only on honey-dominant cells ────────────────
            # Previously this damped EVERY active grid cell (including the ice
            # path through air), which killed ice freefall before it could
            # reach the honey surface. Gating by honey-mass fraction restores
            # the freefall while still slowing motion inside the honey.
            if grid_m_honey[i, j] > 0.5 * grid_m[i, j]:
                grid_v[i, j] *= HONEY_VIS_FACTOR

            # ── Container walls (one-sided reflective) ────────────────────────
            # BC band must cover the 3-cell quadratic stencil read by a
            # particle pinned at advect()'s 1.5*DX position clamp; 2*DX is
            # enough. Otherwise particles get pinned at the position
            # clamp while their interpolated grid velocity keeps
            # accumulating gravity → fake "free fall after rest".
            x_node = i * DX
            y_node = j * DX
            if x_node < BOX_X0 + 2.0 * DX and grid_v[i, j].x < 0.0:
                grid_v[i, j].x = 0.0
            if x_node > BOX_X1 - 2.0 * DX and grid_v[i, j].x > 0.0:
                grid_v[i, j].x = 0.0
            if y_node < BOX_Y0 + 2.0 * DX and grid_v[i, j].y < 0.0:
                grid_v[i, j].y = 0.0

            # ── Outer domain guard ────────────────────────────────────────────
            if i <= 2 or i >= GRID_SIZE - 3 or j <= 2 or j >= GRID_SIZE - 3:
                grid_v[i, j] = ti.Vector([0.0, 0.0])


@ti.kernel
def g2p():
    for p in range(N):
        base  = (x[p] / DX - 0.5).cast(int)
        fx    = x[p] / DX - base.cast(float)
        w     = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw    = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]
        new_v = ti.Vector.zero(float, 2)
        grad_v = ti.Matrix.zero(float, 2, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                weight = w[i][0] * w[j][1]
                grad_w = ti.Vector([
                    dw[i][0] / DX * w[j][1],
                    w[i][0]  * dw[j][1] / DX,
                ])
                node = base + ti.Vector([i, j])
                if 0 <= node[0] < GRID_SIZE and 0 <= node[1] < GRID_SIZE:
                    gv      = grid_v[node[0], node[1]]
                    new_v  += weight * gv
                    grad_v += gv.outer_product(grad_w)
        v[p] = new_v
        F_new = (ti.Matrix.identity(float, 2) + DT * grad_v) @ F_f[p]
        if mat_id[p] == MAT_HONEY:
            # Isotropic fluid: reset F to sqrtJ*I, preserving det(F) = J
            J_det = F_new.determinant()
            if J_det < 0.05:
                J_det = 0.05
            if J_det > 5.0:
                J_det = 5.0
            sqrtJ = ti.sqrt(J_det)
            F_f[p] = ti.Matrix([[sqrtJ, 0.0], [0.0, sqrtJ]])
        else:  # MAT_ICE
            F_f[p] = F_new


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


# ── Run ───────────────────────────────────────────────────────────────────────
def run_simulation():
    frames_honey = []
    frames_ice   = []
    ice_com_y    = []
    ice_vy       = []

    target_frame_dt = 1.0 / FPS
    next_frame_time = 0.0
    sim_time        = 0.0

    def record():
        pos = x.to_numpy()
        vel = v.to_numpy()
        frames_honey.append(pos[:n_honey].copy())
        frames_ice.append(pos[n_honey:].copy())
        ice_com_y.append(float(pos[n_honey:, 1].mean()))
        ice_vy.append(float(vel[n_honey:, 1].mean()))

    record()
    next_frame_time += target_frame_dt

    while sim_time < TOTAL_TIME - 1e-12:
        step()
        sim_time += DT
        if sim_time + 1e-12 >= next_frame_time:
            record()
            next_frame_time += target_frame_dt

    return frames_honey, frames_ice, np.array(ice_com_y), np.array(ice_vy)


# ── Render ────────────────────────────────────────────────────────────────────
def render_gif(frames_honey, frames_ice):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.02, 0.90)
    ax.set_aspect("equal")
    ax.set_title("2D Ice Into Honey — Basic MPM")
    ax.plot([BOX_X0, BOX_X0], [BOX_Y0, 0.85], color="black", lw=2)
    ax.plot([BOX_X1, BOX_X1], [BOX_Y0, 0.85], color="black", lw=2)
    ax.plot([BOX_X0, BOX_X1], [BOX_Y0, BOX_Y0], color="black", lw=2)

    scat_h = ax.scatter([], [], s=3, c="#f6c90e", alpha=0.8, label="Honey")
    scat_i = ax.scatter([], [], s=6, c="#dfe6e9",
                        edgecolors="#34495e", linewidths=0.2, label="Ice")
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def update(fid):
        scat_h.set_offsets(frames_honey[fid])
        scat_i.set_offsets(frames_ice[fid])
        txt.set_text(f"t = {fid / FPS:.2f} s")
        return scat_h, scat_i, txt

    ani = animation.FuncAnimation(fig, update, frames=len(frames_honey),
                                  interval=1000 // FPS, blit=False)
    ani.save(OUT_GIF, writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)


def plot_metrics(ice_com_y, ice_vy):
    times = np.arange(len(ice_com_y)) / FPS
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(times, ice_com_y, color="#c0392b", lw=1.8)
    axes[0].set_ylabel("Ice COM y (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("2D Ice Into Honey — Ice Metrics")
    axes[1].plot(times, ice_vy, color="#2980b9", lw=1.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Ice vy (m/s)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_COM, dpi=140)
    plt.close(fig)
    np.savez(OUT_NPZ, times=times, ice_com_y=ice_com_y, ice_vy=ice_vy)


if __name__ == "__main__":
    frames_honey, frames_ice, ice_com_y, ice_vy = run_simulation()
    print(f"[ice2d-honey/basic] frames={len(frames_honey)}  "
          f"final_com_y={ice_com_y[-1]:.4f}")
    render_gif(frames_honey, frames_ice)
    plot_metrics(ice_com_y, ice_vy)
    print(f"[ice2d-honey/basic] GIF     → {OUT_GIF}")
    print(f"[ice2d-honey/basic] metrics → {OUT_NPZ}")
