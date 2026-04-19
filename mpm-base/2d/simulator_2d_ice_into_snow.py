"""
2D Ice Drop Benchmark — Snow variant — Basic MPM (Taichi / PIC)
===============================================================

Drop the same ice block into a snow bed and compare solver behaviour
across MPM kernels.  Snow is modelled with the Stomakhin et al. 2013
elastoplastic model:

  • Elastic part : Fixed-corotated stress with Lamé parameters
                   that harden with plastic compaction (Jp).
  • Plastic part : SVD clamping of singular values to
                   [1 - theta_c, 1 + theta_s] — the yield surface.
  • Hardening    : mu(Jp) = mu0 * exp(xi*(1-Jp)),
                   lambda(Jp) = lam0 * exp(xi*(1-Jp))

Key observable differences from water/honey:
  - Ice decelerates abruptly (impact on stiff granular bed)
  - Snow particles form a permanent crater + side piles (plasticity)
  - Ice may partially rebound in a stiff snow bed

Geometry (identical to other variants):
  - Domain      : [0, 1] × [0, 1]
  - Container   : rigid box, x in [0.25, 0.75], floor at y = 0.10
  - Snow bed    : x in [0.28, 0.72],  y in [0.10, 0.32]
  - Ice block   : square, side 0.08 m, centre (0.50, 0.62),
                  v0 = (0, -1.2) m/s

Output:
  - output/ice_into_snow_2d_basic.gif
  - output/ice_into_snow_2d_basic_com.png
  - output/ice_into_snow_2d_basic_metrics.npz   (includes crater_depth)
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
OUT_GIF = os.path.join(OUT_DIR, "ice_into_snow_2d_basic.gif")
OUT_COM = os.path.join(OUT_DIR, "ice_into_snow_2d_basic_com.png")
OUT_NPZ = os.path.join(OUT_DIR, "ice_into_snow_2d_basic_metrics.npz")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Simulation constants ──────────────────────────────────────────────────────
GRID_SIZE  = 160
DX         = 1.0 / GRID_SIZE
PPC        = 4            # 2×2 sub-cell
FPS        = 48
TOTAL_TIME = 2.5
GRAVITY_Y  = -9.8
CFL        = 0.5  # match CKMPM tasty_meal_water (kCfl_ = 0.5)

# ── Geometry ──────────────────────────────────────────────────────────────────
BOX_X0, BOX_X1   = 0.25, 0.75
BOX_Y0            = 0.10
SNOW_X0, SNOW_X1  = 0.28, 0.72
SNOW_Y0, SNOW_Y1  = 0.10, 0.32
ICE_CX, ICE_CY    = 0.50, 0.62
ICE_HALF           = 0.04
ICE_V0             = (0.0, -1.2)

# ── Material IDs ──────────────────────────────────────────────────────────────
MAT_SNOW = 0
MAT_ICE  = 1

# ── Snow parameters (Stomakhin 2013) ──────────────────────────────────────────
SNOW_RHO     = 400.0     # kg/m³   (powder snow)
SNOW_E       = 1.4e5     # Pa      (Young's modulus)
SNOW_NU      = 0.2
SNOW_MU0     = SNOW_E / (2.0 * (1.0 + SNOW_NU))
SNOW_LAM0    = SNOW_E * SNOW_NU / ((1.0 + SNOW_NU) * (1.0 - 2.0 * SNOW_NU))
SNOW_THETA_C = 0.025     # critical compression (yield onset under compression)
SNOW_THETA_S = 0.0075    # critical stretch     (yield onset under tension)
SNOW_XI      = 10.0      # hardening coefficient

# ── Ice parameters — match CKMPM kIce* ──────────────────────────────────────
# Stiff Fixed Corotated. The previous E=2e5 was ~50× too soft; the ice would
# itself collapse on impact instead of pushing into the snow bed.
ICE_RHO = 900.0
ICE_E   = 1e7
ICE_NU  = 0.40
ICE_MU  = ICE_E / (2.0 * (1.0 + ICE_NU))
ICE_LAM = ICE_E * ICE_NU / ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU))

# timestep driven by stiffest material — here ice (SNOW_E < ICE_E)
C_S_ICE     = ((ICE_E * (1.0 - ICE_NU)) /
               ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU) * ICE_RHO)) ** 0.5
DT          = CFL * DX / C_S_ICE
PARTICLE_VOL = DX ** 2 / PPC

print(f"[ice2d-snow/basic] dx={DX:.4e}  dt={DT:.4e}  grid={GRID_SIZE}")
print(f"[ice2d-snow/basic] snow: E={SNOW_E:.1e}  nu={SNOW_NU}  "
      f"theta_c={SNOW_THETA_C}  theta_s={SNOW_THETA_S}  xi={SNOW_XI}")


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


snow_pos = make_rect(SNOW_X0, SNOW_X1, SNOW_Y0, SNOW_Y1)
ice_pos  = make_rect(ICE_CX - ICE_HALF, ICE_CX + ICE_HALF,
                     ICE_CY - ICE_HALF, ICE_CY + ICE_HALF)

n_snow = len(snow_pos)
n_ice  = len(ice_pos)
N      = n_snow + n_ice
print(f"[ice2d-snow/basic] snow={n_snow}  ice={n_ice}  total={N}")

all_pos = np.concatenate([snow_pos, ice_pos], axis=0)
all_vel = np.zeros((N, 2), dtype=np.float32)
all_vel[n_snow:, 0] = ICE_V0[0]
all_vel[n_snow:, 1] = ICE_V0[1]

mat_id_np         = np.empty(N, dtype=np.int32)
mat_id_np[:n_snow] = MAT_SNOW
mat_id_np[n_snow:] = MAT_ICE

mass_np          = np.empty(N, dtype=np.float32)
mass_np[:n_snow]  = PARTICLE_VOL * SNOW_RHO
mass_np[n_snow:]  = PARTICLE_VOL * ICE_RHO

# initial plastic volume ratio = 1 (undeformed)
Jp_init = np.ones(N, dtype=np.float32)


# ── Taichi fields ─────────────────────────────────────────────────────────────
x      = ti.Vector.field(2, float, N)
v      = ti.Vector.field(2, float, N)
F_f    = ti.Matrix.field(2, 2, float, N)   # elastic deformation gradient
Jp     = ti.field(float, N)                # plastic volume ratio (snow only)
mat_id = ti.field(ti.i32, N)
m_p    = ti.field(float, N)
vol_p  = ti.field(float, N)

grid_m = ti.field(float, (GRID_SIZE, GRID_SIZE))
grid_v = ti.Vector.field(2, float, (GRID_SIZE, GRID_SIZE))

x.from_numpy(all_pos)
v.from_numpy(all_vel)
F_f.from_numpy(np.tile(np.eye(2, dtype=np.float32), (N, 1, 1)))
Jp.from_numpy(Jp_init)
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
def snow_pk1(F_e, Jp_p):
    """Corotated PK1 with exponential hardening from Stomakhin 2013."""
    hardening = ti.exp(SNOW_XI * (1.0 - Jp_p))
    mu  = SNOW_MU0  * hardening
    lam = SNOW_LAM0 * hardening
    return corotated_pk1(F_e, lam, mu)


@ti.func
def particle_pk1(p):
    Fp = F_f[p]
    P = snow_pk1(Fp, Jp[p])
    if mat_id[p] == MAT_ICE:
        P = corotated_pk1(Fp, ICE_LAM, ICE_MU)
    return P


# ── Simulation kernels ────────────────────────────────────────────────────────
@ti.kernel
def reset_grid():
    for i, j in grid_m:
        grid_m[i, j] = 0.0
        grid_v[i, j] = ti.Vector([0.0, 0.0])


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
                    grid_m[node[0], node[1]] += weight * m_p[p]
                    grid_v[node[0], node[1]] += weight * m_p[p] * v[p]
                    fi = -vol_p[p] * P @ F_f[p].transpose() @ grad_w
                    grid_v[node[0], node[1]] += DT * fi


@ti.kernel
def update_grid():
    for i, j in grid_m:
        if grid_m[i, j] > 0.0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y += DT * GRAVITY_Y

            # container walls (one-sided reflective)
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

            # outer domain guard
            if i <= 2 or i >= GRID_SIZE - 3 or j <= 2 or j >= GRID_SIZE - 3:
                grid_v[i, j] = ti.Vector([0.0, 0.0])


@ti.kernel
def g2p():
    for p in range(N):
        base   = (x[p] / DX - 0.5).cast(int)
        fx     = x[p] / DX - base.cast(float)
        w      = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw     = [fx - 1.5, 2.0 * (1.0 - fx), fx - 0.5]
        new_v  = ti.Vector.zero(float, 2)
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
        F_trial = (ti.Matrix.identity(float, 2) + DT * grad_v) @ F_f[p]

        if mat_id[p] == MAT_SNOW:
            # ── Snow plasticity: SVD clamp + Jp hardening (Stomakhin 2013) ──
            U, sig, V = ti.svd(F_trial)
            # Clamp each singular value to the elastic range
            for d in ti.static(range(2)):
                sig_clamped = ti.min(
                    ti.max(sig[d, d], 1.0 - SNOW_THETA_C),
                    1.0 + SNOW_THETA_S
                )
                # Amount of plastic deformation on this axis
                Jp[p] *= sig[d, d] / sig_clamped
                sig[d, d] = sig_clamped
            # Clamp Jp to physically sensible range [0.6, 20]
            if Jp[p] < 0.6:
                Jp[p] = 0.6
            if Jp[p] > 20.0:
                Jp[p] = 20.0
            F_f[p] = U @ sig @ V.transpose()   # elastic F (after return mapping)
        else:
            F_f[p] = F_trial                   # ice: no plasticity


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


# ── Crater-depth helper ───────────────────────────────────────────────────────
def compute_crater_depth(pos_snow, initial_snow_top_y):
    """Return max downward displacement of snow surface below ice impact zone."""
    cx_lo, cx_hi = ICE_CX - ICE_HALF - 0.02, ICE_CX + ICE_HALF + 0.02
    in_col = (pos_snow[:, 0] >= cx_lo) & (pos_snow[:, 0] <= cx_hi)
    if in_col.sum() == 0:
        return 0.0
    min_y = pos_snow[in_col, 1].min()
    return float(max(0.0, initial_snow_top_y - min_y))


# ── Run ───────────────────────────────────────────────────────────────────────
def run_simulation():
    frames_snow  = []
    frames_ice   = []
    ice_com_y    = []
    ice_vy       = []
    crater_depth = []

    # Record initial snow surface (top y of snow column above ice centre)
    initial_top_y = SNOW_Y1

    target_frame_dt = 1.0 / FPS
    next_frame_time = 0.0
    sim_time        = 0.0

    def record():
        pos = x.to_numpy()
        vel = v.to_numpy()
        frames_snow.append(pos[:n_snow].copy())
        frames_ice.append(pos[n_snow:].copy())
        ice_com_y.append(float(pos[n_snow:, 1].mean()))
        ice_vy.append(float(vel[n_snow:, 1].mean()))
        crater_depth.append(compute_crater_depth(pos[:n_snow], initial_top_y))

    record()
    next_frame_time += target_frame_dt

    while sim_time < TOTAL_TIME - 1e-12:
        step()
        sim_time += DT
        if sim_time + 1e-12 >= next_frame_time:
            record()
            next_frame_time += target_frame_dt

    return (frames_snow, frames_ice,
            np.array(ice_com_y), np.array(ice_vy), np.array(crater_depth))


# ── Render ────────────────────────────────────────────────────────────────────
def render_gif(frames_snow, frames_ice):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.02, 0.90)
    ax.set_aspect("equal")
    ax.set_title("2D Ice Into Snow — Basic MPM")
    ax.plot([BOX_X0, BOX_X0], [BOX_Y0, 0.85], color="black", lw=2)
    ax.plot([BOX_X1, BOX_X1], [BOX_Y0, 0.85], color="black", lw=2)
    ax.plot([BOX_X0, BOX_X1], [BOX_Y0, BOX_Y0], color="black", lw=2)

    scat_s = ax.scatter([], [], s=3, c="#b2bec3", alpha=0.7, label="Snow")
    scat_i = ax.scatter([], [], s=6, c="#dfe6e9",
                        edgecolors="#34495e", linewidths=0.2, label="Ice")
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def update(fid):
        scat_s.set_offsets(frames_snow[fid])
        scat_i.set_offsets(frames_ice[fid])
        txt.set_text(f"t = {fid / FPS:.2f} s")
        return scat_s, scat_i, txt

    ani = animation.FuncAnimation(fig, update, frames=len(frames_snow),
                                  interval=1000 // FPS, blit=False)
    ani.save(OUT_GIF, writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)


def plot_metrics(ice_com_y, ice_vy, crater_depth):
    times = np.arange(len(ice_com_y)) / FPS
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(times, ice_com_y, color="#c0392b", lw=1.8)
    axes[0].set_ylabel("Ice COM y (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("2D Ice Into Snow — Ice & Crater Metrics")

    axes[1].plot(times, ice_vy, color="#2980b9", lw=1.8)
    axes[1].set_ylabel("Ice vy (m/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, crater_depth, color="#8e44ad", lw=1.8)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Crater depth (m)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_COM, dpi=140)
    plt.close(fig)
    np.savez(OUT_NPZ, times=times, ice_com_y=ice_com_y,
             ice_vy=ice_vy, crater_depth=crater_depth)


if __name__ == "__main__":
    frames_snow, frames_ice, ice_com_y, ice_vy, crater_depth = run_simulation()
    print(f"[ice2d-snow/basic] frames={len(frames_snow)}  "
          f"final_com_y={ice_com_y[-1]:.4f}  "
          f"final_crater={crater_depth[-1]:.4f} m")
    render_gif(frames_snow, frames_ice)
    plot_metrics(ice_com_y, ice_vy, crater_depth)
    print(f"[ice2d-snow/basic] GIF     → {OUT_GIF}")
    print(f"[ice2d-snow/basic] metrics → {OUT_NPZ}")
