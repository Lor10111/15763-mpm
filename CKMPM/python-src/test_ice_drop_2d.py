"""
2D Ice Drop Benchmark — CKMPM (Taichi port of the compact-kernel + dual
interlocked grid scheme from CKMPM/python-src/python_demo.py)
======================================================================

CKMPM kernel (this file):
  * dual interlocked grid : two sub-grids offset by 0.25*dx
  * compact kernel        : 2-cell support per sub-grid (vs. 3-cell
                            quadratic B-spline used by basic / MLS-MPM)
  * weights               : stencil[i,0] = 1 - r + sin(2π r)/(2π)
                            stencil[i,1] = 1 - stencil[i,0]
  * gradient              : grad = sign(r) · (cos(2π|r|) − 1) · n_grid
  * G2P average           : v *= 0.5,  F = (I + 0.5·dt·grad_v) @ F

Run with:
    python test_ice_drop_2d.py water
    python test_ice_drop_2d.py honey
    python test_ice_drop_2d.py snow

Materials  (parameters match the basic / MLS-MPM 2D versions and CKMPM
            tasty_meal_water for water/ice):
  water  : Tait weakly compressible (K=5e4, γ=7.15, viscosity=0.1, ρ=1000)
  honey  : Tait + per-cell grid damping on honey-dominant cells
  snow   : Stomakhin 2013 corotated + SVD-clamp + exponential hardening
  ice    : Fixed Corotated  (E=1e7, ν=0.4, ρ=900)   — CKMPM kIce*

Geometry (identical to the basic and MLS-MPM 2D versions):
  Domain      : [0,1]²
  Container   : rigid walls at x = 0.25, 0.75; floor y = 0.10
  Medium pool : x ∈ [0.28, 0.72], y ∈ [0.10, 0.32]
  Ice cube    : 0.08 m square, centre (0.5, 0.62), v0 = (0, −1.2) m/s
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f32)

# ── Medium selector ───────────────────────────────────────────────────────────
VALID_MEDIA = ("water", "honey", "snow")
MEDIUM = sys.argv[1].lower() if len(sys.argv) > 1 else "water"
if MEDIUM not in VALID_MEDIA:
    raise ValueError(f"MEDIUM must be one of {VALID_MEDIA}, got '{MEDIUM}'")
print(f"[ice2d/{MEDIUM}/ckmpm] medium={MEDIUM}")

# ── Simulation constants ──────────────────────────────────────────────────────
GRID_SIZE  = 256
DX         = 1.0 / GRID_SIZE
INV_DX     = float(GRID_SIZE)
TWOPI      = 2.0 * np.pi
PPC        = 4
FPS        = 48
TOTAL_TIME = 2.5
GRAVITY_Y  = -9.8
CFL        = 0.5  # match CKMPM tasty_meal_water (kCfl_ = 0.5)

# ── Geometry (identical across all three sub-experiments) ────────────────────
BOX_X0, BOX_X1     = 0.25, 0.75
BOX_Y0              = 0.10
MEDIUM_X0, MEDIUM_X1 = 0.28, 0.72
MEDIUM_Y0, MEDIUM_Y1 = 0.10, 0.32
ICE_CX, ICE_CY     = 0.50, 0.62
ICE_HALF            = 0.04
ICE_V0              = (0.0, -1.2)

# ── Material IDs ──────────────────────────────────────────────────────────────
MAT_MEDIUM = 0
MAT_ICE    = 1

# ── Ice (Fixed Corotated, match CKMPM kIce*) ─────────────────────────────────
ICE_RHO = 900.0
ICE_E   = 1e7
ICE_NU  = 0.40
ICE_MU  = ICE_E / (2.0 * (1.0 + ICE_NU))
ICE_LAM = ICE_E * ICE_NU / ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU))

# ── Per-medium parameters ─────────────────────────────────────────────────────
if MEDIUM == "water":
    MEDIUM_RHO       = 1000.0
    WATER_K          = 5e4    # CKMPM kWaterBulk_
    WATER_GAMMA      = 7.15   # CKMPM kWaterGamma_
    WATER_VISCO      = 0.1    # CKMPM kWaterVisco_
    HONEY_VIS_FACTOR = 1.0
elif MEDIUM == "honey":
    MEDIUM_RHO       = 1400.0
    WATER_K          = 5e4
    WATER_GAMMA      = 7.15
    WATER_VISCO      = 0.1
    HONEY_VIS_FACTOR = 0.99   # gated to honey-dominant cells only
else:  # snow — Stomakhin 2013
    MEDIUM_RHO   = 400.0
    SNOW_E       = 1.4e5
    SNOW_NU      = 0.2
    SNOW_MU0     = SNOW_E / (2.0 * (1.0 + SNOW_NU))
    SNOW_LAM0    = SNOW_E * SNOW_NU / ((1.0 + SNOW_NU) * (1.0 - 2.0 * SNOW_NU))
    SNOW_THETA_C = 0.025
    SNOW_THETA_S = 0.0075
    SNOW_XI      = 10.0

# dt sized by stiffest material (ice)
C_S_ICE      = ((ICE_E * (1.0 - ICE_NU)) /
                ((1.0 + ICE_NU) * (1.0 - 2.0 * ICE_NU) * ICE_RHO)) ** 0.5
DT           = CFL * DX / C_S_ICE
PARTICLE_VOL = DX ** 2 / PPC

print(f"[ice2d/{MEDIUM}/ckmpm] dx={DX:.4e} dt={DT:.4e} grid={GRID_SIZE}")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output_ice_drop_2d_ckmpm")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_GIF = os.path.join(OUT_DIR, f"ice_into_{MEDIUM}_2d_ckmpm.gif")
OUT_COM = os.path.join(OUT_DIR, f"ice_into_{MEDIUM}_2d_ckmpm_com.png")
OUT_NPZ = os.path.join(OUT_DIR, f"ice_into_{MEDIUM}_2d_ckmpm_metrics.npz")


# ── Particle generation ───────────────────────────────────────────────────────
def make_rect(x0, x1, y0, y1):
    pos = []
    sub = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
    i0 = int(x0 / DX) - 1; i1 = int(x1 / DX) + 2
    j0 = int(y0 / DX) - 1; j1 = int(y1 / DX) + 2
    for i in range(i0, i1):
        for j in range(j0, j1):
            for ox, oy in sub:
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
print(f"[ice2d/{MEDIUM}/ckmpm] medium={n_medium} ice={n_ice} total={N}")

all_pos = np.concatenate([medium_pos, ice_pos], axis=0)
all_vel = np.zeros((N, 2), dtype=np.float32)
all_vel[n_medium:, 0] = ICE_V0[0]
all_vel[n_medium:, 1] = ICE_V0[1]

mat_id_np            = np.empty(N, dtype=np.int32)
mat_id_np[:n_medium]  = MAT_MEDIUM
mat_id_np[n_medium:]  = MAT_ICE

mass_np              = np.empty(N, dtype=np.float32)
mass_np[:n_medium]    = PARTICLE_VOL * MEDIUM_RHO
mass_np[n_medium:]    = PARTICLE_VOL * ICE_RHO


# ── Taichi fields ─────────────────────────────────────────────────────────────
x       = ti.Vector.field(2, float, N)
v       = ti.Vector.field(2, float, N)
F_p     = ti.Matrix.field(2, 2, float, N)   # elastic deformation gradient
C_fluid = ti.Matrix.field(2, 2, float, N)   # cached velocity gradient (fluid stress)
J_p     = ti.field(float, N)                # fluid volume ratio
Jp_p    = ti.field(float, N)                # snow plastic Jp
mat_id  = ti.field(ti.i32, N)
m_p     = ti.field(float, N)
vol_p   = ti.field(float, N)

# Dual interlocked grid: extra "w" dimension of size 2 (sub-grid index)
grid_m         = ti.field(float, (GRID_SIZE, GRID_SIZE, 2))
grid_m_medium  = ti.field(float, (GRID_SIZE, GRID_SIZE, 2))
grid_v         = ti.Vector.field(2, float, (GRID_SIZE, GRID_SIZE, 2))

x.from_numpy(all_pos)
v.from_numpy(all_vel)
F_p.from_numpy(np.tile(np.eye(2, dtype=np.float32), (N, 1, 1)))
C_fluid.fill(0.0)
J_p.fill(1.0)
Jp_p.fill(1.0)
mat_id.from_numpy(mat_id_np)
m_p.from_numpy(mass_np)
vol_p.fill(PARTICLE_VOL)


# ── Compact kernel (sin/cos) — 2-cell support per sub-grid ───────────────────
@ti.func
def compact_stencil(r):
    """Per-axis compact-kernel weights at fractional offset r ∈ [0, 1)."""
    # stencil[a, 0] for sub-cell 0;  stencil[a, 1] = 1 − stencil[a, 0]
    s = ti.Matrix.zero(float, 2, 2)
    for a in ti.static(range(2)):
        s[a, 0] = 1.0 - r[a] + ti.sin(TWOPI * r[a]) / TWOPI
        s[a, 1] = 1.0 - s[a, 0]
    return s


@ti.func
def compact_grad(I, sten, dw):
    """Gradient of the 2D compact kernel at sub-cell I given offset dw.

    Returns ∇w as a 2-vector with units of 1/length.
    """
    kx = sten[0, I[0]]
    ky = sten[1, I[1]]
    g  = ti.Vector([
        ti.math.sign(dw[0]) * (ti.cos(TWOPI * ti.abs(dw[0])) - 1.0),
        ti.math.sign(dw[1]) * (ti.cos(TWOPI * ti.abs(dw[1])) - 1.0),
    ])
    return INV_DX * ti.Vector([ky * g[0], kx * g[1]])


# ── Constitutive models ───────────────────────────────────────────────────────
@ti.func
def corotated_pf(Fp, lam, mu):
    """Return  P @ F^T  (Kirchhoff form) for fixed-corotated."""
    U, sig, V = ti.svd(Fp)
    R = U @ V.transpose()
    J = Fp.determinant()
    return 2.0 * mu * (Fp - R) @ Fp.transpose() + lam * (J - 1.0) * J * ti.Matrix.identity(float, 2)


@ti.func
def fluid_pf_tait(J, C, K, gamma, viscosity):
    """CKMPM-style fluid Kirchhoff stress (mpm_material.cuh:466)."""
    Jc = ti.max(0.05, ti.min(20.0, J))
    pressure = K * (ti.pow(Jc, -gamma) - 1.0)
    P = (C + C.transpose()) * viscosity
    P[0, 0] -= pressure
    P[1, 1] -= pressure
    return Jc * P


@ti.func
def snow_pf(F_e, Jp, mu0, lam0, xi):
    """Stomakhin 2013: hardened corotated stress."""
    h = ti.exp(xi * (1.0 - Jp))
    return corotated_pf(F_e, lam0 * h, mu0 * h)


@ti.func
def particle_pf(p):
    pf = ti.Matrix.zero(float, 2, 2)
    if mat_id[p] == MAT_ICE:
        pf = corotated_pf(F_p[p], ICE_LAM, ICE_MU)
    else:
        if ti.static(MEDIUM == "snow"):
            pf = snow_pf(F_p[p], Jp_p[p], SNOW_MU0, SNOW_LAM0, SNOW_XI)
        else:
            pf = fluid_pf_tait(J_p[p], C_fluid[p], WATER_K, WATER_GAMMA, WATER_VISCO)
    return pf


# ── Simulation kernels ────────────────────────────────────────────────────────
@ti.kernel
def reset_grid():
    for i, j, w in grid_m:
        grid_m[i, j, w]        = 0.0
        grid_m_medium[i, j, w] = 0.0
        grid_v[i, j, w]        = ti.Vector([0.0, 0.0])


@ti.kernel
def p2g():
    for p in range(N):
        PF = particle_pf(p)
        # Two interleaved sub-grids
        for w in ti.static(range(2)):
            sign = ti.static(-1.0 if w == 0 else 1.0)
            base = ti.cast(x[p] * INV_DX - sign * 0.25, ti.i32)
            offset = x[p] * INV_DX - (base.cast(float) + 0.25 * sign)
            sten = compact_stencil(offset)
            for i, j in ti.static(ti.ndrange(2, 2)):
                I  = ti.Vector([i, j])
                weight = sten[0, i] * sten[1, j]
                gw = compact_grad(I, sten, offset - I.cast(float))
                node = base + I
                if 0 <= node[0] < GRID_SIZE and 0 <= node[1] < GRID_SIZE:
                    mw = weight * m_p[p]
                    grid_m[node[0], node[1], w] += mw
                    if mat_id[p] == MAT_MEDIUM:
                        grid_m_medium[node[0], node[1], w] += mw
                    grid_v[node[0], node[1], w] += mw * v[p]
                    fi = -vol_p[p] * PF @ gw
                    grid_v[node[0], node[1], w] += DT * fi


@ti.kernel
def update_grid():
    for i, j, w in grid_m:
        if grid_m[i, j, w] > 0.0:
            grid_v[i, j, w] /= grid_m[i, j, w]
            grid_v[i, j, w].y += DT * GRAVITY_Y

            # honey: damp only honey-dominant cells (both sub-grids share
            # the same per-cell decision)
            if ti.static(MEDIUM == "honey"):
                if grid_m_medium[i, j, w] > 0.5 * grid_m[i, j, w]:
                    grid_v[i, j, w] *= HONEY_VIS_FACTOR

            # container walls — 2-cell band so the BC covers the stencil
            # that a particle pinned at advect()'s 1.5*DX position clamp
            # reads.  Because each sub-grid only spans 2 cells, this is
            # already as wide as a single sub-grid stencil — fine.
            x_node = i * DX
            y_node = j * DX
            if x_node < BOX_X0 + 2.0 * DX and grid_v[i, j, w].x < 0.0:
                grid_v[i, j, w].x = 0.0
            if x_node > BOX_X1 - 2.0 * DX and grid_v[i, j, w].x > 0.0:
                grid_v[i, j, w].x = 0.0
            if y_node < BOX_Y0 + 2.0 * DX and grid_v[i, j, w].y < 0.0:
                grid_v[i, j, w].y = 0.0

            # outer domain guard
            if i <= 2 or i >= GRID_SIZE - 3 or j <= 2 or j >= GRID_SIZE - 3:
                grid_v[i, j, w] = ti.Vector([0.0, 0.0])


@ti.kernel
def g2p():
    for p in range(N):
        new_v   = ti.Vector.zero(float, 2)
        cov_v   = ti.Matrix.zero(float, 2, 2)   # accumulated velocity gradient
        for w in ti.static(range(2)):
            sign = ti.static(-1.0 if w == 0 else 1.0)
            base = ti.cast(x[p] * INV_DX - sign * 0.25, ti.i32)
            offset = x[p] * INV_DX - (base.cast(float) + 0.25 * sign)
            sten = compact_stencil(offset)
            for i, j in ti.static(ti.ndrange(2, 2)):
                I  = ti.Vector([i, j])
                weight = sten[0, i] * sten[1, j]
                gw = compact_grad(I, sten, offset - I.cast(float))
                node = base + I
                if 0 <= node[0] < GRID_SIZE and 0 <= node[1] < GRID_SIZE:
                    gv = grid_v[node[0], node[1], w]
                    new_v += weight * gv
                    cov_v += gv.outer_product(gw)

        # CKMPM: average the two sub-grids
        new_v *= 0.5
        cov_v *= 0.5

        v[p] = new_v
        F_new = (ti.Matrix.identity(float, 2) + DT * cov_v) @ F_p[p]

        if mat_id[p] == MAT_MEDIUM:
            if ti.static(MEDIUM == "snow"):
                # SVD clamp + Jp hardening (Stomakhin 2013)
                U, sig, V = ti.svd(F_new)
                for d in ti.static(range(2)):
                    sigc = ti.min(ti.max(sig[d, d], 1.0 - SNOW_THETA_C),
                                  1.0 + SNOW_THETA_S)
                    Jp_p[p] *= sig[d, d] / sigc
                    sig[d, d] = sigc
                Jp_p[p] = ti.max(0.6, ti.min(20.0, Jp_p[p]))
                F_p[p] = U @ sig @ V.transpose()
            else:
                # fluid: track J via trace, reset F to sqrt(J)*I
                C_fluid[p] = cov_v
                J_new = J_p[p] + DT * cov_v.trace() * J_p[p]
                J_p[p] = ti.max(0.05, ti.min(20.0, J_new))
                F_p[p] = ti.Matrix.identity(float, 2)
        else:
            F_p[p] = F_new


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


# ── Run loop ──────────────────────────────────────────────────────────────────
def run_simulation():
    frames_medium = []
    frames_ice    = []
    ice_com_y     = []
    ice_vy        = []
    crater_depth  = []
    initial_top_y = MEDIUM_Y1

    target_dt = 1.0 / FPS
    next_t    = 0.0
    sim_t     = 0.0

    def record():
        pos = x.to_numpy()
        vel = v.to_numpy()
        frames_medium.append(pos[:n_medium].copy())
        frames_ice.append(pos[n_medium:].copy())
        ice_com_y.append(float(pos[n_medium:, 1].mean()))
        ice_vy.append(float(vel[n_medium:, 1].mean()))
        if MEDIUM == "snow":
            mp = pos[:n_medium]
            mask = ((mp[:, 0] >= ICE_CX - ICE_HALF - 0.02) &
                    (mp[:, 0] <= ICE_CX + ICE_HALF + 0.02))
            d = float(max(0.0, initial_top_y - mp[mask, 1].min())) if mask.any() else 0.0
            crater_depth.append(d)

    record()
    next_t += target_dt

    while sim_t < TOTAL_TIME - 1e-12:
        step()
        sim_t += DT
        if sim_t + 1e-12 >= next_t:
            record()
            next_t += target_dt

    return (frames_medium, frames_ice,
            np.array(ice_com_y), np.array(ice_vy),
            np.array(crater_depth) if MEDIUM == "snow" else None)


# ── Render ────────────────────────────────────────────────────────────────────
COLOURS = {"water": "#2e86de", "honey": "#f6c90e", "snow": "#b2bec3"}

def render_gif(frames_medium, frames_ice):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.02, 0.90)
    ax.set_aspect("equal")
    ax.set_title(f"2D Ice Into {MEDIUM.capitalize()} — CKMPM")
    ax.plot([BOX_X0, BOX_X0], [BOX_Y0, 0.85], "k-", lw=2)
    ax.plot([BOX_X1, BOX_X1], [BOX_Y0, 0.85], "k-", lw=2)
    ax.plot([BOX_X0, BOX_X1], [BOX_Y0, BOX_Y0], "k-", lw=2)

    sm = ax.scatter([], [], s=3, c=COLOURS[MEDIUM], alpha=0.8, label=MEDIUM.capitalize())
    si = ax.scatter([], [], s=6, c="#dfe6e9", edgecolors="#34495e",
                    linewidths=0.2, label="Ice")
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def update(fid):
        sm.set_offsets(frames_medium[fid])
        si.set_offsets(frames_ice[fid])
        txt.set_text(f"t = {fid / FPS:.2f} s")
        return sm, si, txt

    ani = animation.FuncAnimation(fig, update, frames=len(frames_medium),
                                  interval=1000 // FPS, blit=False)
    ani.save(OUT_GIF, writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)


def plot_metrics(ice_com_y, ice_vy, crater):
    times = np.arange(len(ice_com_y)) / FPS
    n_panels = 3 if crater is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 3 * n_panels), sharex=True)
    axes[0].plot(times, ice_com_y, color="#c0392b", lw=1.8)
    axes[0].set_ylabel("Ice COM y (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"2D Ice Into {MEDIUM.capitalize()} — CKMPM Metrics")
    axes[1].plot(times, ice_vy, color="#2980b9", lw=1.8)
    axes[1].set_ylabel("Ice vy (m/s)")
    axes[1].grid(True, alpha=0.3)
    if crater is not None:
        axes[2].plot(times, crater, color="#8e44ad", lw=1.8)
        axes[2].set_ylabel("Crater depth (m)")
        axes[2].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(OUT_COM, dpi=140)
    plt.close(fig)
    extra = {"crater_depth": crater} if crater is not None else {}
    np.savez(OUT_NPZ, times=times, ice_com_y=ice_com_y, ice_vy=ice_vy, **extra)


if __name__ == "__main__":
    fm, fi, com_y, vy, crater = run_simulation()
    print(f"[ice2d/{MEDIUM}/ckmpm] frames={len(fm)} final_com_y={com_y[-1]:.4f}")
    if crater is not None:
        print(f"[ice2d/{MEDIUM}/ckmpm] final_crater={crater[-1]:.4f} m")
    render_gif(fm, fi)
    plot_metrics(com_y, vy, crater)
    print(f"[ice2d/{MEDIUM}/ckmpm] GIF     → {OUT_GIF}")
    print(f"[ice2d/{MEDIUM}/ckmpm] metrics → {OUT_NPZ}")
