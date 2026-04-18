# Twisting Bar – Basic MPM 3D  (Taichi / PIC)
# =============================================
# Rectangular elastic bar; ends rotated in opposite directions about the z-axis.
# Outputs: output/twisting_bar_basic3d_ppc8.gif
#
# Shared parameters (identical across CKMPM / MLS-MPM / Basic-MPM-3D):
#   E=100 Pa, nu=0.4, rho=2 kg/m³, omega=1.0 rad/s
#   ppc=8, CFL=0.5, fps=48, total_time=5.0 s, gravity=0
#
# Bar geometry (dx=1/256):
#   x,y: cells [118,138) → 20 cells = 0.078 m
#   z:   cells [88,168)  → 80 cells = 0.313 m
#   Top end (+ω): z∈[148,168)    Bot end (−ω): z∈[88,108)
#   N = 20×20×80×8 = 256000 particles

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import taichi as ti

ti.init(arch=ti.gpu)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUT_GIF = os.path.join(OUT_DIR, "twisting_bar_basic3d_ppc8.gif")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
grid_size  = 256
dx         = 1.0 / grid_size
E, nu      = 100.0, 0.4
density    = 2.0
mu         = E / (2.0 * (1.0 + nu))
lam        = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
c_s        = ((E * (1.0 - nu)) / ((1.0 + nu) * (1.0 - 2.0 * nu) * density)) ** 0.5
CFL        = 0.5
dt         = CFL * dx / c_s
fps        = 48
total_time = 5.0
OMEGA      = 1.0    # rad/s

particle_vol  = dx**3 / 8
particle_mass = particle_vol * density

print(f"[basic3d-twist-ppc8]  dx={dx:.4e}  c_s={c_s:.4f}  dt={dt:.4e}")

# ── Bar geometry ──────────────────────────────────────────────────────────────
X_LO, X_HI         = 118, 138
Y_LO, Y_HI         = 118, 138
Z_LO,  Z_HI        = 88,  168
Z_TOP_LO, Z_TOP_HI = 148, 168
Z_BOT_LO, Z_BOT_HI = 88,  108

pos_list, col_list = [], []
for i in range(X_LO, X_HI):
    for j in range(Y_LO, Y_HI):
        for k in range(Z_LO, Z_HI):
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        pos_list.append([(i + 0.25 + di*0.5)*dx,
                                         (j + 0.25 + dj*0.5)*dx,
                                         (k + 0.25 + dk*0.5)*dx])
                        col_list.append((k - Z_LO) / (Z_HI - Z_LO))

all_pos = np.array(pos_list, dtype=np.float32)
z_nrm   = np.array(col_list, dtype=np.float32)
N = len(all_pos)
print(f"[basic3d-twist-ppc8]  particles: {N}")

# ── Taichi fields ─────────────────────────────────────────────────────────────
x      = ti.Vector.field(3, float, N)
v      = ti.Vector.field(3, float, N)
F_f    = ti.Matrix.field(3, 3, float, N)
vol_f  = ti.field(float, N)
m_f    = ti.field(float, N)

grid_m = ti.field(float, (grid_size, grid_size, grid_size))
grid_v = ti.Vector.field(3, float, (grid_size, grid_size, grid_size))
color_f = ti.Vector.field(3, float, N)

x.from_numpy(all_pos)
v.fill(0.0)
vol_f.fill(particle_vol)
m_f.fill(particle_mass)
F_f.from_numpy(np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)))

cmap   = plt.cm.coolwarm
rgb    = cmap(z_nrm)[:, :3].astype(np.float32)
color_f.from_numpy(rgb)

# ── Fixed Corotated ───────────────────────────────────────────────────────────
@ti.func
def fixed_corotated_pk1(Fp):
    U, sig, V = ti.svd(Fp)
    R   = U @ V.transpose()
    J   = Fp.determinant()
    Fit = Fp.inverse().transpose()
    return 2.0*mu*(Fp - R) + lam*(J - 1.0)*J*Fit

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
        w    = [0.5*(1.5-fx)**2, 0.75-(fx-1.0)**2, 0.5*(fx-0.5)**2]
        dw   = [fx-1.5, 2.0*(1.0-fx), fx-0.5]
        P    = fixed_corotated_pk1(F_f[p])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    weight = w[i][0]*w[j][1]*w[k][2]
                    grad_w = ti.Vector([
                        dw[i][0]/dx * w[j][1]        * w[k][2],
                        w[i][0]       * dw[j][1]/dx  * w[k][2],
                        w[i][0]       * w[j][1]        * dw[k][2]/dx,
                    ])
                    node = base + ti.Vector([i, j, k])
                    grid_m[node[0], node[1], node[2]] += weight * m_f[p]
                    grid_v[node[0], node[1], node[2]] += weight * m_f[p] * v[p]
                    fi = -vol_f[p] * P @ F_f[p].transpose() @ grad_w
                    grid_v[node[0], node[1], node[2]] += dt * fi

# ── Grid update + Dirichlet BC ────────────────────────────────────────────────
@ti.kernel
def update_grid():
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0.0:
            grid_v[i, j, k] /= grid_m[i, j, k]

        # Domain boundary: zero velocity guard
        if (i <= 3 or i >= grid_size-3 or
            j <= 3 or j >= grid_size-3 or
            k <= 3 or k >= grid_size-3):
            grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

        # Dirichlet BC: top end rotates +OMEGA about z-axis
        in_top = (Z_TOP_LO <= k < Z_TOP_HI and
                  100 <= i <= 155 and
                  100 <= j <= 155)
        if in_top:
            rx = i * dx - 0.5
            ry = j * dx - 0.5
            grid_v[i, j, k] = ti.Vector([-OMEGA * ry, OMEGA * rx, 0.0])

        # Dirichlet BC: bottom end rotates −OMEGA about z-axis
        in_bot = (Z_BOT_LO <= k < Z_BOT_HI and
                  100 <= i <= 155 and
                  100 <= j <= 155)
        if in_bot:
            rx = i * dx - 0.5
            ry = j * dx - 0.5
            grid_v[i, j, k] = ti.Vector([OMEGA * ry, -OMEGA * rx, 0.0])

# ── G2P ──────────────────────────────────────────────────────────────────────
@ti.kernel
def g2p():
    for p in range(N):
        base = (x[p] / dx - 0.5).cast(int)
        fx   = x[p] / dx - base.cast(float)
        w    = [0.5*(1.5-fx)**2, 0.75-(fx-1.0)**2, 0.5*(fx-0.5)**2]
        dw   = [fx-1.5, 2.0*(1.0-fx), fx-0.5]
        new_v  = ti.Vector.zero(float, 3)
        grad_v = ti.Matrix.zero(float, 3, 3)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    weight = w[i][0]*w[j][1]*w[k][2]
                    grad_w = ti.Vector([
                        dw[i][0]/dx * w[j][1]        * w[k][2],
                        w[i][0]       * dw[j][1]/dx  * w[k][2],
                        w[i][0]       * w[j][1]        * dw[k][2]/dx,
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

def step():
    reset_grid()
    p2g()
    update_grid()
    g2p()
    advect()

# ── Simulation loop ───────────────────────────────────────────────────────────
steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
total_frames    = int(round(total_time * fps))
print(f"[basic3d-twist-ppc8]  steps_per_frame={steps_per_frame}  total_frames={total_frames}")

frames_pos = []

def record():
    frames_pos.append(x.to_numpy().copy())

record()
for frame in range(1, total_frames + 1):
    for _ in range(steps_per_frame):
        step()
    record()
    if frame % 24 == 0:
        print(f"  Frame {frame:4d}/{total_frames}  t={frame/fps:.3f}s")

print("[basic3d-twist-ppc8]  simulation done, rendering …")

# ── GIF ───────────────────────────────────────────────────────────────────────
cmap_np = plt.cm.coolwarm
colors  = cmap_np(z_nrm)[:, :3].tolist()

fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111, projection="3d")
ax.set_xlim(0.35, 0.65); ax.set_ylim(0.35, 0.65); ax.set_zlim(0.25, 0.75)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Twisting Bar – Basic MPM 3D (dx=1/256, ppc=8)")
ax.view_init(elev=20, azim=45)

pts0 = frames_pos[0]
scat = ax.scatter(pts0[:,0], pts0[:,1], pts0[:,2],
                  c=colors, s=3, alpha=0.7, depthshade=True)
txt  = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)

def update(fid):
    p = frames_pos[fid]
    scat._offsets3d = (p[:,0], p[:,1], p[:,2])
    txt.set_text(f"t = {fid/fps:.2f} s")
    return scat, txt

ani = animation.FuncAnimation(fig, update, frames=len(frames_pos),
                               interval=1000//fps, blit=False)
ani.save(OUT_GIF, writer="pillow", fps=fps, dpi=100)
plt.close(fig)
print(f"[basic3d-twist-ppc8]  GIF → {OUT_GIF}")
