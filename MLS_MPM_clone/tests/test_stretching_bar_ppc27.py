# Stretching Bar – MLS-MPM  (PyTorch / MLS-APIC)
# =================================================
# Same rectangular bar and material as the twisting test, but the two ends
# are pulled apart along the z-axis instead of rotated.  Run until the bar
# necks and particles in the middle separate (MPM "fracture").
#
# Shared parameters (identical across CKMPM / MLS-MPM / Basic-MPM-3D):
#   E=100 Pa, nu=0.4, rho=2 kg/m³   (same as twisting_bar)
#   v_stretch=0.02 m/s per end, total_time=5.0 s, gravity=0
#   → each end travels 0.1 m → bar elongates from 0.313 m to 0.513 m (64% strain)

import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mpm_pytorch import MPMSolver
from mpm_pytorch.constitutive_models.elasticity import CorotatedElasticity
from mpm_pytorch.constitutive_models.plasticity import IdentityPlasticity

# ── Parameters ────────────────────────────────────────────────────────────────
GRID_SIZE  = 64
DX         = 1.0 / GRID_SIZE
E          = 100.0
NU         = 0.4
DENSITY    = 2.0
CFL        = 0.5
FPS        = 48
TOTAL_TIME = 5.0
V_STRETCH  = 0.02   # m/s per end (top +z, bot −z)
PPC        = 27

C_S           = ((E * (1 - NU)) / ((1 + NU) * (1 - 2*NU) * DENSITY)) ** 0.5
DT            = CFL * DX / C_S
PARTICLE_VOL  = DX**3 / PPC
PARTICLE_MASS = PARTICLE_VOL * DENSITY

OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
OUT_GIF  = os.path.join(OUT_DIR, "stretching_bar_mlsmpm_ppc27.gif")
OUT_PLOT = os.path.join(OUT_DIR, "stretching_bar_length_mlsmpm_ppc27.png")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[mlsmpm-stretch]  dx={DX:.4e}  c_s={C_S:.4f}  dt={DT:.4e}")

# ── Bar geometry ──────────────────────────────────────────────────────────────
X_LO, X_HI      = 30, 35
Y_LO, Y_HI      = 30, 35
Z_LO,  Z_HI     = 22, 42
Z_TOP_LO, Z_TOP_HI = 37, 42   # top end: +z pull
Z_BOT_LO, Z_BOT_HI = 22, 27   # bot end: −z pull

pos_list, col_list = [], []
for i in range(X_LO, X_HI):
    for j in range(Y_LO, Y_HI):
        for k in range(Z_LO, Z_HI):
            for di in range(3):
                for dj in range(3):
                    for dk in range(3):
                        pos_list.append([(i + (di + 0.5)/3.0)*DX,
                                         (j + (dj + 0.5)/3.0)*DX,
                                         (k + (dk + 0.5)/3.0)*DX])
                        col_list.append((k - Z_LO) / (Z_HI - Z_LO))

all_pos = np.array(pos_list, dtype=np.float32)
z_color = np.array(col_list, dtype=np.float32)
N = len(all_pos)
print(f"[mlsmpm-stretch]  particles: {N}")

# ── Precompute Dirichlet BC grid-node indices ─────────────────────────────────
n = GRID_SIZE

def _bc_nodes(z_lo, z_hi, vz):
    idxs, vels = [], []
    for gi in range(X_LO - 1, X_HI + 1):
        for gj in range(Y_LO - 1, Y_HI + 1):
            for gk in range(z_lo, z_hi):
                idxs.append(gi * n * n + gj * n + gk)
                vels.append([0.0, 0.0, vz])
    return idxs, vels

top_idxs, top_vels = _bc_nodes(Z_TOP_LO, Z_TOP_HI, +V_STRETCH)
bot_idxs, bot_vels = _bc_nodes(Z_BOT_LO, Z_BOT_HI, -V_STRETCH)


# ── Simulation ────────────────────────────────────────────────────────────────
def run(device):
    x_t = torch.tensor(all_pos, dtype=torch.float32, device=device)
    v_t = torch.zeros(N, 3,    dtype=torch.float32, device=device)
    C   = torch.zeros(N, 3, 3, dtype=torch.float32, device=device)
    F   = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(N,-1,-1).clone()

    solver = MPMSolver(
        init_pos=x_t, rho=DENSITY, num_grids=GRID_SIZE,
        dt=DT, gravity=[0.0,0.0,0.0], clip_bound=0.0, damping=1.0, device=device,
    )
    solver.vol    = float(PARTICLE_VOL)
    solver.p_mass = float(PARTICLE_MASS)
    solver.pre_particle_process = []

    top_idx_t = torch.tensor(top_idxs, dtype=torch.long,    device=device)
    bot_idx_t = torch.tensor(bot_idxs, dtype=torch.long,    device=device)
    top_vel_t = torch.tensor(top_vels, dtype=torch.float32, device=device)
    bot_vel_t = torch.tensor(bot_vels, dtype=torch.float32, device=device)

    def stretch_bc(slvr):
        slvr.grid_mv[top_idx_t] = top_vel_t
        slvr.grid_mv[bot_idx_t] = bot_vel_t

    solver.post_grid_process = [stretch_bc]

    elasticity = CorotatedElasticity(E=E, nu=NU).to(device)
    plasticity = IdentityPlasticity().to(device)

    total_frames    = int(TOTAL_TIME * FPS)
    target_frame_dt = 1.0 / FPS
    next_frame_time = 0.0
    sim_time        = 0.0

    frames      = []
    bar_lengths = []   # track z-extent of bar each frame

    def record():
        pts = x_t.detach().cpu().numpy()
        frames.append(pts.copy())
        bar_lengths.append(pts[:, 2].max() - pts[:, 2].min())

    record()
    next_frame_time += target_frame_dt

    while sim_time < TOTAL_TIME - 1e-12:
        stress          = elasticity(F)
        x_t, v_t, C, F = solver(x_t, v_t, C, F, stress)
        F               = plasticity(F)
        sim_time       += DT

        if sim_time + 1e-12 >= next_frame_time:
            record()
            if len(frames) % 24 == 0:
                print(f"  Frame {len(frames):4d}/{total_frames}  "
                      f"t={sim_time:.3f}s  bar_length={bar_lengths[-1]:.4f}m")
            next_frame_time += target_frame_dt

    return frames, bar_lengths


# ── GIF ───────────────────────────────────────────────────────────────────────
def render_gif(frames):
    cmap   = plt.cm.coolwarm
    colors = cmap(z_color)[:, :3].tolist()

    # Side view (XZ plane) to see elongation clearly
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0.35, 0.65); ax.set_ylim(0.35, 0.65); ax.set_zlim(0.1, 0.9)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Stretching Bar – MLS-MPM")
    ax.view_init(elev=20.0, azim=-65.0)   # side view along y-axis

    pts0 = frames[0]
    scat = ax.scatter(pts0[:,0], pts0[:,1], pts0[:,2],
                      c=colors, s=3, alpha=0.7, depthshade=True)
    txt  = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)

    def update(fid):
        p = frames[fid]
        scat._offsets3d = (p[:,0], p[:,1], p[:,2])
        txt.set_text(f"t = {fid/FPS:.2f} s")
        return scat, txt

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000//FPS, blit=False)
    ani.save(OUT_GIF, writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"[mlsmpm-stretch]  GIF → {OUT_GIF}")


# ── Bar-length plot ───────────────────────────────────────────────────────────
def plot_length(bar_lengths):
    times = np.arange(len(bar_lengths)) / FPS
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, bar_lengths, lw=1.5, color="#e74c3c")
    ax.axhline(bar_lengths[0], color="k", ls="--", lw=0.8, label=f"initial {bar_lengths[0]:.3f} m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bar z-extent (m)")
    ax.set_title("Bar Length vs Time – MLS-MPM Stretching")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=120)
    plt.close(fig)
    print(f"[mlsmpm-stretch]  length plot → {OUT_PLOT}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mlsmpm-stretch]  device={device}")
    frames, bar_lengths = run(device)
    render_gif(frames)
    plot_length(bar_lengths)
    print(f"[mlsmpm-stretch]  final bar length: {bar_lengths[-1]:.4f} m  "
          f"(initial: {bar_lengths[0]:.4f} m, strain: {(bar_lengths[-1]/bar_lengths[0]-1)*100:.1f}%)")
