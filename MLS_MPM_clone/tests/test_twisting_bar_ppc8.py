# Twisting Bar – MLS-MPM  (PyTorch / MLS-APIC)
# ===============================================
# A rectangular elastic bar; ends rotated in opposite directions about z-axis.
#
# Shared parameters (identical across CKMPM / MLS-MPM / Basic-MPM-3D):
#   E=100 Pa, nu=0.4, rho=2 kg/m³  (matches CKMPM twisting_bar)
#   omega=1.0 rad/s, ppc=8, CFL=0.5, fps=48, total_time=5.0 s, gravity=0
#
# Bar geometry (dx=1/256):
#   x,y: cells [118,138)  →  20 cells = 0.078 m
#   z:   cells [88,168)   →  80 cells = 0.313 m
#   Top Dirichlet (+ω): z∈[148,168)    Bot Dirichlet (−ω): z∈[88,108)
#   N = 20×20×80×8 = 256000 particles

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
GRID_SIZE  = 256
DX         = 1.0 / GRID_SIZE
E          = 100.0
NU         = 0.4
DENSITY    = 2.0
CFL        = 0.5
FPS        = 48
TOTAL_TIME = 5.0
OMEGA      = 1.0    # rad/s (each end, opposite signs)

C_S           = ((E * (1 - NU)) / ((1 + NU) * (1 - 2*NU) * DENSITY)) ** 0.5
DT            = CFL * DX / C_S
PARTICLE_VOL  = DX**3 / 8      # ppc=8
PARTICLE_MASS = PARTICLE_VOL * DENSITY

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUT_GIF = os.path.join(OUT_DIR, "twisting_bar_mlsmpm_ppc8.gif")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[mlsmpm-twist-ppc8]  dx={DX:.4e}  c_s={C_S:.4f}  dt={DT:.4e}")
print(f"[mlsmpm-twist-ppc8]  particle_vol={PARTICLE_VOL:.4e}  particle_mass={PARTICLE_MASS:.4e}")

# ── Bar geometry ──────────────────────────────────────────────────────────────
X_LO, X_HI         = 118, 138
Y_LO, Y_HI         = 118, 138
Z_LO,  Z_HI        = 88,  168
Z_TOP_LO, Z_TOP_HI = 148, 168   # top end Dirichlet zone (+ω)
Z_BOT_LO, Z_BOT_HI = 88,  108   # bot end Dirichlet zone (−ω)

pos_list, col_list = [], []
for i in range(X_LO, X_HI):
    for j in range(Y_LO, Y_HI):
        for k in range(Z_LO, Z_HI):
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        pos_list.append([(i + 0.25 + di*0.5)*DX,
                                         (j + 0.25 + dj*0.5)*DX,
                                         (k + 0.25 + dk*0.5)*DX])
                        col_list.append((k - Z_LO) / (Z_HI - Z_LO))  # colour by z

all_pos = np.array(pos_list, dtype=np.float32)
z_color = np.array(col_list, dtype=np.float32)
N = len(all_pos)
print(f"[mlsmpm-twist-ppc8]  particles: {N}  (20×20×80×8 = 256000)")

# ── Precompute Dirichlet BC: grid-node indices + prescribed velocities ─────────
# grid_mv is flat: index = i*n*n + j*n + k
n = GRID_SIZE

def _bc_indices_vels(z_lo, z_hi, sign):
    idxs, vels = [], []
    for gi in range(X_LO - 1, X_HI + 1):   # include one extra node on each side
        for gj in range(Y_LO - 1, Y_HI + 1):
            for gk in range(z_lo, z_hi + 1):
                rx = gi * DX - 0.5   # relative to bar axis (domain centre)
                ry = gj * DX - 0.5
                idxs.append(gi * n * n + gj * n + gk)
                vels.append([sign * (-OMEGA * ry),
                              sign * ( OMEGA * rx),
                              0.0])
    return idxs, vels

top_idxs, top_vels = _bc_indices_vels(Z_TOP_LO, Z_TOP_HI, +1.0)
bot_idxs, bot_vels = _bc_indices_vels(Z_BOT_LO, Z_BOT_HI, -1.0)


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

    # Dirichlet BC injected into grid_mv after grid_update
    top_idx_t = torch.tensor(top_idxs, dtype=torch.long,    device=device)
    bot_idx_t = torch.tensor(bot_idxs, dtype=torch.long,    device=device)
    top_vel_t = torch.tensor(top_vels, dtype=torch.float32, device=device)
    bot_vel_t = torch.tensor(bot_vels, dtype=torch.float32, device=device)

    def twist_bc(slvr):
        slvr.grid_mv[top_idx_t] = top_vel_t
        slvr.grid_mv[bot_idx_t] = bot_vel_t

    solver.post_grid_process = [twist_bc]

    elasticity = CorotatedElasticity(E=E, nu=NU).to(device)
    plasticity = IdentityPlasticity().to(device)

    total_frames    = int(TOTAL_TIME * FPS)
    target_frame_dt = 1.0 / FPS
    next_frame_time = 0.0
    sim_time        = 0.0
    frames = []

    def record():
        frames.append(x_t.detach().cpu().numpy().copy())

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
                print(f"  Frame {len(frames):4d}/{total_frames}  t={sim_time:.3f}s")
            next_frame_time += target_frame_dt

    return frames


# ── GIF ───────────────────────────────────────────────────────────────────────
def render(frames):
    cmap   = plt.cm.coolwarm
    colors = cmap(z_color)[:, :3].tolist()

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0.35, 0.65); ax.set_ylim(0.35, 0.65); ax.set_zlim(0.25, 0.75)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Twisting Bar – MLS-MPM (dx=1/256, ppc=8)")
    ax.view_init(elev=20, azim=45)

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
    print(f"[mlsmpm-twist-ppc8]  GIF → {OUT_GIF}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mlsmpm-twist-ppc8]  device={device}")
    frames = run(device)
    render(frames)
