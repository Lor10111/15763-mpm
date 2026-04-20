"""CKMPM component timing in the SAME visual style as the MLS-MPM chart."""
import matplotlib.pyplot as plt

# ---- raw averages (s -> ms) from efficiency_benchmark log ----
ms = {
    "act_ext":   3.7238e-05 * 1e3,   # 0.0372
    "act_nbr":   2.6363e-05 * 1e3,   # 0.0264
    "copy_grid": 9.4844e-05 * 1e3,   # 0.0948
    "g2p2g":     3.9161e-03 * 1e3,   # 3.9161
}
n_substeps = 38868
n_part     = 19683            # active partitions from log header (proxy)
device     = "gpu (V100)"

# ---------- 3-bar version (cleanest mapping to MLS chart) ----------
labels3 = ["G2P2G", "Grid Update", "Boundary\nConditions"]
vals3   = [ms["g2p2g"],
           ms["copy_grid"],
           ms["act_ext"] + ms["act_nbr"]]
colors3 = ["#c0504d", "#e08a3c", "#7d8c4e"]   # match MLS palette ordering by component type

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels3, vals3, color=colors3, edgecolor="black", linewidth=0.5)
for b, m in zip(bars, vals3):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
            f"{m:.2f}" if m >= 1 else f"{m:.3f}",
            ha="center", fontsize=10)
ax.set_ylabel("Time (ms per substep)")
ax.set_title(f"CKMPM Component Timing\n({n_part} particles, {device})")
ax.set_ylim(0, max(vals3) * 1.18)
plt.tight_layout()
plt.savefig("/Users/lelinw/Desktop/15763-mpm/plots/ckmpm_components_3bar.png", dpi=150)
plt.close()

# ---------- 4-bar version (keep MLS's P2G / Grid / BC / G2P slots) ----------
# CKMPM merges P2G + G2P → split G2P2G evenly across the two slots so the
# total still equals the measured G2P2G time. (Annotated in title.)
half = ms["g2p2g"] / 2.0
labels4 = ["P2G\n(merged)", "Grid Update", "Boundary\nConditions", "G2P\n(merged)"]
vals4   = [half, ms["copy_grid"], ms["act_ext"]+ms["act_nbr"], half]
colors4 = ["#3b6fb6", "#e08a3c", "#7d8c4e", "#c0504d"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels4, vals4, color=colors4, edgecolor="black", linewidth=0.5)
for b, m in zip(bars, vals4):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
            f"{m:.2f}" if m >= 1 else f"{m:.3f}",
            ha="center", fontsize=10)
ax.set_ylabel("Time (ms per substep)")
ax.set_title(f"CKMPM Component Timing  (P2G+G2P fused as G2P2G)\n"
             f"({n_part} particles, {device})")
ax.set_ylim(0, max(vals4) * 1.25)
plt.tight_layout()
plt.savefig("/Users/lelinw/Desktop/15763-mpm/plots/ckmpm_components_4bar.png", dpi=150)
plt.close()

print("saved 3bar + 4bar PNGs to /Users/lelinw/Desktop/15763-mpm/plots/")
