"""Plot CKMPM CUDA efficiency_benchmark component timing (V100, 3D)."""
import matplotlib.pyplot as plt

# ---- numbers from MPMBenchmark log (s -> ms) ----
labels = ["Activate\nExterior", "Activate\nNeighbor", "Copy\nGrid Data", "G2P2G\n(CUDA)"]
means_ms = [
    3.7238e-05 * 1e3,   # 0.0372
    2.6363e-05 * 1e3,   # 0.0264
    9.4844e-05 * 1e3,   # 0.0948
    3.9161e-03 * 1e3,   # 3.9161
]
colors = ["#3b6fb6", "#e08a3c", "#7d8c4e", "#c0504d"]

# scene info from log
n_substeps = 38868
wall_total = 174.679           # s
n_part     = 19683 * 1         # active partition count from log header (proxy)
device     = "V100 (CUDA)"

fig, ax = plt.subplots(figsize=(7.5, 5))
bars = ax.bar(labels, means_ms,
              color=colors, edgecolor="black", linewidth=0.5)
for b, m in zip(bars, means_ms):
    ax.text(b.get_x() + b.get_width() / 2,
            b.get_height() + 0.08,
            f"{m:.3f}" if m < 1 else f"{m:.2f}",
            ha="center", fontsize=10)

ax.set_ylabel("Time (ms per substep)")
ax.set_title(f"CKMPM Component Timing (efficiency_benchmark, 3D)\n"
             f"{n_substeps} substeps, {wall_total:.1f}s wall, {device}")
ax.set_ylim(0, max(means_ms) * 1.18)

plt.tight_layout()
out = "/Users/lelinw/Desktop/15763-mpm/plots/ckmpm_components.png"
plt.savefig(out, dpi=150)
print("saved", out)
