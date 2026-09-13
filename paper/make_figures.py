"""Generates the paper's figures from the experiment results (numbers from E_results docs / run logs)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

C_BC = "#c44e52"    # behavior-cloned (red)
C_AA = "#4c72b0"    # anchored (blue)
C_BASE = "#555555"  # pretrained (gray)
C_GREEN = "#55a868"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 9.5,
    "axes.titlesize": 9.5, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "figure.dpi": 100,
})

# ---------------------------------------------------------------- fig 1
fig, ax = plt.subplots(figsize=(3.3, 2.6))
layers = np.arange(25)
bc = [100,100,99,99,98,98,97,96,97,97,97,96,96,95,95,92,91,92,92,90,90,83,59,47,63]
aa = [100,100,99,99,99,99,99,99,99,99,99,98,98,98,99,99,99,99,99,98,98,98,97,97,93]
ax.plot(layers, bc, color=C_BC, marker="o", ms=2.5, label="Behavior-cloned (BC)")
ax.plot(layers, aa, color=C_AA, marker="s", ms=2.5, label="Anchored")
ax.axhline(100, color=C_BASE, ls="--", lw=1.0, label="Pretrained VLM")
ax.axvspan(18, 24, color="0.9", zorder=0)
ax.set_xlabel("Transformer layer (hidden-states index)")
ax.set_ylabel("Text-token CKA to pretrained VLM (%)")
ax.set_ylim(40, 102)
ax.legend(loc="lower left", frameon=False)
ax.annotate("layers 21--24", xy=(22, 41.4), fontsize=8, ha="center", va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_cka_curves.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---------------------------------------------------------------- fig 2
alphas = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
vis = {"deep6": [53.9,60.0,61.7,62.0,61.9,61.5],
       "band": [48.9,51.3,52.4,53.1,53.6,53.7],
       "all":  [62.3,59.4,58.1,57.5,56.7,56.4]}
ap  = {"deep6": [96,90,88,82,77,72],
       "band": [96,91,84,83,85,84],
       "all":  [45,39,41,41,41,41]}
styles = {"deep6": (C_AA, "o", "layers 18--23"),
          "band":  (C_GREEN, "s", "layers 21--24"),
          "all":   (C_BC, "^", "all layers")}
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1))
handles, labels = [], []
for key, (col, mk, lab) in styles.items():
    h, = axes[0].plot([0] + list(alphas), [46.6] + vis[key], color=col, marker=mk, ms=3.5, label=lab)
    axes[1].plot([0] + list(alphas), [93] + ap[key], color=col, marker=mk, ms=3.5, label=lab)
    handles.append(h); labels.append(lab)
axes[0].set_xlabel("Correction strength $\\alpha$")
axes[0].set_ylabel("Vision CKA to pretrained VLM (%)")
axes[0].set_ylim(40, 68)
axes[0].set_title("Semantic recovery")
axes[0].axvspan(0, 0.25, color="0.92", zorder=0)
axes[1].set_xlabel("Correction strength $\\alpha$")
axes[1].set_ylabel("Action direction accuracy (%)")
axes[1].set_ylim(30, 100)
axes[1].set_title("Action fidelity")
axes[1].axvspan(0, 0.25, color="0.92", zorder=0)
axes[0].annotate("free-lunch\n$\\alpha{=}.25$", xy=(0.05, 0.05), xycoords="axes fraction",
                 fontsize=7.5, ha="left", va="bottom", color="0.25")
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig(os.path.join(OUT, "fig2_tradeoff.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---------------------------------------------------------------- fig 3
k = np.array([1, 4, 8, 16, 32])
err = np.array([53.95, 39.13, 29.90, 19.39, 9.98])
kf = np.array([0, 1, 4, 8, 16, 32])
visf = np.array([44.9, 47.2, 52.6, 57.3, 59.4, 60.1])
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1))
axes[0].plot(k, err, color=C_AA, marker="o", ms=3.5)
axes[0].set_xscale("log", base=2)
axes[0].set_xticks([1, 2, 4, 8, 16, 32]); axes[0].set_xticklabels(["1", "2", "4", "8", "16", "32"])
axes[0].set_xlabel("Bundle rank $k$ (directions per layer)")
axes[0].set_ylabel("Relative drift reconstruction error (%)", fontsize=9)
axes[0].set_ylim(0, 60)
axes[0].set_title("Drift structure")
h1, = axes[1].plot(kf, visf, color=C_AA, marker="o", ms=3.5, label="k-direction bundle")
h2 = axes[1].axhline(60.9, color=C_BASE, ls="--", lw=1.0, label="full tensor ($k{=}$full)")
h3 = axes[1].axhline(46.6, color=C_BC, ls=":", lw=1.2, label="BC baseline")
axes[1].set_xlabel("Bundle rank $k$")
axes[1].set_ylabel("Vision CKA to pretrained VLM (%)", fontsize=9)
axes[1].set_ylim(40, 68)
axes[1].set_title("Recovery vs. $k$")
fig.legend([h1, h2, h3], ["k-direction bundle", "full tensor ($k{=}$full)", "BC baseline"],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig(os.path.join(OUT, "fig3_bundle.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---------------------------------------------------------------- fig 4
names = ["Anchored\n(trained proj.)", "BC via same proj.\n(transfer)", "Anchored\n(identity proj.)"]
vals = [90.0, 58.8, 22.0]
cols = [C_AA, C_BC, "#999999"]
fig, ax = plt.subplots(figsize=(3.5, 2.7))
bars = ax.bar(names, vals, color=cols, width=0.6)
ax.axhline(16.7, color=C_BASE, ls="--", lw=1.0, label="chance (16.7%)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.5)
ax.legend(loc="upper right", frameon=False)
ax.set_ylabel("Frozen-head direction readout (%)")
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_readability.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f)
