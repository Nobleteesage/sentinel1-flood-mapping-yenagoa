import os
import shutil
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
DATE = "2022-10-25"
os.makedirs("figures", exist_ok=True)

d = np.load(f"data/dataset_{DATE}.npz")
VVb, VVf, VHb, VHf = d["VVb"], d["VVf"], d["VHb"], d["VHf"]
label, valid = d["label"], d["valid"]
dVV = VVf - VVb


def show(ax, img, **kw):
    cmap = matplotlib.colormaps[kw.pop("cmap", "gray")].copy()
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_where(~valid, img), cmap=cmap, interpolation="nearest", **kw)
    ax.set_axis_off()


# Figure 1: the data
fig, axs = plt.subplots(2, 2, figsize=(11, 11.5))
show(axs[0, 0], VVb, vmin=-25, vmax=0)
axs[0, 0].set_title("A. VV before the flood\nmedian of 5 scenes, 2 Aug to 19 Sep 2022", fontsize=10)
show(axs[0, 1], VVf, vmin=-25, vmax=0)
axs[0, 1].set_title("B. VV on 25 Oct 2022", fontsize=10)
show(axs[1, 0], dVV, cmap="RdBu", vmin=-6, vmax=6)
axs[1, 0].set_title("C. Change in VV, B minus A\nblue is darker than before, red is brighter", fontsize=10)
lab_cmap = ListedColormap(["#e6e6e6", "#1f5fbf", "#555555", "#f2c14e"])
lab = np.ma.masked_where(~valid, label)
lab_cmap.set_bad("white")
axs[1, 1].imshow(lab, cmap=lab_cmap, vmin=0, vmax=3, interpolation="nearest")
axs[1, 1].set_axis_off()
axs[1, 1].set_title("D. UNOSAT VIIRS reference, 1 to 25 Oct 2022\nabout 375 m pixels", fontsize=10)
axs[1, 1].legend(handles=[Patch(color="#e6e6e6", label="Dry"), Patch(color="#1f5fbf", label="Flood"),
                          Patch(color="#555555", label="Permanent water (ignored)"),
                          Patch(color="#f2c14e", label="Cloud (ignored)")],
                 loc="lower left", fontsize=8)
fig.suptitle("Sentinel-1 radar and the UNOSAT flood reference, Yenagoa LGA", fontsize=12)
fig.tight_layout()
fig.savefig("figures/fig1_data.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Figure 2: agreement map made earlier by s1_rf.py
if os.path.exists("preview/rf_agreement.png"):
    shutil.copy("preview/rf_agreement.png", "figures/fig2_agreement.png")

# Figure 3: scores over 5 block splits
r = np.loadtxt("results/seed_results.csv", delimiter=",", skiprows=1)
mean, sd = r.mean(axis=0), r.std(axis=0, ddof=1)
items = [("Chance (flood share)", mean[7], sd[7], "#b8b8b8"),
         ("Single pixel VH change rule", mean[3], sd[3], "#8a8a8a"),
         ("Smoothed 15 pixel VH change rule", mean[4], sd[4], "#5c7a99"),
         ("Random Forest, 13 features", mean[0], sd[0], "#1f5fbf")]
fig, ax = plt.subplots(figsize=(8, 3.6))
for i, (nm, m, s, c) in enumerate(items[::-1]):
    ax.barh(i, m, xerr=s, color=c, capsize=3, height=0.62)
    ax.text(m + s + 0.012, i, f"{m:.2f}", va="center", fontsize=9)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([x[0] for x in items[::-1]], fontsize=9)
ax.set_xlim(0, 0.85)
ax.set_xlabel("Average precision on held out spatial blocks (mean and spread over 5 splits)", fontsize=9)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("figures/fig3_scores.png", dpi=150, bbox_inches="tight")
print("Saved figures/fig1_data.png, figures/fig2_agreement.png, figures/fig3_scores.png")
