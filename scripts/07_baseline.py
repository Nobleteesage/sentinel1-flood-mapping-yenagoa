import os
import warnings
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import box
from shapely.validation import make_valid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

DATE = "2022-10-25"
NODATA = -32768
base = os.path.expanduser("~/nigeria-flood-access/data/flood")
nat = f"{base}/FL20221019NGA_SHP"

# 1. Radar bands
with rasterio.open(f"data/s1_yenagoa_{DATE}.tif") as src:
    arr = src.read()
    transform, crs = src.transform, src.crs
    H, W = src.height, src.width
    bounds = src.bounds
print(f"Raster {W} x {H}, bands {arr.shape[0]}, CRS {crs}")
valid = (arr != NODATA).all(axis=0) & np.load("data/inside_lga.npy")
a = np.where(arr == NODATA, np.nan, arr / 100.0).astype("float32")
VVb, VHb, VVf, VHf = a
dVV = VVf - VVb
dVH = VHf - VHb
print(f"Valid pixels: {valid.sum():,} ({valid.mean() * 100:.1f}% of the bounding box)")

# 2. UNOSAT labels on the same grid
bb = gpd.GeoSeries([box(*bounds)], crs=crs).to_crs(4326).total_bounds


def burn(path):
    g = gpd.read_file(path, bbox=tuple(bb))
    if len(g) == 0:
        return np.zeros((H, W), bool)
    g["geometry"] = g.geometry.apply(make_valid)
    g = g.explode(index_parts=False)
    g = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    g = g.to_crs(4326).to_crs(crs)
    shapes = [(geom, 1) for geom in g.geometry if not geom.is_empty]
    if not shapes:
        return np.zeros((H, W), bool)
    return features.rasterize(shapes, out_shape=(H, W), transform=transform,
                              fill=0, dtype="uint8").astype(bool)


flood = burn(f"{base}/yenagoa_clipped/yenagoa_flood_final.shp")
perm = burn(f"{nat}/VIIRS_20221001_20221025_PermanentWater_Nigeria.shp")
cloud = burn(f"{nat}/VIIRS_20221001_20221025_CloudObstruction_Nigeria.shp")

label = np.zeros((H, W), np.uint8)   # 0 dry, 1 flood, 2 permanent water, 3 cloud
label[cloud] = 3
label[flood] = 1
label[perm] = 2
for k, nm in enumerate(["dry land", "UNOSAT flood", "permanent water (ignored)", "cloud (ignored)"]):
    n = int(((label == k) & valid).sum())
    print(f"  label {k} {nm:<28}{n:>10,}  {n / valid.sum() * 100:5.1f}% of valid pixels")

truth = label == 1
eval_m = valid & (label < 2)

# 3. Spatial blocks (about 2 km), the same split will be reused for Random Forest
BS = 80
rows, cols = np.indices((H, W))
block = (rows // BS) * ((W + BS - 1) // BS) + (cols // BS)
rng = np.random.default_rng(42)
train_blocks = rng.random(block.max() + 1) < 0.6
is_train = train_blocks[block]
train_m = eval_m & is_train
test_m = eval_m & ~is_train
print(f"Train pixels {train_m.sum():,}, test pixels {test_m.sum():,} (blocks of {BS} x {BS} pixels, seed 42)")
print(f"Flood share: train {truth[train_m].mean() * 100:.1f}%, test {truth[test_m].mean() * 100:.1f}%")


def metrics(pred, m):
    p, t = pred[m], truth[m]
    tp = np.sum(p & t)
    fp = np.sum(p & ~t)
    fn = np.sum(~p & t)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return prec, rec, f1, iou


def best_threshold(feature, cands):
    best = None
    for t in cands:
        r = metrics(feature < t, train_m)
        if best is None or r[2] > best[1][2]:
            best = (float(t), r)
    return best[0]


t_vv = best_threshold(VVf, np.arange(-22, -9, 1.0))
t_dvv = best_threshold(dVV, np.arange(-8, -0.4, 0.5))
t_dvh = best_threshold(dVH, np.arange(-8, -0.4, 0.5))

rules = {
    "VV < -18 dB (fixed)": VVf < -18,
    "VV change < -3 dB (fixed)": dVV < -3,
    f"VV < {t_vv:g} dB (tuned on train)": VVf < t_vv,
    f"VV change < {t_dvv:g} dB (tuned)": dVV < t_dvv,
    f"VH change < {t_dvh:g} dB (tuned)": dVH < t_dvh,
    f"VV < {t_vv:g} and VV change < {t_dvv:g}": (VVf < t_vv) & (dVV < t_dvv),
}

print("\nScores on held out test blocks, against UNOSAT (permanent water and cloud pixels excluded)")
print(f"{'rule':<42}{'precision':<11}{'recall':<9}{'F1':<8}{'IoU'}")
scores = {}
for name, pred in rules.items():
    pr, rc, f1, iou = metrics(pred, test_m)
    scores[name] = f1
    print(f"{name:<42}{pr:<11.3f}{rc:<9.3f}{f1:<8.3f}{iou:.3f}")

# 4. Save the dataset for the Random Forest step
np.savez_compressed(f"data/dataset_{DATE}.npz", VVb=VVb, VHb=VHb, VVf=VVf, VHf=VHf,
                    label=label, valid=valid, block=block, is_train=is_train)
print(f"\nSaved data/dataset_{DATE}.npz")

# 5. Agreement map for the best rule
best = max(scores, key=scores.get)
pred = rules[best]
rgb = np.ones((H, W, 3))
rgb[valid] = (0.92, 0.92, 0.92)
rgb[valid & (label >= 2)] = (0.55, 0.55, 0.55)
rgb[eval_m & pred & truth] = (0.10, 0.35, 0.80)
rgb[eval_m & pred & ~truth] = (0.85, 0.15, 0.15)
rgb[eval_m & ~pred & truth] = (0.95, 0.60, 0.10)
os.makedirs("preview", exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(rgb)
ax.set_axis_off()
ax.set_title(f"Radar rule against UNOSAT flood, {DATE}\nRule: {best}", fontsize=11)
ax.legend(handles=[
    Patch(color=(0.10, 0.35, 0.80), label="Radar and UNOSAT agree: flood"),
    Patch(color=(0.85, 0.15, 0.15), label="Radar flood, UNOSAT dry"),
    Patch(color=(0.95, 0.60, 0.10), label="UNOSAT flood missed by radar"),
    Patch(color=(0.92, 0.92, 0.92), label="Agree: dry"),
    Patch(color=(0.55, 0.55, 0.55), label="Ignored (permanent water, cloud)"),
], loc="lower left", fontsize=8, frameon=True)
fig.savefig("preview/baseline_agreement.png", dpi=130, bbox_inches="tight")
print("Saved preview/baseline_agreement.png using rule:", best)
