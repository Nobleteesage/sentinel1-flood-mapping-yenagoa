import os
import time
import warnings
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import box
from shapely.validation import make_valid
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
NODATA = -32768
S = 15
base = os.path.expanduser("~/nigeria-flood-access/data/flood")
nat = f"{base}/FL20221019NGA_SHP"


def fill(x):
    y = x.copy()
    y[np.isnan(y)] = np.nanmedian(x)
    return y


def lmean(x, s):
    return ndi.uniform_filter(fill(x), size=s)


def lstd(x, s):
    xf = fill(x)
    m = ndi.uniform_filter(xf, size=s)
    m2 = ndi.uniform_filter(xf * xf, size=s)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def build(VVb, VHb, VVf, VHf):
    dVV = VVf - VVb
    dVH = VHf - VHb
    feats = {
        "VV_flood": VVf, "VH_flood": VHf, "VV_before": VVb, "VH_before": VHb,
        "dVV": dVV, "dVH": dVH, "VV_minus_VH": VVf - VHf,
        "dVV_mean5": lmean(dVV, 5), "dVV_mean15": lmean(dVV, 15),
        "dVH_mean5": lmean(dVH, 5), "dVH_mean15": lmean(dVH, 15),
        "VV_flood_mean5": lmean(VVf, 5), "VV_flood_std5": lstd(VVf, 5),
    }
    X = np.stack([fill(v) for v in feats.values()], axis=-1)
    return X.reshape(-1, X.shape[-1]).astype("float32"), dVH


# Oct 25 data, used for training only
d = np.load("data/dataset_2022-10-25.npz")
label25, valid25, is_train = d["label"], d["valid"], d["is_train"]
H, W = label25.shape
X25, _ = build(d["VVb"], d["VHb"], d["VVf"], d["VHf"])
eval25 = valid25 & (label25 < 2)
tr = np.flatnonzero((eval25 & is_train).ravel())
y25 = (label25 == 1).ravel()
rng = np.random.default_rng(0)
samp = rng.choice(tr, size=min(250000, len(tr)), replace=False)
print(f"Training on Oct 25 radar and labels, {len(samp):,} pixels from the training blocks ...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=20,
                            class_weight="balanced_subsample", n_jobs=-1, random_state=42)
rf.fit(X25[samp], y25[samp])
print(f"  trained in {time.time() - t0:.0f} s")
del X25

# Oct 13 radar and Oct 13 to 17 UNOSAT window, never seen by the model
with rasterio.open("data/s1_yenagoa_2022-10-13.tif") as src:
    arr = src.read()
    transform, crs, bounds = src.transform, src.crs, src.bounds
    shape13 = (src.height, src.width)
with rasterio.open("data/s1_yenagoa_2022-10-25.tif") as src:
    same_grid = (src.transform == transform) and ((src.height, src.width) == shape13)
print("Oct 13 and Oct 25 rasters share the same grid:", same_grid)
if not same_grid or shape13 != (H, W):
    raise SystemExit("The two rasters are not on the same grid, stop here and send me this message.")

valid13 = (arr != NODATA).all(axis=0) & np.load("data/inside_lga.npy")
a = np.where(arr == NODATA, np.nan, arr / 100.0).astype("float32")
X13, dVH13 = build(a[0], a[1], a[2], a[3])

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


w = "20221013_20221017"
flood = burn(f"{nat}/VIIRS_{w}_MaximumFloodWaterExtent_Nigeria.shp")
perm = burn(f"{nat}/VIIRS_{w}_PermanentWater_Nigeria.shp")
cloud = burn(f"{nat}/VIIRS_{w}_CloudObstruction_Nigeria.shp")
label13 = np.zeros((H, W), np.uint8)
label13[cloud] = 3
label13[flood] = 1
label13[perm] = 2
for k, nm in enumerate(["dry land", "UNOSAT flood", "permanent water (ignored)", "cloud (ignored)"]):
    n = int(((label13 == k) & valid13).sum())
    print(f"  Oct 13 to 17 label {k} {nm:<28}{n:>10,}  {n / valid13.sum() * 100:5.1f}%")

eval13 = valid13 & (label13 < 2)
truth13 = label13 == 1
te = np.flatnonzero((eval13 & ~is_train).ravel())
yt = truth13.ravel()[te]
print(f"\nTransfer test pixels (Oct 13, spatially held out blocks): {len(te):,}, flood share {yt.mean() * 100:.1f}%")

prob = np.full(H * W, np.nan, dtype="float32")
idx = np.flatnonzero(eval13.ravel())
for i in range(0, len(idx), 500000):
    ch = idx[i:i + 500000]
    prob[ch] = rf.predict_proba(X13[ch])[:, 1]

p = prob[te] >= 0.5
tp, fp, fn = np.sum(p & yt), np.sum(p & ~yt), np.sum(~p & yt)
pr, rc = tp / (tp + fp), tp / (tp + fn)
print("\nPixel level, Random Forest trained on Oct 25, tested on Oct 13")
print(f"  precision {pr:.3f}  recall {rc:.3f}  F1 {2 * pr * rc / (pr + rc):.3f}  IoU {tp / (tp + fp + fn):.3f}")
print(f"{'score':<28}{'avg precision':<15}{'ROC AUC'}")
for nm, s in [("Random Forest probability", prob[te]), ("minus dVH (simple rule)", -dVH13.ravel()[te])]:
    print(f"{nm:<28}{average_precision_score(yt, s):<15.3f}{roc_auc_score(yt, s):.3f}")

# Same test at the 375 m cell scale
Hc, Wc = H // S * S, W // S * S


def cells(x):
    return x[:Hc, :Wc].reshape(Hc // S, S, Wc // S, S)


n_eval = cells(eval13.astype(float)).sum(axis=(1, 3))
n_flood = cells((truth13 & eval13).astype(float)).sum(axis=(1, 3))
train_share = cells(is_train.astype(float)).mean(axis=(1, 3))
pf = prob.reshape(H, W)
rf_sum = cells(np.where(eval13, np.nan_to_num(pf), 0.0)).sum(axis=(1, 3))
bl_sum = cells(np.where(eval13, np.nan_to_num(-dVH13.reshape(H, W)), 0.0)).sum(axis=(1, 3))
ok = (n_eval >= 0.8 * S * S) & (train_share == 0)
frac = n_flood[ok] / n_eval[ok]
rfc, blc = rf_sum[ok] / n_eval[ok], bl_sum[ok] / n_eval[ok]
tc = frac >= 0.5
print(f"\nCell level (about 375 m): {ok.sum():,} cells, {tc.sum():,} UNOSAT flood cells ({tc.mean() * 100:.1f}%)")
print(f"  correlation with UNOSAT flood fraction: Random Forest {np.corrcoef(frac, rfc)[0, 1]:.3f}, minus dVH {np.corrcoef(frac, blc)[0, 1]:.3f}")
print(f"{'score':<28}{'avg precision':<15}{'ROC AUC'}")
for nm, s in [("Random Forest, cell mean", rfc), ("minus dVH, cell mean", blc)]:
    print(f"{nm:<28}{average_precision_score(tc, s):<15.3f}{roc_auc_score(tc, s):.3f}")
