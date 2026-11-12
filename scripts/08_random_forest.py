import os
import time
import warnings
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
DATE = "2022-10-25"

d = np.load(f"data/dataset_{DATE}.npz")
VVb, VHb, VVf, VHf = d["VVb"], d["VHb"], d["VVf"], d["VHf"]
label, valid, is_train = d["label"], d["valid"], d["is_train"]
H, W = label.shape
truth = label == 1
eval_m = valid & (label < 2)
train_m = (eval_m & is_train).ravel()
test_m = (eval_m & ~is_train).ravel()


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


dVV = VVf - VVb
dVH = VHf - VHb
feats = {
    "VV_flood": VVf, "VH_flood": VHf, "VV_before": VVb, "VH_before": VHb,
    "dVV": dVV, "dVH": dVH, "VV_minus_VH": VVf - VHf,
    "dVV_mean5": lmean(dVV, 5), "dVV_mean15": lmean(dVV, 15),
    "dVH_mean5": lmean(dVH, 5), "dVH_mean15": lmean(dVH, 15),
    "VV_flood_mean5": lmean(VVf, 5), "VV_flood_std5": lstd(VVf, 5),
}
names = list(feats)
X = np.stack([fill(v) for v in feats.values()], axis=-1).reshape(-1, len(names)).astype("float32")
y = truth.ravel()
print(f"{len(names)} features, {H} x {W} pixels")

tr_all = np.flatnonzero(train_m)
te = np.flatnonzero(test_m)
yt = y[te]
print(f"Test pixels {len(te):,}, flood share {yt.mean() * 100:.1f}% (a random guess would score about this precision)")


def metrics(p, t):
    tp = np.sum(p & t)
    fp = np.sum(p & ~t)
    fn = np.sum(~p & t)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return pr, rc, f1, iou


def tune(col, cands):
    best_t, best_f = None, -1
    for t in cands:
        f1 = metrics(X[tr_all, col] < t, y[tr_all])[2]
        if f1 > best_f:
            best_t, best_f = float(t), f1
    return best_t


rows = []
for nm, cands in [("VV_flood", np.arange(-22, -4, 1.0)),
                  ("dVV", np.arange(-8, -0.4, 0.5)),
                  ("dVH", np.arange(-8, -0.4, 0.5))]:
    col = names.index(nm)
    t = tune(col, cands)
    rows.append((f"Threshold: {nm} < {t:g} dB (tuned)", metrics(X[te, col] < t, yt)))

# Random Forest
rng = np.random.default_rng(0)
samp = rng.choice(tr_all, size=min(250000, len(tr_all)), replace=False)
print(f"Training Random Forest on {len(samp):,} pixels from the training blocks ...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=20,
                            class_weight="balanced_subsample", n_jobs=-1, random_state=42)
rf.fit(X[samp], y[samp])
print(f"  trained in {time.time() - t0:.0f} s")

vidx = np.flatnonzero(valid.ravel())
prob = np.full(H * W, np.nan, dtype="float32")
for i in range(0, len(vidx), 500000):
    ch = vidx[i:i + 500000]
    prob[ch] = rf.predict_proba(X[ch])[:, 1]
np.save(f"data/rf_prob_{DATE}.npy", prob.reshape(H, W))
rows.append(("Random Forest (probability >= 0.5)", metrics(prob[te] >= 0.5, yt)))

print("\nScores on held out test blocks against UNOSAT")
print(f"{'method':<44}{'precision':<11}{'recall':<9}{'F1':<8}{'IoU'}")
for nm, (pr, rc, f1, iou) in rows:
    print(f"{nm:<44}{pr:<11.3f}{rc:<9.3f}{f1:<8.3f}{iou:.3f}")

print("\nThreshold free comparison on the same test pixels (higher is better)")
print(f"{'score':<28}{'avg precision':<15}{'ROC AUC'}")
for nm, s in [("Random Forest probability", prob[te]),
              ("minus dVH", -X[te, names.index('dVH')]),
              ("minus dVV", -X[te, names.index('dVV')]),
              ("minus VV_flood", -X[te, names.index('VV_flood')])]:
    print(f"{nm:<28}{average_precision_score(yt, s):<15.3f}{roc_auc_score(yt, s):.3f}")

print("\nFeature importance")
for i in np.argsort(rf.feature_importances_)[::-1]:
    print(f"  {names[i]:<18}{rf.feature_importances_[i]:.3f}")

# Map of the held out blocks only
pred = (prob >= 0.5).reshape(H, W)
em = eval_m & ~is_train
rgb = np.ones((H, W, 3))
rgb[valid & ~is_train] = (0.92, 0.92, 0.92)
rgb[valid & ~is_train & (label >= 2)] = (0.55, 0.55, 0.55)
rgb[em & pred & truth] = (0.10, 0.35, 0.80)
rgb[em & pred & ~truth] = (0.85, 0.15, 0.15)
rgb[em & ~pred & truth] = (0.95, 0.60, 0.10)
os.makedirs("preview", exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(rgb)
ax.set_axis_off()
ax.set_title(f"Random Forest against UNOSAT, {DATE}\nHeld out test blocks only (training blocks left white)", fontsize=11)
ax.legend(handles=[
    Patch(color=(0.10, 0.35, 0.80), label="Agree: flood"),
    Patch(color=(0.85, 0.15, 0.15), label="Radar model flood, UNOSAT dry"),
    Patch(color=(0.95, 0.60, 0.10), label="UNOSAT flood missed by model"),
    Patch(color=(0.92, 0.92, 0.92), label="Agree: dry"),
    Patch(color=(0.55, 0.55, 0.55), label="Ignored (permanent water, cloud)"),
], loc="lower left", fontsize=8, frameon=True)
fig.savefig("preview/rf_agreement.png", dpi=130, bbox_inches="tight")
print("\nSaved preview/rf_agreement.png and data/rf_prob_2022-10-25.npy")
