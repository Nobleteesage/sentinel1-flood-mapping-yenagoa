import os
import time
import warnings
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
DATE = "2022-10-25"
SEEDS = [42, 1, 2, 3, 4]


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


d = np.load(f"data/dataset_{DATE}.npz")
VVb, VHb, VVf, VHf = d["VVb"], d["VHb"], d["VVf"], d["VHf"]
label, valid, block = d["label"], d["valid"], d["block"]
dVV, dVH = VVf - VVb, VHf - VHb
feats = {
    "VV_flood": VVf, "VH_flood": VHf, "VV_before": VVb, "VH_before": VHb,
    "dVV": dVV, "dVH": dVH, "VV_minus_VH": VVf - VHf,
    "dVV_mean5": lmean(dVV, 5), "dVV_mean15": lmean(dVV, 15),
    "dVH_mean5": lmean(dVH, 5), "dVH_mean15": lmean(dVH, 15),
    "VV_flood_mean5": lmean(VVf, 5), "VV_flood_std5": lstd(VVf, 5),
}
names = list(feats)
eval_m = valid & (label < 2)
ie = np.flatnonzero(eval_m.ravel())
Xe = np.stack([fill(v).ravel()[ie] for v in feats.values()], axis=-1).astype("float32")
y = (label.ravel()[ie] == 1)
be = block.ravel()[ie]
col = {n: i for i, n in enumerate(names)}
print(f"{len(ie):,} evaluable pixels inside Yenagoa, flood share {y.mean() * 100:.1f}%")


def f1_of(p, t):
    tp, fp, fn = np.sum(p & t), np.sum(p & ~t), np.sum(~p & t)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def tune(x, t, cands):
    best_t, best_f = None, -1
    for c in cands:
        f = f1_of(x < c, t)
        if f > best_f:
            best_t, best_f = c, f
    return best_t


results = []
print(f"\n{'seed':<6}{'RF AP':<9}{'RF AUC':<9}{'RF F1':<9}{'dVH AP':<9}{'dVH15 AP':<10}{'dVV15 AP':<10}{'dVH15 rule F1'}")
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    tb = rng.random(be.max() + 1) < 0.6
    istr = tb[be]
    tr, te = np.flatnonzero(istr), np.flatnonzero(~istr)
    samp = rng.choice(tr, size=min(150000, len(tr)), replace=False)
    rf = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=20,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    rf.fit(Xe[samp], y[samp])
    p = rf.predict_proba(Xe[te])[:, 1]
    yt = y[te]
    rf_ap, rf_auc, rf_f1 = average_precision_score(yt, p), roc_auc_score(yt, p), f1_of(p >= 0.5, yt)
    s_dvh = average_precision_score(yt, -Xe[te, col["dVH"]])
    s15h = average_precision_score(yt, -Xe[te, col["dVH_mean15"]])
    s15v = average_precision_score(yt, -Xe[te, col["dVV_mean15"]])
    thr = tune(Xe[tr, col["dVH_mean15"]], y[tr], np.arange(-6, -0.2, 0.25))
    s15f = f1_of(Xe[te, col["dVH_mean15"]] < thr, yt)
    results.append([rf_ap, rf_auc, rf_f1, s_dvh, s15h, s15v, s15f, yt.mean()])
    print(f"{seed:<6}{rf_ap:<9.3f}{rf_auc:<9.3f}{rf_f1:<9.3f}{s_dvh:<9.3f}{s15h:<10.3f}{s15v:<10.3f}{s15f:.3f}")

r = np.array(results)
labels = ["RF average precision", "RF ROC AUC", "RF F1 (cutoff 0.5)",
          "Single pixel dVH rule, AP", "Smoothed 15 px dVH rule, AP",
          "Smoothed 15 px dVV rule, AP", "Smoothed 15 px dVH rule, F1 (tuned)",
          "Flood share in test blocks (chance AP)"]
print("\nMean and spread over 5 different random block splits")
for i, nm in enumerate(labels):
    print(f"  {nm:<42}{r[:, i].mean():.3f}  +/- {r[:, i].std(ddof=1):.3f}")
os.makedirs("results", exist_ok=True)
np.savetxt("results/seed_results.csv", r, delimiter=",", header=",".join(labels), fmt="%.4f")
print("\nSaved results/seed_results.csv")
