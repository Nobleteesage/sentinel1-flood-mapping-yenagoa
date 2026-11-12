import warnings
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
DATE = "2022-10-25"
S = 15   # 15 pixels of 25 m is about 375 m, the size of one VIIRS pixel

d = np.load(f"data/dataset_{DATE}.npz")
prob = np.load(f"data/rf_prob_{DATE}.npy")
label, valid, is_train = d["label"], d["valid"], d["is_train"]
dVH = d["VHf"] - d["VHb"]
H, W = label.shape
Hc, Wc = H // S * S, W // S * S


def cells(a):
    return a[:Hc, :Wc].reshape(Hc // S, S, Wc // S, S)


eval_m = valid & (label < 2)
truth = label == 1
n_eval = cells(eval_m.astype(float)).sum(axis=(1, 3))
n_flood = cells((truth & eval_m).astype(float)).sum(axis=(1, 3))
train_share = cells(is_train.astype(float)).mean(axis=(1, 3))
rf_sum = cells(np.where(eval_m, np.nan_to_num(prob), 0.0)).sum(axis=(1, 3))
bl_sum = cells(np.where(eval_m, np.nan_to_num(-dVH), 0.0)).sum(axis=(1, 3))

ok = (n_eval >= 0.8 * S * S) & (train_share == 0)
frac = n_flood[ok] / n_eval[ok]
rf = rf_sum[ok] / n_eval[ok]
bl = bl_sum[ok] / n_eval[ok]
truth_c = frac >= 0.5

print(f"Cells of about 375 m used (test blocks only, at least 80% evaluable): {ok.sum():,}")
print(f"UNOSAT flood cells (at least half the cell flooded): {truth_c.sum():,} ({truth_c.mean() * 100:.1f}%)")
print(f"Correlation between UNOSAT flood fraction and mean Random Forest probability: {np.corrcoef(frac, rf)[0, 1]:.3f}")
print(f"Correlation between UNOSAT flood fraction and mean minus dVH:               {np.corrcoef(frac, bl)[0, 1]:.3f}")
print(f"Mean RF probability in UNOSAT flood cells: {rf[truth_c].mean():.3f}, in dry cells: {rf[~truth_c].mean():.3f}")

print("\nCell level ranking quality")
print(f"{'score':<28}{'avg precision':<15}{'ROC AUC'}")
for nm, s in [("Random Forest, cell mean", rf), ("minus dVH, cell mean", bl)]:
    print(f"{nm:<28}{average_precision_score(truth_c, s):<15.3f}{roc_auc_score(truth_c, s):.3f}")

print("\nRandom Forest at cell level, flagging a cell when its mean probability reaches the value")
print(f"{'cutoff':<9}{'precision':<11}{'recall':<9}{'F1'}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    p = rf >= t
    tp = np.sum(p & truth_c)
    fp = np.sum(p & ~truth_c)
    fn = np.sum(~p & truth_c)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    print(f"{t:<9}{pr:<11.3f}{rc:<9.3f}{f1:.3f}")
