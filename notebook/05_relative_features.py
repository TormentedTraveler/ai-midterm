"""
One time-boxed feature iteration. Decision metric: within-file preictal AUC
(preictal vs the same file's earlier interictal/buffer windows).

Part 1: per-feature within-file AUC on the raw features. AUC < 0.5 means the
        feature FALLS before seizures (an anomaly detector that only flags
        "unusual high" values will miss that).
Part 2: Isolation Forest / autoencoder on baseline-relative + trend features.

Run next to the other scripts:  python 05_relative_features.py
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

M = importlib.import_module("02_models_and_threshold")

BASE = 60      # causal baseline length (windows) = 30 min
MIN_HIST = 10  # windows of history required
SLOPE = 10     # trend over the last 10 windows = 5 min

df = pd.read_csv(M.CSV)
df[M.LOG_FEATS] = np.log1p(df[M.LOG_FEATS])


def add_relative(g):
    g = g.sort_values("t_start").copy()
    for c in M.FEATS:
        past = g[c].shift(1)                      # causal: excludes current window
        mu = past.rolling(BASE, min_periods=MIN_HIST).mean()
        sd = past.rolling(BASE, min_periods=MIN_HIST).std()
        g[f"{c}_z"] = (g[c] - mu) / (sd + 1e-6)
        g[f"{c}_slope"] = (g[c] - g[c].shift(SLOPE)) / SLOPE
    return g


df = pd.concat([add_relative(g) for _, g in df.groupby("file")])
REL = [f"{c}_z" for c in M.FEATS] + [f"{c}_slope" for c in M.FEATS]
df = df.dropna(subset=REL).reset_index(drop=True)
seiz = sorted(df.loc[df.label == "ictal", "file"].unique())


def within_auc(te, s):
    labels = te.label.values
    pre = np.where(labels == "preictal")[0]
    if len(pre) == 0:
        return None
    base = [i for i in range(pre.min()) if labels[i] in ("interictal", "buffer")]
    if len(base) < 5:
        return None
    y = np.r_[np.zeros(len(base)), np.ones(len(pre))]
    return roc_auc_score(y, np.r_[s[base], s[pre]])


pd.set_option("display.width", 170)

# ---------------- Part 1: direction check, one feature at a time ----------------
rows = []
for f in seiz:
    te = df[df.file == f].sort_values("t_start").reset_index(drop=True)
    for c in M.FEATS:
        a = within_auc(te, te[c].values)
        if a is not None:
            rows.append(dict(feature=c, file=f, auc=a))
uni = pd.DataFrame(rows).pivot_table(index="feature", columns="file", values="auc")
uni["mean"] = uni.mean(axis=1)
print("=== Raw feature, within-file AUC (>0.5 rises before seizure, <0.5 falls) ===")
print(uni.round(2).sort_values("mean"))

# ---------------- Part 2: models on relative features ----------------
res = []
sets = [("raw", M.FEATS), ("relative", REL), ("raw+relative", M.FEATS + REL)]
for f in seiz:
    train = df[(df.file != f) & (df.label == "interictal")]
    te = df[df.file == f].sort_values("t_start").reset_index(drop=True)
    for set_name, cols in sets:
        sc = StandardScaler().fit(train[cols])
        Xtr, Xte = sc.transform(train[cols]), sc.transform(te[cols])
        iso = IsolationForest(n_estimators=50, max_samples=256, random_state=0).fit(Xtr)
        ae = M.train_ae(Xtr)
        for name, s in (("IsolationForest", -iso.score_samples(Xte)),
                        ("Autoencoder", M.ae_score(ae, Xte))):
            a = within_auc(te, s)
            if a is not None:
                res.append(dict(features=set_name, model=name, file=f, auc=a))
    print("done", f)

r = pd.DataFrame(res).pivot_table(index=["features", "model"], columns="file", values="auc")
r["mean"] = r.mean(axis=1)
print("\n=== Within-file preictal AUC by feature set ===")
print(r.round(2))
uni.to_csv("results_univariate_within_auc.csv")
r.to_csv("results_relative_within_auc.csv")