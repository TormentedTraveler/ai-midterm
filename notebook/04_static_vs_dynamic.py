"""
Two checks after 03_diagnostics.py:
  A) within-file AUC: preictal vs the SAME file's earlier windows (removes the
     cross-file / state confound that can inflate the pooled AUC)
  B) static percentile threshold (calibrated on training scores) vs the dynamic one

Run next to the other scripts:  python 04_static_vs_dynamic.py
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

M = importlib.import_module("02_models_and_threshold")

QS = (99, 99.5, 99.9)
PERSIST = (2, 3, 5)
MODELS = ["IsolationForest", "Autoencoder"]

df = pd.read_csv(M.CSV)
df[M.LOG_FEATS] = np.log1p(df[M.LOG_FEATS])
files = sorted(df.file.unique())
seiz = sorted(df.loc[df.label == "ictal", "file"].unique())
pure = [f for f in files if f not in seiz]
folds = [[s] for s in seiz]
for i, f in enumerate(pure):
    folds[i % len(seiz)].append(f)


def static_alarm(s, thr, p):
    out, run = np.zeros(len(s), bool), 0
    for t, flag in enumerate(s > thr):
        run = run + 1 if flag else 0
        out[t] = run >= p
    return out


def evaluate_alarm(te, alarm):
    labels, t_end = te.label.values, te.t_end.values
    ev = M.alarm_events(alarm)
    pre = [t for t in ev if labels[t] == "preictal"]
    onset = te.loc[te.label == "ictal", "t_start"].min()
    lead = (onset - t_end[pre[0]]) / 60 if pre else np.nan
    fa = sum(labels[t] == "interictal" for t in ev)
    hrs = (labels == "interictal").sum() * M.WIN_S / 3600
    return bool(pre), lead, fa, hrs


within, rows = [], []
for held in folds:
    train = df[(~df.file.isin(held)) & (df.label == "interictal")]
    sc = StandardScaler().fit(train[M.FEATS])
    Xtr = sc.transform(train[M.FEATS])
    iso = IsolationForest(n_estimators=50, max_samples=256, random_state=0).fit(Xtr)
    ae = M.train_ae(Xtr)
    train_scores = {"IsolationForest": -iso.score_samples(Xtr),
                    "Autoencoder": M.ae_score(ae, Xtr)}

    for f in held:
        te = df[df.file == f].sort_values("t_start").reset_index(drop=True)
        X = sc.transform(te[M.FEATS])
        scores = {"IsolationForest": -iso.score_samples(X),
                  "Autoencoder": M.ae_score(ae, X)}
        labels = te.label.values

        # ---- A) within-file AUC ----
        if f in seiz:
            pre_idx = np.where(labels == "preictal")[0]
            base_idx = [i for i in range(pre_idx.min())
                        if labels[i] in ("interictal", "buffer")]
            if len(base_idx) >= 5:
                for name in MODELS:
                    y = np.r_[np.zeros(len(base_idx)), np.ones(len(pre_idx))]
                    s = np.r_[scores[name][base_idx], scores[name][pre_idx]]
                    within.append(dict(file=f, model=name, n_baseline=len(base_idx),
                                       auc=roc_auc_score(y, s)))

        # ---- B) static percentile threshold ----
        for name in MODELS:
            for q in QS:
                thr = np.percentile(train_scores[name], q)
                for p in PERSIST:
                    det, lead, fa, hrs = evaluate_alarm(te, static_alarm(scores[name], thr, p))
                    rows.append(dict(model=name, q=q, persist=p, file=f,
                                     detected=det and f in seiz, lead_min=lead,
                                     false_alarms=fa, interictal_h=hrs))
    print("done fold", held[0])

pd.set_option("display.width", 160)
w = pd.DataFrame(within)
print("\n=== Within-file preictal AUC (vs same file's earlier windows) ===")
print(w.pivot_table(index="file", columns="model", values="auc").round(2))
print("mean:", w.groupby("model").auc.mean().round(3).to_dict())
print("(chb01_21 is skipped: its seizure is too early in the file for a baseline)")

r = pd.DataFrame(rows)
summ = (r.groupby(["model", "q", "persist"])
        .agg(detected=("detected", "sum"), mean_lead_min=("lead_min", "mean"),
             false_alarms=("false_alarms", "sum"), interictal_h=("interictal_h", "sum"))
        .reset_index())
summ["FAR_per_24h"] = summ.false_alarms / summ.interictal_h * 24
summ["sensitivity"] = summ.detected / len(seiz)
print("\n=== Static percentile threshold ===")
print(summ.round(2).to_string(index=False))
w.to_csv("results_within_auc.csv", index=False)
summ.to_csv("results_static.csv", index=False)