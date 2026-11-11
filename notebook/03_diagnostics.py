"""
Fixes the FAR denominator and adds a threshold-free check of whether the features
carry any preictal signal.

Put this next to 02_models_and_threshold.py and features_chb01.csv, then:
    python 03_diagnostics.py

Design: 7 folds. Each fold holds out ONE seizure file plus ~4 seizure-free files.
Models train on interictal windows from all other files. FAR is therefore measured
on ~30 h of data instead of ~1 h.
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

M = importlib.import_module("02_models_and_threshold")   # reuse its functions

df = pd.read_csv(M.CSV)
df[M.LOG_FEATS] = np.log1p(df[M.LOG_FEATS])

files = sorted(df.file.unique())
seiz = sorted(df.loc[df.label == "ictal", "file"].unique())
pure = [f for f in files if f not in seiz]

folds = [[s] for s in seiz]
for i, f in enumerate(pure):
    folds[i % len(seiz)].append(f)

MODELS = ["IsolationForest", "Autoencoder"]
auc_rows, eval_rows = [], []

for held in folds:
    train = df[(~df.file.isin(held)) & (df.label == "interictal")]
    sc = StandardScaler().fit(train[M.FEATS])
    Xtr = sc.transform(train[M.FEATS])
    iso = IsolationForest(n_estimators=50, max_samples=256, random_state=0).fit(Xtr)
    ae = M.train_ae(Xtr)

    parts = {}
    for f in held:
        te = df[df.file == f].sort_values("t_start").reset_index(drop=True)
        X = sc.transform(te[M.FEATS])
        parts[f] = (te, {"IsolationForest": -iso.score_samples(X),
                         "Autoencoder": M.ae_score(ae, X)})

    # ---- threshold-free AUC on all held-out windows of this fold ----
    labels = np.concatenate([parts[f][0]["label"].values for f in held])
    for name in MODELS:
        s = np.concatenate([parts[f][1][name] for f in held])
        for target in ("preictal", "ictal"):
            m = np.isin(labels, ["interictal", target])
            auc_rows.append(dict(fold=held[0], model=name, target=target,
                                 auc=roc_auc_score(labels[m] == target, s[m])))

    # ---- dynamic-threshold evaluation per file ----
    for name in MODELS:
        for k, p in M.GRID:
            for f in held:
                te, sc_dict = parts[f]
                det, lead, fa, hrs = M.evaluate_file(te, sc_dict[name], k, p)
                eval_rows.append(dict(model=name, k=k, persist=p, file=f,
                                      is_seizure=f in seiz, detected=det and f in seiz,
                                      lead_min=lead, false_alarms=fa, interictal_h=hrs))
    print("done fold", held[0])

pd.set_option("display.width", 160)

auc = pd.DataFrame(auc_rows)
print("\n=== Threshold-free AUC (0.5 = no signal, 1.0 = perfect) ===")
print(auc.pivot_table(index=["model", "target"], columns="fold", values="auc").round(2))
print("\nMean AUC:")
print(auc.groupby(["model", "target"]).auc.mean().round(3))

ev = pd.DataFrame(eval_rows)
summ = (ev.groupby(["model", "k", "persist"])
        .agg(detected=("detected", "sum"), mean_lead_min=("lead_min", "mean"),
             false_alarms=("false_alarms", "sum"), interictal_h=("interictal_h", "sum"))
        .reset_index())
summ["FAR_per_24h"] = summ.false_alarms / summ.interictal_h * 24
summ["sensitivity"] = summ.detected / len(seiz)
print("\n=== Dynamic threshold, FAR over all held-out files ===")
print(summ.round(2).to_string(index=False))
auc.to_csv("results_auc.csv", index=False)
summ.to_csv("results_summary_v2.csv", index=False)