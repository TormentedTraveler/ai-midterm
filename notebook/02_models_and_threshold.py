"""
Phase 3-4: anomaly models + dynamic threshold + leave-one-seizure-out evaluation

Install:  pip install scikit-learn torch pandas numpy
Run:      python 02_models_and_threshold.py     (needs features_chb01.csv from script 01)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

CSV = "features_chb01.csv"
FEATS = ["line_length", "variance", "rms", "hjorth_mobility", "hjorth_complexity",
         "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
         "theta_beta_ratio"]
LOG_FEATS = ["line_length", "variance", "rms", "theta_beta_ratio"]  # heavy-tailed
WIN_S = 30
BASE_WIN = 60        # rolling baseline length in windows (60 x 30 s = 30 min)
MIN_HIST = 10        # windows of history needed before an alarm is allowed
REFRACT = 10         # windows (5 min) between counted alarm events
GRID = [(k, m) for k in (2, 3, 4) for m in (2, 3, 5)]   # (k, persistence)

torch.manual_seed(0)


# ----------------------------- models ----------------------------------
class AE(nn.Module):
    def __init__(self, d, h=8, z=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(X, epochs=300):
    m = AE(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    Xt = torch.tensor(X, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((m(Xt) - Xt) ** 2).mean()
        loss.backward()
        opt.step()
    return m


def ae_score(m, X):
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return ((m(Xt) - Xt) ** 2).mean(1).numpy()


# ------------------------ dynamic threshold -----------------------------
def alarm_stream(scores, k, persist):
    """Alarm at t if score > mean + k*std of the previous BASE_WIN scores
    for `persist` consecutive windows."""
    n = len(scores)
    flag = np.zeros(n, bool)
    for t in range(n):
        hist = scores[max(0, t - BASE_WIN):t]
        if len(hist) < MIN_HIST:
            continue
        flag[t] = scores[t] > hist.mean() + k * hist.std()
    alarm = np.zeros(n, bool)
    run = 0
    for t in range(n):
        run = run + 1 if flag[t] else 0
        alarm[t] = run >= persist
    return alarm


def alarm_events(alarm):
    ev, last = [], -10 ** 9
    for t in np.where(alarm)[0]:
        if t - last >= REFRACT:
            ev.append(t)
            last = t
    return ev


def evaluate_file(test, scores, k, persist):
    """Returns (detected, lead_min, n_false_alarms, interictal_hours)."""
    labels = test["label"].values
    t_end = test["t_end"].values
    ev = alarm_events(alarm_stream(scores, k, persist))
    onset = test.loc[test.label == "ictal", "t_start"].min()   # approx. onset
    pre_ev = [t for t in ev if labels[t] == "preictal"]
    detected = len(pre_ev) > 0
    lead = (onset - t_end[pre_ev[0]]) / 60 if detected else np.nan
    false_alarms = sum(1 for t in ev if labels[t] == "interictal")
    hours = (labels == "interictal").sum() * WIN_S / 3600
    return detected, lead, false_alarms, hours


# ------------------------------ main ------------------------------------
def main():
    df = pd.read_csv(CSV)
    df[LOG_FEATS] = np.log1p(df[LOG_FEATS])
    seizure_files = sorted(df.loc[df.label == "ictal", "file"].unique())
    print(f"{len(seizure_files)} seizure files -> leave-one-seizure-out folds\n")

    rows = []
    for F in seizure_files:
        train = df[(df.file != F) & (df.label == "interictal")]
        test = df[df.file == F].sort_values("t_start").reset_index(drop=True)

        sc = StandardScaler().fit(train[FEATS])
        Xtr, Xte = sc.transform(train[FEATS]), sc.transform(test[FEATS])

        iso = IsolationForest(n_estimators=50, max_samples=256, random_state=0).fit(Xtr)
        scores = {"IsolationForest": -iso.score_samples(Xte),
                  "Autoencoder": ae_score(train_ae(Xtr), Xte)}

        for name, s in scores.items():
            for k, m in GRID:
                det, lead, fa, hrs = evaluate_file(test, s, k, m)
                rows.append(dict(file=F, model=name, k=k, persist=m,
                                 detected=det, lead_min=lead,
                                 false_alarms=fa, interictal_h=hrs))
        print("done", F)

    res = pd.DataFrame(rows)
    res.to_csv("results_loso.csv", index=False)

    summ = (res.groupby(["model", "k", "persist"])
            .agg(detected=("detected", "sum"),
                 mean_lead_min=("lead_min", "mean"),
                 false_alarms=("false_alarms", "sum"),
                 interictal_h=("interictal_h", "sum"))
            .reset_index())
    summ["FAR_per_24h"] = summ.false_alarms / summ.interictal_h * 24
    summ["sensitivity"] = summ.detected / len(seizure_files)
    pd.set_option("display.width", 160)
    print("\n=== Summary (pooled over folds) ===")
    print(summ.round(2).to_string(index=False))
    summ.to_csv("results_summary.csv", index=False)


if __name__ == "__main__":
    main()