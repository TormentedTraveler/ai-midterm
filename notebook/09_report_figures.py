"""
Builds the report figures from the CSV files written by scripts 01-04.
Run from the folder that contains them:   python 09_report_figures.py
Output: report_figures/*.png  (150 dpi)
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("report_figures")
OUT.mkdir(exist_ok=True)
COLORS = {"IsolationForest": "#1f77b4", "Autoencoder": "#d95f02"}


def have(path):
    ok = Path(path).exists()
    if not ok:
        print(f"missing {path} -> figure skipped")
    return ok


# ---- Figure: class balance ----
if have("features_chb01.csv"):
    df = pd.read_csv("features_chb01.csv")
    order = ["interictal", "postictal", "buffer", "preictal", "ictal"]
    counts = df["label"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(order, counts.values,
                  color=["#7f7f7f", "#bcbd22", "#c7c7c7", "#ff9f1c", "#d62728"])
    ax.set_yscale("log")
    ax.set_ylabel("30 s windows (log scale)")
    ax.set_title("chb01: window labels")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{int(v)}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_class_balance.png", dpi=150)
    plt.close(fig)

# ---- Figure: pooled vs within-file preictal AUC ----
if have("results_auc.csv") and have("results_within_auc.csv"):
    pooled = pd.read_csv("results_auc.csv")
    pooled = pooled[pooled.target == "preictal"]
    within = pd.read_csv("results_within_auc.csv")
    models = ["IsolationForest", "Autoencoder"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.36
    rng = np.random.default_rng(0)
    for j, (label, data) in enumerate([("pooled (all held-out files)", pooled),
                                       ("within-file", within)]):
        for i, m in enumerate(models):
            vals = data[data.model == m].auc.values
            x = i + (j - 0.5) * w
            ax.bar(x, vals.mean(), w * 0.95, color=COLORS[m], alpha=0.45 + 0.4 * j,
                   label=f"{label}" if i == 0 else None, edgecolor="black", linewidth=0.5)
            ax.scatter(x + rng.uniform(-0.05, 0.05, len(vals)), vals, color="black", s=12, zorder=3)
    ax.axhline(0.5, ls="--", color="gray")
    ax.text(1.55, 0.51, "chance", color="gray", ha="right")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("preictal vs interictal AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pooled AUC overstates the signal (dots = individual seizures)")
    ax.legend(loc="upper right", fontsize=8, title="lighter = pooled, darker = within-file")
    fig.tight_layout()
    fig.savefig(OUT / "fig_auc_pooled_vs_within.png", dpi=150)
    plt.close(fig)

    # per-seizure within-file AUC
    pv = within.pivot_table(index="file", columns="model", values="auc")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(pv))
    for i, m in enumerate(models):
        if m in pv:
            ax.bar(x + (i - 0.5) * 0.38, pv[m].values, 0.36, label=m, color=COLORS[m])
    ax.axhline(0.5, ls="--", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(".edf", "") for f in pv.index], rotation=30)
    ax.set_ylabel("within-file preictal AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Within-file AUC per seizure")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_within_file_auc_by_seizure.png", dpi=150)
    plt.close(fig)

# ---- Figure: sensitivity vs false alarms, dynamic and static thresholds ----
if have("results_summary_v2.csv") and have("results_static.csv"):
    dyn = pd.read_csv("results_summary_v2.csv")
    sta = pd.read_csv("results_static.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in COLORS:
        d, s = dyn[dyn.model == m], sta[sta.model == m]
        ax.scatter(d.FAR_per_24h, d.sensitivity, color=COLORS[m], marker="o", s=45,
                   label=f"{m}, dynamic", alpha=0.8)
        ax.scatter(s.FAR_per_24h, s.sensitivity, color=COLORS[m], marker="s", s=45,
                   label=f"{m}, static", alpha=0.8, facecolors="none")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("false alarms per 24 h (symlog)")
    ax.set_ylabel("sensitivity (seizures predicted / 7)")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("All threshold settings: at most 1 of 7 seizures predicted")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sensitivity_vs_far.png", dpi=150)
    plt.close(fig)

# ---- reuse the feature plot from script 01 ----
if Path("chb01_features_plot.png").exists():
    shutil.copy("chb01_features_plot.png", OUT / "fig_features_chb01_03.png")

print("figures written to", OUT.resolve())
for p in sorted(OUT.glob("*.png")):
    print("  ", p.name)