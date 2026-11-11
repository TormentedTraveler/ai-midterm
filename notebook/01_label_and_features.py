"""
Phase 1-2: CHB-MIT labeling + windowed features (chb01)

Install (in your venv):
    pip install mne numpy pandas scipy matplotlib

Run (VS Code: run cell by cell, or just `python 01_label_and_features.py`):
    1. Edit DATA_DIR below to point at your chb01 folder.
    2. Run it. It writes features_chb01.csv and plots one seizure file.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch

# ----------------------------- CONFIG ---------------------------------
DATA_DIR = Path(r"C:/Users/esyas/Work/Personal/ai-final/data/chb01")   # <-- CHANGE THIS
SUMMARY = DATA_DIR / "chb01-summary.txt"
OUT_CSV = Path("features_chb01.csv")

WIN_S = 30            # window length (s)
STEP_S = 30           # step (s); use 10 for overlapping windows later
PREICTAL_S = 5 * 60   # target prediction horizon: 5 min before onset
BUFFER_S = 30 * 60    # 5-30 min before onset: excluded from training
POSTICTAL_S = 30 * 60 # after seizure end: excluded from training

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 70)}


# --------------------- 1. parse the summary file -----------------------
def parse_summary(path: Path) -> dict:
    """Return {edf_filename: [(onset_s, end_s), ...]} (empty list = no seizure)."""
    text = path.read_text()
    out = {}
    for block in re.split(r"File Name:\s*", text)[1:]:
        fname = block.split()[0]
        starts = re.findall(r"Seizure(?: \d+)? Start Time:\s*(\d+)\s*seconds", block)
        ends = re.findall(r"Seizure(?: \d+)? End Time:\s*(\d+)\s*seconds", block)
        out[fname] = [(int(s), int(e)) for s, e in zip(starts, ends)]
    return out


# ----------------------- 2. label one window ---------------------------
def label_window(t0, t1, seizures):
    """Labels: ictal, preictal, buffer, postictal, interictal."""
    for on, off in seizures:
        if t0 < off and t1 > on:
            return "ictal"
    for on, off in seizures:
        if on - PREICTAL_S <= t1 <= on:
            return "preictal"
    for on, off in seizures:
        if on - BUFFER_S <= t1 < on - PREICTAL_S:
            return "buffer"
        if off <= t0 < off + POSTICTAL_S:
            return "postictal"
    return "interictal"


# ------------------------- 3. features ---------------------------------
def window_features(x, fs):
    """x: (n_channels, n_samples). Returns dict of features averaged over channels."""
    f = {}
    f["line_length"] = np.mean(np.sum(np.abs(np.diff(x, axis=1)), axis=1))
    f["variance"] = np.mean(np.var(x, axis=1))
    f["rms"] = np.mean(np.sqrt(np.mean(x ** 2, axis=1)))
    d1 = np.diff(x, axis=1)
    d2 = np.diff(d1, axis=1)
    mob = np.sqrt(np.var(d1, axis=1) / (np.var(x, axis=1) + 1e-12))
    mob2 = np.sqrt(np.var(d2, axis=1) / (np.var(d1, axis=1) + 1e-12))
    f["hjorth_mobility"] = np.mean(mob)
    f["hjorth_complexity"] = np.mean(mob2 / (mob + 1e-12))
    freqs, psd = welch(x, fs=fs, nperseg=int(fs * 4), axis=1)
    total = psd[:, (freqs >= 0.5) & (freqs <= 70)].sum(axis=1) + 1e-12
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        f[f"rel_{name}"] = np.mean(psd[:, m].sum(axis=1) / total)
    f["theta_beta_ratio"] = f["rel_theta"] / (f["rel_beta"] + 1e-12)
    return f


# ------------------------- 4. main loop --------------------------------
def process_file(edf_path: Path, seizures):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    keep = [c for c in raw.ch_names if not c.startswith(("-", "ECG", "VNS"))]
    raw.pick(keep)
    fs = raw.info["sfreq"]
    data = raw.get_data() * 1e6           # volts -> microvolts
    n = data.shape[1]
    rows = []
    t = 0
    while t + WIN_S * fs <= n:
        t0_s, t1_s = t / fs, (t + WIN_S * fs) / fs
        feats = window_features(data[:, t:t + int(WIN_S * fs)], fs)
        feats.update(file=edf_path.name, t_start=t0_s, t_end=t1_s,
                     label=label_window(t0_s, t1_s, seizures))
        rows.append(feats)
        t += int(STEP_S * fs)
    return pd.DataFrame(rows)


def main():
    summary = parse_summary(SUMMARY)
    print(f"Summary lists {len(summary)} files, "
          f"{sum(len(v) for v in summary.values())} seizures total.")

    frames = []
    for fname, seizures in summary.items():
        p = DATA_DIR / fname
        if not p.exists():
            continue                      # you only downloaded some files
        print(f"Processing {fname}  seizures={seizures}")
        frames.append(process_file(p, seizures))

    if not frames:
        raise SystemExit("No EDF files found. Check DATA_DIR.")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print("\nSaved", OUT_CSV, "shape", df.shape)
    print("\nClass balance (windows):")
    print(df["label"].value_counts())

    # Plot first file that has a seizure
    seizure_files = df.loc[df.label == "ictal", "file"].unique()
    if len(seizure_files):
        sub = df[df.file == seizure_files[0]]
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        for ax, col in zip(axes, ["line_length", "variance", "theta_beta_ratio"]):
            ax.plot(sub.t_start / 60, sub[col])
            ax.set_ylabel(col)
            for lab, color in [("ictal", "red"), ("preictal", "orange")]:
                m = sub.label == lab
                ax.fill_between(sub.t_start / 60, ax.get_ylim()[0], ax.get_ylim()[1],
                                where=m, color=color, alpha=0.25, label=lab)
        axes[0].legend(loc="upper left")
        axes[-1].set_xlabel("time in file (min)")
        fig.suptitle(f"{seizure_files[0]}: proxy features vs. labels")
        plt.tight_layout()
        plt.savefig("chb01_features_plot.png", dpi=120)
        plt.show()


if __name__ == "__main__":
    main()