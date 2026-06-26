"""
Choose the interpolation ceiling for the motion channels: the longest NaN run, in
samples, that is still filled rather than left as a break.

The ceiling rests on two measurements:
  1. Gap-size distribution. The lengths of the real NaN runs across all subjects and all
     27 motion channels. This sets how much data each candidate ceiling recovers.
  2. Error knee. The linear-interpolation mean absolute error (MAE), as a percentage of
     the signal's standard deviation, as a function of gap length. This sets the fill
     error at each length.

The chosen ceiling is the smaller of the length at which the error crosses tolerance and
the length beyond which negligible gap mass remains.

Linear interpolation here is .interpolate(method="index") on the contiguous integer index,
which is linear in sample number and matches the ML pipeline's fill with limit=N.
"""
import sys
from pathlib import Path
from itertools import groupby
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from read_data import load_subject, get_pamap2_headers

SAMPLE_RATE = 100
SUBJECTS = [str(s) for s in range(101, 110)]

# The 27 motion channels kept downstream: three IMUs, each with acc16, gyro, and mag on x, y, z.
PARTS = ["hand", "chest", "ankle"]
SENSORS = ["acc16", "gyro", "mag"]
AXES = ["x", "y", "z"]
MOTION_COLS = [f"{p}_{s}_{a}" for p in PARTS for s in SENSORS for a in AXES]


def gap_lengths(null_mask: pd.Series) -> np.ndarray:
    """Lengths of the runs of consecutive True (NaN) values in an index-ordered boolean Series."""
    idx = null_mask[null_mask].index.values
    if len(idx) == 0:
        return np.array([], dtype=int)
    return np.array([len(list(g)) for _, g in groupby(enumerate(idx), lambda x: x[0] - x[1])])


def collect_gaps():
    all_gaps = []
    per_sensor = {s: [] for s in SENSORS}
    for subj in SUBJECTS:
        try:
            df = load_subject(subj)
        except FileNotFoundError:
            continue
        cols = [c for c in MOTION_COLS if c in df.columns]
        for c in cols:
            g = gap_lengths(df[c].isnull())
            all_gaps.append(g)
            for s in SENSORS:
                if f"_{s}_" in c:
                    per_sensor[s].append(g)
    all_gaps = np.concatenate(all_gaps) if all_gaps else np.array([])
    per_sensor = {s: (np.concatenate(v) if v else np.array([])) for s, v in per_sensor.items()}
    return all_gaps, per_sensor


def longest_gap_free_segment(s: pd.Series, want=2000):
    ok = s.notna().to_numpy()
    idx = s.index.to_numpy()
    blocks, start = [], None
    for i in range(len(ok)):
        if ok[i] and start is None:
            start = i
        elif not ok[i] and start is not None:
            blocks.append((start, i)); start = None
    if start is not None:
        blocks.append((start, len(ok)))
    blocks = [(a, b) for a, b in blocks if b - a >= want]
    if not blocks:
        return None
    a, b = max(blocks, key=lambda ab: ab[1] - ab[0])
    return idx[a:b]


def sensor_of(col):
    for s in SENSORS:
        if f"_{s}_" in col:
            return s
    return "?"


def mae_sweep(gap_sizes, n_cols_per_subj=9, n_gaps=8, pad=30):
    """For each gap size, introduce synthetic gaps of that length into the gap-free segments
    of each column across subjects, linearly interpolate them, and return the mean MAE as a
    percentage of the signal's standard deviation, both overall and per sensor type.

    Also return the intrinsic floor, mean |x[t] - x[t-1]| as a percentage of the standard
    deviation. This is the sample-to-sample change, computed without the interpolator, and
    no interpolator beats it."""
    rows = {gs: [] for gs in gap_sizes}
    rows_by_sensor = {s: {gs: [] for gs in gap_sizes} for s in SENSORS}
    floor = {s: [] for s in SENSORS}
    for subj in SUBJECTS:
        try:
            df = load_subject(subj)
        except FileNotFoundError:
            continue
        cols = [c for c in MOTION_COLS if c in df.columns][:n_cols_per_subj]
        for c in cols:
            block = longest_gap_free_segment(df[c], want=2000)
            if block is None:
                continue
            block = block[:2000]
            truth = df.loc[block, c].astype(float)
            scale = truth.std()
            if not np.isfinite(scale) or scale == 0:
                continue
            sens = sensor_of(c)
            floor[sens].append(100 * np.mean(np.abs(np.diff(truth.to_numpy()))) / scale)
            for gs in gap_sizes:
                stride = max(gs + pad, (len(block) - 2 * pad) // n_gaps)
                starts = pad + np.arange(n_gaps) * stride
                starts = starts[starts + gs <= len(block) - pad]
                if len(starts) == 0:
                    continue
                masked_idx = np.concatenate([block[st:st + gs] for st in starts])
                col = df[c].astype(float).copy()
                col.loc[masked_idx] = np.nan
                filled = col.interpolate(method="index")  # linear in sample number
                yt = truth.loc[masked_idx].to_numpy()
                yp = filled.loc[masked_idx].to_numpy()
                keep = ~np.isnan(yp)
                if keep.sum() == 0:
                    continue
                val = 100 * np.mean(np.abs(yt[keep] - yp[keep])) / scale
                rows[gs].append(val)
                rows_by_sensor[sens][gs].append(val)
    overall = {gs: (np.mean(v) if v else np.nan, len(v)) for gs, v in rows.items()}
    per_sensor = {s: {gs: (np.mean(v) if v else np.nan) for gs, v in d.items()}
                  for s, d in rows_by_sensor.items()}
    floor = {s: (np.mean(v) if v else np.nan) for s, v in floor.items()}
    return overall, per_sensor, floor


if __name__ == "__main__":
    print("=" * 70)
    print("1. REAL GAP-SIZE DISTRIBUTION  (all subjects, 27 motion channels)")
    print("=" * 70)
    all_gaps, per_sensor = collect_gaps()
    if len(all_gaps):
        total = len(all_gaps)
        print(f"\ntotal gaps: {total:,}")
        for label, g in [("ALL", all_gaps)] + list(per_sensor.items()):
            if len(g) == 0:
                continue
            print(f"\n[{label}]  n={len(g):,}")
            print(f"  median={np.median(g):.0f}  mean={g.mean():.2f}  "
                  f"p90={np.percentile(g,90):.0f}  p99={np.percentile(g,99):.0f}  max={g.max()}")
        # Cumulative coverage: the fraction of gaps, and of gapped samples, at or below each ceiling.
        print("\n--- coverage by candidate ceiling (ALL channels) ---")
        print(f"{'ceiling':>8} {'gap_ms':>7} {'% gaps <=':>10} {'% NaN-samples kept':>20}")
        tot_samples = all_gaps.sum()
        for ceil in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 300]:
            pct_gaps = 100 * np.mean(all_gaps <= ceil)
            pct_samples = 100 * all_gaps[all_gaps <= ceil].sum() / tot_samples
            print(f"{ceil:>8} {ceil*10:>7} {pct_gaps:>9.2f}% {pct_samples:>19.2f}%")

    print("\n" + "=" * 70)
    print("2. ERROR KNEE: linear-interp MAE as % of signal std vs gap length")
    print("=" * 70)
    gap_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100, 150]
    overall, per_sensor, floor = mae_sweep(gap_sizes)
    print("\nintrinsic floor = mean|x[t]-x[t-1]| as % of std (no interpolator beats this):")
    for s in SENSORS:
        print(f"   {s:>6}: {floor[s]:.1f}%")
    print(f"\n{'gap':>5} {'gap_ms':>7} {'MAE_all':>9} {'acc16':>8} {'gyro':>8} {'magn':>8} {'n':>5}")
    for gs in gap_sizes:
        m, n = overall[gs]
        print(f"{gs:>5} {gs*10:>7} {m:>8.1f}% "
              f"{per_sensor['acc16'][gs]:>7.1f}% {per_sensor['gyro'][gs]:>7.1f}% "
              f"{per_sensor['mag'][gs]:>7.1f}% {n:>5}")
