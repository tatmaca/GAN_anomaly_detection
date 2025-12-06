# batch_q2_tests.py

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, skew, kurtosis

IN_DIR = "sorted_returns"
OUT_DIR = "q2_stats"

MERGED_RE = re.compile(r"^(?P<code>\d{4})_merged_disret_.*\.csv$")
UNMERGED_RE = re.compile(r"^(?P<code>\d{4})_unmerged_disret_.*\.csv$")

def _ensure_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "dt_index" in df.columns:

        df["dt_index"] = pd.to_datetime(df["dt_index"], errors="coerce")
        
    elif {"date", "time"}.issubset(df.columns):

        df["dt_index"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce"
        )

    else:
        raise ValueError("Need either 'dt_index' or ('date' and 'time') to order snapshots.")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    else:
        df["date"] = df["dt_index"].dt.date

    return df


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:

    df = _ensure_datetime_cols(df)
    df = df.sort_values(["dt_index"]).copy()

    # mid-quote
    df["mid"] = (df["BP1"].astype(float) + df["SP1"].astype(float)) / 2.0

    # spread
    df["spread"] = df["SP1"].astype(float) - df["BP1"].astype(float)

    # intraday vs daily
    # if any date has >1 row, treat as intraday and diff within-day else diff across days
    counts = df.groupby("date")["dt_index"].transform("size")
    is_intraday = bool((counts > 1).any())

    if is_intraday:

        df = df.sort_values(["date", "dt_index"]).copy()

        df["trade_ret"] = df.groupby("date")["lastPx"].transform(
            lambda s: np.log(s.astype(float)).diff()
        )

        df["mid_ret"] = df.groupby("date")["mid"].transform(
            lambda s: np.log(s.astype(float)).diff()
        )

        df["d_spread"] = df.groupby("date")["spread"].transform(lambda s: s.diff())

    else:
        # daily: compute diffs across time
        df = df.sort_values(["dt_index"]).copy()

        df["trade_ret"] = np.log(df["lastPx"].astype(float)).diff()
        df["mid_ret"] = np.log(df["mid"].astype(float)).diff()
        df["d_spread"] = df["spread"].diff()

    # trade size
    df["trade_size"] = df["size"].astype(float)

    # 1-level pressure
    denom1 = (df["BV1"].astype(float) + df["SV1"].astype(float)).replace(0, np.nan)
    df["press_1"] = (df["BV1"].astype(float) - df["SV1"].astype(float)) / denom1

    # 5-level pressure
    bv_cols = [f"BV{i}" for i in range(1, 6)]
    sv_cols = [f"SV{i}" for i in range(1, 6)]

    missing = [c for c in (bv_cols + sv_cols) if c not in df.columns]

    if missing:
        raise ValueError(f"Missing volume columns needed for 5-level pressure: {missing}")

    bv5 = df[bv_cols].astype(float).sum(axis=1)
    sv5 = df[sv_cols].astype(float).sum(axis=1)

    denom5 = (bv5 + sv5).replace(0, np.nan)
    df["press_5"] = (bv5 - sv5) / denom5

    # clean infinities
    feat_cols = ["trade_ret", "mid_ret", "trade_size", "spread", "d_spread", "press_1", "press_5"]

    for c in feat_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df


def summarize_series(x: pd.Series) -> dict:
    x = x.dropna().to_numpy(dtype=float)

    if x.size == 0:
        return {"n": 0, "mean": np.nan, "var": np.nan, "skew": np.nan, "kurtosis": np.nan}
    
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "var": float(np.var(x, ddof=1)) if x.size > 1 else np.nan,
        "skew": float(skew(x, bias=False)) if x.size > 2 else np.nan,
        "kurtosis": float(kurtosis(x, bias=False)) if x.size > 3 else np.nan,  # excess
    }


def ks_test(a: pd.Series, b: pd.Series) -> dict:

    a = a.dropna().to_numpy(dtype=float)
    b = b.dropna().to_numpy(dtype=float)

    if a.size == 0 or b.size == 0:
        return {"ks_stat": np.nan, "ks_pvalue": np.nan}
    
    res = ks_2samp(a, b, alternative="two-sided", mode="auto")

    return {"ks_stat": float(res.statistic), "ks_pvalue": float(res.pvalue)}


def run_all_tests(merged: pd.DataFrame, unmerged: pd.DataFrame) -> pd.DataFrame:

    ab = add_microstructure_features(merged)
    nm = add_microstructure_features(unmerged)

    variables = [
        ("trade_ret", "Trade price log returns"),
        ("mid_ret", "Mid-quote log returns"),
        ("trade_size", "Trade size"),
        ("spread", "Bid-ask spread"),
        ("d_spread", "First diff of spread"),
        ("press_1", "1-level book pressure"),
        ("press_5", "5-level book pressure"),
    ]

    rows = []
    for var, label in variables:

        ab_sum = summarize_series(ab[var])
        nm_sum = summarize_series(nm[var])

        ks = ks_test(ab[var], nm[var])

        rows.append(
            {
                "variable": var,
                "label": label,
                **{f"ab_{k}": v for k, v in ab_sum.items()},
                **{f"nm_{k}": v for k, v in nm_sum.items()},
                **ks,
            }
        )

    return pd.DataFrame(rows).sort_values(["ks_pvalue", "variable"], na_position="last")


def _pick_latest(files):
    return sorted(files)[-1]


def find_pairs(in_dir: str):
    merged_map, unmerged_map = {}, {}

    for path in glob.glob(os.path.join(in_dir, "*.csv")):

        name = os.path.basename(path)
        m = MERGED_RE.match(name)

        if m:
            merged_map.setdefault(m.group("code"), []).append(path)
            continue

        u = UNMERGED_RE.match(name)

        if u:
            unmerged_map.setdefault(u.group("code"), []).append(path)

    codes = sorted(set(merged_map) & set(unmerged_map))
    pairs = [(c, _pick_latest(merged_map[c]), _pick_latest(unmerged_map[c])) for c in codes]

    return pairs

def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    pairs =find_pairs(IN_DIR)

    if not pairs:
        raise SystemExit(
            f"No merged/unmerged pairs found in ./{IN_DIR}/ matching "
            f"'####_merged_disret_*.csv' and '####_unmerged_disret_*.csv'."
        )

    combined = []
    for code, merged_path, unmerged_path in pairs:

        print(f"\n[{code}] Reading:")
        print("  merged  :", merged_path)
        print("  unmerged:", unmerged_path)

        merged = pd.read_csv(merged_path)
        unmerged = pd.read_csv(unmerged_path)

        results = run_all_tests(merged, unmerged)
        results.insert(0, "stock_code", code)

        out_csv = os.path.join(OUT_DIR, f"{code}_q2_abnormal_vs_normal_stats_ks.csv")
        out_txt = os.path.join(OUT_DIR, f"{code}_q2_abnormal_vs_normal_stats_ks.txt")

        results.to_csv(out_csv, index=False)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(results.to_string(index=False))

        print(f"Saved:\n  {out_csv}\n  {out_txt}")
        combined.append(results)

    all_df = pd.concat(combined, ignore_index=True)

    all_path = os.path.join(OUT_DIR, "ALL_q2_abnormal_vs_normal_stats_ks.csv")

    all_df.to_csv(all_path, index=False)

    print(f"\nSaved combined: {all_path}")


if __name__ == "__main__":
    main()
