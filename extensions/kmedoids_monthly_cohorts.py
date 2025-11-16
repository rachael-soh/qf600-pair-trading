#!/usr/bin/env python3
"""
Build monthly Gatev-style Top-N cohorts (formation only) for the 10-year horizon,
**with formation filters**:
  (1) EXCLUDE any stock that has *any* day with vol == 0 in the 12M formation window.
  (2) EXCLUDE any stock that has a delist date (dlret non-null) *inside* the 12M formation window.

Formation months: 1962-12 .. 1972-12 (inclusive)
Trading for each cohort starts next business day after the formation month.

Inputs:
  - A CRSP daily panel CSV produced by your extractor, e.g.:
      crsp_daily_1961_1973.csv
    Required columns: date, permno, permco, adj_prc, vol, (optional: dlret)

Outputs in ./monthly_cohorts_top20_10y_filtered:
  - matched_pairs_monthly_YYYYMM.csv  (Top-20 by SSD of normalized-price spread)
  - matched_pairs_monthly_index.csv   (combined index of all months)

Notes:
- We keep the same-company exclusion (permco_1 != permco_2).
- We also keep shrcd/exchcd filters if present (common stocks on NYSE/AMEX/NASDAQ).
- MIN_DAYS ensures reasonably complete 12M windows.
"""

import os, math, itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
from joblib import Parallel, delayed

# -------- Parameters you can tweak --------
PANEL_PATH      = "../crsp_daily_1961_1973.csv"
OUT_DIR         = "kmedoids_ssd_monthly_cohorts_top20"
TOP_N           = 20
MIN_DAYS        = 200       # require at least this many price days in the 12M window
START_YYYYMM    = 196201    # first formation month
END_YYYYMM      = 197212    # last formation month

# -------- Helpers --------
def yyyymm_to_period(x):
    return pd.Period(str(int(x)), freq="M")

def month_bounds_from_period(p):
    start = (p - 11).start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = p.end_time.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, end

def load_panel(path=PANEL_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    need = ["date","permno","permco","adj_prc","vol"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in panel: {missing}")
    # Optional filters for common stocks/exchanges if available
    if "shrcd" in df.columns:
        df = df[df["shrcd"].isin([10,11])]
    if "exchcd" in df.columns:
        df = df[df["exchcd"].isin([1,2,3])]
    df = df[df["adj_prc"] > 0].drop_duplicates().sort_values(["permno","date"])
    return df

def normalized_price(series):
    if series.empty: return series
    base = series.iloc[0]
    if pd.isna(base) or base == 0: 
        return pd.Series(index=series.index, dtype=float)
    return series / base

def ssd_spread(ni: pd.Series, nj: pd.Series) -> float:
    merged = pd.concat([ni, nj], axis=1, join="inner").dropna()
    if len(merged) == 0: return np.nan
    diff = merged.iloc[:,0] - merged.iloc[:,1]
    return float(np.sum(np.square(diff)))

def has_zero_volume(win_df: pd.DataFrame) -> bool:
    # Any zero-volume day in the 12M window?
    return (win_df["vol"] == 0).any()

def delisted_inside_window(win_df: pd.DataFrame) -> bool:
    # If panel has dlret, dlret is only non-null on the delist date.
    if "dlret" not in win_df.columns:
        return False
    return win_df["dlret"].notna().any()

def compute_pairwise_ssd(permnos, permco_map, norm_map, p):
    # Compute SSD for all distinct-company pairs; pick Top-N
    print(f"[{p}] Computing pairwise SSD for {len(permnos)} stocks...")
    results = []
    for i, j in itertools.combinations(permnos, 2):
        if permco_map[i] == permco_map[j]:
            continue  # exclude same-company pairs
        ssd = ssd_spread(norm_map[i], norm_map[j])
        if math.isnan(ssd):
            continue
        results.append({
            "formation_month": int(str(p).replace("-", "")),
            "permno_1": int(i),  "permco_1": permco_map[i],
            "permno_2": int(j),  "permco_2": permco_map[j],
            "ssd": ssd
        })
    return results

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = load_panel(PANEL_PATH)
    scaler = StandardScaler()

    start_p = yyyymm_to_period(START_YYYYMM)
    end_p   = yyyymm_to_period(END_YYYYMM)

    all_rows = []
    for p in pd.period_range(start_p, end_p, freq="M"):
        f_start, f_end = month_bounds_from_period(p)
        win_cols = ["date","permco","permno","adj_prc","vol"]
        if "dlret" in panel.columns:
            win_cols.append("dlret")

        win = panel.loc[(panel["date"] >= f_start) & (panel["date"] <= f_end), win_cols].copy()

        # Count days per permno for MIN_DAYS eligibility
        counts = win.groupby("permno")["date"].count().sort_values(ascending=False)
        eligible_permnos = counts[counts >= MIN_DAYS].index.tolist()

        # Apply formation filters per permno: zero-volume days & delist-inside-window
        filtered_permnos = []
        for permno in eligible_permnos:
            sdf = win[win["permno"]==permno]
            if has_zero_volume(sdf):
                continue  # exclude if any vol==0 day in formation window
            if delisted_inside_window(sdf):
                continue  # exclude if dlret present inside formation window
            filtered_permnos.append(permno)

        if not filtered_permnos:
            print(f"[{p}] No eligible stocks after filters. Skipping.")
            continue

        pool = win[win["permno"].isin(filtered_permnos)].copy()

        # Precompute normalized price series for each stock in this window
        norm_map = {}
        permco_map = {}
        for permno, sdf in pool.groupby("permno", sort=False):
            sdf = sdf.sort_values("date")
            ni = normalized_price(sdf["adj_prc"])
            if ni.notna().sum() == 0:
                continue
            norm_map[permno] = pd.Series(ni.values, index=sdf["date"])
            permco_map[permno] = int(sdf["permco"].iloc[0])

        permnos = list(norm_map.keys())
        if len(permnos) < 2:
            print(f"[{p}] Not enough valid stocks after normalization. Skipping.")
            continue

        # First form K clusters
        n_clusters = min(10, len(permnos) // 2)
        if n_clusters < 2:
            print(f"[{p}] Not enough stocks for clustering. Skipping.")
            continue
        
        # Create feature matrix: rows=stocks, cols=dates (fill missing with NaN)
        all_dates = sorted(set().union(*(norm_map[pn].index for pn in permnos)))
        feature_matrix = np.full((len(permnos), len(all_dates)), np.nan)
        for i, pn in enumerate(permnos):
            ni = norm_map[pn]
            date_idx = [all_dates.index(d) for d in ni.index]
            feature_matrix[i, date_idx] = ni.values
        
        # Impute missing values with column means
        col_means = np.nanmean(feature_matrix, axis=0)
        inds = np.where(np.isnan(feature_matrix))
        feature_matrix[inds] = np.take(col_means, inds[1])
        
        # Standardize features
        feature_matrix = scaler.fit_transform(feature_matrix)

        # Compute squared-squared distance matrix
        ssd_matrix = pairwise_distances(feature_matrix, metric='sqeuclidean')

        kmedoids = KMedoids(
                n_clusters=4,
                metric='precomputed',   # we’re providing a distance matrix, not raw features
                method='pam',
                init='k-medoids++',
                random_state=42
            )
        labels = kmedoids.fit_predict(ssd_matrix)
        clusters = pd.Series(labels, index=permnos, name='cluster')

        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(compute_pairwise_ssd)(
                members.index.tolist(), permco_map, norm_map, p
            )
            for cluster_id, members in clusters.groupby(clusters)
        ) 
        
        if not results:
            print(f"[{p}] No pair distances after filters. Skipping.")
            continue
        results = [item for sublist in results for item in sublist]  # flatten
        print(pd.DataFrame(results))
        df_pairs = pd.DataFrame(results).sort_values("ssd").head(TOP_N).reset_index(drop=True)
        out_path = os.path.join(OUT_DIR, f"matched_pairs_monthly_{int(str(p).replace('-', ''))}.csv")
        df_pairs.to_csv(out_path, index=False)
        print(f"[{p}] Saved Top-{TOP_N} pairs → {out_path} (universe={len(permnos)})")
        all_rows.append(df_pairs)

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_df.to_csv(os.path.join(OUT_DIR, "matched_pairs_monthly_index.csv"), index=False)
        print("Wrote combined index:", os.path.join(OUT_DIR, "matched_pairs_monthly_index.csv"))
    else:
        print("No monthly cohorts produced. Check data/parameters.")

if __name__ == "__main__":
    main()
