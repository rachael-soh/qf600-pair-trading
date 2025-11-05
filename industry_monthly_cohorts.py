#!/usr/bin/env python3
"""
Build monthly Gatev-style Top-N cohorts (formation only) for the 10-year horizon,
grouped by major industry divisions first, then finding pairs within each industry.

**with formation filters**:
  (1) EXCLUDE any stock that has *any* day with vol == 0 in the 12M formation window.
  (2) EXCLUDE any stock that has a delist date (dlret non-null) *inside* the 12M formation window.
  (3) Group stocks by major industry divisions based on SIC codes

Formation months: 
- 1962-12 .. 1972-12 (inclusive)
- 2013-01 .. 2024-06 (inclusive)
Trading for each cohort starts next business day after the formation month.

Inputs:
  - A CRSP daily panel CSV produced by your extractor, e.g.:
      crsp_daily_1961_1973.csv
      crsp_daily_2013_2024.csv
    Required columns: date, permno, permco, siccd, ret, vol, (optional: dlret)

Outputs in ./industry_monthly_cohorts_top20:
  - matched_pairs_monthly_YYYYMM.csv  (Top-20 by SSD of normalized-price spread per industry)
  - matched_pairs_monthly_index.csv   (combined index of all months)
  - industry_stats_YYYYMM.csv        (statistics about pairs found in each industry)
"""

import os, math, itertools
import numpy as np
import pandas as pd

# -------- Parameters you can tweak --------
PANEL_PATH      = "crsp_daily_1961_1973.csv"
OUT_DIR         = "industry_monthly_cohorts_top20"
TOP_N           = 20        # top N pairs per industry
MIN_DAYS        = 200      # require at least this many price days in the 12M window
MIN_STOCKS_PER_INDUSTRY = 5  # minimum number of stocks needed in an industry to form pairs
START_YYYYMM    = 196201   # first formation month
END_YYYYMM      = 196210   # last formation month

# Industry classification based on SIC codes
INDUSTRY_CODE = {
    "Agriculture, Forestry and Fishing": (0, 999),
    "Mining": (1000, 1499),
    "Construction": (1500, 1799),
    "Manufacturing": (2000, 3999),
    "Transportation, Communications, Electric, Gas and Sanitary service": (4000, 4999),
    "Wholesale Trade": (5000, 5199),
    "Retail Trade": (5200, 5999),
    "Finance, Insurance and Real estate": (6000, 6799),
    "Services": (7000, 8999),
    "Public administration": (9100, 9729)
}

def get_industry_name(sic_code):
    """Map SIC code to industry name"""
    try:
        sic_int = int(sic_code)
        for industry, (start, end) in INDUSTRY_CODE.items():
            if start <= sic_int <= end:
                return industry
    except (ValueError, TypeError):
        pass
    return "Other"

# -------- Helpers --------
def yyyymm_to_period(x):
    return pd.Period(str(int(x)), freq="M")

def month_bounds_from_period(p):
    start = (p - 11).start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = p.end_time.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, end

def load_panel(path=PANEL_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    need = ["date", "permno", "permco", "siccd", "ret", "vol"]

    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in panel: {missing}")
    
    # Optional filters for common stocks/exchanges if available
    if "shrcd" in df.columns:
        df = df[df["shrcd"].isin([10,11])]
    if "exchcd" in df.columns:
        df = df[df["exchcd"].isin([1,2,3])]
    
    # Map SIC codes to industry names
    df["industry"] = df["siccd"].apply(get_industry_name)
    
    df = df.drop_duplicates().sort_values(["permno", "date"])
    return df

def normalized_ctri(ret_series):
    if ret_series.empty: return ret_series
    ctri = (1 + ret_series).cumprod()
    base = ctri.iloc[0]
    if pd.isna(base) or base == 0:
        return pd.Series(index=ret_series.index, dtype=float)
    return ctri / base

def ssd_spread(ni: pd.Series, nj: pd.Series) -> float:
    merged = pd.concat([ni, nj], axis=1, join="inner").dropna()
    if len(merged) == 0: return np.nan
    diff = merged.iloc[:,0] - merged.iloc[:,1]
    return float(np.sum(np.square(diff)))

def has_zero_volume(win_df: pd.DataFrame) -> bool:
    return (win_df["vol"] == 0).any()

def delisted_inside_window(win_df: pd.DataFrame) -> bool:
    if "dlret" not in win_df.columns:
        return False
    return win_df["dlret"].notna().any()

def process_industry_group(industry_stocks, permco_map, norm_map, siccd_map):
    permnos = list(industry_stocks)
    if len(permnos) < 2:
        return []
    
    results = []
    for i, j in itertools.combinations(permnos, 2):
        if permco_map[i] == permco_map[j]:
            continue  # exclude same-company pairs
        ssd = ssd_spread(norm_map[i], norm_map[j])
        if math.isnan(ssd):
            continue
        results.append({
            "permno_1": int(i), "permco_1": permco_map[i], "siccd_1": siccd_map[i],
            "permno_2": int(j), "permco_2": permco_map[j], "siccd_2": siccd_map[j],
            "ssd": ssd
        })
    return results

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = load_panel(PANEL_PATH)

    start_p = yyyymm_to_period(START_YYYYMM)
    end_p   = yyyymm_to_period(END_YYYYMM)

    all_rows = []

    for p in pd.period_range(start_p, end_p, freq="M"):
        f_start, f_end = month_bounds_from_period(p)
        win_cols = ["date", "permco", "permno", "siccd", "industry", "ret", "vol"]
        if "dlret" in panel.columns:
            win_cols.append("dlret")

        win = panel.loc[(panel["date"] >= f_start) & (panel["date"] <= f_end), win_cols].copy()

        # Count days per permno for MIN_DAYS eligibility
        counts = win.groupby("permno")["date"].count().sort_values(ascending=False)
        eligible_permnos = counts[counts >= MIN_DAYS].index.tolist()

        # Apply formation filters per permno
        filtered_permnos = []
        for permno in eligible_permnos:
            sdf = win[win["permno"]==permno]
            if has_zero_volume(sdf):
                continue
            if delisted_inside_window(sdf):
                continue
            filtered_permnos.append(permno)

        if not filtered_permnos:
            print(f"[{p}] No eligible stocks after filters. Skipping.")
            continue

        pool = win[win["permno"].isin(filtered_permnos)].copy()
        print(f"[{p}] Begin processing {len(filtered_permnos)} stocks after filters.")

        # Group stocks by industry
        industry_groups = {}
        norm_map = {}
        permco_map = {}
        siccd_map = {}
        industry_stats = []

        # First pass: compute normalized returns and group by industry
        for permno, sdf in pool.groupby("permno", sort=False):
            industry = sdf["industry"].iloc[0]
            sdf = sdf.sort_values("date")
            ni = normalized_ctri(sdf["ret"])
            if ni.notna().sum() == 0:
                continue
                
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(permno)
            
            norm_map[permno] = pd.Series(ni.values, index=sdf["date"])
            permco_map[permno] = int(sdf["permco"].iloc[0])
            siccd_map[permno] = int(sdf["siccd"].iloc[0])

        # Process each industry group
        monthly_results = []
        
        for industry, stock_list in industry_groups.items():
            if len(stock_list) < MIN_STOCKS_PER_INDUSTRY:
                print(f"[{p}] Industry '{industry}' has only {len(stock_list)} stocks, skipping.")
                continue
                
            industry_pairs = process_industry_group(stock_list, permco_map, norm_map, siccd_map)
            
            if industry_pairs:
                # Take top N pairs from this industry
                industry_pairs = sorted(industry_pairs, key=lambda x: x["ssd"])[:TOP_N]
                for pair in industry_pairs:
                    pair["formation_month"] = int(str(p).replace("-", ""))
                    pair["industry"] = industry
                monthly_results.extend(industry_pairs)
                
            # Record industry statistics
            industry_stats.append({
                "formation_month": int(str(p).replace("-", "")),
                "industry": industry,
                "n_stocks": len(stock_list),
                "n_pairs": len(industry_pairs)
            })

        if not monthly_results:
            print(f"[{p}] No pair distances after filters. Skipping.")
            continue

        # Save monthly results
        df_pairs = pd.DataFrame(monthly_results).sort_values(["industry", "ssd"]).reset_index(drop=True)
        out_path = os.path.join(OUT_DIR, f"matched_pairs_monthly_{int(str(p).replace('-', ''))}.csv")
        df_pairs.to_csv(out_path, index=False)
        print(f"[{p}] Saved pairs across industries → {out_path}")
        
        # Save industry statistics
        stats_df = pd.DataFrame(industry_stats)
        stats_path = os.path.join(OUT_DIR, f"industry_stats_{int(str(p).replace('-', ''))}.csv")
        stats_df.to_csv(stats_path, index=False)
        
        all_rows.append(df_pairs)

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_df.to_csv(os.path.join(OUT_DIR, "matched_pairs_monthly_index.csv"), index=False)
        print("Wrote combined index:", os.path.join(OUT_DIR, "matched_pairs_monthly_index.csv"))
    else:
        print("No monthly cohorts produced. Check data/parameters.")

if __name__ == "__main__":
    main()
