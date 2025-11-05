#!/usr/bin/env python3
"""
Run monthly Gatev-style trading + FFMR (10-year horizon), pair-level equal-weighting.

Reads:
  - Panel: crsp_daily_1961_1973.csv  (includes ret, dlret, ret_total, adj_prc)
  - Cohorts: monthly_cohorts_top20/matched_pairs_monthly_YYYYMM.csv (196212..197212)
  - French monthly factors: F-F_Research_Data_Factors.csv, F-F_Momentum_Factor.csv, F-F_ST_Reversal_Factor.csv

Outputs -> monthly_trading_ffmr_top20_10y/ :
  - portfolio_daily.csv
  - portfolio_monthly.csv
  - ffmr_regression_table.csv
  - ffmr_performance_stats.csv
"""
import os, re, io, glob, math
import numpy as np
import pandas as pd
import statsmodels.api as sm

# -------- Parameters you can tweak --------
PANEL = "crsp_daily_1961_1973.csv"
COHORT_DIR = "monthly_cohorts_top20"
TRADING_BDAYS = 126
WAIT_ONE_DAY = False
TOP_N_CAP = 9999

# TODO: changed for extension
Z_ENTRY = 2.0

# -------- Helpers --------
def yyyymm_to_period(x):
    return pd.Period(str(int(x)), freq="M")

def form_window_bounds(p):
    start = (p - 11).start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = p.end_time.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, end

def trading_bounds(p):
    start = p.end_time + pd.offsets.BDay(1)
    end   = start + pd.offsets.BDay(TRADING_BDAYS-1)
    return start, end

def load_panel():
    df = pd.read_csv(PANEL, parse_dates=["date"])
    need = ["date","permno","adj_prc"]
    if not set(need).issubset(df.columns):
        raise ValueError(f"Panel missing required columns: {need}")
    # Ensure ret_total exists; if not, compute from ret/dlret
    if "ret_total" not in df.columns:
        if "ret" not in df.columns:
            raise ValueError("Need ret or ret_total in panel")
        df["ret_total"] = df["ret"].astype(float)
        if "dlret" in df.columns:
            has_dl = df["dlret"].notna()
            df.loc[has_dl, "ret_total"] = (1.0 + df.loc[has_dl, "ret"].astype(float)) * (1.0 + df.loc[has_dl, "dlret"].astype(float)) - 1.0
    # Keep positive adj prices
    df = df[df["adj_prc"] > 0].drop_duplicates().sort_values(["permno","date"])
    return df

def normalized_price(series):
    if series.empty: return series
    base = series.iloc[0]
    if pd.isna(base) or base==0: return pd.Series(index=series.index, dtype=float)
    return series / base

def compute_mu_sd_from_formation(panel, i, j, p):
    f_start, f_end = form_window_bounds(p)
    si = panel[(panel["permno"]==i) & (panel["date"].between(f_start,f_end))]\
            .sort_values("date")["adj_prc"]
    sj = panel[(panel["permno"]==j) & (panel["date"].between(f_start,f_end))]\
            .sort_values("date")["adj_prc"]
    if si.empty or sj.empty: 
        return None
    form = pd.concat([si.reset_index(drop=True), sj.reset_index(drop=True)], axis=1).dropna()
    if len(form) < 60: 
        return None
    ni = normalized_price(form.iloc[:,0])
    nj = normalized_price(form.iloc[:,1])
    spread = ni - nj
    mu = float(spread.mean())
    sd = float(spread.std(ddof=1))
    if sd <= 0 or math.isnan(sd): 
        return None
    base_i = float(si.iloc[0])
    base_j = float(sj.iloc[0])
    return {"mu":mu,"sd":sd,"base_i":base_i,"base_j":base_j}

def find_trades(z, wait_day=0):
    trades = []
    state=0
    entry=None
    side=0
    for k in range(len(z)):
        zk = z.iloc[k]
        if state==0:
            # only enter new trades when the previous period's spread was within normal bounds
            prev_ok = (k == 0) or (abs(z.iloc[k-1]) < Z_ENTRY)
            if prev_ok and abs(zk) >= Z_ENTRY:
                entry = k + wait_day
                if entry >= len(z):
                    continue
                side = int(-np.sign(zk))  # z>0 -> short i/long j (-1); z<0 -> long i/short j (+1)
                state = 1
        else:
            z_prev = z.iloc[k-1] if k>0 else z.iloc[k]
            crossed_zero = (zk == 0) or (z_prev > 0 and zk <= 0) or (z_prev < 0 and zk >= 0)
            if crossed_zero:
                trades.append({"entry_idx":entry, "exit_idx":k, "side":side})
                state=0
                entry=None
                side=0
    if state==1 and entry is not None:
        trades.append({"entry_idx":entry, "exit_idx":len(z)-1, "side":side, "forced_exit":True})
    return trades

def pair_daily_return_series(trad, z, trades):
    ri = trad["ret_total_i"].fillna(0.0).to_numpy()
    rj = trad["ret_total_j"].fillna(0.0).to_numpy()
    pos = np.zeros(len(z))

    transaction_costs = np.zeros(len(z))

    for tr in trades:
        a,b,s = tr["entry_idx"], tr["exit_idx"], tr["side"]
        start = min(a+1, len(z)-1)  # P&L accrues from day AFTER entry signal close
        end = b
        if end >= start:
            pos[start:end+1] = s
            # Add transaction cost for both entry and exit
            transaction_costs[start] -= 0.0020
            transaction_costs[end] -= 0.0020 

    pair_returns = 0.5*(pos*ri) + 0.5*(-pos*rj) + transaction_costs
    return pd.Series(pair_returns, index=trad["date"])

def build_portfolio_series_pair_level():
    panel = load_panel()
    files = sorted(glob.glob(os.path.join(COHORT_DIR, "matched_pairs_monthly_*.csv")))
    if not files:
        raise RuntimeError("No cohort files found in monthly_cohorts_top20/.")

    pair_series = []
    for path in files:
        m = re.search(r"matched_pairs_monthly_(\d{6})\.csv$", path)
        if not m: continue
        yyyymm = int(m.group(1))
        p = yyyymm_to_period(yyyymm)

        df = pd.read_csv(path).head(TOP_N_CAP).copy()
        if "permno_1" not in df.columns: 
            continue

        t_start, t_end = trading_bounds(p)
        for _, row in df.iterrows():
            i = int(row["permno_1"]); j = int(row["permno_2"])
            fstats = compute_mu_sd_from_formation(panel, i, j, p)
            if fstats is None:
                continue
            mu, sd, base_i, base_j = fstats["mu"], fstats["sd"], fstats["base_i"], fstats["base_j"]

            si = panel[(panel["permno"]==i) & (panel["date"].between(t_start,t_end))]\
                    .sort_values("date")[["date","adj_prc","ret_total"]]\
                    .rename(columns={"adj_prc":"adj_prc_i","ret_total":"ret_total_i"})
            sj = panel[(panel["permno"]==j) & (panel["date"].between(t_start,t_end))]\
                    .sort_values("date")[["date","adj_prc","ret_total"]]\
                    .rename(columns={"adj_prc":"adj_prc_j","ret_total":"ret_total_j"})
            trad = si.merge(sj, on="date", how="inner")
            if len(trad) < 30:
                continue

            ni = trad["adj_prc_i"]/base_i
            nj = trad["adj_prc_j"]/base_j
            z = (ni - nj - mu)/sd

            # TODO: if WAIT_ONE_DAY, set wait_day = 1
            wait_day = 1 if WAIT_ONE_DAY else 0
            trades = find_trades(z, wait_day=wait_day)
            if len(trades)==0:
                series = pd.Series(0.0, index=trad["date"])
            else:
                series = pair_daily_return_series(trad, z, trades)

            pair_series.append(series)

    if not pair_series:
        raise RuntimeError("No pair-level series built.")

    mat = pd.concat(pair_series, axis=1)
    portfolio_daily = mat.mean(axis=1).sort_index()

    portfolio_monthly = portfolio_daily.groupby([portfolio_daily.index.year, portfolio_daily.index.month])\
                                       .apply(lambda s: (1.0+s).prod()-1.0)
    portfolio_monthly.index = [f"{y:04d}{m:02d}" for (y,m) in portfolio_monthly.index]
    return portfolio_daily, portfolio_monthly

def load_french_monthly_csv(path, rename_map=None):
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        parts = [p.strip() for p in line.strip().split(",")]
        if not parts or len(parts[0])==0: continue
        if re.match(r"^\d{6}$", parts[0]):
            rows.append(parts)
        elif rows and parts[0].lower().startswith("annual"):
            break
    if not rows: raise RuntimeError(f"Could not parse monthly block in {path}")
    ncol = max(len(r) for r in rows)
    if ncol==5:
        cols = ["YYYYMM","Mkt-RF","SMB","HML","RF"]
    elif ncol==2:
        cols = ["YYYYMM","VAL"]
    else:
        cols = ["YYYYMM"] + [f"V{i}" for i in range(1,ncol)]
    csv_text = ",".join(cols) + "\n" + "\n".join([",".join(r[:ncol]) for r in rows])
    df = pd.read_csv(io.StringIO(csv_text), dtype={"YYYYMM":str})
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def run_ffmr(portfolio_monthly):
    os.makedirs(OUTDIR, exist_ok=True)
    pm_df = pd.DataFrame({"YYYYMM": portfolio_monthly.index, "R_p": portfolio_monthly.values})
    pm_df.to_csv(os.path.join(OUTDIR, "portfolio_monthly.csv"), index=False)

    ff3 = load_french_monthly_csv("F-F_Research_Data_Factors.csv")
    mom = load_french_monthly_csv("F-F_Momentum_Factor.csv", rename_map={"VAL":"UMD"})
    strev = load_french_monthly_csv("F-F_ST_Reversal_Factor.csv", rename_map={"VAL":"ST_Rev"})
    factors = ff3.merge(mom, on="YYYYMM", how="left").merge(strev, on="YYYYMM", how="left")

    data = pm_df.merge(factors, on="YYYYMM", how="inner").copy()
    data["R_p_pct"] = data["R_p"] * 100.0
    data["R_excess"] = data["R_p_pct"] - data["RF"]

    X = data[["Mkt-RF","SMB","HML","UMD","ST_Rev"]].astype(float)
    X = sm.add_constant(X)
    y = data["R_excess"].astype(float)

    model = sm.OLS(y, X, missing='drop').fit(cov_type="HAC", cov_kwds={"maxlags":6})

    rows = []
    for name, label in [("const","Intercept (Alpha)"),("Mkt-RF","Mkt-RF"),("SMB","SMB"),
                        ("HML","HML"),("UMD","UMD"),("ST_Rev","ST_Rev")]:
        rows.append({"Model":"FFMR Monthly Cohorts (Pair-level EW)","Factor":label,
                     "Coef":model.params.get(name, np.nan),"t-stat":model.tvalues.get(name, np.nan)})
    res_table = pd.DataFrame(rows)[["Model","Factor","Coef","t-stat"]]
    res_table.to_csv(os.path.join(OUTDIR, "ffmr_regression_table.csv"), index=False)

    R = data["R_excess"]/100.0
    mean_m = R.mean(); std_m = R.std(ddof=1)
    sharpe_m = mean_m/std_m if std_m!=0 else np.nan
    ac1 = R.autocorr(lag=1)
    perf = pd.DataFrame({
        "Series":["FFMR Monthly Cohorts (Pair-level EW)"],
        "Mean excess (mo)":[mean_m],
        "Std (mo)":[std_m],
        "Sharpe (mo)":[sharpe_m],
        "AC(1)":[ac1],
        "T months":[len(R.dropna())],
        "R^2":[model.rsquared]
    })
    perf.to_csv(os.path.join(OUTDIR, "ffmr_performance_stats.csv"), index=False)

def main():
    topN = COHORT_DIR[-3:]

    global OUTDIR
    # TODO: set to true, if want to delay trade
    if WAIT_ONE_DAY:
        OUTDIR = f"monthly_trading_ffmr_{topN}_wait1d"
    else:
        OUTDIR = f"monthly_trading_ffmr_{topN}"

    os.makedirs(OUTDIR, exist_ok=True)
    daily, monthly = build_portfolio_series_pair_level()
    daily.to_csv(os.path.join(OUTDIR, "portfolio_daily.csv"), header=["r_p_daily"])
    run_ffmr(monthly)
    print("Done. Outputs in", OUTDIR)

if __name__ == "__main__":
    main()
