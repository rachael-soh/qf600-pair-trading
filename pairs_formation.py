import wrds
import pandas as pd
import numpy as np
from itertools import combinations
from typing import Tuple, List


def fetch_crsp_data(db: wrds.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    """    
    Get common shares (SHRCD in (10 Securities which have not been further defined, 11 Securities which need not be further defined)) 
    on major exchanges  EXCHCD in (1 NYSE,2 ASE,3 NASDAQ))
    See this link for details on the tables: https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_a_stock/
    See details of codes: https://wrds-www.wharton.upenn.edu/documents/399/Data_Descriptions_Guide.pdf 
    """
    sql = f"""
    select a.permco, a.date, a.ret, a.prc, a.vol,
       b.comnam, b.shrcd, b.exchcd, b.siccd,
       c.dlret
    from crsp.dsf a
      left join crsp.dsenames b
        on a.permco = b.permco
        and a.date >= b.namedt
        and a.date <= b.nameendt
      left join crsp.dsedelist c
        on a.permco = c.permco
        and a.date = c.dlstdt
        where a.date between '{start_date}' and '{end_date}'
        and b.shrcd in (10,11)
        and b.exchcd in (1,2,3)
    """
    df = db.raw_sql(sql, date_cols=['date'])
    df.to_csv(f'crsp_data_{start_date}_{end_date}.csv', index=False)
    return df

def build_cum_total_return_index(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build cumulative total-return index for each permco,
    - Screen out stocks that have one or more days with no trade
    - Normalize to 1 at the start date
    """
    # Set values first and coerce any errors to NaN 
    df_daily['ret_num'] = pd.to_numeric(df_daily['ret'], errors='coerce')
    df_daily['date'] = pd.to_datetime(df_daily['date'], errors='coerce')

     # Check for duplicates in the combination of 'date' and 'permco'
    if df_daily.duplicated(subset=['date', 'permco']).any():
        print("Duplicate entries found in 'date' and 'permco'. Resolving by taking the mean.")
        # Resolve duplicates by grouping and taking the mean
        numeric_cols = df_daily.select_dtypes(include=['number']).columns  # Select only numeric columns
        df_daily = df_daily.groupby(['date', 'permco'], as_index=False)[numeric_cols].mean()



    # Create new table where date is row, permco is col and the values are the daily returns 
    ret_wide = df_daily.pivot(index='date', columns='permco', values='ret_num').sort_index()
    
    # Screen out stocks that have one or more days with no trade
    ret_wide_clean = ret_wide.dropna(axis=1)

    # Build the cumulative total returns index, which apparently is: 
    # Total Return Index = Previous TR * [1+(Today’s PR Index + Indexed Dividend/Previous PR Index-1)]
    # but ret already includes the reinvested dividend, so we can just do 1+ret, then get the cumulative product over each row
    # See https://www.angelone.in/knowledge-center/share-market/what-is-total-return-index
    one_plus_r = 1.0 + ret_wide_clean
    cum_index = one_plus_r.cumprod()
    
    # Normalize it
    cum_index = cum_index.div(cum_index.iloc[0, :])

    return cum_index

def compute_pairwise_ssd(cum_index: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SSD for all pairs from cumulative returns index
    """
    pairs = []
    for stock_1, stock_2 in combinations(cum_index.columns, 2):
        diff = cum_index[stock_1].values - cum_index[stock_2].values
        ssd = float(np.sum(diff * diff))
        pairs.append((stock_1, stock_2, ssd))
    # Sort by ssd in ascending order
    pairs_df = pd.DataFrame(pairs, columns=['permco_1', 'permco_2', 'ssd']).sort_values('ssd').reset_index(drop=True)
    return pairs_df

def pair_matching(pairs_df: pd.DataFrame, max_pairs: int = None) -> pd.DataFrame:
    """
    Algo to create pairs:
    - iterate through sorted pairs 
    - pick a pair if neither stock was matched yet,
    - stop when we've matched all or reached max_pairs (if provided).
    """
    # keep track of stocks that have been matched
    matched = set()
    pairs = []
    for _, row in pairs_df.iterrows():
        stock_1, stock_2, ssd = int(row['permco_1']), int(row['permco_2']), float(row['ssd'])
        if stock_1 == stock_2:
            continue
        if (stock_1 not in matched) and (stock_2 not in matched):
            pairs.append({'permco_1': stock_1, 'permco_2': stock_2, 'ssd': ssd})
            matched.add(stock_1); matched.add(stock_2)
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pd.DataFrame(pairs)


def form_pairs_wrds(
    db: wrds.Connection,
    formation_start: str,
    formation_end: str,
    max_pairs_to_return
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pairs formation following Gatev et al. 2006 Section 2.1:
      - fetch CRSP daily for formation window
      - construct cumulative total-return indexes (normalize to 1 at start)
      - compute SSDs 
      - form pairs
    Returns: (matched_pairs_df, cum_index_df)
    - matched_pairs_df: columns permco_1, permco_2, ssd (sorted ascending by ssd)
    - cum_index_df: normalized cum-return series used for SSD calculation (dates x permco)
    """
    # Fetch CRSP data if the csv doesn't exist locally
    filename = f'crsp_data_{formation_start}_{formation_end}.csv'
    try:
        df_daily = pd.read_csv(filename)
        print(f"Loaded CRSP data from {filename}")
    except FileNotFoundError:
        df_daily = fetch_crsp_data(db, formation_start, formation_end)

    # Create mormalized cumulative return index
    print("Building cumulative return index")
    normalize_cum_returns_index = build_cum_total_return_index(df_daily)

    # Compute SSDs 
    print("Computing pairwise SSD")
    ssd_df = compute_pairwise_ssd(normalize_cum_returns_index)

    # Match into pairs
    print("Matching into pairs")
    matched_pairs = pair_matching(ssd_df, max_pairs=max_pairs_to_return)
    matched_pairs = matched_pairs.sort_values('ssd').reset_index(drop=True)

    return matched_pairs, normalize_cum_returns_index


if __name__ == "__main__":

    db = wrds.Connection(wrds_username='sohrac')

    formation_start = "2020-01-01"
    formation_end   = "2021-01-01"

    matched_pairs_df, cum_index_df = form_pairs_wrds(db, formation_start, formation_end, 3000)

    print(f"Total pairs formed: {len(matched_pairs_df)}")
    print("Top 20 pairs (smallest SSD):")
    print(matched_pairs_df.head(20))

    # Save outputs
    matched_pairs_df.to_csv("matched_pairs.csv", index=False)
    cum_index_df.to_csv("cum_returns_index.csv")
