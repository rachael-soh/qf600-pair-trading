import wrds
import pandas as pd
import numpy as np
from itertools import combinations
from typing import Tuple, List
from datetime import date,timedelta
from dateutil.relativedelta import relativedelta

def fetch_crsp_data(db: wrds.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    """    
    Get common shares (SHRCD in (10 Securities which have not been further defined, 11 Securities which need not be further defined)) 
    on major exchanges  EXCHCD in (1 NYSE,2 ASE,3 NASDAQ))
    See this link for details on the tables: https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_a_stock/
    See details of codes: https://wrds-www.wharton.upenn.edu/documents/399/Data_Descriptions_Guide.pdf 
    """
    sql = f"""
    select a.permco, a.permno, a.date, a.ret, a.prc, a.vol, a,naics,
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
    df.to_csv(f'data/crsp_data/crsp_data_{start_date}_{end_date}.csv', index=False)
    return df

def build_cum_total_return_index(df_daily: pd.DataFrame, formation_start, formation_end) -> pd.DataFrame:
    """
    Build cumulative total-return index for each permco,
    - Screen out stocks that have one or more days with no trade
    - Normalize to 1 at the start date
    """
    # Set values types first and coerce any errors to NaN 
    df_daily['ret_num'] = pd.to_numeric(df_daily['ret'], errors='coerce')
    df_daily['date'] = pd.to_datetime(df_daily['date'], errors='coerce')

    # Check for duplicates in the combination of 'date' and 'permco'
    if df_daily.duplicated(subset=['date', 'permno']).any():
        print("Duplicate entries found in 'date' and 'permno'. Resolving by taking the mean.")
        # Resolve duplicates by grouping and taking the mean
        numeric_cols = df_daily.select_dtypes(include=['number']).columns  # Select only numeric columns
        df_daily = df_daily.groupby(['date', 'permno'], as_index=False)[numeric_cols].mean()

    # drop any permnos which have any 0 vol trading days
    df_daily = df_daily[df_daily['vol'].notna() & (df_daily['vol'] != 0)] 
    vol_wide = df_daily.pivot(index='date', columns='permno', values='vol').sort_index()
    df_daily = df_daily[df_daily['permno'].isin(vol_wide.columns)]

    # Create new table where date is row, permco is col and the values are the daily returns 
    ret_wide = df_daily.pivot(index='date', columns='permno', values='ret_num').sort_index()
    # # Screen out stocks that have missing returs
    ret_wide_clean = ret_wide.dropna(axis=1)
    ret_wide_clean
    
    # Build the cumulative total returns index, which apparently is: 
    # Total Return Index = Previous TR * [1+(Today’s PR Index + Indexed Dividend/Previous PR Index-1)]
    # but ret already includes the reinvested dividend, so we can just do 1+ret, then get the cumulative product over each row
    # See https://www.angelone.in/knowledge-center/share-market/what-is-total-return-index
    one_plus_r = 1.0 + ret_wide_clean
    cum_index = one_plus_r.cumprod()
    
    # Normalize it
    cum_index = cum_index.div(cum_index.iloc[0, :])

    # Save it
    print(f"Saving cumulative returns index to cum_returns_index_{formation_start}_{formation_end}.csv")
    cum_index.to_csv(f"data/cum_returns/cum_returns_index_{formation_start}_{formation_end}.csv")

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
    pairs_df = pd.DataFrame(pairs, columns=['permno_1', 'permno_2', 'ssd']).sort_values('ssd').reset_index(drop=True)
    return pairs_df

def pair_matching(pairs_df: pd.DataFrame, crsp_data: pd.DataFrame, max_pairs: int = None) -> pd.DataFrame:
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
        stock_1, stock_2, ssd = int(float(row['permno_1'])), int(float(row['permno_2'])), float(row['ssd'])
        permco_1 = crsp_data[crsp_data['permno'] == stock_1]['permco'].iloc[0]
        permco_2 = crsp_data[crsp_data['permno'] == stock_2]['permco'].iloc[0]

        if stock_1 == stock_2:
            continue
        # match if not matched yet and if they're not from the same permco
        if (stock_1 not in matched) and (stock_2 not in matched) and (permco_1 != permco_2):
            pairs.append({'permno_1': stock_1, 'permno_2': stock_2, 'ssd': ssd})
            matched.add(stock_1); matched.add(stock_2)
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pd.DataFrame(pairs)


def form_pairs_wrds(
    db: wrds.Connection,
    formation_start: str,
    formation_end: str,
    max_pairs_to_return
) -> pd.DataFrame:
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
    filename = f'data/crsp_data/crsp_data_{formation_start}_{formation_end}.csv'
    try:
        crsp_data = pd.read_csv(filename)
        print(f"Loaded CRSP data from {filename}")
    except FileNotFoundError:
        print("Fetching CRSP data from WRDS")
        crsp_data = fetch_crsp_data(db, formation_start, formation_end)

    # Create mormalized cumulative return index
    print("Building cumulative return index")
    normalize_cum_returns_index = build_cum_total_return_index(crsp_data, formation_start, formation_end)

    # Compute SSDs 
    print("Computing pairwise SSD")
    ssd_df = compute_pairwise_ssd(normalize_cum_returns_index)

    # Match into pairs
    print("Matching into pairs")
    matched_pairs = pair_matching(ssd_df, crsp_data, max_pairs=max_pairs_to_return)
    matched_pairs = matched_pairs.sort_values('ssd').reset_index(drop=True)

    return matched_pairs

def generate_yearly_tuples(start_date_range, end_date_range):
    date_tuples = []
    current_start = start_date_range
    
    start_date_range = date.fromisoformat(start_date_range)
    end_date_range = date.fromisoformat(end_date_range)
    current_start = start_date_range

    one_year_delta = relativedelta(years=1)
    one_day_delta = timedelta(days=1)
    year_step = relativedelta(years=1)

    while current_start <= end_date_range:
        # Calculate the one-year anniversary date (e.g., 2024-01-01)
        current_end = current_start + one_year_delta - one_day_delta

        start_str = current_start.isoformat()
        end_str = current_end.isoformat()
        date_tuples.append((start_str, end_str))
        current_start += year_step
        
    return date_tuples

def get_stock(df, db) -> pd.DataFrame:
  
  permnos = set(df['permno_1'].unique()).union(df['permno_2'].unique())
  comnames = db.raw_sql(
    f""" 
    SELECT distinct on (permno) comnam, permno, permco, ticker, siccd
    FROM crsp.dsenames
    WHERE permno IN ({','.join(map(str, permnos))})
    ORDER BY permno, namedt DESC
    """
  )
  df = df.merge(comnames, how='left', left_on='permno_1', right_on='permno').rename(columns={"permco": "permco_1", "comnam": "comnam_1", "ticker": "ticker_1", "naics": "naics_1"})
  df = df.merge(comnames, how='left', left_on='permno_2', right_on='permno').rename(columns={"permco": "permco_2", "comnam": "comnam_2", "ticker": "ticker_2", "naics": "naics_2"})
  df = df.drop(columns=['permno_x', 'permno_y'])
  return df


if __name__ == "__main__":
    # TODO: replace with your own username
    db = wrds.Connection(wrds_username='sohrac')
    
    # # TODO: CHANGE THE DATES
    a = generate_yearly_tuples("1962-01-01","1966-01-01")
    b = generate_yearly_tuples("2000-01-01","2004-01-01")
    c = generate_yearly_tuples("2018-01-01","2023-01-01")
    date_ranges = a + b + c
    # date_ranges = [("1962-01-01","1962-12-31")]
   
    for (formation_start, formation_end) in date_ranges:

        print(f"Formation dates: {formation_start} UNTIL {formation_end}")
        matched_pairs_df= form_pairs_wrds(db, formation_start, formation_end, 100)

        print(f"Total pairs formed: {len(matched_pairs_df)}")
        print("Top 100 pairs (smallest SSD):")
        print(matched_pairs_df.head(100))

        # save in directory data
        matched_pairs_named = get_stock(matched_pairs_df, db)

        # TODO: REDEFINE WHERE YOU WANT TO SAVE THE FILE. or create a to_upload/total_matched_pairs directory in your folder
        matched_pairs_named.to_csv(f"permno_matched_pairs_{formation_start}_{formation_end}.csv", index=False)
