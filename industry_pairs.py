import wrds
import pandas as pd
from pairs_formation import fetch_crsp_data, compute_pairwise_ssd, pair_matching, get_stock
import os
from datetime import date,timedelta
from dateutil.relativedelta import relativedelta

# Industry to code
INDUSTRY_CODE = {
    "Agriculture, Forestry and Fishing" : (0, 999),
	"Mining" : (1000, 1499),
	"Construction" : (1500, 1799), 
	"Manufacturing":(2000, 3999),
	"Transportation, Communications, Electric, Gas and Sanitary service": (4000,4999),
    "Wholesale Trade": (5000,5199),
    "Retail Trade": (5200,5999),
    "Finance, Insurance and Real estate" : (6000,6799),
    "Services" : (7000,8999),
    "Public administration":(9100,9729)
}

def build_industry_cum_total_return_index(df_daily: pd.DataFrame, formation_start, formation_end, industry: str):
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
    
    # # drop any permnos with vol 0 on any day in the formation period
    vol_wide = df_daily.pivot(index='date', columns='permno', values='vol').sort_index()
    vol_wide_clean = vol_wide.dropna(axis=1)
    df_daily = df_daily[df_daily['permno'].isin(vol_wide_clean.columns)]
    
    # Create new table where date is row, permco is col and the values are the daily returns 
    ret_wide = df_daily.pivot(index='date', columns='permno', values='ret_num').sort_index()
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

    # Save it
    print(f"Saving cumulative returns index to {industry}_{formation_start}_{formation_end}.csv")
    cum_index.to_csv(f"data/industry_returns/{industry}_{formation_start}_{formation_end}.csv")

    return cum_index

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
        df_daily = pd.read_csv(filename)
        print(f"Loaded CRSP data from {filename}")
    except FileNotFoundError:
        print("Fetching CRSP data from WRDS")
        df_daily = fetch_crsp_data(db, formation_start, formation_end)
    
    df_daily['siccd'] = df_daily['siccd'].astype(int)
    industry_groups = df_daily.groupby('siccd')['permno'].unique().to_dict()
    industry_dict = {}
    for code, permnos in industry_groups.items():
        for industry_name, (lower,upper) in INDUSTRY_CODE.items():
            industry_name = industry_name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
            if lower <= code <= upper:
                industry_dict.setdefault(industry_name, []).extend(permnos)
    
    # for each industry group, get list of permcos
    for industry_name, permnos in industry_dict.items():
        
        print(f"Processing industry: {industry_name} with {len(permnos)} stocks")
        if len(permnos) < 2:
            print(f"Not enough stocks in industry {industry_name} to form pairs. Skipping.")
            continue

        # Create mormalized cumulative return index
        print("Building cumulative return index")
        # get all prices for these permcos
        df_industry = df_daily[df_daily['permno'].isin(permnos)]
        normalize_cum_returns_index = build_industry_cum_total_return_index(df_industry, formation_start, formation_end, industry_name)

        # Compute SSDs 
        print("Computing pairwise SSD")
        ssd_df = compute_pairwise_ssd(normalize_cum_returns_index)

        # Match into pairs
        print("Matching into pairs")
        matched_pairs = pair_matching(ssd_df, df_daily, max_pairs=max_pairs_to_return)
        if matched_pairs.size != 0:
            matched_pairs = matched_pairs.sort_values('ssd').reset_index(drop=True)
            
            # get permco comname and ticker by merging to df_daily
            matched_pairs = matched_pairs.merge(df_daily[['permno', 'comnam']].drop_duplicates(), how='left', left_on='permno_1', right_on='permno').rename(columns={"comnam": "comnam_1"})
            matched_pairs = matched_pairs.merge(df_daily[['permno', 'comnam']].drop_duplicates(), how='left', left_on='permno_2', right_on='permno').rename(columns={"comnam": "comnam_2"})
            matched_pairs = matched_pairs.drop(columns=['permno_x', 'permno_y'])
            matched_pairs = get_stock(matched_pairs,db)
            matched_pairs.to_csv(f"permno_data/industry_pairs/{industry_name}/{industry_name}_{formation_start}_{formation_end}.csv", index=False)

# Report the ssd stats of each industry
def find_best_industry(directory='permno_data/industry_pairs'):
    industry_stats = {}

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            if not df.empty:
                # Report the ssd stats of each industry
                min_ssd = df['ssd'].min()
                avg_ssd = df['ssd'].mean()
                var_ssd = df['ssd'].var() 

                industry_stats[filename] = (min_ssd, avg_ssd, var_ssd)
    return industry_stats

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

if __name__ == "__main__":
    # TODO: replace with your own username
    db = wrds.Connection(wrds_username='sohrac')
    
    # TODO: CHANGE THE DATES
    a = generate_yearly_tuples("1962-01-01","1966-01-01")
    b = generate_yearly_tuples("2000-01-01","2004-01-01")
    c = generate_yearly_tuples("2018-01-01","2023-01-01")
    date_ranges = a + b + c
    
    for (formation_start, formation_end) in date_ranges:
        form_pairs_wrds(db, formation_start, formation_end, 100)
        
        # Find 'best' industry
        stats = find_best_industry()
        stats = dict(sorted(stats.items(), key=lambda item: item[1][0]))

        for industry, (min_ssd, avg_ssd, var_ssd) in stats.items():
            industry_name = industry.split('_2024')[0]
            print(f"Industry: {industry_name}, Min SSD: {min_ssd:.6f}, Avg SSD: {avg_ssd:.6f}, Var SSD: {var_ssd:.6f}")