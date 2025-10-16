import wrds
import pandas as pd
from pairs_formation import fetch_crsp_data, compute_pairwise_ssd, pair_matching
import os

# Industry to code
INDUSTRY_CODE = {
    'Agriculture, Forestry, Fishing and Hunting': ['11'],
    'Mining, Quarrying, and Oil and Gas Extraction': ['21'],
    'Utilities': ['22'],
    'Construction': ['23'],
    'Manufacturing': ['31', '32', '33'],
    'Wholesale Trade': ['42'],
    'Retail Trade': ['44', '45'],
    'Transportation and Warehousing': ['48', '49'],
    'Information': ['51'],
    'Finance and Insurance': ['52'],
    'Real Estate and Rental and Leasing': ['53'],
    'Professional, Scientific, and Technical Services': ['54'],
    'Management of Companies and Enterprises': ['55'],
    'Administrative and Support and Waste Management and Remediation Services': ['56'],
    'Educational Services': ['61'],
    'Health Care and Social Assistance': ['62'],
    'Arts, Entertainment, and Recreation': ['71'],
    'Accommodation and Food Services': ['72'],
    'Other Services (except Public Administration)': ['81'],
    'Public Administration': ['92']
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

    # Save it
    print(f"Saving cumulative returns index to {industry}_{formation_start}_{formation_end}.csv")
    cum_index.to_csv(f"data/returns/{industry}_{formation_start}_{formation_end}.csv")

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
    filename = f'data/crsp_data_{formation_start}_{formation_end}.csv'
    try:
        df_daily = pd.read_csv(filename)
        print(f"Loaded CRSP data from {filename}")
    except FileNotFoundError:
        print("Fetching CRSP data from WRDS")
        df_daily = fetch_crsp_data(db, formation_start, formation_end)
    
    # group by first 2 digits of naics
    df_daily['naics_2d'] = df_daily['naics'].astype(str).str[:2]
    industry_groups = df_daily.groupby('naics_2d')['permco'].unique().to_dict()
    industry_dict = {}
    for code, permcos in industry_groups.items():
        for industry_name, codeList in INDUSTRY_CODE.items():
            industry_name = industry_name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
            if code in codeList:
                industry_dict.setdefault(industry_name, []).extend(permcos)

    # for each industry group, get list of permcos
    for industry_name, permcos in industry_dict.items():
        
        print(f"Processing industry: {industry_name} with {len(permcos)} stocks")

        # Create mormalized cumulative return index
        print("Building cumulative return index")
        # get all prices for these permcos
        df_industry = df_daily[df_daily['permco'].isin(permcos)]
        normalize_cum_returns_index = build_industry_cum_total_return_index(df_industry, formation_start, formation_end, industry_name)

        # Compute SSDs 
        print("Computing pairwise SSD")
        ssd_df = compute_pairwise_ssd(normalize_cum_returns_index)

        # Match into pairs
        print("Matching into pairs")
        matched_pairs = pair_matching(ssd_df, max_pairs=max_pairs_to_return)
        matched_pairs = matched_pairs.sort_values('ssd').reset_index(drop=True)
        
        # get permco comname and ticker by merging to df_daily
        matched_pairs = matched_pairs.merge(df_daily[['permco', 'comnam']].drop_duplicates(), how='left', left_on='permco_1', right_on='permco').rename(columns={"comnam": "comnam_1"})
        matched_pairs = matched_pairs.merge(df_daily[['permco', 'comnam']].drop_duplicates(), how='left', left_on='permco_2', right_on='permco').rename(columns={"comnam": "comnam_2"})
        matched_pairs = matched_pairs.drop(columns=['permco_x', 'permco_y'])
        matched_pairs.to_csv(f"data/matched_pairs/{industry_name}_{formation_start}_{formation_end}.csv", index=False)

# Report the ssd stats of each industry
def find_best_industry(directory='data/matched_pairs'):
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


if __name__ == "__main__":

    db = wrds.Connection(wrds_username='sohrac')

    formation_start = "2024-01-01"
    formation_end   = "2024-12-31"

    form_pairs_wrds(db, formation_start, formation_end, 3000)
    
    # Find 'best' industry
    stats = find_best_industry()
    stats = dict(sorted(stats.items(), key=lambda item: item[1][0]))

    for industry, (min_ssd, avg_ssd, var_ssd) in stats.items():
        industry_name = industry.split('_2024')[0]
        print(f"Industry: {industry_name}, Min SSD: {min_ssd:.6f}, Avg SSD: {avg_ssd:.6f}, Var SSD: {var_ssd:.6f}")