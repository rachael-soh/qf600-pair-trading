import wrds
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---LOAD THE CSV FILE ---
pair_file = 'Finance_and_Insurance_2024-01-01_2024-12-31.csv'

# Load the data first
returns = pd.read_csv(pair_file)

# Convert to collect Data types 
print("columns found:", returns.columns.tolist())       

# Convert to appropriate data types
type_map = {
    'permco_1': 'int64',
    'permco_2': 'int64',
    'ssd': 'float64',
    'comnam_1': 'string',
    'ticker_1': 'string',
    'naics_1': 'Int64',
    'comnam_2': 'string',
    'ticker_2': 'string',
    'naics_2': 'Int64'
}

# Apply conversions safely
for col, dtype in type_map.items():
    if col in returns.columns:
        if dtype in ['int64', 'Int64', 'float64']:
            returns[col] = pd.to_numeric(returns[col], errors='coerce')
        else:
            returns[col] = returns[col].astype(dtype)

print("\nPairs data loaded successfully!")
print(returns.head())

# LOAD PRICE DATA from WRDS
def fetch_crsp_data(db: wrds.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily stock data from CRSP (daily stock file - dsf)"""
    query = f"""
        SELECT a.permco, a.date, a.ret, a.prc
        FROM crsp.dsf a
        WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
          AND a.prc IS NOT NULL
          AND a.ret IS NOT NULL
          AND ABS(a.ret) < 1  
    """
    return db.raw_sql(query, date_cols=['date'])

# Connect to WRDS
db = wrds.Connection(wrds_username='andygse')

# Pull CRSP data for formation period
data = fetch_crsp_data(db, '2024-01-01', '2024-06-30')
print(f"\nCRSP data fetched: {data.shape}")
print(data.head())  

# NORMALIZE PRICES
def normalize_prices(price_series):
    """Normalize prices to start at 1.0"""
    return price_series / price_series.iloc[0]

# CALCULATE SPREAD STATISTICS
def calculate_spread_stats(formation_data, pair_row):
    """
    Calculate mean and std dev of spread during formation period
    These will be used for 2-sigma trading thresholds
    """
    s1, s2 = pair_row['permco_1'], pair_row['permco_2']
    
    # Get price series for both stocks
    s1_data = formation_data[formation_data['permco'] == s1].sort_values('date').set_index('date')['prc']
    s2_data = formation_data[formation_data['permco'] == s2].sort_values('date').set_index('date')['prc']
    
    # Check if both stocks have data
    if len(s1_data) == 0 or len(s2_data) == 0:
        return None
    
    # Normalize prices
    norm_s1 = normalize_prices(s1_data)
    norm_s2 = normalize_prices(s2_data)
    
    # Align indices (forward fill if needed)
    combined = pd.DataFrame({'s1': norm_s1, 's2': norm_s2})
    combined = combined.fillna(method='ffill').dropna()
    
    if len(combined) == 0:
        return None
    
    # Calculate spread
    spread = combined['s1'] - combined['s2']
    mu_spread = spread.mean()
    sigma_spread = spread.std()

    return {
        'stock1': s1,
        'stock2': s2,
        'ticker1': pair_row['ticker_1'],
        'ticker2': pair_row['ticker_2'],
        'ssd': pair_row['ssd'],
        'mu_spread': mu_spread,
        'sigma_spread': sigma_spread,
        'threshold_upper': mu_spread + 2 * sigma_spread,
        'threshold_lower': mu_spread - 2 * sigma_spread
    }

# TRADING SIMULATION
def simulate_trading(trading_data, pair_stats):
    """
    Simulate trading strategy for one pair
    
    Entry: When spread > mu + 2*sigma OR spread < mu - 2*sigma
    Exit: When spread crosses back through mean
    """
    if pair_stats is None:
        return None
    
    s1, s2 = pair_stats['stock1'], pair_stats['stock2']
    mu = pair_stats['mu_spread']
    upper = pair_stats['threshold_upper']
    lower = pair_stats['threshold_lower']
    
    # Get price series for both stocks
    s1_data = trading_data[trading_data['permco'] == s1].sort_values('date').set_index('date')['prc']
    s2_data = trading_data[trading_data['permco'] == s2].sort_values('date').set_index('date')['prc']
    
    # Check if both stocks exist
    if len(s1_data) == 0 or len(s2_data) == 0:
        return None
    
    # Normalize prices
    norm_s1 = normalize_prices(s1_data)
    norm_s2 = normalize_prices(s2_data)
    
    # Align indices
    combined = pd.DataFrame({'s1': norm_s1, 's2': norm_s2})
    combined = combined.fillna(method='ffill').dropna()
    
    if len(combined) == 0:
        return None
    
    spread = combined['s1'] - combined['s2']
    
    # Track positions and trades
    position = 0  # 0=closed, 1=long s1/short s2, -1=short s1/long s2
    entry_date = None
    entry_spread = None
    trades = []
    
    for i in range(len(spread)):
        current_date = spread.index[i]
        current_spread = spread.iloc[i]
        
        # ENTRY LOGIC
        if position == 0:
            if current_spread > upper:
                position = -1
                entry_date = current_date
                entry_spread = current_spread
                print(f"  {current_date.date()}: OPEN Short {s1}/Long {s2}, Spread={current_spread:.4f} > {upper:.4f}")
            
            elif current_spread < lower:
                position = 1
                entry_date = current_date
                entry_spread = current_spread
                print(f"  {current_date.date()}: OPEN Long {s1}/Short {s2}, Spread={current_spread:.4f} < {lower:.4f}")
        
        # EXIT LOGIC: Close when spread crosses mean
        else:
            if i > 0:
                prev_spread = spread.iloc[i-1]
                
                # Check if we crossed the mean
                crossed_mean = (prev_spread > mu and current_spread <= mu) or \
                               (prev_spread < mu and current_spread >= mu)
                
                if crossed_mean:
                    exit_spread = current_spread
                    pnl = -position * (exit_spread - entry_spread)
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_spread': entry_spread,
                        'exit_spread': exit_spread,
                        'position': 'Long S1/Short S2' if position == 1 else 'Short S1/Long S2',
                        'pnl': pnl,
                        'days_held': (current_date - entry_date).days
                    })
                    
                    print(f"  {current_date.date()}: CLOSE, Spread={current_spread:.4f}, PnL={pnl:.4f}")
                    position = 0
                    entry_date = None
                    entry_spread = None
    
    # Force close at end of period if still open
    if position != 0:
        exit_spread = spread.iloc[-1]
        pnl = -position * (exit_spread - entry_spread)
        
        trades.append({
            'entry_date': entry_date,
            'exit_date': spread.index[-1],
            'entry_spread': entry_spread,
            'exit_spread': exit_spread,
            'position': 'Long S1/Short S2' if position == 1 else 'Short S1/Long S2',
            'pnl': pnl,
            'days_held': (spread.index[-1] - entry_date).days
        })
        
        print(f"  {spread.index[-1].date()}: FORCE CLOSE (end of period), PnL={pnl:.4f}")
    
    return {
        'pair': f"{pair_stats['ticker1']}-{pair_stats['ticker2']}",
        'stock1': s1,
        'stock2': s2,
        'num_trades': len(trades),
        'trades': trades,
        'total_pnl': sum([t['pnl'] for t in trades]),
        'avg_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0,
        'spread_series': spread
    }

# MAIN EXECUTION
if __name__ == "__main__":
    # Get trading data (next 6 months after formation)
    trading_data = fetch_crsp_data(db, '2024-07-01', '2024-12-31')
    
    # Process first 5 pairs for testing
    results = []
    for idx, (_, pair_row) in enumerate(returns.head(5).iterrows()):
        print(f"\n--- Processing Pair {idx+1}: {pair_row['ticker_1']} vs {pair_row['ticker_2']} ---")
        
        pair_stats = calculate_spread_stats(data, pair_row)
        if pair_stats:
            result = simulate_trading(trading_data, pair_stats)
            if result:
                results.append(result)
                print(f"  Summary: {result['num_trades']} trades, Total PnL: {result['total_pnl']:.4f}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL PAIRS")
    print("="*60)
    if results:
        summary_df = pd.DataFrame([{
            'Pair': r['pair'],
            'Num_Trades': r['num_trades'],
            'Total_PnL': r['total_pnl'],
            'Avg_PnL': r['avg_pnl']
        } for r in results])
        print(summary_df)
    else:
        print("No valid trades generated")