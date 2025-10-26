import wrds
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import csv
from datetime import date,timedelta
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')

# LOAD PRICE DATA from WRDS
def fetch_crsp_data(db: wrds.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily stock data from CRSP using returns (RET) instead of prices"""
    data_filename = f'data/crsp_data/crsp_data_{start_date}_{end_date}.csv'
    try:
        df = pd.read_csv(data_filename)
        print(f"Loaded CRSP data from {data_filename}")
    except FileNotFoundError:
        print("Fetching CRSP data from WRDS")
        query = f"""
            SELECT a.permno, a.permco, a.date, a.ret, a.prc
            FROM crsp.dsf a
            WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
            AND a.ret IS NOT NULL
            AND ABS(a.ret) < 1  
        """
        #TODO: consider saving this file.. it's quite slow to pull everytime. 
        df =  db.raw_sql(query, date_cols=['date'])
        df.to_csv(f'data/crsp_data/crsp_data_{start_date}_{end_date}.csv', index=False)

    # Check for duplicates in the combination of 'date' and 'permco'
    if df.duplicated(subset=['date', 'permno']).any():
        print("Duplicate entries found in 'date' and 'permno'. Resolving by taking the mean.")
        # Resolve duplicates by grouping and taking the mean
        numeric_cols = df.select_dtypes(include=['number']).columns  # Select only numeric columns
        df = df.groupby(['date', 'permno'], as_index=False)[numeric_cols].mean()
    df['date'] = pd.to_datetime(df['date'])
    return df

# CALCULATE CUMULATIVE RETURNS (for normalized series)
def calculate_cumulative_returns(return_series):
    """
    Calculate cumulative returns from daily returns
    Cumulative return = Product of (1 + daily_ret) - 1
    This creates a price-like series normalized to start at 1.0
    """
    cum_returns = (1 + return_series).cumprod() - 1
    return cum_returns

# CALCULATE SPREAD STATISTICS USING RETURNS
def calculate_spread_stats(formation_data, pair_row):
    """
    Calculate mean and std dev of spread during formation period using returns
    Spread = cumulative return of stock1 - cumulative return of stock2
    """
    s1, s2 = pair_row['permno_1'], pair_row['permno_2']

    s1_data = formation_data[formation_data['permno'] == s1].sort_values('date').set_index('date')['ret']
    s2_data = formation_data[formation_data['permno'] == s2].sort_values('date').set_index('date')['ret']

    if len(s1_data) == 0 or len(s2_data) == 0:
        return None

    # Calculate cumulative returns
    cum_s1 = calculate_cumulative_returns(s1_data)
    cum_s2 = calculate_cumulative_returns(s2_data)

    # Align indices (forward fill if needed)
    combined = pd.DataFrame({'s1': cum_s1, 's2': cum_s2})
    combined = combined.fillna(method='ffill').dropna()

    if len(combined) == 0:
        return None

    # Calculate spread of cumulative returns
    spread = combined['s1'] - combined['s2']
    mu_spread = spread.mean()
    sigma_spread = spread.std()

    return {
        'stock1': s1,
        'stock2': s2,
        'permco1': pair_row['permco_1'],
        'permco2': pair_row['permco_2'],
        'comnam1': pair_row['comnam_1'],
        'comnam2': pair_row['comnam_2'],
        'ssd': pair_row['ssd'],
        'mu_spread': mu_spread,
        'sigma_spread': sigma_spread,
        'threshold_upper': mu_spread + 2 * sigma_spread,
        'threshold_lower': mu_spread - 2 * sigma_spread
    }

# TRADING SIMULATION WITH DETAILED TRACKING
def simulate_trading(trading_data, pair_stats):
    """
    Simulate trading strategy for one pair using returns
    Track divergence/convergence dates and PnL series
    """
    if pair_stats is None:
        return None

    s1, s2 = pair_stats['stock1'], pair_stats['stock2']
    mu = pair_stats['mu_spread']
    upper = pair_stats['threshold_upper']
    lower = pair_stats['threshold_lower']

    s1_data = trading_data[trading_data['permno'] == s1].sort_values('date').set_index('date')['ret']
    s2_data = trading_data[trading_data['permno'] == s2].sort_values('date').set_index('date')['ret']

    if len(s1_data) == 0 or len(s2_data) == 0:
        return None

    # Calculate cumulative returns
    cum_s1 = calculate_cumulative_returns(s1_data)
    cum_s2 = calculate_cumulative_returns(s2_data)

    # Align indices
    combined = pd.DataFrame({'s1': cum_s1, 's2': cum_s2})
    combined = combined.fillna(method='ffill').dropna()
    combined.index = pd.to_datetime(combined.index)

    if len(combined) == 0:
        return None

    spread = combined['s1'] - combined['s2']
    
    position = 0
    entry_date = None
    diverge_date = None
    entry_spread = None
    trades = []
    pnl_series = []
    
    for i in range(len(spread)):
        current_date = spread.index[i]
        current_spread = spread.iloc[i]

        # ENTRY LOGIC
        if position == 0:
            if current_spread > upper:
                position = -1
                diverge_date = current_date
                entry_date = current_date
                entry_spread = current_spread
                
            elif current_spread < lower:
                position = 1
                diverge_date = current_date
                entry_date = current_date
                entry_spread = current_spread

        # EXIT LOGIC
        else:
            if i > 0:
                prev_spread = spread.iloc[i-1]
                crossed_mean = (prev_spread > mu and current_spread <= mu) or (prev_spread < mu and current_spread >= mu) or (current_date == spread.index[-1])

                if crossed_mean:
                    exit_spread = current_spread
                    converge_date = current_date
                    pnl = position * (exit_spread - entry_spread)

                    trades.append({
                        'diverge_date': diverge_date,
                        'converge_date': converge_date,
                        'entry_spread': entry_spread,
                        'exit_spread': exit_spread,
                        'position': 'Long S1/Short S2' if position == 1 else 'Short S1/Long S2',
                        'pnl': pnl,
                        'days_held': (converge_date - diverge_date).days
                    })

                    pnl_series.append(pnl)
                    position = 0
                    entry_date = None
                    diverge_date = None
                    entry_spread = None

    return {
        'pair': f"{pair_stats['comnam1']}-{pair_stats['comnam2']}",
        'permco1': pair_stats['permco1'],
        'permco2': pair_stats['permco2'], 
        'stock1': s1,
        'stock2': s2,
        'num_trades': len(trades),
        'trades': trades,
        'total_pnl': sum([t['pnl'] for t in trades]),
        'avg_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0,
        'pnl_series': np.array(pnl_series),
        'spread_series': spread
    }

# TRADING SIMULATION WITH DETAILED TRACKING AND 1-DAY DELAY
def simulate_trading_delay(trading_data, pair_stats, entry_delay_days=1):
    """
    Simulate trading strategy for one pair using returns
    Track divergence/convergence dates and PnL series
    
    Args:
        trading_data: DataFrame with trading period data
        pair_stats: Dictionary with pair statistics
        entry_delay_days: Number of days to delay entry after signal (default: 1)
    
    Returns:
        Dictionary with trade results
    """
    if pair_stats is None:
        return None

    s1, s2 = pair_stats['stock1'], pair_stats['stock2']
    mu = pair_stats['mu_spread']
    upper = pair_stats['threshold_upper']
    lower = pair_stats['threshold_lower']

    s1_data = trading_data[trading_data['permno'] == s1].sort_values('date').set_index('date')['ret']
    s2_data = trading_data[trading_data['permno'] == s2].sort_values('date').set_index('date')['ret']

    if len(s1_data) == 0 or len(s2_data) == 0:
        return None

    # Calculate cumulative returns
    cum_s1 = calculate_cumulative_returns(s1_data)
    cum_s2 = calculate_cumulative_returns(s2_data)

    # Align indices
    combined = pd.DataFrame({'s1': cum_s1, 's2': cum_s2})
    combined = combined.ffill().dropna()

    if len(combined) == 0:
        return None

    spread = combined['s1'] - combined['s2']
    
    position = 0
    entry_date = None
    diverge_date = None
    signal_date = None  # NEW: Track when signal occurred
    entry_spread = None
    signal_spread = None  # NEW: Track spread value at signal
    trades = []
    pnl_series = []
    
    for i in range(len(spread)):
        current_date = spread.index[i]
        current_spread = spread.iloc[i]

        # ENTRY LOGIC WITH DELAY
        if position == 0:
            # Check if we have a pending signal to execute
            if signal_date is not None:
                days_since_signal = (current_date - signal_date).days
                
                if days_since_signal >= entry_delay_days:
                    # Execute the delayed entry
                    if signal_spread > upper:
                        position = -1
                    elif signal_spread < lower:
                        position = 1
                    
                    diverge_date = signal_date  # Original signal date
                    entry_date = current_date   # Actual entry date (delayed)
                    entry_spread = current_spread  # Entry price after delay
                    
                    print(f"⏱️ Delayed entry: Signal on {signal_date.date()}, Entry on {entry_date.date()}, "
                          f"Signal spread: {signal_spread:.4f}, Entry spread: {entry_spread:.4f}")
                    
                    # Reset signal tracking
                    signal_date = None
                    signal_spread = None
            
            # Check for new signals (only if no pending signal)
            if signal_date is None:
                if current_spread > upper:
                    signal_date = current_date
                    signal_spread = current_spread
                    print(f"📊 Signal detected on {signal_date.date()}: Spread {signal_spread:.4f} > Upper {upper:.4f}")
                    
                elif current_spread < lower:
                    signal_date = current_date
                    signal_spread = current_spread
                    print(f"📊 Signal detected on {signal_date.date()}: Spread {signal_spread:.4f} < Lower {lower:.4f}")

        # EXIT LOGIC (unchanged - exits immediately on convergence)
        else:
            if i > 0:
                prev_spread = spread.iloc[i-1]
                crossed_mean = (prev_spread > mu and current_spread <= mu) or \
                               (prev_spread < mu and current_spread >= mu)

                if crossed_mean:
                    exit_spread = current_spread
                    converge_date = current_date
                    pnl = -position * (exit_spread - entry_spread)

                    trades.append({
                        'signal_date': diverge_date,  # When signal occurred
                        'diverge_date': diverge_date,  # Same as signal_date for compatibility
                        'entry_date': entry_date,      # When we actually entered (delayed)
                        'converge_date': converge_date,
                        'signal_spread': signal_spread if signal_spread else entry_spread,  # Spread at signal
                        'entry_spread': entry_spread,  # Spread at actual entry
                        'exit_spread': exit_spread,
                        'position': 'Long S1/Short S2' if position == 1 else 'Short S1/Long S2',
                        'pnl': pnl,
                        'days_held': (converge_date - entry_date).days,
                        'total_days': (converge_date - diverge_date).days  # Including delay
                    })

                    pnl_series.append(pnl)
                    position = 0
                    entry_date = None
                    diverge_date = None
                    entry_spread = None

    # Force close at end of period if still open
    if position != 0:
        exit_spread = spread.iloc[-1]
        converge_date = spread.index[-1]
        pnl = -position * (exit_spread - entry_spread)

        trades.append({
            'signal_date': diverge_date,
            'diverge_date': diverge_date,
            'entry_date': entry_date,
            'converge_date': converge_date,
            'signal_spread': signal_spread if signal_spread else entry_spread,
            'entry_spread': entry_spread,
            'exit_spread': exit_spread,
            'position': 'Long S1/Short S2' if position == 1 else 'Short S1/Long S2',
            'pnl': pnl,
            'days_held': (converge_date - entry_date).days,
            'total_days': (converge_date - diverge_date).days
        })
        pnl_series.append(pnl)

    return {
        'pair': f"{pair_stats['comnam1']}-{pair_stats['comnam2']}",
        'permco1': pair_stats['permco1'],
        'permco2': pair_stats['permco2'], 
        'stock1': s1,
        'stock2': s2,
        'num_trades': len(trades),
        'trades': trades,
        'total_pnl': sum([t['pnl'] for t in trades]),
        'avg_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0,
        'pnl_series': np.array(pnl_series),
        'spread_series': spread
    }

def export_monthly_returns(results, output_file='monthly_returns.csv'):
    """
    Combines monthly returns (realized + unrealized) for all pairs into one CSV.
    """
    pair_monthly_returns = []  # store Series per pair
    
    for r in results:
        trades = r.get('trades', [])
        spread = r.get('spread_series')
        pair_name = r.get('pair', 'Unknown')
        
        if trades and isinstance(spread, pd.Series) and not spread.empty:
            trades_df = pd.DataFrame(trades)

            # --- Realized monthly PnL ---
            trades_df['month'] = trades_df['converge_date'].dt.to_period('M')
            realized_monthly = trades_df.groupby('month')['pnl'].sum().to_timestamp()

            # --- Full monthly timeline ---
            months = pd.period_range(spread.index.min().to_period('M'),
                                     spread.index.max().to_period('M'))
            monthly_returns = pd.Series(0.0, index=months.to_timestamp(how='end'))

            # Add realized PnL
            monthly_returns = monthly_returns.add(realized_monthly, fill_value=0)

            # --- Unrealized (mark-to-market) PnL for open trades ---
            for m in months:
                month_end = m.to_timestamp(how='end')
                open_trade = None

                for t in trades:
                    if t['diverge_date'] <= month_end and t['converge_date'] > month_end:
                        open_trade = t
                        break

                if open_trade:
                    if not isinstance(spread.index, pd.DatetimeIndex):
                        spread.index = pd.to_datetime(spread.index)

                    valid_spread = spread.loc[spread.index <= month_end]
                    if not valid_spread.empty:
                        month_spread = valid_spread.iloc[-1]
                    else:
                        month_spread = spread.iloc[0]

                    direction = -1 if open_trade['position'] == 'Short S1/Long S2' else 1
                    unrealized_pnl = -direction * (month_spread - open_trade['entry_spread'])

                    # Ensure index alignment
                    if month_end not in monthly_returns.index:
                        monthly_returns.loc[month_end] = 0.0
                    monthly_returns.loc[month_end] += unrealized_pnl

            # Name and store
            monthly_returns.name = pair_name
            pair_monthly_returns.append(monthly_returns)

    # --- Combine all pairs ---
    if pair_monthly_returns:
        all_monthly_returns = pd.concat(pair_monthly_returns, axis=1).fillna(0)

        # Compounded cumulative returns (not additive)
        all_cumulative_returns = (1 + all_monthly_returns).cumprod() - 1

        cumulative_output_file = output_file.replace('.csv', '_cumulative.csv')
        all_monthly_returns.to_csv(output_file)
        all_cumulative_returns.to_csv(cumulative_output_file)

        print(f"\n✅ Exported monthly returns to: {output_file}")
        print(f"✅ Exported cumulative returns to: {cumulative_output_file}")

        return all_monthly_returns, all_cumulative_returns

    else:
        print("\n⚠️ No valid monthly returns found. CSV not generated.")
        # Return empty frames so calling code still works safely
        return pd.DataFrame(), pd.DataFrame()
    
# CALCULATE SHARPE RATIO
def calculate_sharpe_ratio(pnl_series, risk_free_rate=0.04, periods_per_year=252):
    """
    Calculate Sharpe ratio from PnL series
    risk_free_rate: annual risk-free rate (default 4%)
    periods_per_year: trading periods per year (default 252 for daily)
    """
    if len(pnl_series) == 0:
        return np.nan
    
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = pnl_series - daily_rf
    
    if np.std(excess_returns) == 0:
        return np.nan
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    return sharpe

# MAIN EXECUTION
if __name__ == "__main__":
    # TODO: change username
    db = wrds.Connection(wrds_username='sohrac')
    # industries = ["Agriculture_Forestry_and_Fishing", "Construction", "Finance_Insurance_and_Real_estate", "Manufacturing", "Mining", "Public_administration", 
    #              "Retail_Trade", "Services", "Transportation_Communications_Electric_Gas_and_Sanitary_service", "Wholesale_Trade"]
    # TODO: change dates
    # date_ranges = [('1962-01-01', '1962-12-31')]
    date_ranges = [('1962-01-01', '1962-12-31'), ('1963-01-01', '1963-12-31'), ('1964-01-01', '1964-12-31'), ('1965-01-01', '1965-12-31'), ('1966-01-01', '1966-12-31'), 
                   ('2000-01-01', '2000-12-31'), ('2001-01-01', '2001-12-31'), ('2002-01-01', '2002-12-31'), ('2003-01-01', '2003-12-31'), ('2004-01-01', '2004-12-31'), 
                   ('2018-01-01', '2018-12-31'), ('2019-01-01', '2019-12-31'), ('2020-01-01', '2020-12-31'), ('2021-01-01', '2021-12-31'), ('2022-01-01', '2022-12-31'), ('2023-01-01', '2023-12-31')] 
    # for industry in industries:        
    for formation_start, formation_end in date_ranges:
        # ---LOAD THE CSV FILE ---
        # TODO: CHANGE FILEPATH if needed
        try:
            # pair_file = f'permno_data/industry_pairs/{industry}/{industry}_{formation_start}_{formation_end}.csv'
            pair_file = f'permno_data/total_matched_pairs/permno_matched_pairs_{formation_start}_{formation_end}.csv'
            # pair_file = f'permno_matched_pairs_{formation_start}_{formation_end}.csv'
            returns = pd.read_csv(pair_file)
        except FileNotFoundError:
            print(f"{pair_file} not found skipping")
            continue

        print("columns found:", returns.columns.tolist())

        # Convert to appropriate data types
        type_map = {
            'permno_1': 'int64',
            'permno_2': 'int64',
            'ssd': 'float64',
            'comnam_1': 'string',
            'comnam_2': 'string',
        }

        for col, dtype in type_map.items():
            if col in returns.columns:
                if dtype in ['int64', 'Int64', 'float64']:
                    returns[col] = pd.to_numeric(returns[col], errors='coerce')
                else:
                    returns[col] = returns[col].astype(dtype)

        print("\nPairs data loaded successfully!")
        print(returns.head())

        formation_data = fetch_crsp_data(db, formation_start, formation_end)
        print(f"\nCRSP data fetched for Formation Period: {formation_data.shape}")
        print(formation_data.head())
        print(formation_data.size)

        trading_start_date = date.fromisoformat(formation_start) + relativedelta(years=1)
        trading_end_date = date.fromisoformat(formation_end) + relativedelta(years=1)
        trading_start = trading_start_date.isoformat()
        trading_end = trading_end_date.isoformat()


        trading_data = fetch_crsp_data(db, trading_start, trading_end)
        print(f"\nCRSP data fetched for Trading Period: {trading_data.shape}")
        print(trading_data.head())
        print(trading_data.size)

        # Process all pairs
        results = []
        for idx, (_, pair_row) in enumerate(returns.iterrows()):
            print(f"\n--- Processing Pair {idx+1}: {pair_row['permno_1']} vs {pair_row['permno_2']} ---")

            pair_stats = calculate_spread_stats(formation_data, pair_row)
            if pair_stats:
                # TODO: run with the delay code too!! 
                result = simulate_trading(trading_data, pair_stats)
                if result and result['num_trades'] > 0:
                    sharpe = calculate_sharpe_ratio(result['pnl_series'])
                    result['sharpe_ratio'] = sharpe
                    results.append(result)
                    print(f"  Summary: {result['num_trades']} trades, Total PnL: {result['total_pnl']:.4f}, Sharpe: {sharpe:.4f}")

        tops = [5, 20, 100]
        for i in tops:
            # Sort by Sharpe ratio and get top 20
            print("\n" + "="*80)
            print("TOP PAIRS BY SHARPE RATIO (USING RETURNS-BASED METHODOLOGY)")
            print("="*80)
            
            if results:
                results_sorted = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
                top = results_sorted[:i]

                # Create detailed output
                top_data = []
                for rank, result in enumerate(top, 1):
                    for trade in result['trades']:
                        top_data.append({
                            'Rank': rank,
                            'Pair': result['pair'],
                            'Stock1_permco': result['permco1'],
                            'Stock2_permco': result['permco2'],
                            'Stock1_Permno': result['stock1'],
                            'Stock2_Permno': result['stock2'],
                            'Num_Trades': result['num_trades'],
                            'Total_PnL': result['total_pnl'],
                            'Avg_PnL': result['avg_pnl'],
                            'Sharpe_Ratio': result['sharpe_ratio'],
                            'Diverge_Date': trade['diverge_date'].date(),
                            'Converge_Date': trade['converge_date'].date(),
                            'Entry_Spread': trade['entry_spread'],
                            'Exit_Spread': trade['exit_spread'],
                            'Position_Type': trade['position'],
                            'Trade_PnL': trade['pnl'],
                            'Days_Held': trade['days_held']
                        })

                top_df = pd.DataFrame(top_data)
                
                # top_df.to_csv(f'delay_trade/trading_strategy/top_{i}/top_{i}_pairs_with_dates_{trading_start}_{trading_end}.csv', index=False)
                top_df.to_csv(f'permno_data/trading_strategy/top_{i}/top_{i}_pairs_with_dates_{trading_start}_{trading_end}.csv', index=False)
                # top_df.to_csv(f'permno_data/industry_pairs/{industry}/trading_strategy/top_{i}/top_{i}_pairs_with_dates_{trading_start}_{trading_end}.csv', index=False)
                # top_df.to_csv(f'delay_trade/industry_pairs/{industry}/trading_strategy/top_{i}/top_{i}_pairs_with_dates_{trading_start}_{trading_end}.csv', index=False)
                # top_df.to_csv(f'top_{i}_pairs_with_dates_{trading_start}_{trading_end}.csv', index=False)

                # Summary by pair
                summary_df = top_df.groupby('Pair').agg({
                    'Rank': 'first',
                    'Total_PnL': 'first',
                    'Avg_PnL': 'first',
                    'Sharpe_Ratio': 'first',
                    'Num_Trades': 'first'
                }).reset_index()
                
                print("\nSummary:")
                print(summary_df.to_string(index=False))
                # summary_df.to_csv(f'delay_trade/trading_strategy/top_{i}/top_{i}_pairs_summary_{trading_start}_{trading_end}.csv', index=False)
                summary_df.to_csv(f'permno_data/trading_strategy/top_{i}/top_{i}_pairs_summary_{trading_start}_{trading_end}.csv', index=False)
                # summary_df.to_csv(f'permno_data/industry_pairs/{industry}/trading_strategy/top_{i}/top_{i}_pairs_summary_{trading_start}_{trading_end}.csv', index=False)
                # summary_df.to_csv(f'delay_trade/industry_pairs/{industry}/trading_strategy/top_{i}/top_{i}_pairs_summary_{trading_start}_{trading_end}.csv', index=False)
                # summary_df.to_csv(f'top_{i}_pairs_summary_{trading_start}_{trading_end}.csv', index=False)
                
                print("\nGenerating monthly returns CSV for all pairs...")
                # export_monthly_returns(results, output_file=f'delay_trade/trading_strategy/top_{i}/top_{i}_monthly_returns_{trading_start}_{trading_end}.csv')
                export_monthly_returns(results, output_file=f'permno_data/trading_strategy/top_{i}/top_{i}_monthly_returns_{trading_start}_{trading_end}.csv')
                # export_monthly_returns(results, output_file=f'permno_data/industry_pairs/{industry}/trading_strategy/top_{i}/top_{i}_monthly_returns_{trading_start}_{trading_end}.csv')
                # export_monthly_returns(results, output_file=f'delay_trade/industry_pairs/{industry}/trading_strategy/top_{i}/top_{i}_monthly_returns_{trading_start}_{trading_end}.csv')
                # export_monthly_returns(results, output_file=f'top_{i}_monthly_returns_{trading_start}_{trading_end}.csv')
                print("✅ Monthly returns successfully exported!")

                # # Visualization: Top 5 pairs - Normalized price charts
                # print("\n" + "="*80)
                # print("GENERATING CHARTS FOR TOP 5 PAIRS")
                # print("="*80)
                
                # # # Get top 5 pairs (already sorted above)
                # top_5 = results_sorted[:5]
                
                # # Create figure with 5 subplots (one for each pair)
                # fig, axes = plt.subplots(2, 1, figsize=(14, 20))
                # fig.suptitle('Top 5 Pairs Trading Strategy - Normalized Cumulative Returns', 
                #              fontsize=16, fontweight='bold', y=0.995)
                
                # ax = axes[0]
                # # Get individual stock data for this pair
                # s1, s2 = result['stock1'], result['stock2']
                
                # # Fetch formation + trading period data for visualization
                # full_data = pd.concat([formation_data,trading_data])
                
                # s1_data = full_data[full_data['permno'] == s1].sort_values('date').set_index('date')['ret']
                # s2_data = full_data[full_data['permno'] == s2].sort_values('date').set_index('date')['ret']
                
                # # Calculate cumulative returns (normalized to start at 1.0)
                # cum_s1 = (1 + s1_data).cumprod()
                # cum_s2 = (1 + s2_data).cumprod()
                
                # # Align data
                # combined = pd.DataFrame({'stock1': cum_s1, 'stock2': cum_s2})
                # combined = combined.fillna(method='ffill').dropna()
                
                # # Plot both stocks
                # ax.plot(combined.index, combined['stock1'], 
                #         label=f'Stock 1 (PERMNO: {s1})', 
                #         linewidth=2, color='#1f77b4', alpha=0.8)
                # ax.plot(combined.index, combined['stock2'], 
                #         label=f'Stock 2 (PERMNO: {s2})', 
                #         linewidth=2, color='#ff7f0e', alpha=0.8)
                
                # # Mark trade entry and exit points
                # for trade in result['trades']:
                #     diverge_date = trade['diverge_date']
                #     converge_date = trade['converge_date']
                    
                #     # Entry point (divergence)
                #     ax.axvline(x=diverge_date, color='red', linestyle='--', 
                #                 alpha=0.3, linewidth=1)
                    
                #     # Exit point (convergence)
                #     ax.axvline(x=converge_date, color='green', linestyle='--', 
                #                 alpha=0.3, linewidth=1)
                
                # # Add formation period separator
                # formation_end = pd.Timestamp(formation_end)
                # ax.axvline(x=formation_end, color='purple', linestyle='-', 
                #             linewidth=2, alpha=0.5, label='Formation/Trading Split')
                
                # # Formatting
                # ax.set_title(f'Rank #1: {result["pair"]}\n'
                #             f'Sharpe: {result["sharpe_ratio"]:.3f} | '
                #             f'Trades: {result["num_trades"]} | '
                #             f'Total PnL: {result["total_pnl"]:.4f}',
                #             fontsize=11, fontweight='bold', pad=10)
                
                # ax.set_xlabel('Date', fontsize=10)
                # ax.set_ylabel('Normalized Price (Starting at 1.0)', fontsize=10)
                # ax.legend(loc='best', fontsize=9)
                # ax.grid(True, alpha=0.3, linestyle=':')
                
                # # Rotate x-axis labels
                # ax.tick_params(axis='x', rotation=45)
                
                # # Format y-axis
                # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

                # # PLot the spread in axes[1]
                # ax2 = axes[1]
                # spread = result['spread_series']
                # ax2.plot(spread.index, spread, 
                #          label='Spread (Cumulative Return Difference)', 
                #          color='darkcyan', linewidth=2, alpha=0.8)
                # ax2.axhline(y=pair_stats['mu_spread'], color='black', linestyle='-', linewidth=1, label='Mean Spread')
                # ax2.axhline(y=pair_stats['threshold_upper'], color='red', linestyle='--', linewidth=1, label='Upper Threshold')
                # ax2.axhline(y=pair_stats['threshold_lower'], color='green', linestyle='--', linewidth=1, label='Lower Threshold')
                # ax2.set_title('Spread with Entry/Exit Points', fontsize=11, fontweight='bold', pad=10)
                # ax2.set_xlabel('Date', fontsize=10)
                # ax2.set_ylabel('Spread Value', fontsize=10)
                # ax2.legend(loc='best', fontsize=9)
                # ax2.grid(True, alpha=0.3, linestyle=':')
                # ax2.tick_params(axis='x', rotation=45)

            
                # # Adjust layout
                # plt.tight_layout()
                
                # # Save figure
                # # plt.savefig('top_5_pairs_normalized_prices.png', dpi=300, bbox_inches='tight')
                # # print("\n✓ Chart saved as 'top_5_pairs_normalized_prices.png'")
                
                # # Show plot
                # plt.show()
                
                # print("\nVisualization complete!")
                # print("="*80)
                
                print("\n\nDetailed trades:")
                print(top_df.to_string(index=False))
                
                