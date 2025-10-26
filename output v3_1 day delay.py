import wrds
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import warnings
import csv

warnings.filterwarnings('ignore')

# LOAD PRICE DATA from WRDS
def fetch_crsp_data(db: wrds.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily stock data from CRSP"""
    query = f"""
        SELECT a.permco, a.date, a.ret, a.prc
        FROM crsp.dsf a
        WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
          AND a.ret IS NOT NULL
          AND ABS(a.ret) < 1  
    """
    df = db.raw_sql(query, date_cols=['date'])
    df.to_csv(f'data/crsp_data/crsp_data_{start_date}_{end_date}.csv', index=False)
    return df

# Connect to WRDS (will be initialized in main)
db = None

# CALCULATE NORMALIZED PRICE INDEX FROM RETURNS
def calculate_price_index(return_series, starting_value=100):
    """
    Convert returns to normalized price index
    Starting at 100 for easy interpretation
    """
    return starting_value * (1 + return_series).cumprod()

# CALCULATE SPREAD STATISTICS USING LOG RETURNS
def calculate_spread_stats(formation_data, pair_row):
    """
    Calculate mean and std dev of spread during formation period using log returns
    This is more theoretically sound for mean-reversion strategies
    
    Spread = log(Price_S1) - log(Price_S2) = log(Price_S1 / Price_S2)
    """
    s1, s2 = pair_row['permco_1'], pair_row['permco_2']

    s1_data = formation_data[formation_data['permco'] == s1].sort_values('date').set_index('date')['ret']
    s2_data = formation_data[formation_data['permco'] == s2].sort_values('date').set_index('date')['ret']

    if len(s1_data) == 0 or len(s2_data) == 0:
        print(f"   No data found for pair {s1}-{s2}")
        return None

    # Convert to log returns
    log_ret_s1 = np.log(1 + s1_data)
    log_ret_s2 = np.log(1 + s2_data)
    
    # Cumulative log returns (equivalent to log of price index)
    cum_log_s1 = log_ret_s1.cumsum()
    cum_log_s2 = log_ret_s2.cumsum()

    # Align indices (forward fill if needed)
    combined = pd.DataFrame({'l1': cum_log_s1, 'l2': cum_log_s2})
    combined = combined.ffill().dropna()

    if len(combined) < 20:  # Need minimum data points for statistics
        print(f"   Insufficient data points ({len(combined)}) for pair {s1}-{s2}")
        return None

    # Calculate spread of log prices (log spread = log ratio)
    spread = combined['l1'] - combined['l2']
    mu_spread = spread.mean()
    sigma_spread = spread.std()
    
    if sigma_spread == 0:
        print(f"   Zero volatility for pair {s1}-{s2}")
        return None

    return {
        'stock1': s1,
        'stock2': s2,
        'comnam1': pair_row['comnam_1'],
        'comnam2': pair_row['comnam_2'],
        'ssd': pair_row['ssd'],
        'mu_spread': mu_spread,
        'sigma_spread': sigma_spread,
        'threshold_upper': mu_spread + 2 * sigma_spread,
        'threshold_lower': mu_spread - 2 * sigma_spread
    }

# DETECT DELISTING
def detect_delisting(trading_data, permco, current_date, lookback_days=5):
    """
    Detect if a stock has been delisted by checking for missing data
    
    Args:
        trading_data: Full trading period DataFrame
        permco: Stock identifier
        current_date: Current trading date
        lookback_days: Days to look back for missing data (default: 5)
    
    Returns:
        True if stock appears delisted, False otherwise
    """
    stock_data = trading_data[trading_data['permco'] == permco].set_index('date')
    
    # Get all dates in the trading period up to current date
    all_dates = trading_data['date'].unique()
    all_dates = pd.to_datetime(all_dates)
    all_dates = all_dates[all_dates <= current_date]
    
    if len(all_dates) < lookback_days:
        return False
    
    # Check last N trading days
    recent_dates = all_dates[-lookback_days:]
    
    # Count how many days the stock has data
    stock_dates = stock_data.index
    present_count = sum(date in stock_dates for date in recent_dates)
    
    # If stock missing for most recent days, consider delisted
    if present_count == 0:
        return True
    
    return False

# TRADING SIMULATION WITH REALISTIC PNL CALCULATION AND DELISTING HANDLING
def simulate_trading(trading_data, pair_stats, entry_delay_days=1, capital_per_leg=10000):
    """
    Simulate trading strategy for one pair using log returns spread
    Calculate realistic PnL based on actual dollar positions
    Handles delisting by forcing position closure
    
    Args:
        trading_data: DataFrame with trading period data
        pair_stats: Dictionary with pair statistics from formation period
        entry_delay_days: Number of days to delay entry after signal (default: 1)
        capital_per_leg: Dollar amount to invest in each leg (default: $10,000)
    
    Returns:
        Dictionary with trade results and PnL
    """
    if pair_stats is None:
        return None

    s1, s2 = pair_stats['stock1'], pair_stats['stock2']
    mu = pair_stats['mu_spread']
    upper = pair_stats['threshold_upper']
    lower = pair_stats['threshold_lower']

    s1_data = trading_data[trading_data['permco'] == s1].sort_values('date').set_index('date')['ret']
    s2_data = trading_data[trading_data['permco'] == s2].sort_values('date').set_index('date')['ret']

    if len(s1_data) == 0 or len(s2_data) == 0:
        return None
    
    # Store original trading_data for delisting detection
    full_trading_data = trading_data.copy()

    # Calculate log returns
    log_ret_s1 = np.log(1 + s1_data)
    log_ret_s2 = np.log(1 + s2_data)
    
    # Cumulative log returns
    cum_log_s1 = log_ret_s1.cumsum()
    cum_log_s2 = log_ret_s2.cumsum()
    
    # Also calculate price indices for PnL calculation
    price_s1 = calculate_price_index(s1_data)
    price_s2 = calculate_price_index(s2_data)

    # Align all data
    combined = pd.DataFrame({
        'l1': cum_log_s1, 
        'l2': cum_log_s2,
        'p1': price_s1,
        'p2': price_s2
    })
    combined = combined.ffill().dropna()

    if len(combined) == 0:
        return None

    # Calculate log spread
    spread = combined['l1'] - combined['l2']
    
    # Trading state variables
    position = 0
    signal_date = None
    signal_spread = None
    entry_date = None
    entry_spread = None
    diverge_date = None
    
    # Position tracking
    shares_s1 = 0
    shares_s2 = 0
    entry_price_s1 = 0
    entry_price_s2 = 0
    
    trades = []
    pnl_series = []
    
    for i in range(len(spread)):
        current_date = spread.index[i]
        current_spread = spread.iloc[i]
        current_price_s1 = combined.loc[current_date, 'p1']
        current_price_s2 = combined.loc[current_date, 'p2']

        # CHECK FOR DELISTING WHILE IN POSITION
        if position != 0:
            s1_delisted = detect_delisting(full_trading_data, s1, current_date)
            s2_delisted = detect_delisting(full_trading_data, s2, current_date)
            
            if s1_delisted or s2_delisted:
                # Force close position due to delisting
                exit_spread = current_spread
                exit_price_s1 = current_price_s1
                exit_price_s2 = current_price_s2
                converge_date = current_date
                
                # Calculate PnL at delisting
                pnl_s1 = shares_s1 * (exit_price_s1 - entry_price_s1)
                pnl_s2 = shares_s2 * (exit_price_s2 - entry_price_s2)
                pnl = pnl_s1 + pnl_s2
                
                total_capital = 2 * capital_per_leg
                return_pct = (pnl / total_capital) * 100
                
                delisted_stock = []
                if s1_delisted:
                    delisted_stock.append('S1')
                if s2_delisted:
                    delisted_stock.append('S2')
                delisted_str = ' & '.join(delisted_stock)

                trades.append({
                    'signal_date': diverge_date,
                    'diverge_date': diverge_date,
                    'entry_date': entry_date,
                    'converge_date': converge_date,
                    'signal_spread': signal_spread if signal_spread else entry_spread,
                    'entry_spread': entry_spread,
                    'exit_spread': exit_spread,
                    'position': 'Long S1/Short S2' if position == 1 else 'Short S1/Long S2',
                    'shares_s1': shares_s1,
                    'shares_s2': shares_s2,
                    'entry_price_s1': entry_price_s1,
                    'entry_price_s2': entry_price_s2,
                    'exit_price_s1': exit_price_s1,
                    'exit_price_s2': exit_price_s2,
                    'pnl': pnl,
                    'pnl_s1': pnl_s1,
                    'pnl_s2': pnl_s2,
                    'return_pct': return_pct,
                    'days_held': (converge_date - entry_date).days,
                    'total_days': (converge_date - diverge_date).days,
                    'exit_reason': f'Delisting ({delisted_str})'
                })

                pnl_series.append(pnl)
                
                print(f"     DELISTING DETECTED ({delisted_str}): Forced close on {converge_date.date()}")
                print(f"       PnL=${pnl:.2f} ({return_pct:.2f}%), S1 PnL=${pnl_s1:.2f}, S2 PnL=${pnl_s2:.2f}")
                
                # Reset position
                position = 0
                shares_s1 = 0
                shares_s2 = 0
                entry_date = None
                diverge_date = None
                entry_spread = None
                
                # Continue to next iteration
                continue

        # ENTRY LOGIC WITH DELAY
        if position == 0:
            # Check if we have a pending signal to execute
            if signal_date is not None:
                days_since_signal = (current_date - signal_date).days
                
                if days_since_signal >= entry_delay_days:
                    # Execute the delayed entry
                    if signal_spread > upper:
                        # Spread too high: Short S1, Long S2 (bet on convergence)
                        position = -1
                        shares_s1 = -capital_per_leg / current_price_s1  # Negative = short
                        shares_s2 = capital_per_leg / current_price_s2   # Positive = long
                        
                    elif signal_spread < lower:
                        # Spread too low: Long S1, Short S2
                        position = 1
                        shares_s1 = capital_per_leg / current_price_s1   # Positive = long
                        shares_s2 = -capital_per_leg / current_price_s2  # Negative = short
                    
                    diverge_date = signal_date
                    entry_date = current_date
                    entry_spread = current_spread
                    entry_price_s1 = current_price_s1
                    entry_price_s2 = current_price_s2
                    
                    print(f"     Delayed entry: Signal {signal_date.date()}, Entry {entry_date.date()}, "
                          f"Signal spread: {signal_spread:.4f}, Entry spread: {entry_spread:.4f}")
                    print(f"       Position: {shares_s1:.2f} shares S1 @ ${entry_price_s1:.2f}, "
                          f"{shares_s2:.2f} shares S2 @ ${entry_price_s2:.2f}")
                    
                    # Reset signal tracking
                    signal_date = None
                    signal_spread = None
            
            # Check for new signals (only if no pending signal)
            if signal_date is None:
                if current_spread > upper:
                    signal_date = current_date
                    signal_spread = current_spread
                    print(f"     Signal detected on {signal_date.date()}: "
                          f"Spread {signal_spread:.4f} > Upper {upper:.4f}")
                    
                elif current_spread < lower:
                    signal_date = current_date
                    signal_spread = current_spread
                    print(f"     Signal detected on {signal_date.date()}: "
                          f"Spread {signal_spread:.4f} < Lower {lower:.4f}")

        # EXIT LOGIC
        else:
            if i > 0:
                prev_spread = spread.iloc[i-1]
                crossed_mean = (prev_spread > mu and current_spread <= mu) or \
                               (prev_spread < mu and current_spread >= mu)

                if crossed_mean:
                    exit_spread = current_spread
                    exit_price_s1 = current_price_s1
                    exit_price_s2 = current_price_s2
                    converge_date = current_date
                    
                    # Calculate actual PnL from each leg
                    pnl_s1 = shares_s1 * (exit_price_s1 - entry_price_s1)
                    pnl_s2 = shares_s2 * (exit_price_s2 - entry_price_s2)
                    pnl = pnl_s1 + pnl_s2
                    
                    # Calculate return on capital
                    total_capital = 2 * capital_per_leg
                    return_pct = (pnl / total_capital) * 100

                    trades.append({
                        'signal_date': diverge_date,
                        'diverge_date': diverge_date,
                        'entry_date': entry_date,
                        'converge_date': converge_date,
                        'signal_spread': signal_spread if signal_spread else entry_spread,
                        'entry_spread': entry_spread,
                        'exit_spread': exit_spread,
                        'position': 'Long S1/Short S2' if position == 1 else 'Short S1/Long S2',
                        'shares_s1': shares_s1,
                        'shares_s2': shares_s2,
                        'entry_price_s1': entry_price_s1,
                        'entry_price_s2': entry_price_s2,
                        'exit_price_s1': exit_price_s1,
                        'exit_price_s2': exit_price_s2,
                        'pnl': pnl,
                        'pnl_s1': pnl_s1,
                        'pnl_s2': pnl_s2,
                        'return_pct': return_pct,
                        'days_held': (converge_date - entry_date).days,
                        'total_days': (converge_date - diverge_date).days
                    })

                    pnl_series.append(pnl)
                    
                    print(f"    ✅ Exit: PnL=${pnl:.2f} ({return_pct:.2f}%), "
                          f"S1 PnL=${pnl_s1:.2f}, S2 PnL=${pnl_s2:.2f}")
                    
                    # Reset position
                    position = 0
                    shares_s1 = 0
                    shares_s2 = 0
                    entry_date = None
                    diverge_date = None
                    entry_spread = None

    # Force close at end of period if still open
    if position != 0:
        exit_spread = spread.iloc[-1]
        exit_price_s1 = combined.iloc[-1]['p1']
        exit_price_s2 = combined.iloc[-1]['p2']
        converge_date = spread.index[-1]
        
        pnl_s1 = shares_s1 * (exit_price_s1 - entry_price_s1)
        pnl_s2 = shares_s2 * (exit_price_s2 - entry_price_s2)
        pnl = pnl_s1 + pnl_s2
        
        total_capital = 2 * capital_per_leg
        return_pct = (pnl / total_capital) * 100

        trades.append({
            'signal_date': diverge_date,
            'diverge_date': diverge_date,
            'entry_date': entry_date,
            'converge_date': converge_date,
            'signal_spread': signal_spread if signal_spread else entry_spread,
            'entry_spread': entry_spread,
            'exit_spread': exit_spread,
            'position': 'Long S1/Short S2' if position == 1 else 'Short S1/Long S2',
            'shares_s1': shares_s1,
            'shares_s2': shares_s2,
            'entry_price_s1': entry_price_s1,
            'entry_price_s2': entry_price_s2,
            'exit_price_s1': exit_price_s1,
            'exit_price_s2': exit_price_s2,
            'pnl': pnl,
            'pnl_s1': pnl_s1,
            'pnl_s2': pnl_s2,
            'return_pct': return_pct,
            'days_held': (converge_date - entry_date).days,
            'total_days': (converge_date - diverge_date).days,
            'exit_reason': 'Period End'
        })
        pnl_series.append(pnl)
        
        print(f"    🔒 Forced close: PnL=${pnl:.2f} ({return_pct:.2f}%)")

    return {
        'pair': f"{pair_stats['comnam1']}-{pair_stats['comnam2']}",
        'stock1': s1,
        'stock2': s2,
        'num_trades': len(trades),
        'trades': trades,
        'total_pnl': sum([t['pnl'] for t in trades]),
        'avg_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0,
        'total_return_pct': sum([t['return_pct'] for t in trades]),
        'avg_return_pct': np.mean([t['return_pct'] for t in trades]) if trades else 0,
        'pnl_series': np.array(pnl_series),
        'spread_series': spread,
        'capital_per_leg': capital_per_leg
    }

# CALCULATE SHARPE RATIO
def calculate_sharpe_ratio(pnl_series, risk_free_rate=0.04, periods_per_year=252):
    """
    Calculate Sharpe ratio from PnL series
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
    # Connect to WRDS
    db = wrds.Connection(wrds_username='sohrac')

    # Define date ranges for backtesting
    date_ranges = [
        ('1962-01-01', '1962-12-31'), ('1963-01-01', '1963-12-31'),
        ('2018-01-01', '2018-12-31'), ('2019-01-01', '2019-12-31'),
        ('2020-01-01', '2020-12-31'), ('2021-01-01', '2021-12-31'),
        ('2022-01-01', '2022-12-31'), ('2023-01-01', '2023-12-31')
    ]
    
    for (formation_start, formation_end) in date_ranges:
        print("\n" + "="*80)
        print(f"PROCESSING FORMATION PERIOD: {formation_start} to {formation_end}")
        print("="*80)
        
        # Load pair file
        pair_file = f'to_upload/total_matched_pairs/matched_pairs_{formation_start}_{formation_end}.csv'
        try:
            returns = pd.read_csv(pair_file)
        except FileNotFoundError:
            print(f" Pair file not found: {pair_file}")
            continue
            
        print(f"Loaded {len(returns)} pairs from CSV")
        print("Columns found:", returns.columns.tolist())

        # Convert to appropriate data types
        type_map = {
            'permco_1': 'int64',
            'permco_2': 'int64',
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

        # Fetch or load formation data
        formation_data_file = f'data/crsp_data/crsp_data_{formation_start}_{formation_end}.csv'
        try:
            formation_data = pd.read_csv(formation_data_file)
            formation_data['date'] = pd.to_datetime(formation_data['date'])
            print(f"✓ Loaded formation data from local file: {formation_data.shape}")
        except FileNotFoundError:
            print("Fetching formation data from WRDS...")
            formation_data = fetch_crsp_data(db, formation_start, formation_end)
            print(f"✓ Fetched formation data: {formation_data.shape}")

        # Calculate trading period dates
        trading_start_date = date.fromisoformat(formation_start) + relativedelta(years=1)
        trading_end_date = date.fromisoformat(formation_end) + relativedelta(years=1)
        trading_start = trading_start_date.isoformat()
        trading_end = trading_end_date.isoformat()

        # Fetch or load trading data
        trading_data_file = f'data/crsp_data/crsp_data_{trading_start}_{trading_end}.csv'
        try:
            trading_data = pd.read_csv(trading_data_file)
            trading_data['date'] = pd.to_datetime(trading_data['date'])
            print(f"✓ Loaded trading data from local file: {trading_data.shape}")
        except FileNotFoundError:
            print("Fetching trading data from WRDS...")
            trading_data = fetch_crsp_data(db, trading_start, trading_end)
            print(f"✓ Fetched trading data: {trading_data.shape}")

        # Process all pairs
        results = []
        for idx, (_, pair_row) in enumerate(returns.iterrows()):
            print(f"\n--- Processing Pair {idx+1}/{len(returns)}: "
                  f"{pair_row['permco_1']} vs {pair_row['permco_2']} ---")

            pair_stats = calculate_spread_stats(formation_data, pair_row)
            if pair_stats:
                result = simulate_trading(trading_data, pair_stats, 
                                        entry_delay_days=1, 
                                        capital_per_leg=10000)
                if result and result['num_trades'] > 0:
                    sharpe = calculate_sharpe_ratio(result['pnl_series'])
                    result['sharpe_ratio'] = sharpe
                    results.append(result)
                    print(f"  Summary: {result['num_trades']} trades, "
                          f"Total PnL: ${result['total_pnl']:.2f} ({result['total_return_pct']:.2f}%), "
                          f"Sharpe: {sharpe:.4f}")

        # Sort by Sharpe ratio and get top 20
        print("\n" + "="*80)
        print("TOP 20 PAIRS BY SHARPE RATIO (LOG RETURNS METHODOLOGY)")
        print("="*80)
        
        if results:
            results_sorted = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
            top_20 = results_sorted[:20]

            # Create detailed output
            top_20_data = []
            for rank, result in enumerate(top_20, 1):
                for trade in result['trades']:
                    top_20_data.append({
                        'Rank': rank,
                        'Pair': result['pair'],
                        'Stock1_Permco': result['stock1'],
                        'Stock2_Permco': result['stock2'],
                        'Num_Trades': result['num_trades'],
                        'Total_PnL': result['total_pnl'],
                        'Total_Return_Pct': result['total_return_pct'],
                        'Avg_PnL': result['avg_pnl'],
                        'Avg_Return_Pct': result['avg_return_pct'],
                        'Sharpe_Ratio': result['sharpe_ratio'],
                        'Capital_Per_Leg': result['capital_per_leg'],
                        'Signal_Date': trade['signal_date'].date(),
                        'Entry_Date': trade['entry_date'].date(),
                        'Converge_Date': trade['converge_date'].date(),
                        'Signal_Spread': trade.get('signal_spread', trade['entry_spread']),
                        'Entry_Spread': trade['entry_spread'],
                        'Exit_Spread': trade['exit_spread'],
                        'Position_Type': trade['position'],
                        'Shares_S1': trade['shares_s1'],
                        'Shares_S2': trade['shares_s2'],
                        'Entry_Price_S1': trade['entry_price_s1'],
                        'Entry_Price_S2': trade['entry_price_s2'],
                        'Exit_Price_S1': trade['exit_price_s1'],
                        'Exit_Price_S2': trade['exit_price_s2'],
                        'Trade_PnL': trade['pnl'],
                        'PnL_S1': trade['pnl_s1'],
                        'PnL_S2': trade['pnl_s2'],
                        'Trade_Return_Pct': trade['return_pct'],
                        'Days_Held': trade['days_held'],
                        'Total_Days': trade.get('total_days', trade['days_held']),
                        'Exit_Reason': trade.get('exit_reason', 'Unknown')
                    })

            top_20_df = pd.DataFrame(top_20_data)
            output_file = f'to_upload/trading_strategy/top_20_pairs_detailed_{trading_start}_{trading_end}.csv'
            top_20_df.to_csv(output_file, index=False)
            print(f"\n✓ Saved detailed results to: {output_file}")

            # Summary by pair
            summary_df = top_20_df.groupby('Pair').agg({
                'Rank': 'first',
                'Total_PnL': 'first',
                'Total_Return_Pct': 'first',
                'Avg_PnL': 'first',
                'Avg_Return_Pct': 'first',
                'Sharpe_Ratio': 'first',
                'Num_Trades': 'first',
                'Capital_Per_Leg': 'first'
            }).reset_index()
            
            print("\n" + "="*80)
            print("SUMMARY OF TOP 20 PAIRS")
            print("="*80)
            print(summary_df.to_string(index=False))
            
            summary_file = f'to_upload/trading_strategy/top_20_pairs_summary_{trading_start}_{trading_end}.csv'
            summary_df.to_csv(summary_file, index=False)
            print(f"\n✓ Saved summary to: {summary_file}")
            
        else:
            print(" No valid trades generated for this period")

    # Close database connection
    if db:
        db.close()
        print("\n✓ WRDS connection closed.")
        
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE!")
    print("="*80)