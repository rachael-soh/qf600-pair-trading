import wrds
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import csv

warnings.filterwarnings('ignore')

# ---LOAD THE CSV FILE ---
pair_file = 'Finance_and_Insurance_2024-01-01_2024-12-31.csv'
returns = pd.read_csv(pair_file)

print("columns found:", returns.columns.tolist())

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

print("\nPairs data loaded successfully!")
print(returns.head())

# LOAD PRICE DATA from WRDS
def fetch_crsp_data(db: wrds.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily stock data from CRSP using returns (RET) instead of prices"""
    query = f"""
        SELECT a.permco, a.date, a.ret, a.prc
        FROM crsp.dsf a
        WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
          AND a.ret IS NOT NULL
          AND ABS(a.ret) < 1  
    """
    return db.raw_sql(query, date_cols=['date'])

# Connect to WRDS (will be initialized in main)
db = None

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
    s1, s2 = pair_row['permco_1'], pair_row['permco_2']

    s1_data = formation_data[formation_data['permco'] == s1].sort_values('date').set_index('date')['ret']
    s2_data = formation_data[formation_data['permco'] == s2].sort_values('date').set_index('date')['ret']

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

    s1_data = trading_data[trading_data['permco'] == s1].sort_values('date').set_index('date')['ret']
    s2_data = trading_data[trading_data['permco'] == s2].sort_values('date').set_index('date')['ret']

    if len(s1_data) == 0 or len(s2_data) == 0:
        return None

    # Calculate cumulative returns
    cum_s1 = calculate_cumulative_returns(s1_data)
    cum_s2 = calculate_cumulative_returns(s2_data)

    # Align indices
    combined = pd.DataFrame({'s1': cum_s1, 's2': cum_s2})
    combined = combined.fillna(method='ffill').dropna()

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
                crossed_mean = (prev_spread > mu and current_spread <= mu) or \
                               (prev_spread < mu and current_spread >= mu)

                if crossed_mean:
                    exit_spread = current_spread
                    converge_date = current_date
                    pnl = -position * (exit_spread - entry_spread)

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

    # Force close at end of period if still open
    if position != 0:
        exit_spread = spread.iloc[-1]
        converge_date = spread.index[-1]
        pnl = -position * (exit_spread - entry_spread)

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
    
    #getting my monthly returns for 1 pair
    trades_df=pd.DataFrame(trades)

    if not trades_df.empty:
        trades_df['month']=trades_df['converge-date'].dt.to.period('M')
        monthly_returns = trades_df.groupby('month')['pnl'].sum().to.timestamp() #this returns a monthly pnl series
    else:
        monthly_returns = pd.Series(dtype=float) #this returns empty series



    return {
        'pair': f"{pair_stats['comnam1']}-{pair_stats['comnam2']}",
        'stock1': s1,
        'stock2': s2,
        'num_trades': len(trades),
        'trades': trades,
        'total_pnl': sum([t['pnl'] for t in trades]),
        'avg_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0,
        'pnl_series': np.array(pnl_series),
        'spread_series': spread,
        'monthly_returns': monthly_returns  
    }
        

#GETTING MONTHLY RETURNS CSV
def export_monthly_returns(results):
    import pandas as pd

    # Check if there are results
    if not results:
        print("No results to export.")
        return

    # Combine all pairs' monthly returns into one DataFrame
    all_monthly_returns = pd.concat(
        [r['monthly_returns'] for r in results if 'monthly_returns' in r],
        axis=1
    )

    # Label the columns by pair name
    all_monthly_returns.columns = [r['pair'] for r in results if 'monthly_returns' in r]

    # (Optional) Add a portfolio-level total column
    all_monthly_returns['Portfolio_Total'] = all_monthly_returns.sum(axis=1)

    # Export to CSV
    all_monthly_returns.to_csv('monthly_returns.csv', index_label='Month')

    print("✅ Monthly returns have been exported to 'monthly_returns.csv'")
  
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
    # Connect to WRDS
    db = wrds.Connection(wrds_username='andygse')
    
    # Pull CRSP data for formation period
    data = fetch_crsp_data(db, '2024-01-01', '2024-06-30')
    print(f"\nCRSP data fetched: {data.shape}")
    print(data.head())

    # Get trading data (next 6 months after formation)
    trading_data = fetch_crsp_data(db, '2024-07-01', '2024-12-31')

    # Process all pairs
    results = []
    for idx, (_, pair_row) in enumerate(returns.iterrows()):
        print(f"\n--- Processing Pair {idx+1}: {pair_row['permco_1']} vs {pair_row['permco_2']} ---")

        pair_stats = calculate_spread_stats(data, pair_row)
        if pair_stats:
            result = simulate_trading(trading_data, pair_stats)
            if result and result['num_trades'] > 0:
                sharpe = calculate_sharpe_ratio(result['pnl_series'])
                result['sharpe_ratio'] = sharpe
                results.append(result)
                print(f"  Summary: {result['num_trades']} trades, Total PnL: {result['total_pnl']:.4f}, Sharpe: {sharpe:.4f}")

    # Sort by Sharpe ratio and get top 20
    print("\n" + "="*80)
    print("TOP 20 PAIRS BY SHARPE RATIO (USING RETURNS-BASED METHODOLOGY)")
    print("="*80)
    
    if results:
        # 👇 Export all monthly returns before filtering top 20
        export_monthly_returns(results)

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

        top_20_df = pd.DataFrame(top_20_data)
        top_20_df.to_csv('top_20_pairs_with_dates.csv', index=False)
        

        # Summary by pair
        summary_df = top_20_df.groupby('Pair').agg({
            'Rank': 'first',
            'Total_PnL': 'first',
            'Avg_PnL': 'first',
            'Sharpe_Ratio': 'first',
            'Num_Trades': 'first'
        }).reset_index()
        
        print("\nSummary:")
        print(summary_df.to_string(index=False))
        summary_df.to_csv('top_20_pairs_summary.csv', index=False)
        
        print("\n\nDetailed trades:")
        print(top_20_df.to_string(index=False))
        
    #     # Visualization: Top 5 pairs - Normalized price charts
    #     print("\n" + "="*80)
    # print("GENERATING CHARTS FOR TOP 5 PAIRS")
    # print("="*80)
    
    # # Get top 5 pairs (already sorted above)
    # top_5 = results_sorted[:5]
    
    # # Create figure with 5 subplots (one for each pair)
    # fig, axes = plt.subplots(5, 1, figsize=(14, 20))
    # fig.suptitle('Top 5 Pairs Trading Strategy - Normalized Cumulative Returns', 
    #              fontsize=16, fontweight='bold', y=0.995)
    
    # for idx, result in enumerate(top_5):
    #     ax = axes[idx]
        
    #     # Get individual stock data for this pair
    #     s1, s2 = result['stock1'], result['stock2']
        
    #     # Fetch formation + trading period data for visualization
    #     full_data = fetch_crsp_data(db, '2024-01-01', '2024-12-31')
        
    #     s1_data = full_data[full_data['permco'] == s1].sort_values('date').set_index('date')['ret']
    #     s2_data = full_data[full_data['permco'] == s2].sort_values('date').set_index('date')['ret']
        
    #     # Calculate cumulative returns (normalized to start at 1.0)
    #     cum_s1 = (1 + s1_data).cumprod()
    #     cum_s2 = (1 + s2_data).cumprod()
        
    #     # Align data
    #     combined = pd.DataFrame({'stock1': cum_s1, 'stock2': cum_s2})
    #     combined = combined.fillna(method='ffill').dropna()
        
    #     # Plot both stocks
    #     ax.plot(combined.index, combined['stock1'], 
    #             label=f'Stock 1 (PERMCO: {s1})', 
    #             linewidth=2, color='#1f77b4', alpha=0.8)
    #     ax.plot(combined.index, combined['stock2'], 
    #             label=f'Stock 2 (PERMCO: {s2})', 
    #             linewidth=2, color='#ff7f0e', alpha=0.8)
        
    #     # Mark trade entry and exit points
    #     for trade in result['trades']:
    #         diverge_date = trade['diverge_date']
    #         converge_date = trade['converge_date']
            
    #         # Entry point (divergence)
    #         ax.axvline(x=diverge_date, color='red', linestyle='--', 
    #                   alpha=0.3, linewidth=1)
            
    #         # Exit point (convergence)
    #         ax.axvline(x=converge_date, color='green', linestyle='--', 
    #                   alpha=0.3, linewidth=1)
        
    #     # Add formation period separator
    #     formation_end = pd.Timestamp('2024-06-30')
    #     ax.axvline(x=formation_end, color='purple', linestyle='-', 
    #               linewidth=2, alpha=0.5, label='Formation/Trading Split')
        
    #     # Formatting
    #     ax.set_title(f'Rank #{idx+1}: {result["pair"]}\n'
    #                 f'Sharpe: {result["sharpe_ratio"]:.3f} | '
    #                 f'Trades: {result["num_trades"]} | '
    #                 f'Total PnL: {result["total_pnl"]:.4f}',
    #                 fontsize=11, fontweight='bold', pad=10)
        
    #     ax.set_xlabel('Date', fontsize=10)
    #     ax.set_ylabel('Normalized Price (Starting at 1.0)', fontsize=10)
    #     ax.legend(loc='best', fontsize=9)
    #     ax.grid(True, alpha=0.3, linestyle=':')
        
    #     # Rotate x-axis labels
    #     ax.tick_params(axis='x', rotation=45)
        
    #     # Format y-axis
    #     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    
    # # Adjust layout
    # plt.tight_layout()
    
    # # Save figure
    # plt.savefig('top_5_pairs_normalized_prices.png', dpi=300, bbox_inches='tight')
    # print("\n✓ Chart saved as 'top_5_pairs_normalized_prices.png'")
    
    # # Show plot
    # plt.show()
    
    # print("\nVisualization complete!")
    # print("="*80)
    
        
else:
        print("No valid trades generated")

if db:
    db.close()
    print("\nWRDS connection closed.")
