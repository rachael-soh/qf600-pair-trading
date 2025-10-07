import pandas as pd
import numpy as np
from itertools import combinations
import yfinance as yf
import matplotlib.pyplot as plt


def normalize_prices(prices: pd.Series):
    """Normalize a price series by first value."""
    return prices / prices.iloc[0]

def sum_squared_deviation(x: pd.Series, y: pd.Series):
    """Compute SSD between two normalized price series."""
    return np.sum((x - y) ** 2)

def form_pairs_from_yf(tickers, start, end, top_n):
    """
    Download Yahoo Finance data and find top N pairs with smallest SSD.
    
    Parameters
    ----------
    tickers : list of str
        Stock tickers (e.g. ["AAPL", "MSFT", "GOOG"])
    start, end : str
        Date range for price data
    top_n : int
        Number of pairs to return
    
    Returns
    -------
    list of tuples (ticker1, ticker2, SSD)
    """
    # Used Adj. Close prices because it accounts for dividends and splits. 
    # Alternatively, we can consider calculating the cumulative total return ourselves.
    price_data = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    
    # Drop any stocks with missing data
    price_data = price_data.dropna(axis=1)

    # Normalize prices
    norm_prices = price_data.apply(normalize_prices)
    
    pairs = []
    for s1, s2 in combinations(norm_prices.columns, 2):
        # TODO: can replace sum_squared_deviation with any distance metric (R^2, cointegration, etc.)
        ssd = sum_squared_deviation(norm_prices[s1], norm_prices[s2])
        print(f"Comparing {s1} and {s2}: SSD = {ssd}")
        pairs.append((s1, s2, ssd))

    # Sort by SSD (lower = more similar)
    pairs_sorted = sorted(pairs, key=lambda x: x[2])

    return pairs_sorted[:top_n], norm_prices

def plot_pair(pair, norm_prices):
    s1, s2, ssd = pair
    
    plt.figure(figsize=(10,6))
    plt.plot(norm_prices[s1], label=s1, linewidth=2)
    plt.plot(norm_prices[s2], label=s2, linewidth=2)
    plt.title(f"Top Pair: {s1} & {s2} (SSD={ssd:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True)
    plt.show()
    
# ---------------- Main function ---------------- #
if __name__ == "__main__":
    # TODO: Replace with the actual tickers
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA"]
    n = 5
    top_pairs, norm_prices = form_pairs_from_yf(tickers, start="2024-01-01", end="2025-01-01", top_n=n)
    print(f"Top {n} Pairs:")
    for p in top_pairs:
        print(p)
        # uncomment line below to see the plots.
        # plot_pair(p, norm_prices)
















# ---------- Portfolio Returns Computation ------------ #


raw_returns = {
#To clean CRSP data to dynamically pull out returns of pairs based on trading strategy... 
    "P1": [0.01, -0.05,0.03],
    "P2": [0.005,0.01,-0.08]
}

returns = pd.DataFrame(raw_returns)

#risk free rate
Rf_rate= 0.001

#weights (inital holdings of each pair)
holdings = np.full(returns.shape[1], 100) #populate initial holdings dynamically

#storage for tracking
portfolio_returns = []
portfolio_value = [holdings.sum()]

#need to code a loop to continuously mulitply returns with weights across time 
for t in range(len(returns)):
    total_value = holdings.sum()
    daily_returns= np.dot(holdings, returns.iloc[t])/ total_value #reiterates through every time period, later must update holding values at the end
    portfolio_returns.append(daily_returns)
    holdings = holdings * (1 + returns.iloc[t]) #updating holding weights
    portfolio_value.append(holdings.sum())

#convert to pandas series
portfolio_returns = pd.Series(portfolio_returns)
portfolio_value = pd.Series(portfolio_value[1:])


print("Daily Portfolio Returns :")
print(portfolio_returns)
print("Portfolio Value :")
print(portfolio_value)