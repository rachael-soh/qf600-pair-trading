import pandas as pd
import matplotlib.pyplot as plt

def plot_pair(pair_number, filename):
    """ Plot the price and spread of a given pair from the matched pairs file.
    Args:
        pair_number (int): The pair number to plot (1-indexed).
        filename (str): The path to the matched pairs CSV file.
    """
    pairs = pd.read_csv('data/matched_pairs/' + filename)
    
    # get row number of index pair_number
    pair_info = pairs.iloc[pair_number-1]
    if pair_info.empty:
        print(f"No data found for pair number {pair_number}")
        return

    # use rthe permco to get the returns
    permco_1 = pair_info['permco_1']
    permco_2 = pair_info['permco_2']
    comnam_1 = pair_info['comnam_1']
    comnam_2 = pair_info['comnam_2']

    returns_file = 'data/returns/' + filename
    returns = pd.read_csv(
        returns_file,
        index_col=0,              # Use first column (dates) as index
        parse_dates=True,         # Parse index as dates
        dtype=float               # Try to load columns as float (permco IDs)
        )
    returns.columns = returns.columns.astype(float)

    returns1 = returns[permco_1]
    returns2 = returns[permco_2]
    returns_both = pd.merge(returns1, on='date', right=returns2)

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot price1 and price2 on primary y-axis
    ax1.plot(returns_both[permco_1], label=f'{comnam_1} ({permco_1}) Price', color='blue')
    ax1.plot(returns_both[permco_2], label=f'{comnam_2} ({permco_2}) Price', color='orange')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.tick_params(axis='y')
    plt.xticks(rotation=45)
    
    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc='upper left')
    plt.title(f'Price and Spread for {comnam_1} ({permco_1}) and {comnam_2} ({permco_2})')
    plt.grid()
    plt.tight_layout()
    plt.show()

# just give the filename. 1 gives top pair
plot_pair(1, 'Accommodation_and_Food_Services_2024-01-01_2024-12-31.csv')