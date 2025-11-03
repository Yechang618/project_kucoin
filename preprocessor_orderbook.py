import pandas as pd
import numpy as np
import warnings, csv
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def fast_2d_reg(x, y):
  b1 = np.cov(x, y)[0][1]/(1e-10 + np.var(x))
  b0 = np.mean(y) - b1*np.mean(x)
  # print(np.cov(x, y), np.var(x), np.mean(x))
  return b1, b0

def get_cumulative_amount_at_price(df, price_col_prefix, amount_col_prefix, target_price):
    """Finds the cumulative amount at or near the target price."""
    # Find the index of the price level closest to the target price
    price_columns = [f'{price_col_prefix}[{i}].price' for i in range(25)]
    prices = df[price_columns].values

    # Calculate the absolute difference between target price and all price levels
    # Reshape target_price to match the dimensions of prices for broadcasting
    price_diffs = np.abs(prices - target_price.values.reshape(-1, 1))

    # Find the index of the minimum difference for each row
    closest_price_indices = np.argmin(price_diffs, axis=1)

    # print(target_price, closest_price_indices)

    # Get the cumulative amount at the closest price index
    cumulative_amounts = [df.iloc[i][f'{amount_col_prefix}[{idx}].camt'] for i, idx in enumerate(closest_price_indices)]

    return cumulative_amounts


def generate_df_one_day_one_symbol(symbol, date = None, resample_rate = '20s'):
    # Load trade data
    address = 'datasets'
    fde_name = f'binance-futures_derivative_ticker_{date}_{symbol}.csv.gz'
    spot_name = f'binance_book_snapshot_25_{date}_{symbol}.csv.gz'
    swap_name = f'binance-futures_book_snapshot_25_{date}_{symbol}.csv.gz'
    spot_trade_name = f'binance_trades_{date}_{symbol}.csv.gz'
    swap_trade_name = f'binance-futures_trades_{date}_{symbol}.csv.gz'
    print(f"Processing symbol: {symbol}, resample_rate: {resample_rate}")
    # Add sub-folder
    spot_trade_name = os.path.join('trades', spot_trade_name)
    swap_trade_name = os.path.join('ftrades', swap_trade_name)
    spot_name = os.path.join('book', spot_name)
    swap_name = os.path.join('fbook', swap_name)
    fde_name = os.path.join('ftick', fde_name)
    # Add dataset
    spot_name = os.path.join(address, spot_name)
    swap_name = os.path.join(address, swap_name)
    fde_name = os.path.join(address, fde_name)
    spot_trade_name = os.path.join(address, spot_trade_name)
    swap_trade_name = os.path.join(address, swap_trade_name)
    df_spot = pd.read_csv(spot_name, compression='gzip')
    df_swap = pd.read_csv(swap_name, compression='gzip')
    df_fde = pd.read_csv(fde_name, compression='gzip')
    df_spot_trd = pd.read_csv(spot_trade_name, compression='gzip')
    df_swap_trd = pd.read_csv(swap_trade_name, compression='gzip')
    df_spot['local_timestamp'] = pd.to_datetime(df_spot['local_timestamp'], unit='us', origin='unix')
    df_swap['local_timestamp'] = pd.to_datetime(df_swap['local_timestamp'], unit='us', origin='unix')
    df_fde['local_timestamp'] = pd.to_datetime(df_fde['local_timestamp'], unit='us', origin='unix')
    df_fde['funding_timestamp'] = pd.to_datetime(df_fde['funding_timestamp'], unit='us', origin='unix')
    df_spot_trd['local_timestamp'] = pd.to_datetime(df_spot_trd['local_timestamp'], unit='us', origin='unix')
    df_swap_trd['local_timestamp'] = pd.to_datetime(df_swap_trd['local_timestamp'], unit='us', origin='unix')

    # print(df_swap.info())
    df_spot = df_spot.set_index(pd.DatetimeIndex(df_spot['local_timestamp']))
    df_swap = df_swap.set_index(pd.DatetimeIndex(df_swap['local_timestamp']))
    df_spot_trd = df_spot_trd.set_index(pd.DatetimeIndex(df_spot_trd['local_timestamp']))
    df_swap_trd = df_swap_trd.set_index(pd.DatetimeIndex(df_swap_trd['local_timestamp']))
    df_fde = df_fde.set_index(pd.DatetimeIndex(df_fde['local_timestamp']))

    # Remove duplicate indices
    df_spot = df_spot[~df_spot.index.duplicated(keep='first')]
    df_swap = df_swap[~df_swap.index.duplicated(keep='first')]
    df_fde = df_fde[~df_fde.index.duplicated(keep='first')]

    # Resample and fill NaN values
    df_spot = df_spot.resample(resample_rate).ffill().bfill()
    df_swap = df_swap.resample(resample_rate).ffill().bfill()
    df_fde = df_fde.resample(resample_rate).ffill().bfill()

    # Process trade data
    df_spot_buy = df_spot_trd[df_spot_trd['side'] == 'buy'].copy()
    df_spot_sell = df_spot_trd[df_spot_trd['side'] == 'sell'].copy()
    df_swap_buy = df_swap_trd[df_swap_trd['side'] == 'buy'].copy()
    df_swap_sell = df_swap_trd[df_swap_trd['side'] == 'sell'].copy()
    
    # Calculate total amount for each trade
    df_spot_buy = df_spot_trd[df_spot_trd['side'] == 'buy'].copy()
    df_spot_sell = df_spot_trd[df_spot_trd['side'] == 'sell'].copy()
    df_swap_buy = df_swap_trd[df_swap_trd['side'] == 'buy'].copy()
    df_swap_sell = df_swap_trd[df_swap_trd['side'] == 'sell'].copy()

    # Resample df_spot_trd to the same time index as df_spot and aggregate buy and sell trades
    df_spot_buy_agg = df_spot_buy.resample(resample_rate).agg({
        'amount': 'sum',
        'price': ['mean', 'std']
    })
    df_spot_sell_agg = df_spot_sell.resample(resample_rate).agg({
        'amount': 'sum',
        'price': ['mean', 'std']
    })
    df_swap_buy_agg = df_swap_buy.resample(resample_rate).agg({
        'amount': 'sum',
        'price': ['mean', 'std']
    })
    df_swap_sell_agg = df_swap_sell.resample(resample_rate).agg({
        'amount': 'sum',
        'price': ['mean', 'std']
    })

    # Rename columns for single level
    df_spot_buy_agg.columns = ['spot_buy_amount_sum', 'spot_buy_price_mean', 'spot_buy_price_std']
    df_spot_sell_agg.columns = ['spot_sell_amount_sum', 'spot_sell_price_mean', 'spot_sell_price_std']
    df_swap_buy_agg.columns = ['swap_buy_amount_sum', 'swap_buy_price_mean', 'swap_buy_price_std']
    df_swap_sell_agg.columns = ['swap_sell_amount_sum', 'swap_sell_price_mean', 'swap_sell_price_std']

    # Merge the aggregated dataframes
    df_trades = df_spot_buy_agg.merge(df_spot_sell_agg, left_index=True, right_index=True, how='inner')
    df_trades = df_trades.merge(df_swap_buy_agg, left_index=True, right_index=True, how='inner')
    df_trades = df_trades.merge(df_swap_sell_agg, left_index=True, right_index=True, how='inner')

    # Calculate cumulative amounts for df_spot
    for i in range(25):
        df_spot[f'spot.bids[{i}].camt'] = df_spot[f'bids[{i}].amount'].cumsum()
        df_spot[f'spot.asks[{i}].camt'] = df_spot[f'asks[{i}].amount'].cumsum()

    # Calculate cumulative amounts for df_swap
    for i in range(25):
        df_swap[f'swap.bids[{i}].camt'] = df_swap[f'bids[{i}].amount'].cumsum()
        df_swap[f'swap.asks[{i}].camt'] = df_swap[f'asks[{i}].amount'].cumsum()

    # Convert cumulative amount columns to numeric, coercing errors
    for i in range(25):
        df_spot[f'spot.bids[{i}].camt'] = pd.to_numeric(df_spot[f'spot.bids[{i}].camt'], errors='coerce')
        df_spot[f'spot.asks[{i}].camt'] = pd.to_numeric(df_spot[f'spot.asks[{i}].camt'], errors='coerce')
        df_swap[f'swap.bids[{i}].camt'] = pd.to_numeric(df_swap[f'swap.bids[{i}].camt'], errors='coerce')
        df_swap[f'swap.asks[{i}].camt'] = pd.to_numeric(df_swap[f'swap.asks[{i}].camt'], errors='coerce')

    # Take logarithm of cumulative amounts
    for i in range(25):
        df_spot[f'spot.bids[{i}].camt'] = np.log(df_spot[f'spot.bids[{i}].camt'])
        df_spot[f'spot.asks[{i}].camt'] = np.log(df_spot[f'spot.asks[{i}].camt'])
        df_swap[f'swap.bids[{i}].camt'] = np.log(df_swap[f'swap.bids[{i}].camt'])
        df_swap[f'swap.asks[{i}].camt'] = np.log(df_swap[f'swap.asks[{i}].camt'])
    # Perform linear regression for spot bids
    df_spot['spot.bids_slope'] = None
    df_spot['spot.bids_b'] = None
    for i in range(len(df_spot)):
        x = np.arange(25)
        y = df_spot.iloc[i][[f'spot.bids[{j}].camt' for j in range(25)]].values
        # Ensure y contains only numeric types before checking for NaNs
        y = pd.to_numeric(y, errors='coerce')
        slope, intercept = fast_2d_reg(x, y)
        df_spot.loc[df_spot.index[i], 'spot.bids_slope'] = slope
        df_spot.loc[df_spot.index[i], 'spot.bids_b'] = intercept

    # Perform linear regression for spot asks
    df_spot['spot.asks_slope'] = None
    df_spot['spot.asks_b'] = None
    for i in range(len(df_spot)):
        x = np.arange(25)
        y = df_spot.iloc[i][[f'spot.asks[{j}].camt' for j in range(25)]].values
        # Ensure y contains only numeric types before checking for NaNs
        y = pd.to_numeric(y, errors='coerce')
        slope, intercept = fast_2d_reg(x, y)
        df_spot.loc[df_spot.index[i], 'spot.asks_slope'] = slope
        df_spot.loc[df_spot.index[i], 'spot.asks_b'] = intercept

    # Perform linear regression for swap bids
    df_swap['swap.bids_slope'] = None
    df_swap['swap.bids_b'] = None
    for i in range(len(df_swap)):
        x = np.arange(25)
        y = df_swap.iloc[i][[f'swap.bids[{j}].camt' for j in range(25)]].values
        # Ensure y contains only numeric types before checking for NaNs
        y = pd.to_numeric(y, errors='coerce')
        slope, intercept = fast_2d_reg(x, y)
        df_swap.loc[df_swap.index[i], 'swap.bids_slope'] = slope
        df_swap.loc[df_swap.index[i], 'swap.bids_b'] = intercept

    # Perform linear regression for swap asks
    df_swap['swap.asks_slope'] = None
    df_swap['swap.asks_b'] = None
    for i in range(len(df_swap)):
        x = np.arange(25)
        y = df_swap.iloc[i][[f'swap.asks[{j}].camt' for j in range(25)]].values
        # Ensure y contains only numeric types before checking for NaNs
        y = pd.to_numeric(y, errors='coerce')
        slope, intercept = fast_2d_reg(x, y)
        df_swap.loc[df_swap.index[i], 'swap.asks_slope'] = slope
        df_swap.loc[df_swap.index[i], 'swap.asks_b'] = intercept
    
    # Calculate swap_asks_amt001 for df_swap
    df_swap['swap_asks_amt001'] = get_cumulative_amount_at_price(
        df_swap,
        'asks',
        'swap.asks',
        df_swap['asks[0].price'] + np.log(1.01)
    )

    # Calculate swap_bids_amt001 for df_swap
    df_swap['swap_bids_amt001'] = get_cumulative_amount_at_price(
        df_swap,
        'bids',
        'swap.bids',
        df_swap['bids[0].price'] + np.log(0.99)
    )

    # Calculate spot_asks_amt001 for df_spot
    df_spot['spot_asks_amt001'] = get_cumulative_amount_at_price(
        df_spot,
        'asks',
        'spot.asks',
        df_spot['asks[0].price'] + np.log(1.01)
    )

    # Calculate spot_bids_amt001 for df_spot
    df_spot['spot_bids_amt001'] = get_cumulative_amount_at_price(
        df_spot,
        'bids',
        'spot.bids',
        df_spot['bids[0].price'] + np.log(0.99)
    )
    # Rename columns in df_swap
    df_swap = df_swap.rename(columns={
        'bids[0].price': 'swap.bids[0].price',
        'asks[0].price': 'swap.asks[0].price',
        'bids[0].amount': 'swap.bids[0].amount',
        'asks[0].amount': 'swap.asks[0].amount'
    })

    # Rename columns in df_spot
    df_spot = df_spot.rename(columns={
        'bids[0].price': 'spot.bids[0].price',
        'asks[0].price': 'spot.asks[0].price',
        'bids[0].amount': 'spot.bids[0].amount',
        'asks[0].amount': 'spot.asks[0].amount'
    })

    # Select specified columns from df_spot
    df_spot_selected = df_spot[[
        'spot.asks[0].price', 'spot.asks[0].amount',
        'spot.bids[0].price', 'spot.bids[0].amount',
        'spot.bids_slope', 'spot.bids_b',
        'spot.asks_slope', 'spot.asks_b',
        'spot_bids_amt001', 'spot_asks_amt001',
        'spot.bids[24].camt', 'spot.asks[24].camt'
    ]]

    # Select specified columns from df_swap
    df_swap_selected = df_swap[[
        'swap.asks[0].price', 'swap.asks[0].amount',
        'swap.bids[0].price', 'swap.bids[0].amount',
        'swap.bids_slope', 'swap.bids_b',
        'swap.asks_slope', 'swap.asks_b',
        'swap_bids_amt001', 'swap_asks_amt001',
        'swap.bids[24].camt', 'swap.asks[24].camt'
    ]]

    # Select specified columns from df_fde
    df_fde_selected = df_fde[['funding_rate']]

    # Merge the selected dataframes on their index
    df_merged = df_spot_selected.merge(df_swap_selected, left_index=True, right_index=True, how='inner')
    df_merged = df_merged.merge(df_fde_selected, left_index=True, right_index=True, how='inner')
    df_combined = df_merged.merge(df_trades, left_index=True, right_index=True, how='inner')

    return df_combined

def generate_df_one_symbol(symbol, start_date = None, end_date = None, resample_rate = '1s'):
    df = None
    date_range = pd.date_range(start=start_date, end=end_date)
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        print(f"Processing date: {date_str}")
        df_day = generate_df_one_day_one_symbol(symbol, date_str, resample_rate = resample_rate)
        if df is None:
            df = df_day.copy()
        else:
            df = pd.concat([df, df_day])
        del df_day
        print(df.info())
    return df

if __name__ == "__main__":
    # symbols=["BTCUSDT","SOLUSDT","PNUTUSDT","TURBOUSDT","APTUSDT","AIXBTUSDT",
    #          "TAOUSDT","KAITOUSDT","OMUSDT","XRPUSDT","FETUSDT","UNIUSDT",
    #          "COMPUSDT","THEUSDT","AVAXUSDT","LTCUSDT","ETCUSDT","FORMUSDT",
    #          "TONUSDT","HFTUSDT","DOTUSDT","CHESSUSDT",'ETHUSDT', 'BNBUSDT', 
    #          'TRXUSDT', 'DOGEUSDT', 'ADAUSDT', 'LINKUSDT', 'XLMUSDT', 'BCHUSDT', 
    #          'HBARUSDT','ZECUSDT', 'AAVEUSDT', 'ENAUSDT', 'NEARUSDT','ONDOUSDT'],   
    # symbols = ["SOLUSDT", "COMPUSDT","DOTUSDT","CHESSUSDT"]
    # symbols = ["BTCUSDT","SOLUSDT","PNUTUSDT","TURBOUSDT","APTUSDT","AIXBTUSDT",
    #          "TAOUSDT","KAITOUSDT","OMUSDT","XRPUSDT","FETUSDT","UNIUSDT",
    #          "COMPUSDT","THEUSDT","AVAXUSDT","LTCUSDT","ETCUSDT","FORMUSDT",
    #          "TONUSDT","HFTUSDT","DOTUSDT","CHESSUSDT",'ETHUSDT', 'BNBUSDT', 
    #          'TRXUSDT', 'DOGEUSDT', 'ADAUSDT', 'LINKUSDT', 'XLMUSDT', 'BCHUSDT', 
    #          'HBARUSDT','ZECUSDT', 'AAVEUSDT', 'ENAUSDT', 'NEARUSDT','ONDOUSDT']   
    symbols = ['BNBUSDT', 'ETHUSDT', 'ADAUSDT', 'LINKUSDT','TRXUSDT',"DOTUSDT"]
    # start_date, end_date = "2025-09-21", "2025-09-27"
    start_date, end_date = "2025-09-01", "2025-09-30"
    resample_rate = '1s'
    for symbol in symbols:
        # print(symbol)
        df = generate_df_one_symbol(symbol, start_date=start_date, end_date=end_date,  resample_rate = resample_rate)
        # df = generate_df(symbols, dataset_name)
        print(df.head(5))
        print(df.info())
        df.to_csv(f"processed_obook_{symbol}_{start_date}_{end_date}_{resample_rate}" , index=True, header=True, quoting=csv.QUOTE_NONNUMERIC)
        print(f"DataFrame saved to processed_obook_{symbol}_{start_date}_{end_date}_{resample_rate}.csv  successfully.")
