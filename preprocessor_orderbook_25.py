import pandas as pd
import numpy as np
import warnings, csv
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def generate_df_one_day_one_symbol(symbol, date = None, dataset_name = 'book_snapshot_5_', resample_rate = '1s'):
    df = None
    address = 'datasets'
    thisName = dataset_name + '_' + date
    spot_name1 = 'binance_' + thisName +'_'
    swap_name1 = 'binance-futures_' + thisName +'_'  
    spot_name2 = '.csv.gz'
    swap_name2 = '.csv.gz'
    # for symbol in symbols:
    print(f"Processing symbol: {symbol}, resample_rate: {resample_rate}")
    print(symbol)
    spot_name = spot_name1 + symbol + spot_name2
    spot_name = os.path.join(address, spot_name)
    swap_name = swap_name1 + symbol + swap_name2
    swap_name = os.path.join(address, swap_name)
    df_spot = pd.read_csv(spot_name, compression='gzip')
    df_swap = pd.read_csv(swap_name, compression='gzip')
    df_spot['local_timestamp'] = pd.to_datetime(df_spot['local_timestamp'], unit='us', origin='unix')
    df_swap['local_timestamp'] = pd.to_datetime(df_swap['local_timestamp'], unit='us', origin='unix')
    # print(df_swap.info())
    df_spot = df_spot.set_index(pd.DatetimeIndex(df_spot['local_timestamp']))
    df_swap = df_swap.set_index(pd.DatetimeIndex(df_swap['local_timestamp']))
    # df_swap_qt = pd.DataFrame({'swap_bids': df_swap['bids[0].price'], 'swap_asks': df_swap['asks[0].price']}).drop_duplicates()
    # df_spot_qt = pd.DataFrame({'spot_bids': df_spot['bids[0].price'], 'spot_asks': df_spot['asks[0].price']}).drop_duplicates()
    # print(df_spot.info())
    df_spot.rename(columns={'bids[0].price': 'spot.bids[0].price', 'asks[0].price': 'spot.asks[0].price', 'bids[0].amount': 'spot.bids[0].amount', 'asks[0].amount': 'spot.asks[0].amount'}, inplace=True)
    df_spot.rename(columns={'bids[1].amount': 'spot.bids[1].amount', 'asks[1].amount': 'spot.asks[1].amount'}, inplace=True)
    df_spot.rename(columns={'bids[2].amount': 'spot.bids[2].amount', 'asks[2].amount': 'spot.asks[2].amount'}, inplace=True)
    df_spot.rename(columns={'bids[1].price': 'spot.bids[1].price', 'asks[1].price': 'spot.asks[1].price'}, inplace=True)
    df_spot.rename(columns={'bids[2].price': 'spot.bids[2].price', 'asks[2].price': 'spot.asks[2].price'}, inplace=True)
    df_swap.rename(columns={'bids[0].price': 'swap.bids[0].price', 'asks[0].price': 'swap.asks[0].price'}, inplace=True)
    df_swap.rename(columns={'bids[1].price': 'swap.bids[1].price', 'asks[1].price': 'swap.asks[1].price'}, inplace=True)
    df_swap.rename(columns={'bids[2].price': 'swap.bids[2].price', 'asks[2].price': 'swap.asks[2].price'}, inplace=True)
    df_swap.rename(columns={'bids[0].amount': 'swap.bids[0].amount', 'asks[0].amount': 'swap.asks[0].amount'}, inplace=True)
    df_swap.rename(columns={'bids[1].amount': 'swap.bids[1].amount', 'asks[1].amount': 'swap.asks[1].amount'}, inplace=True)
    df_swap.rename(columns={'bids[2].amount': 'swap.bids[2].amount', 'asks[2].amount': 'swap.asks[2].amount'}, inplace=True)
    # print(df_spot.info())
    df_swap_qt = df_swap[['swap.bids[0].price', 'swap.asks[0].price', 
                            'swap.bids[1].price', 'swap.asks[1].price', 
                            'swap.bids[2].price', 'swap.asks[2].price',
                            'swap.bids[0].amount', 'swap.asks[0].amount', 
                            'swap.bids[1].amount', 'swap.asks[1].amount', 
                            'swap.bids[2].amount', 'swap.asks[2].amount']]#.apply(np.log)
    df_spot_qt = df_spot[['spot.bids[0].price', 'spot.asks[0].price', 
                            'spot.bids[1].price', 'spot.asks[1].price', 
                            'spot.bids[2].price', 'spot.asks[2].price',
                            'spot.bids[0].amount', 'spot.asks[0].amount',
                            'spot.bids[1].amount', 'spot.asks[1].amount',
                            'spot.bids[2].amount', 'spot.asks[2].amount']]#.apply(np.log)
    # Resample df_spot_qt and df_swap_qt to a higher frequency (e.g., 1ms) to allow for finer matching
    df_spot_qt = df_spot_qt[~df_spot_qt.index.duplicated(keep='first')]
    df_swap_qt = df_swap_qt[~df_swap_qt.index.duplicated(keep='first')]
    df_spot_qt_resampled = df_spot_qt.resample(resample_rate).ffill()
    df_swap_qt_resampled = df_swap_qt.resample(resample_rate).ffill()  
    # Merge the resampled dataframes based on the closest time index within a tolerance
    # df_merged = pd.merge_asof(
    #     df_swap_qt_resampled['swap_bids'],
    #     df_spot_qt_resampled['spot_asks'],
    #     left_index=True,
    #     right_index=True,
    #     direction='nearest',
    #     tolerance=pd.Timedelta('1s')
    # )         
    df_merged=pd.merge(df_spot_qt_resampled,df_swap_qt_resampled, how='inner', left_index=True, right_index=True)
        # Calculate the basis
    # print(df_basis.info())
    if df is None:
        df = df_merged.copy()
    else:
        df = df.join(df_merged)
    del df_merged
    return df

def generate_df_one_symbol(symbol, start_date = None, end_date = None, dataset_name = None, resample_rate = '1s'):
    df = None
    date_range = pd.date_range(start=start_date, end=end_date)
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        print(f"Processing date: {date_str}")
        df_day = generate_df_one_day_one_symbol(symbol, date_str, dataset_name, resample_rate = resample_rate)
        if df is None:
            df = df_day.copy()
        else:
            df = pd.concat([df, df_day])
        del df_day
        print(df.info())
    return df

if __name__ == "__main__":
    # symbols = ["AVAXUSDT"]
    # symbols = ["CHESSUSDT"]
    # symbols = ["TONUSDT"]
    symbols=["SOLUSDT","AIXBTUSDT","TAOUSDT","KAITOUSDT","OMUSDT","XRPUSDT","FETUSDT","UNIUSDT","COMPUSDT","THEUSDT","AVAXUSDT","LTCUSDT","ETCUSDT","FORMUSDT","TONUSDT","HFTUSDT","DOTUSDT","CHESSUSDT"]
    
    dataset_name = 'book_snapshot_5'
    # start_date, end_date = "2025-09-21", "2025-09-27"
    start_date, end_date = "2025-10-13", "2025-10-18"
    for symbol in symbols:
        # print(symbol)
        df = generate_df_one_symbol(symbol, start_date=start_date, end_date=end_date, dataset_name=dataset_name, resample_rate = '10s')
        # df = generate_df(symbols, dataset_name)
        print(df.head(5))
        print(df.info())
        df.to_csv('basis_' + dataset_name + f"_{symbol}" +  start_date + '_' + end_date + ".csv" , index=True, header=True, quoting=csv.QUOTE_NONNUMERIC)
        print("DataFrame saved to 'basis_" + dataset_name + f"_{symbol}_" + start_date + '_' + end_date + ".csv  successfully.")
