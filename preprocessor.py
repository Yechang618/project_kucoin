import pandas as pd
import numpy as np
import warnings, csv
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def generate_df(symbols, dataset_name):
    df = None
    address = 'datasets'
    # spot_name1 = 'binance_book_snapshot_5_2025-05-22_'
    # swap_name1 = 'binance-futures_book_snapshot_5_2025-05-22_'
    spot_name1 = 'binance_' + dataset_name +'_'
    swap_name1 = 'binance-futures_' + dataset_name +'_'  
    spot_name2 = '.csv.gz'
    swap_name2 = '.csv.gz'
    for symbol in symbols:
        print(f"Processing symbol: {symbol}")
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
        df_swap_qt = pd.DataFrame({'swap_bids': df_swap['bid_price'], 'swap_asks': df_swap['ask_price']}).drop_duplicates()
        df_spot_qt = pd.DataFrame({'spot_bids': df_spot['bid_price'], 'spot_asks': df_spot['ask_price']}).drop_duplicates()
        # Resample df_spot_qt and df_swap_qt to a higher frequency (e.g., 1ms) to allow for finer matching
        df_spot_qt = df_spot_qt[~df_spot_qt.index.duplicated(keep='first')]
        df_swap_qt = df_swap_qt[~df_swap_qt.index.duplicated(keep='first')]
        df_spot_qt_resampled = df_spot_qt.resample('1s').ffill()
        df_swap_qt_resampled = df_swap_qt.resample('1s').ffill()  
        # Merge the resampled dataframes based on the closest time index within a tolerance
        df_merged = pd.merge_asof(
            df_swap_qt_resampled['swap_bids'],
            df_spot_qt_resampled['spot_asks'],
            left_index=True,
            right_index=True,
            direction='nearest',
            tolerance=pd.Timedelta('1s')
        )         
         # Calculate the basis
        df_basis = pd.DataFrame({
            symbol+'.basis1': np.log(df_merged['swap_bids']) - np.log(df_merged['spot_asks'])
        })
        # print(df_basis.info())
        if df is None:
            df = df_basis.copy()
        else:
            df = df.join(df_basis)
        del df_merged
        del df_basis

        # Merge the resampled dataframes based on the closest time index within a tolerance
        df_merged2 = pd.merge_asof(
            df_swap_qt_resampled['swap_asks'],
            df_spot_qt_resampled['spot_bids'],
            left_index=True,
            right_index=True,
            direction='nearest',
            tolerance=pd.Timedelta('100ms')
        )

        # Calculate the basis
        df_basis2 = pd.DataFrame({
            symbol+'.basis2': np.log(df_merged2['swap_asks']) - np.log(df_merged2['spot_bids'])
        })

        df = df.join(df_basis2)
        del df_basis2
        del df_merged2
    return df



if __name__ == "__main__":
    symbols = ["BTCUSDT","SOLUSDT","PNUTUSDT",
               "TURBOUSDT","APTUSDT","AIXBTUSDT",
               "TAOUSDT","KAITOUSDT","OMUSDT",
               "XRPUSDT","FETUSDT","UNIUSDT",
               "COMPUSDT","THEUSDT","AVAXUSDT",
               "LTCUSDT","ETCUSDT","FORMUSDT",
               "TONUSDT","HFTUSDT","DOTUSDT",
               "CHESSUSDT","MKRUSDT","WIFUSDT"]
    # dataset_name = 'quotes_2025-06-07'
    dataset_name = 'quotes_2025-04-05'
    # dataset_name = 'book_snapshot_5_2025-05-22'
    df = generate_df(symbols, dataset_name)
    print(df.head(5))
    print(df.info())
    df.to_csv('basis_' + dataset_name + '.csv', index=True, header=True, quoting=csv.QUOTE_NONNUMERIC)
    print("DataFrame saved to 'basis_" + dataset_name + ".csv'  successfully.")
