import pandas as pd
import numpy as np  

def generate_features(df):
    df_obook = df.copy()
    df_obook['basis1'] = df_obook['swap.bids[0].price'] - df_obook['spot.asks[0].price']
    df_obook['basis2'] = df_obook['swap.asks[0].price'] - df_obook['spot.bids[0].price']
    df_obook['basis1_dt'] = df_obook['basis1'].diff()
    df_obook['basis2_dt'] = df_obook['basis2'].diff()
    df_obook['spot_bids_slope_d'] = df_obook['spot.bids_slope'].diff()
    df_obook['swap_bids_slope_d'] = df_obook['swap.bids_slope'].diff()
    df_obook['spot_asks_slope_d'] = df_obook['spot.asks_slope'].diff()
    df_obook['swap_asks_slope_d'] = df_obook['swap.asks_slope'].diff()
    df_obook['swap_bids_price_d'] = df_obook['swap.bids[0].price'].diff()
    df_obook['spot_asks_price_d'] = df_obook['spot.asks[0].price'].diff()
    df_obook['spot_bids_price_d'] = df_obook['spot.bids[0].price'].diff()
    df_obook['swap_asks_price_d'] = df_obook['swap.asks[0].price'].diff()
    df_obook = df_obook.fillna(0)
    return df_obook