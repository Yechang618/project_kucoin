# import os
import pandas as pd
import glob
from pathlib import Path

# ==============================
# 配置
# ==============================
base_dir = Path("datasets")
processed_dir = base_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

symbol = "BNB"
quote = "USDT"
pair = f"{symbol}{quote}"

start_date = "2025-01-01"
end_date = "2025-10-29"
date_range = pd.date_range(start=start_date, end=end_date, freq="D")

# ==============================
# 工具函数
# ==============================
def parse_timestamp_series(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        max_val = s.max()
        if max_val > 1e17:
            unit = 'ns'
        elif max_val > 1e14:
            unit = 'us'
        elif max_val > 1e11:
            unit = 'ms'
        else:
            unit = 's'
        return pd.to_datetime(s, unit=unit)
    else:
        return pd.to_datetime(s)

def process_book_df(df, prefix):
    df_out = pd.DataFrame(index=df.index)
    df_out[f"{prefix}_bid0_price"]  = df.get("bids[0].price")
    df_out[f"{prefix}_bid0_amount"] = df.get("bids[0].amount")
    df_out[f"{prefix}_ask0_price"]  = df.get("asks[0].price")
    df_out[f"{prefix}_ask0_amount"] = df.get("asks[0].amount")

    # 前5档
    bid_p5  = [f"bids[{i}].price"  for i in range(5) if f"bids[{i}].price"  in df.columns]
    bid_a5  = [f"bids[{i}].amount" for i in range(5) if f"bids[{i}].amount" in df.columns]
    ask_p5  = [f"asks[{i}].price"  for i in range(5) if f"asks[{i}].price"  in df.columns]
    ask_a5  = [f"asks[{i}].amount" for i in range(5) if f"asks[{i}].amount" in df.columns]

    if bid_a5:
        bid_sum5 = df[bid_a5].sum(axis=1)
        bid_wavg5 = (df[bid_p5] * df[bid_a5]).sum(axis=1) / bid_sum5.replace(0, pd.NA)
        df_out[f"{prefix}_bid5_avg_price"] = bid_wavg5
        df_out[f"{prefix}_bid5_sum_amount"] = bid_sum5
    else:
        df_out[f"{prefix}_bid5_avg_price"] = pd.NA
        df_out[f"{prefix}_bid5_sum_amount"] = pd.NA

    if ask_a5:
        ask_sum5 = df[ask_a5].sum(axis=1)
        ask_wavg5 = (df[ask_p5] * df[ask_a5]).sum(axis=1) / ask_sum5.replace(0, pd.NA)
        df_out[f"{prefix}_ask5_avg_price"] = ask_wavg5
        df_out[f"{prefix}_ask5_sum_amount"] = ask_sum5
    else:
        df_out[f"{prefix}_ask5_avg_price"] = pd.NA
        df_out[f"{prefix}_ask5_sum_amount"] = pd.NA

    # 25档总量
    bid_a25 = [f"bids[{i}].amount" for i in range(25) if f"bids[{i}].amount" in df.columns]
    ask_a25 = [f"asks[{i}].amount" for i in range(25) if f"asks[{i}].amount" in df.columns]
    df_out[f"{prefix}_bid25_sum_amount"] = df[bid_a25].sum(axis=1) if bid_a25 else pd.NA
    df_out[f"{prefix}_ask25_sum_amount"] = df[ask_a25].sum(axis=1) if ask_a25 else pd.NA

    return df_out

def process_trades_df(df, prefix):
    df = df.copy()
    if 'timestamp' not in df.columns:
        raise KeyError("'timestamp' column missing")

    # 处理 side
    if 'side' in df.columns:
        side_series = df['side']
    elif 'isBuyerMaker' in df.columns:
        side_series = df['isBuyerMaker'].map({True: 'sell', False: 'buy'})
    elif 'm' in df.columns:
        side_series = df['m'].map({True: 'sell', False: 'buy'})
    else:
        raise KeyError(f"Cannot determine side in {prefix} trades")

    if side_series.dtype == 'object':
        side_series = side_series.str.upper().map({'B': 'buy', 'S': 'sell', 'BUY': 'buy', 'SELL': 'sell'})
    else:
        side_series = side_series.map({1: 'buy', 0: 'sell', -1: 'sell', True: 'sell', False: 'buy'})

    df['side'] = side_series
    df = df[df['side'].isin(['buy', 'sell'])]

    # 确保 price/amount 存在
    if 'price' not in df.columns and 'p' in df.columns:
        df['price'] = df['p']
    if 'amount' not in df.columns and 'q' in df.columns:
        df['amount'] = df['q']

    required = ['price', 'amount', 'side', 'timestamp']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing {col} in {prefix} trades")

    # 按 1 秒 + side 聚合
    df_agg = df.groupby(['side', pd.Grouper(key='timestamp', freq='1s')]).agg(
        price=('price', 'mean'),
        amount=('amount', 'mean')
    ).reset_index()

    df_wide = df_agg.pivot(index='timestamp', columns='side', values=['price', 'amount'])
    df_wide.columns = [f"{prefix}_{side}_{col}" for col, side in df_wide.columns]

    # 构造完整秒索引（当天）
    day = df['timestamp'].iloc[0].strftime('%Y-%m-%d')
    full_sec = pd.date_range(start=f"{day} 00:00:00", end=f"{day} 23:59:59", freq='1s')
    return df_wide.reindex(full_sec)

# ==============================
# 主循环：按天处理
# ==============================
valid_dfs = []  # 存放每天的有效 df
valid_dates = []  # 对应日期

for single_date in date_range:
    date_str = single_date.strftime("%Y-%m-%d")
    print(f"\n📆 Processing {date_str}...")

    # 构造当日文件路径
    patterns = {
        "book":     f"book/binance_book_snapshot_25_{date_str}_{pair}.csv.gz",
        "fbook":    f"fbook/binance-futures_book_snapshot_25_{date_str}_{pair}.csv.gz",
        "ftick":    f"ftick/binance-futures_derivative_ticker_{date_str}_{pair}.csv.gz",
        "ftrades":  f"ftrades/binance-futures_trades_{date_str}_{pair}.csv.gz",
        "trades":   f"trades/binance_trades_{date_str}_{pair}.csv.gz",
    }
    paths = {k: base_dir / v for k, v in patterns.items()}

    # ✅ 关键：spot 和 swap book 必须都存在
    if not (paths["book"].exists() and paths["fbook"].exists()):
        print(f"  ⚠️ Skipping {date_str}: missing spot or futures book")
        continue

    # 构造当天完整秒索引
    full_second_index = pd.date_range(
        start=f"{date_str} 00:00:00",
        end=f"{date_str} 23:59:59",
        freq="1s"
    )
    dfs_to_merge = []

    try:
        # --- book (spot) ---
        df = pd.read_csv(paths["book"])
        df.index = parse_timestamp_series(df["timestamp"])
        df = df.sort_index()  # ← 必须！
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]
        df_feat = process_book_df(df, "spot")
        df_res = df_feat.reindex(full_second_index, method='pad')
        dfs_to_merge.append(df_res)

        # --- fbook (swap) ---
        df = pd.read_csv(paths["fbook"])
        df.index = parse_timestamp_series(df["timestamp"])
        df = df.sort_index()  # ← 必须！
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]
        df_feat = process_book_df(df, "swap")
        df_res = df_feat.reindex(full_second_index, method='pad')
        dfs_to_merge.append(df_res)

        # --- ftick ---
        if paths["ftick"].exists():
            df = pd.read_csv(paths["ftick"])
            df.index = parse_timestamp_series(df["timestamp"])
            df = df.sort_index()  # ← 必须！
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='last')]
            df = df[["index_price", "mark_price", "funding_rate"]]
            df_res = df.reindex(full_second_index, method='pad')
            dfs_to_merge.append(df_res)
        else:
            print(f"  ⚠️ Ticker missing for {date_str}, skipping ticker")

        # --- spot trades ---
        if paths["trades"].exists():
            df = pd.read_csv(paths["trades"])
            df["timestamp"] = parse_timestamp_series(df["timestamp"])
            df_res = process_trades_df(df, "spot")
            dfs_to_merge.append(df_res)
        else:
            print(f"  ⚠️ Spot trades missing")

        # --- futures trades ---
        if paths["ftrades"].exists():
            df = pd.read_csv(paths["ftrades"])
            df["timestamp"] = parse_timestamp_series(df["timestamp"])
            df_res = process_trades_df(df, "swap")
            dfs_to_merge.append(df_res)
        else:
            print(f"  ⚠️ Futures trades missing")

        # 合并当日数据
        df_day = pd.concat(dfs_to_merge, axis=1)
        df_day.index.name = "timestamp"

        # Trim 前导 NaN（基于 spot_bid0_price）
        first_valid = df_day["spot_bid0_price"].first_valid_index()
        if first_valid is not None:
            df_day = df_day.loc[first_valid:]
        else:
            print(f"  ⚠️ No valid spot book, skipping {date_str}")
            continue

        valid_dfs.append(df_day)
        valid_dates.append(single_date)
        print(f"  ✅ {date_str} processed ({len(df_day)} seconds)")

    except Exception as e:
        print(f"  ❌ Error on {date_str}: {e}")
        continue

# ==============================
# 合并连续日期并保存
# ==============================
# if not valid_dfs:
#     print("No valid days processed.")
# else:
#     # 找出连续日期段
#     segments = []
#     current_seg = [0]
#     for i in range(1, len(valid_dates)):
#         if (valid_dates[i] - valid_dates[i-1]).days == 1:
#             current_seg.append(i)
#         else:
#             segments.append(current_seg)
#             current_seg = [i]
#     segments.append(current_seg)

#     # 保存每个连续段
#     for seg in segments:
#         seg_dfs = [valid_dfs[i] for i in seg]
#         df_seg = pd.concat(seg_dfs, axis=0)

#         start = valid_dates[seg[0]].strftime("%Y-%m-%d")
#         end = valid_dates[seg[-1]].strftime("%Y-%m-%d")
#         output_file = processed_dir / f"{pair}_{start}_{end}.csv.gz"

#         print(f"\n💾 Saving segment: {start} to {end} ({len(df_seg)} rows) -> {output_file.name}")
#         df_seg.to_csv(output_file, compression="gzip")

#     print(f"\n🎉 Done! Processed {len(valid_dates)} days, saved {len(segments)} file(s).")
# ==============================
# 按月份分组保存（保留连续性，但跨月拆分）
# ==============================
if not valid_dfs:
    print("No valid days processed.")
else:
    # 先合并所有有效数据（带日期信息）
    all_df = pd.concat(valid_dfs, axis=0)
    all_df.index.name = "timestamp"

    # 按年月分组
    grouped = all_df.groupby(pd.Grouper(freq='MS'))  # 'MS' = month start

    saved_files = 0
    for month_start, month_df in grouped:
        if month_df.empty:
            continue

        year_month = month_start.strftime("%Y-%m")
        output_file = processed_dir / f"{pair}_{year_month}.csv.gz"

        print(f"\n💾 Saving {year_month} ({len(month_df)} rows) -> {output_file.name}")
        month_df.to_csv(output_file, compression="gzip")
        saved_files += 1

    print(f"\n🎉 Done! Processed {len(valid_dates)} days, saved {saved_files} monthly file(s).")