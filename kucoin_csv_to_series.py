import os
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------
# 配置
# ------------------------
version = '_3'
CSV_DIR = f"kucoin_csv/kucoin_combined_csv{version}"
OUTPUT_DIR = f"kucoin_series/kucoin_combined_series{version}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# 主函数：处理单个 symbol
# ------------------------
def process_symbol(symbol, csv_dir="kucoin_csv", output_dir="kucoin_series"):
    file_path = Path(csv_dir) / f"{symbol}.csv"
    if not file_path.exists():
        print(f"⚠️  {symbol}.csv not found. Skipping.")
        return False

    try:
        # 1. 读取原始数据
        df = pd.read_csv(file_path, encoding='utf-8')
        if df.empty:
            print(f"⚠️  {symbol}.csv is empty. Skipping.")
            return False

        # 2. 统一时间戳字段
        timestamp_col = None
        for col in ['ts', 'time', 'timestamp']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            print(f"⚠️  No timestamp column found in {symbol}.csv. Skipping.")
            return False

        # 3. 过滤无效时间戳
        df = df.copy()
        df['timestamp'] = df[timestamp_col]
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        min_ts = pd.Timestamp('2020-01-01').value // 10**6
        max_ts = pd.Timestamp('2030-01-01').value // 10**6
        valid_mask = (
            (df['timestamp'] >= min_ts) & 
            (df['timestamp'] <= max_ts) &
            df['timestamp'].notna()
        )
        df = df[valid_mask].copy()
        if df.empty:
            print(f"⚠️  No valid timestamps in {symbol}.csv. Skipping.")
            return False

        # 4. 转换时间为 datetime 并设为索引
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime').sort_index()
        
        # 5. 转换数据类型
        for col in df.columns:
            if col != 'timestamp':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 6. 处理重复时间戳
        df_clean = df.groupby(df.index).last()

        # 7. 仅处理 funding_rate 非零的行
        if 'funding_rate' not in df_clean.columns:
            print(f"ℹ️  No funding_rate column in {symbol}.csv. Skipping.")
            return False

        funding_mask = (
            df_clean['funding_rate'].notna() & 
            (df_clean['funding_rate'] != 0)
        )
        funding_events = df_clean[funding_mask].copy()
        
        if funding_events.empty:
            print(f"⚠️  No non-zero funding_rate events in {symbol}.csv. Skipping.")
            return False

        print(f"ℹ️  Found {len(funding_events)} funding events for {symbol}")

        # 8. 【修复】向历史填充关键字段
        aligned_records = []
        key_fields = [
            'index_price', 'mark_price',
            'spot_best_bid', 'spot_best_ask',
            'futures_best_bid', 'futures_best_ask'
        ]
        
        for idx, row in funding_events.iterrows():
            new_row = row.to_dict()
            
            for col in df_clean.columns:
                if col in ['timestamp', 'funding_rate', 'source_symbol']:
                    continue
                    
                # 只处理关键字段
                if col in key_fields and pd.isna(new_row[col]):
                    past_data = df_clean.loc[:idx, col]
                    valid_past = past_data[past_data.notna()]
                    if not valid_past.empty:
                        new_row[col] = valid_past.iloc[-1]
                # 非关键字段保留原值（可能为 NaN）
            
            aligned_records.append(new_row)

        # 9. 【修复】只过滤无任何价格数据的行
        df_aligned = pd.DataFrame(aligned_records, index=funding_events.index)
        
        # 至少需要一个价格字段
        price_fields = [
            'index_price', 'mark_price',
            'spot_best_bid', 'futures_best_bid'
        ]
        has_price = df_aligned[price_fields].notna().any(axis=1)
        df_final = df_aligned[has_price].copy()
        
        if df_final.empty:
            print(f"⚠️  No valid price data for funding events in {symbol}. Skipping.")
            return False

        # 10. 保存结果
        # df_final= df_final.dropna(axis=1)
        output_path = Path(output_dir) / f"{symbol}_funding{version}.csv"
        df_final.to_csv(output_path, index=True)  # 保留 datetime 索引
        print(f"✅ Saved {len(df_final)} funding events for {symbol} to {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")
        return False

# ------------------------
# 自动获取 symbols
# ------------------------
def get_available_symbols(csv_dir="kucoin_csv"):
    symbols = []
    for file in Path(csv_dir).glob("*.csv"):
        symbol = file.stem
        symbols.append(symbol)
    return sorted(symbols)

# ------------------------
# 主程序
# ------------------------
if __name__ == "__main__":
    symbols = get_available_symbols(CSV_DIR)
    print(f"🔍 Found {len(symbols)} symbols in {CSV_DIR}:")
    print("\n".join(symbols))
    print("\n" + "="*50)

    success_count = 0
    for symbol in symbols:
        if process_symbol(symbol, CSV_DIR, OUTPUT_DIR):
            success_count += 1

    print("\n" + "="*50)
    print(f"🎉 Successfully processed {success_count}/{len(symbols)} symbols!")
    print(f"📁 Output files saved to: {OUTPUT_DIR}/")

# import os
# import pandas as pd
# import numpy as np
# from pathlib import Path

# # ------------------------
# # 配置
# # ------------------------
# version = '_3'
# CSV_DIR = f"kucoin_csv/kucoin_combined_csv{version}"
# OUTPUT_DIR = f"kucoin_series/kucoin_combined_series{version}"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ------------------------
# # 主函数：处理单个 symbol
# # ------------------------
# def process_symbol(symbol, csv_dir="kucoin_csv", output_dir="kucoin_series"):
#     file_path = Path(csv_dir) / f"{symbol}.csv"
#     if not file_path.exists():
#         print(f"⚠️  {symbol}.csv not found. Skipping.")
#         return False

#     try:
#         # 1. 读取原始数据
#         df = pd.read_csv(file_path, encoding='utf-8')
#         if df.empty:
#             print(f"⚠️  {symbol}.csv is empty. Skipping.")
#             return False

#         # 2. 统一时间戳字段
#         timestamp_col = None
#         for col in ['ts', 'time', 'timestamp']:
#             if col in df.columns:
#                 timestamp_col = col
#                 break
        
#         if timestamp_col is None:
#             print(f"⚠️  No timestamp column found in {symbol}.csv. Skipping.")
#             return False

#         # 3. 过滤无效时间戳
#         df = df.copy()
#         df['timestamp'] = df[timestamp_col]
#         df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
#         min_ts = pd.Timestamp('2020-01-01').value // 10**6
#         max_ts = pd.Timestamp('2030-01-01').value // 10**6
#         valid_mask = (
#             (df['timestamp'] >= min_ts) & 
#             (df['timestamp'] <= max_ts) &
#             df['timestamp'].notna()
#         )
#         df = df[valid_mask].copy()
#         if df.empty:
#             print(f"⚠️  No valid timestamps in {symbol}.csv. Skipping.")
#             return False

#         # 4. 转换时间为 datetime 并设为索引
#         df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
#         df = df.set_index('datetime').sort_index()
        
#         # 5. 转换数据类型
#         for col in df.columns:
#             if col != 'timestamp':
#                 df[col] = pd.to_numeric(df[col], errors='coerce')

#         # 6. 【核心】处理重复时间戳（保留每时间点最后一条）
#         df_clean = df.groupby(df.index).last()

#         # 7. 【新逻辑】仅处理 funding_rate 非零的行
#         if 'funding_rate' not in df_clean.columns:
#             print(f"ℹ️  No funding_rate column in {symbol}.csv. Skipping.")
#             return False

#         # 筛选 funding_rate 非零且非 NaN 的行
#         funding_mask = (
#             df_clean['funding_rate'].notna() & 
#             (df_clean['funding_rate'] != 0)
#         )
#         funding_events = df_clean[funding_mask].copy()
        
#         if funding_events.empty:
#             print(f"⚠️  No non-zero funding_rate events in {symbol}.csv. Skipping.")
#             return False

#         print(f"ℹ️  Found {len(funding_events)} funding events for {symbol}")

#         # 8. 【新逻辑】向历史填充缺失值
#         # 先对整个 df_clean 做前向填充（ffill），但只用于获取历史值
#         df_filled = df_clean.copy()
#         # 对每列分别进行前向填充（避免跨列污染）
#         for col in df_filled.columns:
#             if col != 'timestamp':
#                 df_filled[col] = df_filled[col].ffill()  # 向历史填充（实际是向未来，但我们需要反向）

#         # 更正：我们需要向**过去**填充，所以应该用 bfill + 反向
#         # 正确做法：对 funding_events 的每个时间点，从 df_clean 中反向搜索
#         aligned_records = []
        
#         for idx, row in funding_events.iterrows():
#             # 创建新记录（以 funding_rate 行为基础）
#             new_row = row.to_dict()
            
#             # 对每个缺失字段，向更早时间查找
#             for col in df_clean.columns:
#                 if col in ['timestamp', 'funding_rate']:
#                     continue
                    
#                 if pd.isna(new_row[col]):
#                     # 从当前时间点向前搜索（更早时间）
#                     past_data = df_clean.loc[:idx, col]
#                     # 获取非 NaN 的最新值（最接近当前时间的过去值）
#                     valid_past = past_data[past_data.notna()]
#                     if not valid_past.empty:
#                         new_row[col] = valid_past.iloc[-1]  # 最新（最接近）的历史值
#                     # 如果找不到，保留 NaN（后续会过滤）
            
#             aligned_records.append(new_row)

#         # 转为 DataFrame
#         df_aligned = pd.DataFrame(aligned_records, index=funding_events.index)

#         # 9. 移除仍包含 NaN 的行
#         df_final = df_aligned.dropna(how='any')
#         if df_final.empty:
#             print(f"⚠️  No complete funding events after alignment for {symbol}. Skipping.")
#             return False

#         # 10. 保存（无需重采样，因为 funding 事件本身稀疏）
#         output_path = Path(output_dir) / f"{symbol}_funding{version}.csv"
#         df_final.to_csv(output_path)
#         print(f"✅ Saved {len(df_final)} funding events for {symbol} to {output_path}")
#         return True

#     except Exception as e:
#         print(f"❌ Error processing {symbol}: {e}")
#         return False

# # ------------------------
# # 自动获取 symbols
# # ------------------------
# def get_available_symbols(csv_dir="kucoin_csv"):
#     symbols = []
#     for file in Path(csv_dir).glob("*.csv"):
#         symbol = file.stem
#         symbols.append(symbol)
#     return sorted(symbols)

# # ------------------------
# # 主程序
# # ------------------------
# if __name__ == "__main__":
#     symbols = get_available_symbols(CSV_DIR)
#     print(f"🔍 Found {len(symbols)} symbols in {CSV_DIR}:")
#     print("\n".join(symbols))
#     print("\n" + "="*50)

#     success_count = 0
#     for symbol in symbols:
#         if process_symbol(symbol, CSV_DIR, OUTPUT_DIR):
#             success_count += 1

#     print("\n" + "="*50)
#     print(f"🎉 Successfully processed {success_count}/{len(symbols)} symbols!")
#     print(f"📁 Output files saved to: {OUTPUT_DIR}/")