import os
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------
# 配置
# ------------------------
version = '_3'
# CSV_DIR = f"kucoin_csv{version}"
# OUTPUT_DIR = f"kucoin_series{version}"
CSV_DIR = f"kucoin_futures_csv{version}"
OUTPUT_DIR = f"kucoin_futures_series{version}"
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
        else:
            print(f"ℹ️  Processing {symbol}.csv with {len(df)} rows.")
            print(df.head())

        if 'timestamp' not in df.columns:
            print(f"⚠️  'timestamp' missing in {symbol}.csv. Skipping.")
            return False

        # 2. 转换时间（毫秒 -> datetime）
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 3. 转换数据类型
        for col in df.columns:
            if col != 'datetime':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. 【关键】处理重复时间戳
        # 先按 datetime 排序
        df = df.sort_values('datetime')
        
        # 定义聚合规则
        agg_rules = {}
        for col in df.columns:
            if col in ['datetime', 'timestamp', 'symbol']:
                continue
            # 对数值列：取非 NaN 的最后一个值
            agg_rules[col] = lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan

        # 按 datetime 聚合（解决重复索引）
        df_clean = df.groupby('datetime').agg(agg_rules).reset_index()
        df_clean = df_clean.set_index('datetime').sort_index()

        # 5. 以 index_price 非 NaN 时间为准
        if 'index_price' not in df_clean.columns:
            print(f"⚠️  'index_price' missing in {symbol}.csv. Skipping.")
            return False

        valid_index_times = df_clean[df_clean['index_price'].notna()].index
        if valid_index_times.empty:
            print(f"⚠️  No valid index_price in {symbol}.csv. Skipping.")
            return False

        # 6. 创建对齐 DataFrame
        df_aligned = pd.DataFrame(index=valid_index_times)

        # 7. 对齐各列
        for col in df_clean.columns:
            if col == 'index_price':
                df_aligned[col] = df_clean[col].reindex(valid_index_times)
            else:
                col_series = df_clean[col].dropna()
                if col_series.empty:
                    df_aligned[col] = np.nan
                else:
                    # 使用最近邻填充（1分钟容差）
                    df_aligned[col] = col_series.reindex(
                        valid_index_times,
                        method='nearest',
                        tolerance=pd.Timedelta('1s')
                    )

        # 8. 移除 NaN 行
        df_final = df_aligned.dropna()
        if df_final.empty:
            print(f"⚠️  No complete rows after alignment for {symbol}. Skipping.")
            return False

        # 9. 1秒重采样
        df_1s = df_final.resample('1s').last()

        # 10. 保存
        output_path = Path(output_dir) / f"{symbol}_1s{version}.csv"
        df_1s.to_csv(output_path)
        print(f"✅ Saved {len(df_1s)} rows for {symbol} to {output_path}")
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