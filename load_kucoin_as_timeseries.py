import os
import pandas as pd
from pathlib import Path

# ------------------------
# 配置
# ------------------------
CSV_DIR = "kucoin_csv"  # CSV 文件所在目录
# SYMBOLS = [
#     "BTCUSDTM", "ETHUSDTM", "SOLUSDTM", "XRPUSDTM", "FETUSDTM",
#     "UNIUSDTM", "COMPUSDTM", "THEUSDTM", "AVAXUSDTM", "LTCUSDTM",
#     "ETCUSDTM", "FORMUSDTM", "TONUSDTM", "HFTUSDTM", "DOTUSDTM",
#     "CHESSUSDTM", "BNBUSDTM", "TRXUSDTM", "DOGEUSDTM", "ADAUSDTM",
#     "LINKUSDTM", "XLMUSDTM", "BCHUSDTM", "HBARUSDTM", "ZECUSDTM",
#     "AAVEUSDTM", "ENAUSDTM", "NEARUSDTM", "ONDOUSDTM"
# ]  # ← 你关心的 symbol 列表
SYMBOLS = [
    "AAVEUSDTM",
    "ADAUSDTM",
    "AVAXUSDTM",
    "BCHUSDTM",
    "BNBUSDTM",
    "COMPUSDTM",
    "DOGEUSDTM",
    "DOTUSDTM",
    "ENAUSDTM",
    "ETCUSDTM",
    "ETHUSDTM",
    "FETUSDTM",
    "FORMUSDTM",
    "HBARUSDTM",
    "HFTUSDTM",
    "LINKUSDTM",
    "LTCUSDTM",
    "NEARUSDTM",
    "ONDOUSDTM",
    "PNUTUSDTM",
    "SOLUSDTM",
    "THEUSDTM",
    "TONUSDTM",
    "TRXUSDTM",
    "UNIUSDTM",
    "XBTUSDTM",
    "XLMUSDTM",
    "XRPUSDTM",
    "ZECUSDTM"
]
SYMBOLS = [SYMBOLS[0]]
# ------------------------
# 主函数
# ------------------------
def load_kucoin_data(symbols, csv_dir="kucoin_csv"):
    """
    读取指定 symbols 的 CSV 文件，合并为一个以 timestamp 为索引的 DataFrame。
    
    Parameters:
        symbols (list): 要加载的 symbol 列表，如 ["BTCUSDTM", "ETHUSDTM"]
        csv_dir (str): CSV 文件目录
    
    Returns:
        pd.DataFrame: MultiIndex columns, index = timestamp (ms)
    """
    all_dfs = {}
    
    for symbol in symbols:
        file_path = Path(csv_dir) / f"{symbol}.csv"
        if not file_path.exists():
            print(f"⚠️  {symbol}.csv not found. Skipping.")
            continue
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            if df.empty:
                print(f"⚠️  {symbol}.csv is empty. Skipping.")
                continue
                
            # 确保 timestamp 列存在
            if 'timestamp' not in df.columns:
                print(f"⚠️  'timestamp' column missing in {symbol}.csv. Skipping.")
                continue
                
            # 设置 timestamp 为索引（毫秒 -> datetime）
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # 删除原始 timestamp 列（避免重复）
            df.drop(columns=['timestamp'], inplace=True, errors='ignore')
            
            # 删除 symbol 列（如果存在）
            df.drop(columns=['symbol'], inplace=True, errors='ignore')
            
            # 保存
            all_dfs[symbol] = df
            
        except Exception as e:
            print(f"❌ Error loading {symbol}.csv: {e}")
            continue

    if not all_dfs:
        raise ValueError("No valid data loaded. Check your CSV files and symbol list.")
    
    # 合并所有 symbol 的 DataFrame，使用 MultiIndex 列
    combined = pd.concat(all_dfs, axis=1)
    
    # 按时间排序
    combined.sort_index(inplace=True)
    
    return combined

# ------------------------
# 使用示例
# ------------------------
if __name__ == "__main__":
    # 加载数据
    df = load_kucoin_data(SYMBOLS, CSV_DIR)
    df = df.resample('1S').mean()

    # 查看结果
    print("✅ Combined DataFrame shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nColumns (MultiIndex):")
    print(df.columns[:10])  # 显示前10列
    
    # 保存为单个 CSV（可选）
    output_file = f"kucoin_combined_timeseries_{SYMBOLS[0]}.csv"
    df.to_csv(output_file)
    print(f"\n💾 Saved combined time series to: {output_file}")