import os
import pandas as pd
from pathlib import Path

# ------------------------
# 配置
# ------------------------
# version = '_3'
version = ''
CSV_DIR = f"kucoin_csv{version}"      # 输入 JSON 文件夹
OUTPUT_DIR = f"kucoin_csv{version}"      # 输出 CSV 文件夹

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# 主函数：加载单个 symbol 数据并重采样
# ------------------------
def process_symbol(symbol, csv_dir="kucoin_csv", output_dir="kucoin_series"):
    """
    加载单个 symbol 的 CSV, 重采样到 1 秒，并保存到 output_dir。
    """
    file_path = Path(csv_dir) / f"{symbol}.csv"
    if not file_path.exists():
        print(f"⚠️  {symbol}.csv not found. Skipping.")
        return False

    try:
        # 读取数据
        df = pd.read_csv(file_path, encoding='utf-8')
        if df.empty:
            print(f"⚠️  {symbol}.csv is empty. Skipping.")
            return False

        # 检查 timestamp
        if 'timestamp' not in df.columns:
            print(f"⚠️  'timestamp' missing in {symbol}.csv. Skipping.")
            return False

        # 转换时间索引
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['timestamp'], errors='ignore', inplace=True)
        df.drop(columns=['symbol'], errors='ignore', inplace=True)

        # 修复数据类型（关键！避免 object dtype）
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 1秒重采样（非 NaN 平均）
        df_1s = df.resample('1s').mean()

        # 保存到 output_dir
        output_path = Path(output_dir) / f"{symbol}_1s{version}.csv"
        df_1s.to_csv(output_path)
        print(f"✅ Saved {len(df_1s)} rows for {symbol} to {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")
        return False

# ------------------------
# 自动获取所有可用 symbols
# ------------------------
def get_available_symbols(csv_dir="kucoin_csv_3"):
    """从 kucoin_csv 目录自动提取所有 symbol 名称"""
    symbols = []
    for file in Path(csv_dir).glob("*.csv"):
        symbol = file.stem  # 移除 .csv
        symbols.append(symbol)
    return sorted(symbols)

# ------------------------
# 主程序
# ------------------------
if __name__ == "__main__":
    # 自动获取所有可用 symbols（无需手动维护列表）
    symbols = get_available_symbols(CSV_DIR)
    print(f"🔍 Found {len(symbols)} symbols in {CSV_DIR}:")
    print("\n".join(symbols))
    print("\n" + "="*50)

    # 处理每个 symbol
    success_count = 0
    for symbol in symbols:
        if process_symbol(symbol, CSV_DIR, OUTPUT_DIR):
            success_count += 1

    print("\n" + "="*50)
    print(f"🎉 Successfully processed {success_count}/{len(symbols)} symbols!")
    print(f"📁 Output files saved to: {OUTPUT_DIR}/")