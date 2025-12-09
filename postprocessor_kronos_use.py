import pandas as pd
from pathlib import Path

# 配置
symbols = ["SOL", "BNB", "ZEC", "KAITO", "DOT"]
symbol = symbols[4]
quote = "USDT"
pair = f"{symbol}{quote}"
processed_dir, output_dir = Path("datasets/training"), Path("datasets/kronos")
# processed_dir, output_dir = Path("datasets/testing"), Path("datasets/kronos_test")
output_dir.mkdir(parents=True, exist_ok=True)

# 收集所有 processed 文件
files = list(processed_dir.glob(f"{pair}_*.csv.gz"))
if not files:
    raise FileNotFoundError(f"No processed files found for {pair} in {processed_dir}")

print(f"Found {len(files)} processed file(s).")

# 按时间顺序合并所有数据
all_dfs = []
for f in sorted(files):
    print(f"Loading {f.name}...")
    df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
    all_dfs.append(df)

df_all = pd.concat(all_dfs, axis=0).sort_index()
print(f"Total rows: {len(df_all)}")

# 检查必要列是否存在
required_cols = [
    "spot_bid0_price", "spot_ask0_price", "index_price",
    "swap_bid0_price", "swap_ask0_price",
    "spot_bid0_amount", "spot_ask0_amount",
    "swap_bid0_amount", "swap_ask0_amount",
    "spot_bid25_sum_amount", "spot_ask25_sum_amount",
    "swap_bid25_sum_amount", "swap_ask25_sum_amount",
    "funding_rate"
]
missing = [c for c in required_cols if c not in df_all.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

# 生成 Kronos OHLCV 字段
df_kronos = pd.DataFrame(index=df_all.index)
df_kronos["open"] = (df_all["spot_bid0_price"] + df_all["spot_ask0_price"]) / 2 - df_all["index_price"]
df_kronos["high"] = df_all["swap_ask0_price"] - df_all["spot_bid0_price"]
df_kronos["low"]  = df_all["swap_bid0_price"] - df_all["spot_ask0_price"]  # ← 修正为 price
df_kronos["close"] = df_all["index_price"]
df_kronos["volume"] = (
    df_all["spot_bid0_amount"] + df_all["spot_ask0_amount"] +
    df_all["swap_bid0_amount"] + df_all["swap_ask0_amount"]
)
df_kronos["amount"] = (
    df_all["spot_bid25_sum_amount"] + df_all["spot_ask25_sum_amount"] +
    df_all["swap_bid25_sum_amount"] + df_all["swap_ask25_sum_amount"]
)

# 可选：移除全 NaN 行（如 funding_rate 初始缺失）
df_kronos = df_kronos.dropna(how="all")

# 保存为 Kronos 格式
output_file = output_dir / f"{pair}_kronos.csv.gz"
if "timestamp" in df_kronos.columns:
    df_kronos = df_kronos.rename(columns={"timestamp": "timestamps"})
print(df_kronos.info())
print(f"Saving Kronos dataset: {output_file}")
# df_kronos.to_csv(output_file, compression="gzip", date_format="%Y-%m-%d %H:%M:%S")
df_kronos.to_csv(output_file,  date_format="%Y-%m-%d %H:%M:%S")
print("✅ Done.")