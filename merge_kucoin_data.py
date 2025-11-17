import os
import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ------------------------
# 配置
# ------------------------
INPUT_DIR = "kucoin_data"      # 输入 JSON 文件夹
OUTPUT_DIR = "kucoin_csv"      # 输出 CSV 文件夹

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# 主函数
# ------------------------
def main():
    print(f"🔍 Scanning JSON files in: {INPUT_DIR}")
    
    # 按 symbol 分组数据
    symbol_data = defaultdict(list)
    
    # 遍历所有 JSON 文件
    for file_path in Path(INPUT_DIR).rglob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    continue
                
                for record in data:
                    # 从文件名或数据中提取 symbol
                    symbol = None
                    # 方法1: 从文件名提取 (如 20250405_1420_BTCUSDTM.json)
                    filename = file_path.stem  # 不含扩展名
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        possible_symbol = parts[-1]
                        # 验证是否是有效 symbol（包含 USDTM）
                        if possible_symbol.endswith(('USDTM', 'USDT')):
                            symbol = possible_symbol
                    
                    # 方法2: 从数据中提取（如果文件名不规范）
                    if not symbol:
                        # 假设数据中包含 symbol 字段（你的脚本未存，但可跳过）
                        # 这里我们依赖文件名
                        continue
                    
                    if symbol:
                        # 添加来源文件信息（可选）
                        record_with_symbol = {
                            "symbol": symbol,
                            **record
                        }
                        # 在 merge_kucoin_data.py 中，过滤掉非数值字段
                        record_filtered = {
                            k: v for k, v in record_with_symbol.items()
                            if k not in ['type', 'symbol'] and pd.notna(v)
                        }
                        symbol_data[symbol].append(record_filtered)
                        
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            continue

    print(f"📊 Found data for {len(symbol_data)} symbols")

    # 为每个 symbol 生成 CSV
    for symbol, records in symbol_data.items():
        if not records:
            continue
            
        # 转为 DataFrame
        df = pd.DataFrame(records)
        
        # 确保 timestamp 列存在
        if 'timestamp' not in df.columns:
            print(f"⚠️  No 'timestamp' in {symbol} data")
            continue
        
        # 按 timestamp 排序（重要！）
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 输出 CSV
        output_file = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Saved {len(df)} records for {symbol} to {output_file}")

    print(f"\n🎉 All done! CSV files are in: {OUTPUT_DIR}")

# ------------------------
# 入口
# ------------------------
if __name__ == "__main__":
    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' not found!")
        print("Please place your JSON files in a folder named 'kucoin_data'")
        exit(1)
    
    main()