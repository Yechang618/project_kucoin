import os
import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ------------------------
# 配置
# ------------------------
version = '_3'
INPUT_DIR = f"kucoin_data/kucoin_data_combined{version}"
OUTPUT_DIR = f"kucoin_csv/kucoin_combined_csv{version}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# 主函数
# ------------------------
def main():
    print(f"🔍 Scanning JSON files in: {INPUT_DIR}")
    
    symbol_data = defaultdict(list)
    
    for file_path in Path(INPUT_DIR).rglob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    continue
                
                for record in data:
                    symbol = None
                    filename = file_path.stem
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        possible_symbol = parts[-1]
                        if possible_symbol.endswith(('USDTM', 'USDT')):
                            symbol = possible_symbol
                    
                    if not symbol:
                        continue
                    
                    # Filter out non-numeric metadata
                    record_filtered = {
                        k: v for k, v in record.items()
                        if k not in ['type'] and v is not None
                    }
                    
                    # Ensure timestamp is integer Unix timestamp (ms)
                    if 'timestamp' in record_filtered:
                        try:
                            # Convert to integer (handles string timestamps)
                            ts = int(float(record_filtered['timestamp']))
                            # Validate reasonable range (2020-2030)
                            if 1577836800000 <= ts <= 1893456000000:
                                record_filtered['timestamp'] = ts
                            else:
                                record_filtered['timestamp'] = int(ts/1000000)
                        except (ValueError, TypeError):
                            continue  # Skip invalid timestamp formats
                    
                    if 'timestamp' in record_filtered:  # Only keep records with valid timestamp
                        symbol_data[symbol].append(record_filtered)
                        
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            continue

    print(f"📊 Found data for {len(symbol_data)} symbols")

    for symbol, records in symbol_data.items():
        if not records:
            continue
            
        df = pd.DataFrame(records)
        
        if 'timestamp' not in df.columns:
            print(f"⚠️  No 'timestamp' in {symbol} data")
            continue
        
        # Ensure timestamp is integer type
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['timestamp'] = df['timestamp'].astype('int64')
        
        # Sort by Unix timestamp (numeric sort)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        output_file = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Saved {len(df)} records for {symbol} to {output_file}")

    print(f"\n🎉 All done! CSV files are in: {OUTPUT_DIR}")

# ------------------------
# 入口
# ------------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' not found!")
        print("Please place your JSON files in the correct folder")
        exit(1)
    
    main()