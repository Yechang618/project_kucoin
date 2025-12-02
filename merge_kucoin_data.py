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
# 提取 base symbol（移除后缀）
# ------------------------
def get_base_symbol(symbol):
    """
    从 symbol 提取基础资产名
    Examples:
        'DOT-USDT' → 'DOT'
        'DOTUSDTM' → 'DOT'
        'XBT-USDT' → 'XBT'
        'XBTUSDTM' → 'XBT'
    """
    if '-USDT' in symbol:
        return symbol.split('-USDT')[0]
    elif symbol.endswith('USDTM'):
        return symbol[:-5]  # 移除 'USDTM'
    elif symbol.endswith('USDT'):
        return symbol[:-4]  # 移除 'USDT'（旧格式）
    else:
        return symbol  # 无法识别时保持原样

# ------------------------
# 判断 symbol 类型
# ------------------------
def detect_symbol_type(symbol):
    if symbol.endswith('USDTM'):
        return 'futures'
    elif '-USDT' in symbol:
        return 'spot'
    elif symbol.endswith('USDT'):
        # 旧格式现货（无连字符）
        return 'spot'
    else:
        return 'unknown'

# ------------------------
# 重命名字段（添加前缀）
# ------------------------
def rename_fields(record, symbol_type):
    new_record = record.copy()
    field_mapping = {
        'best_bid': f'{symbol_type}_best_bid',
        'best_ask': f'{symbol_type}_best_ask',
        'last_price': f'{symbol_type}_last_price'
    }
    for old_key, new_key in field_mapping.items():
        if old_key in new_record:
            new_record[new_key] = new_record.pop(old_key)
    return new_record

# ------------------------
# 主函数
# ------------------------
def main():
    print(f"🔍 Scanning JSON files in: {INPUT_DIR}")
    
    # 按 base symbol 分组数据
    base_symbol_data = defaultdict(list)
    
    for file_path in Path(INPUT_DIR).rglob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    continue
                
                # 提取 symbol
                filename = file_path.stem
                parts = filename.split('_')
                if len(parts) < 3:
                    continue
                symbol = parts[-1]
                
                if not symbol or len(symbol) < 5:
                    continue
                
                # 检测类型和 base symbol
                symbol_type = detect_symbol_type(symbol)
                if symbol_type == 'unknown':
                    print(f"⚠️  Unknown symbol format: {symbol}. Skipping.")
                    continue
                
                base_symbol = get_base_symbol(symbol)
                print(f"ℹ️  Processing {symbol_type}: {symbol} → base: {base_symbol}")
                
                for record in data:
                    record_filtered = {
                        k: v for k, v in record.items()
                        if k not in ['type'] and v is not None
                    }
                    
                    # 处理时间戳
                    if 'timestamp' in record_filtered:
                        try:
                            ts = int(float(record_filtered['timestamp']))
                            if ts > 1893456000000:  # 微秒转毫秒
                                ts = int(ts/1000000)
                            if 1577836800000 <= ts <= 1893456000000:
                                record_filtered['timestamp'] = ts
                            else:
                                continue
                        except (ValueError, TypeError):
                            continue
                    
                    if 'timestamp' not in record_filtered:
                        continue
                    
                    # 重命名字段
                    record_renamed = rename_fields(record_filtered, symbol_type)
                    
                    # 添加原始 symbol 信息（可选）
                    record_renamed['source_symbol'] = symbol
                    
                    # 按 base symbol 分组
                    base_symbol_data[base_symbol].append(record_renamed)
                        
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            continue

    print(f"📊 Found data for {len(base_symbol_data)} base symbols")

    # 合并每个 base symbol 的数据
    for base_symbol, records in base_symbol_data.items():
        if not records:
            continue
            
        df = pd.DataFrame(records)
        
        if 'timestamp' not in df.columns:
            print(f"⚠️  No 'timestamp' in {base_symbol} data")
            continue
        
        # 转换 timestamp 为整数
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['timestamp'] = df['timestamp'].astype('int64')
        
        # 按时间戳排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 保存合并文件
        output_file = os.path.join(OUTPUT_DIR, f"{base_symbol}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Saved {len(df)} records for {base_symbol} to {output_file}")

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
# import os
# import json
# import pandas as pd
# from collections import defaultdict
# from pathlib import Path

# # ------------------------
# # 配置
# # ------------------------
# version = '_3'
# INPUT_DIR = f"kucoin_data/kucoin_data_combined{version}"
# OUTPUT_DIR = f"kucoin_csv/kucoin_combined_csv{version}"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ------------------------
# # 主函数
# # ------------------------
# def main():
#     print(f"🔍 Scanning JSON files in: {INPUT_DIR}")
    
#     symbol_data = defaultdict(list)
    
#     for file_path in Path(INPUT_DIR).rglob("*.json"):
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 if not isinstance(data, list):
#                     continue
                
#                 for record in data:
#                     symbol = None
#                     filename = file_path.stem
#                     parts = filename.split('_')
#                     if len(parts) >= 3:
#                         possible_symbol = parts[-1]
#                         if possible_symbol.endswith(('USDTM', 'USDT')):
#                             symbol = possible_symbol
                    
#                     if not symbol:
#                         continue
                    
#                     # Filter out non-numeric metadata
#                     record_filtered = {
#                         k: v for k, v in record.items()
#                         if k not in ['type'] and v is not None
#                     }
                    
#                     # Ensure timestamp is integer Unix timestamp (ms)
#                     if 'timestamp' in record_filtered:
#                         try:
#                             # Convert to integer (handles string timestamps)
#                             ts = int(float(record_filtered['timestamp']))
#                             # Validate reasonable range (2020-2030)
#                             if 1577836800000 <= ts <= 1893456000000:
#                                 record_filtered['timestamp'] = ts
#                             else:
#                                 record_filtered['timestamp'] = int(ts/1000000)
#                         except (ValueError, TypeError):
#                             continue  # Skip invalid timestamp formats
                    
#                     if 'timestamp' in record_filtered:  # Only keep records with valid timestamp
#                         symbol_data[symbol].append(record_filtered)
                        
#         except Exception as e:
#             print(f"⚠️  Error reading {file_path}: {e}")
#             continue

#     print(f"📊 Found data for {len(symbol_data)} symbols")

#     for symbol, records in symbol_data.items():
#         if not records:
#             continue
            
#         df = pd.DataFrame(records)
        
#         if 'timestamp' not in df.columns:
#             print(f"⚠️  No 'timestamp' in {symbol} data")
#             continue
        
#         # Ensure timestamp is integer type
#         df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
#         df = df.dropna(subset=['timestamp'])
#         df['timestamp'] = df['timestamp'].astype('int64')
        
#         # Sort by Unix timestamp (numeric sort)
#         df = df.sort_values('timestamp').reset_index(drop=True)
        
#         output_file = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
#         df.to_csv(output_file, index=False, encoding='utf-8')
#         print(f"✅ Saved {len(df)} records for {symbol} to {output_file}")

#     print(f"\n🎉 All done! CSV files are in: {OUTPUT_DIR}")

# # ------------------------
# # 入口
# # ------------------------
# if __name__ == "__main__":
#     if not os.path.exists(INPUT_DIR):
#         print(f"❌ Input directory '{INPUT_DIR}' not found!")
#         print("Please place your JSON files in the correct folder")
#         exit(1)
    
#     main()