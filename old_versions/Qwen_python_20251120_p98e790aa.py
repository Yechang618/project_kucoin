import os
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------
# 配置（基于 KuCoin 2025 默认参数）
# ------------------------
DATA_DIR = "kucoin_series"
MAX_FUNDING_RATE = 0.00375  # 0.375% = (1% - 0.5%) * 0.75
EXPECTED_INTERVAL_SEC = 8 * 3600  # 8 hours in seconds
TOLERANCE_SEC = 1800           # ±30 minutes tolerance

# ------------------------
# 验证单个 symbol
# ------------------------
def validate_symbol_offline(symbol, data_dir="kucoin_series"):
    file_path = Path(data_dir) / f"{symbol}_1s.csv"
    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    except Exception as e:
        print(f"❌ {symbol}: CSV error - {e}")
        return None

    # 检查必要字段
    required_cols = ['funding_rate', 'best_bid', 'best_ask', 'index_price']
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️  {symbol}: Missing columns")
        return None

    # 提取 funding_rate 非 NaN 的记录
    funding_df = df[df['funding_rate'].notna()].copy()
    if funding_df.empty:
        print(f"ℹ️  {symbol}: No funding records")
        return None

    total_records = len(funding_df)

    # 1. 检查费率范围
    out_of_range = (funding_df['funding_rate'].abs() > MAX_FUNDING_RATE).sum()

    # 2. 检查结算间隔
    interval_ok = True
    avg_interval = None
    if total_records > 1:
        intervals = funding_df.index.to_series().diff().dt.total_seconds().dropna()
        avg_interval = intervals.mean()
        # 检查所有间隔是否在 [7.5h, 8.5h]
        interval_ok = ((intervals >= EXPECTED_INTERVAL_SEC - TOLERANCE_SEC) & 
                       (intervals <= EXPECTED_INTERVAL_SEC + TOLERANCE_SEC)).all()

    # 3. 检查符号一致性
    funding_df['mid_price'] = (funding_df['best_bid'] + funding_df['best_ask']) / 2
    funding_df['premium'] = (funding_df['mid_price'] - funding_df['index_price']) / funding_df['index_price']
    
    # 只检查 premium 显著非零的情况（避免除零或噪声）
    significant_premium = funding_df['premium'].abs() > 1e-6
    if significant_premium.any():
        sign_match = (
            (funding_df.loc[significant_premium, 'funding_rate'] > 0) == 
            (funding_df.loc[significant_premium, 'premium'] > 0)
        ).sum()
        total_significant = significant_premium.sum()
        sign_consistency = sign_match / total_significant
    else:
        sign_consistency = 1.0

    # 4. 检查数据完整性（funding_rate 行是否有有效价格）
    valid_prices = (
        funding_df['best_bid'].notna() & 
        funding_df['best_ask'].notna() & 
        funding_df['index_price'].notna() &
        (funding_df['best_bid'] > 0) &
        (funding_df['best_ask'] > 0) &
        (funding_df['index_price'] > 0)
    ).sum()
    missing_price_data = total_records - valid_prices

    return {
        'symbol': symbol,
        'total_records': total_records,
        'out_of_range_count': int(out_of_range),
        'interval_ok': bool(interval_ok),
        'avg_interval_sec': avg_interval,
        'sign_consistency': float(sign_consistency),
        'missing_price_data': int(missing_price_data)
    }

# ------------------------
# 主函数
# ------------------------
def main():
    print("🔍 Offline Validation of KuCoin Funding Rate (2025 Rules)\n")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"⚙️  Funding Rate Cap: ±{MAX_FUNDING_RATE:.4f} ({MAX_FUNDING_RATE*100:.3f}%)")
    print(f"⚙️  Expected Interval: {EXPECTED_INTERVAL_SEC//3600} hours ±{TOLERANCE_SEC//60} min\n")
    print("=" * 80)

    results = []
    files = list(Path(DATA_DIR).glob("*_1s.csv"))
    
    if not files:
        print(f"❌ No CSV files found in {DATA_DIR}")
        return

    for file in sorted(files):
        symbol = file.stem.replace("_1s", "")
        print(f"Processing {symbol}...", end=" ")
        result = validate_symbol_offline(symbol, DATA_DIR)
        if result:
            results.append(result)
            # 判断是否通过
            passed = (
                result['out_of_range_count'] == 0 and
                result['interval_ok'] and
                result['sign_consistency'] >= 0.95 and
                result['missing_price_data'] == 0
            )
            status = "✅" if passed else "⚠️"
            print(f"{status}")
        else:
            print("❌")

    if not results:
        print("\n❌ No valid results.")
        return

    # 汇总报告
    print("\n" + "=" * 80)
    print("📊 OFFLINE VALIDATION SUMMARY")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    
    total_symbols = len(df_results)
    range_violations = (df_results['out_of_range_count'] > 0).sum()
    interval_violations = (~df_results['interval_ok']).sum()
    sign_violations = (df_results['sign_consistency'] < 0.95).sum()
    data_missing = (df_results['missing_price_data'] > 0).sum()

    print(f"Total Symbols Analyzed  : {total_symbols}")
    print(f"Rate Range Violations   : {range_violations}")
    print(f"Interval Violations     : {interval_violations}")
    print(f"Sign Consistency < 95%  : {sign_violations}")
    print(f"Missing Price Data      : {data_missing}")

    # 保存详细报告
    report_file = "funding_validation_offline.csv"
    df_results.to_csv(report_file, index=False)
    print(f"\n💾 Detailed report saved to: {report_file}")

    # 总体结论
    if range_violations == 0 and interval_violations == 0 and sign_violations == 0 and data_missing == 0:
        print("\n🎉 ALL SYMBOLS PASS OFFLINE VALIDATION!")
    else:
        issues = range_violations + interval_violations + sign_violations + data_missing
        print(f"\n⚠️  {issues} issues detected. Check detailed report.")

# ------------------------
# 入口
# ------------------------
if __name__ == "__main__":
    main()