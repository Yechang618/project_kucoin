import os
import json
import glob
import argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==================== 配置 ====================
DATA_DIR = "./kucoin_data/raw_data"
REGRESSION_DIR = "./kucoin_data/regression_results"
FIGURE_DIR = "./figure"
os.makedirs(FIGURE_DIR, exist_ok=True)

def load_symbol_data(symbol, hours=8):
    """加载原始数据"""
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=hours)
    
    print(f"🔍 Loading raw data for {symbol} from {window_start.strftime('%Y-%m-%d %H:%M:%S')} to now")
    
    pattern = os.path.join(DATA_DIR, f"{symbol}_*.json")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No raw data files found for symbol: {symbol}")
    
    records = []
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    continue
                    
                for rec in data:
                    ts = rec.get("timestamp")
                    if not isinstance(ts, int) or ts <= 0:
                        continue
                        
                    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                    if dt < window_start or dt > now:
                        continue
                    
                    swap_bid = rec.get("swap_bid_avg")
                    swap_ask = rec.get("swap_ask_avg")
                    spot_bid = rec.get("spot_bid_avg")
                    spot_ask = rec.get("spot_ask_avg")
                    index_price = rec.get("index_price_avg")
                    funding_rate = rec.get("funding_rate")
                    
                    if None in (swap_bid, swap_ask, index_price, funding_rate):
                        continue
                    
                    # 处理特殊 symbol
                    if symbol.startswith("10000"):
                        spot_bid = spot_bid * 1e4
                        spot_ask = spot_ask * 1e4
                    elif symbol.startswith("1000"):
                        spot_bid = spot_bid * 1e3
                        spot_ask = spot_ask * 1e3
                    
                    try:
                        mid_swap = (float(swap_bid) + float(swap_ask)) / 2.0
                        index = float(index_price)
                        if index == 0:
                            continue
                        funding_estimate = mid_swap / index - 1.0
                        spot_price = (float(spot_bid) + float(spot_ask)) / 2.0
                        
                        records.append({
                            'datetime': dt,
                            'timestamp': ts,
                            'funding_estimate': funding_estimate,
                            'funding_rate': float(funding_rate),
                            'spot_price': spot_price,
                            'index_price': float(index_price)
                        })
                    except (ValueError, TypeError, ZeroDivisionError):
                        continue
        except Exception as e:
            print(f"⚠️ Error reading {filepath}: {e}")
            continue
    
    records.sort(key=lambda x: x['timestamp'])
    return records

def load_regression_results(symbol, hours=8):
    """加载过去 N 小时的回归结果"""
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=hours)
    
    print(f"🔍 Loading regression results for {symbol} from {window_start.strftime('%Y-%m-%d %H:%M:%S')} to now")
    
    pattern = os.path.join(REGRESSION_DIR, "regression_results_rolling_8h_*.json")
    files = glob.glob(pattern)
    
    results = []
    for filepath in files:
        try:
            # 从文件名提取时间戳
            filename = os.path.basename(filepath)
            # 文件名格式: regression_results_rolling_8h_YYYYMMDD_HHMMSS.json
            parts = filename.replace("regression_results_rolling_8h_", "").replace(".json", "").split("_")
            if len(parts) >= 2:
                file_time_str = f"{parts[0]}_{parts[1]}"
                file_dt = datetime.strptime(file_time_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                
                if file_dt < window_start or file_dt > now:
                    continue
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if item.get('symbol') == symbol:
                            results.append({
                                'datetime': file_dt,
                                'b0_theoretical': item.get('b0_theoretical'),
                                'b0_estimate': item.get('b0_estimate'),
                                'r2_score': item.get('r2_score'),
                                'spot_price': item.get('spot_price'),
                                'index_price': item.get('index_price')  # 注意：原结果可能没有index_price
                            })
        except Exception as e:
            print(f"⚠️ Error reading regression file {filepath}: {e}")
            continue
    
    results.sort(key=lambda x: x['datetime'])
    return results

def compute_recursive_average(values):
    avg_values = []
    for i, val in enumerate(values):
        if i == 0:
            avg_values.append(val)
        else:
            n = i + 1
            avg_values.append(((n - 1) * avg_values[-1] + val) / n)
    return avg_values

def main():
    parser = argparse.ArgumentParser(description='Plot comprehensive funding analysis')
    parser.add_argument('symbol', type=str, help='Symbol name (e.g., BTC)')
    parser.add_argument('--hours', type=int, default=8, help='Hours of historical data to load (default: 8)')
    args = parser.parse_args()
    
    try:
        # 1. 加载原始数据
        raw_records = load_symbol_data(args.symbol, args.hours)
        print(f"📊 Loaded {len(raw_records)} raw records for {args.symbol}")
        
        if len(raw_records) < 2:
            print("❌ Not enough raw data points (< 2)")
            return
        
        # 2. 加载回归结果
        regression_results = load_regression_results(args.symbol, args.hours)
        print(f"📊 Loaded {len(regression_results)} regression results for {args.symbol}")
        
        # 3. 创建输出文件名
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{args.symbol}_{timestamp_str}"
        
        # 4. 绘制回归散点图（图1）
        fe_vals = [r['funding_estimate'] for r in raw_records]
        fr_vals = [r['funding_rate'] for r in raw_records]
        avg_fe = compute_recursive_average(fe_vals)
        
        X = np.array(avg_fe).reshape(-1, 1)
        y = np.array(fr_vals)
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        
        regression_plot_path = os.path.join(FIGURE_DIR, f"{base_filename}_regression.png")
        plt.figure(figsize=(12, 8))
        plt.scatter(avg_fe, fr_vals, alpha=0.6, s=20, label='Data points')
        plt.plot(avg_fe, y_pred, color='red', linewidth=2, label=f'Regression line (R²={r2:.4f})')
        title1 = f'{args.symbol} Funding Rate Regression\nSlope: {slope:.6f}, Intercept: {intercept:.6f}, R²: {r2:.4f}'
        plt.title(title1, fontsize=14, pad=20)
        plt.xlabel('Recursive Average of Funding Estimate (avg_fe)', fontsize=12)
        plt.ylabel('Actual Funding Rate', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(regression_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved regression plot to: {regression_plot_path}")
        
        # 5. 绘制价格对比图（图2）
        datetimes_raw = [r['datetime'] for r in raw_records]
        index_prices_raw = [r['index_price'] for r in raw_records]
        spot_prices_raw = [r['spot_price'] for r in raw_records]
        
        price_plot_path = os.path.join(FIGURE_DIR, f"{base_filename}_prices.png")
        plt.figure(figsize=(14, 8))
        plt.plot(datetimes_raw, index_prices_raw, label='Index Price', linewidth=2, alpha=0.8)
        plt.plot(datetimes_raw, spot_prices_raw, label='Spot Price', linewidth=2, alpha=0.8)
        plt.fill_between(datetimes_raw, index_prices_raw, spot_prices_raw, alpha=0.2, color='gray')
        
        price_diffs = np.array(index_prices_raw) - np.array(spot_prices_raw)
        mean_diff = np.mean(price_diffs)
        max_diff = np.max(np.abs(price_diffs))
        
        title2 = f'{args.symbol} Price Comparison (Last {args.hours} Hours)\nMean Diff: {mean_diff:.4f}, Max |Diff|: {max_diff:.4f}'
        plt.title(title2, fontsize=14, pad=20)
        plt.xlabel('Time (UTC)', fontsize=12)
        plt.ylabel('Price (USDT)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(price_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved price plot to: {price_plot_path}")
        
        # 6. 【新增】绘制三子图（图3）
        if regression_results:
            # 提取回归结果数据
            datetimes_reg = [r['datetime'] for r in regression_results]
            b0_theoretical = [r['b0_theoretical'] for r in regression_results]
            b0_estimate = [r['b0_estimate'] for r in regression_results]
            r2_scores = [r['r2_score'] for r in regression_results]
            
            # 创建三子图
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
            
            # 子图1: b0_theoretical 和 b0_estimate
            ax1.plot(datetimes_reg, b0_theoretical, label='b0_theoretical', marker='o', markersize=3)
            ax1.plot(datetimes_reg, b0_estimate, label='b0_estimate', marker='s', markersize=3)
            ax1.set_ylabel('b0 Value', fontsize=12)
            ax1.set_title(f'{args.symbol} - b0 Analysis', fontsize=14, pad=10)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 子图2: R² score
            ax2.plot(datetimes_reg, r2_scores, label='R² Score', color='purple', marker='^', markersize=3)
            ax2.set_ylabel('R² Score', fontsize=12)
            ax2.set_title('Model Fit Quality (R²)', fontsize=14, pad=10)
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 子图3: 价格对比（使用原始数据的时间范围）
            ax3.plot(datetimes_raw, index_prices_raw, label='Index Price', linewidth=1.5, alpha=0.8)
            ax3.plot(datetimes_raw, spot_prices_raw, label='Spot Price', linewidth=1.5, alpha=0.8)
            ax3.fill_between(datetimes_raw, index_prices_raw, spot_prices_raw, alpha=0.2, color='gray')
            ax3.set_ylabel('Price (USDT)', fontsize=12)
            ax3.set_title('Price Comparison', fontsize=14, pad=10)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlabel('Time (UTC)', fontsize=12)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            combined_plot_path = os.path.join(FIGURE_DIR, f"{base_filename}_combined_analysis.png")
            plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved combined analysis plot to: {combined_plot_path}")
        else:
            print("⚠️ No regression results found for the time window. Skipping combined analysis plot.")
        
        print(f"✅ All plots saved to: {FIGURE_DIR}/")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()