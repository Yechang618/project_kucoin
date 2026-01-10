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
FIGURE_DIR = "./figure"
os.makedirs(FIGURE_DIR, exist_ok=True)

def load_symbol_data(symbol, hours=8):
    """
    加载指定 symbol 过去 N 小时的原始数据
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=hours)
    
    print(f"🔍 Loading data for {symbol} from {window_start.strftime('%Y-%m-%d %H:%M:%S')} to now")
    
    # 找到所有包含该 symbol 的 JSON 文件
    pattern = os.path.join(DATA_DIR, f"{symbol}_*.json")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No files found for symbol: {symbol}")
    
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
                    
                    # 提取必要字段
                    swap_bid = rec.get("swap_bid_avg")
                    swap_ask = rec.get("swap_ask_avg")
                    spot_bid = rec.get("spot_bid_avg")
                    spot_ask = rec.get("spot_ask_avg")
                    index_price = rec.get("index_price_avg")
                    funding_rate = rec.get("funding_rate")
                    
                    if None in (swap_bid, swap_ask, index_price, funding_rate):
                        continue
                    
                    # 处理特殊 symbol（如 1000SHIB）
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
    
    # 按时间排序
    records.sort(key=lambda x: x['timestamp'])
    return records

def compute_recursive_average(values):
    """计算递归平均值"""
    avg_values = []
    for i, val in enumerate(values):
        if i == 0:
            avg_values.append(val)
        else:
            n = i + 1
            avg_values.append(((n - 1) * avg_values[-1] + val) / n)
    return avg_values

def main():
    parser = argparse.ArgumentParser(description='Plot funding rate regression and price comparison')
    parser.add_argument('symbol', type=str, help='Symbol name (e.g., BTC)')
    parser.add_argument('--hours', type=int, default=8, help='Hours of historical data to load (default: 8)')
    args = parser.parse_args()
    
    try:
        # 1. 加载数据
        records = load_symbol_data(args.symbol, args.hours)
        print(f"📊 Loaded {len(records)} records for {args.symbol}")
        
        if len(records) < 2:
            print("❌ Not enough data points (< 2) for analysis")
            return
        
        # 2. 准备回归数据
        fe_vals = [r['funding_estimate'] for r in records]
        fr_vals = [r['funding_rate'] for r in records]
        avg_fe = compute_recursive_average(fe_vals)
        
        # 3. 线性回归
        X = np.array(avg_fe).reshape(-1, 1)
        y = np.array(fr_vals)
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        
        print(f"📈 Regression results:")
        print(f"   Slope (b): {slope:.6f}")
        print(f"   Intercept (a): {intercept:.6f}")
        print(f"   R²: {r2:.4f}")
        
        # 4. 创建 figure 目录中的文件名
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{args.symbol}_{timestamp_str}"
        regression_plot_path = os.path.join(FIGURE_DIR, f"{base_filename}_regression.png")
        price_plot_path = os.path.join(FIGURE_DIR, f"{base_filename}_prices.png")
        
        # 5. 绘制回归图
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
        
        # 6. 绘制价格对比图
        datetimes = [r['datetime'] for r in records]
        index_prices = [r['index_price'] for r in records]
        spot_prices = [r['spot_price'] for r in records]
        
        plt.figure(figsize=(14, 8))
        plt.plot(datetimes, index_prices, label='Index Price', linewidth=2, alpha=0.8)
        plt.plot(datetimes, spot_prices, label='Spot Price', linewidth=2, alpha=0.8)
        plt.fill_between(datetimes, index_prices, spot_prices, alpha=0.2, color='gray')
        
        # 计算价差统计
        price_diffs = np.array(index_prices) - np.array(spot_prices)
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
        
        print(f"✅ Both plots saved to: {FIGURE_DIR}/")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()