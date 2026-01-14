import os
import sys
import json
import glob
import re
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from redis import Redis
import yaml
from util import DecimalEncoder

import message_bot as mb

# ==================== 配置 ====================
DATA_DIR = "./kucoin_data/raw_data"
OUTPUT_DIR = "./kucoin_data/regression_results"
config_path = "./config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 移除 SETTLEMENT_INTERVALS（不再需要）

def main(b0_estimates, b0_max = 4, mode=None):
    use_all_data = (mode == '-all')

    if use_all_data:
        print("🌍 Mode: -all → Using ALL historical data across all dates and intervals.")
        relevant_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
        print(f"📊 Found {len(relevant_files)} JSON files (all).")
        # 时间窗口：全部历史
        window_start = None
    else:
        # 【关键修改】计算当前时刻往前8小时的时间点
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=8)
        print(f"🕒 Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Processing rolling 8-hour window: from {window_start.strftime('%Y-%m-%d %H:%M:%S')} to now")

        # 加载所有 JSON 文件（不再按 interval 过滤）
        relevant_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
        print(f"📊 Found {len(relevant_files)} JSON files (scanning all for 8h window).")

    # Collect raw data per symbol
    symbol_raw = defaultdict(list)

    for filepath in relevant_files:
        filename = os.path.basename(filepath)
        symbol = filename.split('_')[0]
        if not symbol:
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                records = json.load(f)
            if not isinstance(records, list):
                continue

            for rec in records:
                ts = rec.get("timestamp")
                if not isinstance(ts, int) or ts <= 0:
                    continue
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

                # 【关键修改】时间窗口过滤
                if not use_all_data:
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

                if symbol.startswith("10000"):
                    spot_bid = spot_bid*1e4
                    spot_ask = spot_ask*1e4
                elif symbol.startswith("1000"):
                    spot_bid = spot_bid*1e3
                    spot_ask = spot_ask*1e3
                    
                try:
                    mid_swap = (float(swap_bid) + float(swap_ask)) / 2.0
                    index = float(index_price)
                    if index == 0:
                        continue
                    funding_estimate = mid_swap / index - 1.0

                    # 【关键修改】不再需要 t_val，直接用 timestamp
                    symbol_raw[symbol].append((
                        ts, funding_estimate, float(funding_rate),
                        ts, float(spot_bid), float(spot_ask), float(index_price)
                    ))
                except (ValueError, TypeError, ZeroDivisionError, OverflowError):
                    continue
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
            continue

    # Process each symbol
    results = []
    for symbol, data in symbol_raw.items():
        if len(data) < 2:
            continue

        # 按时间戳排序
        data.sort(key=lambda x: x[0])
        ts_vals = [d[0] for d in data]
        fe_vals = [d[1] for d in data]
        fr_vals = [d[2] for d in data]
        spot_bids = [d[4] for d in data]
        spot_asks = [d[5] for d in data]
        index_prices = [d[6] for d in data]

        # Compute recursive average of funding_estimate
        avg_fe = []
        for i, fe in enumerate(fe_vals):
            n = i + 1
            if i == 0:
                avg_fe.append(fe)
            else:
                avg_fe.append(((n - 1) * avg_fe[-1] + fe) / n)

        # Linear regression: funding_rate = a + b * avg_fe
        X = np.array(avg_fe).reshape(-1, 1)
        y = np.array(fr_vals)

        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        b = float(model.coef_[0])
        a = float(model.intercept_)
        last_ts = int(max(ts_vals))

        # Calculate b0
        spot_price = (spot_bids[-1] + spot_asks[-1]) / 2.0
        if abs(b) > 1e-8:
            b0 = ((index_prices[-1] - spot_price) - (a / b) * index_prices[-1])/spot_price
        else:
            b0 = 0.0

        b0_theoretical = ((index_prices[-1] - spot_price) - 1e-4 * index_prices[-1]) / spot_price
        if r2 >= 0.6:
            b0_estimate = b0_theoretical*(1-r2) + b0*r2
        else:
            b0_estimate = b0_theoretical

        b0_estimates[symbol].append(b0_estimate)
        if len(b0_estimates[symbol]) > b0_max:
            b0_estimates[symbol].pop(0)
        b0_estimate_ema = sum(b0_estimates[symbol]) / len(b0_estimates[symbol])
        results.append({
            "symbol": symbol,
            "b0_estimate": b0_estimate,
            "b0_estimate_ema": b0_estimate_ema,
            "b0_theoretical": b0_theoretical,
            "spot_price": spot_price,
            "b0": b0,
            "b": b,
            "a": a,
            "r2_score": r2,
            "last_timestamp": last_ts
        })

    # Save results
    if use_all_data:
        output_filename = f"regression_results_ALL_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    else:
        # 【关键修改】输出文件名改为 rolling_8h
        output_filename = f"regression_results_rolling_8h_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved {len(results)} regression results to {output_path}")
    print("🎉 Done.")

    # Feishu bot (保持不变)
    url_report = "https://open.feishu.cn/open-apis/bot/v2/hook/c5634e62-18fa-45e7-9af3-a2dfea7be4eb"
    my_url = "https://open.feishu.cn/open-apis/bot/v2/hook/4519b97c-d166-430f-87bc-13a6b8d35dac"
    my_bot = mb.Bot(my_url)
    report_bot = mb.Bot(url_report)
    msg_detail = ''
    msg = ''
    dic = {}
    for res in results:
        dic[res['symbol']] = res['b0_estimate_ema']
        msg += f"Symbol: {res['symbol']}, b0_estimate: {res['b0_estimate_ema']:.6f}\n"
        msg_detail += f"Symbol: {res['symbol']}, b0_estimate_ema: {res['b0_estimate_ema']:.6f}, N_b0: {len(b0_estimates[res['symbol']])},\
                        b0_estimate: {res['b0_estimate']:.6f}, b0_theoretical: {res['b0_theoretical']:.6f}, \
                        b0: {res['b0']:.6f}, spot_price: {res['spot_price']:.6f}, b: {res['b']:.6f}, \
                        a: {res['a']:.6f}, R²: {res['r2_score']:.2f}\n"
    my_bot.text(msg_detail)
    if mode != '-test':  # 非 -test 模式才发报告
        report_bot.text(msg)
        # r = Redis(host=config['redisUrl'], db=1, password=config['redisPass'])
        # signals_str = json.dumps(dic, cls=DecimalEncoder)        
        # r.publish(f'kucoin_zero_fundingrate', signals_str)
    return b0_estimates

# ==================== 主循环 ====================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None

    print("🔄 Regression analysis script started. Press Ctrl+C to stop.")
    b0_estimates = defaultdict(list)
    try:
        while True:
            next_run = datetime.now() + timedelta(minutes=30)
            print(f"\n🕒 Next run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

            b0_estimates = main(b0_estimates, b0_max = 8, mode=mode)

            time.sleep(1800)

    except KeyboardInterrupt:
        print("\n🛑 Script stopped by user.")
        sys.exit(0)