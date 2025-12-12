import os
import json
import glob
import re
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

import message_bot as mb
# ==================== 配置 ====================
DATA_DIR = "./kucoin_data/raw_data"
OUTPUT_DIR = "./kucoin_data/regression_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Settlement intervals: (start_hour, start_minute, end_hour, end_minute)
SETTLEMENT_INTERVALS = [
    (0, 1, 8, 0),   # Interval 0: 00:01 – 08:00
    (8, 1, 16, 0),  # Interval 1: 08:01 – 16:00
    (16, 1, 24, 0)  # Interval 2: 16:01 – 24:00
]

def get_current_interval_info():
    now = datetime.now(timezone.utc)
    # Handle 00:00 edge case
    if now.hour == 0 and now.minute == 0:
        interval_id = 2
        date_str = (now - timedelta(days=1)).strftime("%Y%m%d")
        interval_start = (now - timedelta(days=1)).replace(
            hour=16, minute=1, second=0, microsecond=0, tzinfo=timezone.utc
        )
    else:
        if 0 <= now.hour < 8:
            interval_id = 0
            date_str = now.strftime("%Y%m%d")
            interval_start = now.replace(hour=0, minute=1, second=0, microsecond=0, tzinfo=timezone.utc)
        elif 8 <= now.hour < 16:
            interval_id = 1
            date_str = now.strftime("%Y%m%d")
            interval_start = now.replace(hour=8, minute=1, second=0, microsecond=0, tzinfo=timezone.utc)
        else:
            interval_id = 2
            date_str = now.strftime("%Y%m%d")
            interval_start = now.replace(hour=16, minute=1, second=0, microsecond=0, tzinfo=timezone.utc)
    return interval_id, date_str, interval_start

def time_in_interval(dt, interval_id):
    start_h, start_m, end_h, end_m = SETTLEMENT_INTERVALS[interval_id]
    total_min = dt.hour * 60 + dt.minute
    start_total = start_h * 60 + start_m
    end_total = end_h * 60 + end_m
    return start_total <= total_min < end_total

def get_t_minutes(dt, interval_start):
    """Return t = minutes since interval start (00:01 → t=1)"""
    delta = dt - interval_start
    return int(delta.total_seconds() // 60) + 1

def file_window_overlaps_interval(filename, target_interval, target_date):
    match = re.match(r"[A-Z0-9]+_(\d{8})_(\d{6})_(\d{8})_(\d{6})\.json", filename)
    if not match:
        return False
    start_date, start_time, end_date, end_time = match.groups()
    if start_date != target_date and end_date != target_date:
        return False
    start_dt = datetime.strptime(start_date + start_time, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date + end_time, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    current = start_dt
    while current < end_dt:
        if time_in_interval(current, target_interval):
            return True
        current += timedelta(minutes=10)
    return False

# ==================== 主逻辑 ====================
def main():
    interval_id, date_str, interval_start = get_current_interval_info()
    interval_names = ["00:01-08:00", "08:01-16:00", "16:01-24:00"]
    print(f"🕒 Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Processing interval: {interval_names[interval_id]} on {date_str}")

    # Find relevant files
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    relevant_files = [
        f for f in json_files
        if file_window_overlaps_interval(os.path.basename(f), interval_id, date_str)
    ]
    print(f"📊 Found {len(relevant_files)} relevant JSON files.")

    # Collect raw data per symbol
    symbol_raw = defaultdict(list)  # list of (t, funding_estimate, funding_rate, timestamp)

    for filepath in relevant_files:
        filename = os.path.basename(filepath)
        symbol = filename.split('_')[0]
        try:
            with open(filepath, 'r') as f:
                records = json.load(f)
            if not isinstance(records, list):
                continue
            for rec in records:
                ts = rec.get("timestamp")
                if not isinstance(ts, int):
                    continue
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                if dt.strftime("%Y%m%d") != date_str:
                    continue
                if not time_in_interval(dt, interval_id):
                    continue

                swap_bid = rec.get("swap_bid_avg")
                swap_ask = rec.get("swap_ask_avg")
                spot_bid = rec.get("spot_bid_avg")
                spot_ask = rec.get("spot_ask_avg")                
                index_price = rec.get("index_price_avg")
                funding_rate = rec.get("funding_rate")

                if None in (swap_bid, swap_ask, index_price, funding_rate):
                    continue

                try:
                    mid_swap = (float(swap_bid) + float(swap_ask)) / 2.0
                    index = float(index_price)
                    if index == 0:
                        continue
                    funding_estimate = mid_swap / index - 1.0
                    t = get_t_minutes(dt, interval_start)
                    symbol_raw[symbol].append((t, funding_estimate, float(funding_rate), ts, spot_bid, spot_ask, index_price))
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")

    # Process each symbol
    results = []
    b0s = []
    for symbol, data in symbol_raw.items():
        if len(data) < 2:
            continue

        # Sort by t (time)
        data.sort(key=lambda x: x[0])
        t_vals = [d[0] for d in data]
        fe_vals = [d[1] for d in data]
        fr_vals = [d[2] for d in data]
        ts_vals = [d[3] for d in data]
        spot_bids = [d[4] for d in data]
        spot_asks = [d[5] for d in data]    
        index_prices = [d[6] for d in data]

        # Compute recursive average: avg_fe[t] = mean(funding_estimate from first to current)
        avg_fe = []
        running_sum = 0.0
        for i, (t, fe) in enumerate(zip(t_vals, fe_vals)):
            # t should equal i+1 if data is dense, but we use actual t
            # However, recursive formula uses count = index+1, not t
            # But per your definition: "t1, t2, ..., tN" are the actual minute indices
            # However, the recurrence uses t as the count → we assume points are in order and use position
            # But to be safe: we use the order index (n = i+1) for averaging
            n = i + 1
            if n == 1:
                current_avg = fe
            else:
                # Use previous avg
                current_avg = ((n - 1) * avg_fe[-1] + fe) / n
            avg_fe.append(current_avg)

        # Now X = avg_fe, y = fr_vals
        X = np.array(avg_fe).reshape(-1, 1)
        y = np.array(fr_vals)

        # Linear regression
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        b = float(model.coef_[0])
        a = float(model.intercept_)
        last_ts = int(max(ts_vals))
        b0 = index_prices[-1] - (spot_bids[-1] + spot_asks[-1]) / 2.0 - a/b*index_prices[-1] if b != 0 else 0.0
        b0s.append({
            "symbol": symbol,
            "b0": b0,
            "r2_score": r2,
            "last_timestamp": last_ts
        })
        results.append({
            "symbol": symbol,
            "b0": b0,
            "b": b,
            "a": a,
            "r2_score": r2,
            "last_timestamp": last_ts
        })

    # Save result
    output_filename = f"regression_results_interval{interval_id}_{date_str}.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved {len(results)} regression results to {output_path}")
    print("🎉 Done.")

    url_report = "https://open.feishu.cn/open-apis/bot/v2/hook/c5634e62-18fa-45e7-9af3-a2dfea7be4eb"
    my_url = "https://open.feishu.cn/open-apis/bot/v2/hook/4519b97c-d166-430f-87bc-13a6b8d35dac" # my own one
    my_bot = mb.Bot(my_url)
    report_bot = mb.Bot(url_report) 
    msg = ''
    msg_report = ''
    for res in results:
        msg += f"Symbol: {res['symbol']}, b0: {res['b0']:.6f}, b: {res['b']:.6f}, a: {res['a']:.6f}, R²: {res['r2_score']:.2f}\n"
        msg_report += f"Symbol: {res['symbol']}, b0: {res['b0']:.6f},  R²: {res['r2_score']:.2f}\n"
    my_bot.text(msg)
    # report_bot.text(msg_report)
if __name__ == "__main__":
    main()