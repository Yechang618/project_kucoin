import asyncio
import json
import websockets
import requests
import threading
import time
import os
import signal
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta

# ==================== 配置 ====================
original_symbols = [
    "AGTUSDT", "DASHUSDT", "XMRUSDT", "BNBUSDT", "WCTUSDT", "INJUSDT", 
    "KAITOUSDT", "HAEDALUSDT", "XPLUSDT", "DOTUSDT", "ONDOUSDT",
    "ZECUSDT", "SUIUSDT", "GIGGLEUSDT"
]

spot_symbols = [s[:-4] + "-USDT" for s in original_symbols if s.endswith("USDT")]
futures_symbols = [s + "M" for s in original_symbols]

def validate_spot_symbols(symbols):
    try:
        resp = requests.get("https://api.kucoin.com/api/v1/symbols", timeout=10)
        if resp.status_code == 200:
            valid = {item["symbol"] for item in resp.json().get("data", [])}
            valid_symbols = [s for s in symbols if s in valid]
            invalid = set(symbols) - set(valid_symbols)
            if invalid:
                print(f"⚠️ Skipping invalid spot symbols: {invalid}")
            return valid_symbols or symbols
    except Exception as e:
        print(f"⚠️ Symbol validation failed: {e}")
    return symbols

spot_symbols = validate_spot_symbols(spot_symbols)
futures_to_spot = {fut: spot for fut, spot in zip(futures_symbols, spot_symbols)}

DATA_DIR = "./kucoin_data/raw_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==================== 全局状态 ====================
stop_flag = False
buffers = defaultdict(lambda: {
    'spot_bid': deque(),
    'spot_ask': deque(),
    'swap_bid': deque(),
    'swap_ask': deque(),
    'index_price': deque(),
    'mark_price': deque(),
})

# 每个 symbol 的 10 分钟聚合窗口缓冲区
# ten_min_buffers[symbol] = list of aggregated rows (each triggered by funding)
ten_min_buffers = defaultdict(list)

file_write_lock = threading.Lock()

# ==================== 工具 ====================
def now_ms():
    return int(time.time() * 1000)

def ms_to_dt(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

# def get_10min_window_label(ts_ms):
#     """返回当前时间所属的10分钟窗口起止时间（UTC）"""
#     dt = ms_to_dt(ts_ms)
#     minute_floor = (dt.minute // 10) * 10
#     window_start = dt.replace(minute=minute_floor, second=0, microsecond=0)
#     window_end = window_start.replace(minute=minute_floor + 10)
#     return window_start, window_end

# from datetime import datetime, timezone, timedelta

def get_10min_window_label(ts_ms):
    """返回当前时间所属的10分钟窗口起止时间(UTC)"""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    # 计算当前时间向下取整到最近的10分钟（例如 14:27 → 14:20）
    minutes = (dt.hour * 60 + dt.minute) // 10 * 10
    window_start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc) + timedelta(minutes=minutes)
    window_end = window_start + timedelta(minutes=10)
    return window_start, window_end

def dt_to_str(dt):
    return dt.strftime("%Y%m%d_%H%M%S")

def avg_or_nan(dq):
    return sum(dq) / len(dq) if dq else None  # JSON 不支持 NaN，改用 null

# ==================== 聚合逻辑 ====================
def aggregate_on_funding(futures_symbol, funding_rate, funding_ts):
    buf = buffers[futures_symbol]
    row = {
        'timestamp': funding_ts,  # funding rate timestamp (ms)
        'funding_rate': funding_rate,
        'spot_bid_avg': avg_or_nan(buf['spot_bid']),
        'spot_ask_avg': avg_or_nan(buf['spot_ask']),
        'swap_bid_avg': avg_or_nan(buf['swap_bid']),
        'swap_ask_avg': avg_or_nan(buf['swap_ask']),
        'index_price_avg': avg_or_nan(buf['index_price']),
        'mark_price_avg': avg_or_nan(buf['mark_price']),
    }

    # 加入 10 分钟窗口缓冲
    with file_write_lock:
        ten_min_buffers[futures_symbol].append(row)

    # 清空瞬时缓冲
    for dq in buf.values():
        dq.clear()

# ==================== 文件写入线程（每10分钟触发） ====================
def file_writer():
    last_window_start = None
    while not stop_flag:
        now = now_ms()
        window_start, window_end = get_10min_window_label(now)
        current_window_str = dt_to_str(window_start)

        # 每整10分钟窗口结束前1~2秒写入（避免跨窗口）
        if window_end.timestamp() - time.time() <= 1.5:
            with file_write_lock:
                for sym, rows in ten_min_buffers.items():
                    if rows:
                        filename = f"{sym}_{dt_to_str(window_start)}_{dt_to_str(window_end)}.json"
                        filepath = os.path.join(DATA_DIR, filename)
                        # 写入 JSON
                        with open(filepath, 'w') as f:
                            json.dump(rows, f, indent=2)
                        print(f"📁 Saved {len(rows)} rows for {sym} to {filename}")
                        ten_min_buffers[sym].clear()  # 清空该窗口数据

            # 避免重复写（等待进入新窗口）
            time.sleep(2)

        time.sleep(0.5)

# ==================== WebSocket 监听器（保持不变） ====================
# [此处 futures_listener() 和 spot_listener() 与上一版完全相同]
# 为节省篇幅，此处复用逻辑，仅需确保调用的是上方定义的 `aggregate_on_funding`
# 实际代码中应保留完整监听器（见上一版），此处略去重复内容

async def futures_listener():
    reconnect_delay = 1
    max_reconnect_delay = 60

    while not stop_flag:
        try:
            token_info = get_futures_token()
            endpoint = token_info["instanceServers"][0]["endpoint"]
            token = token_info["token"]
            ws_url = f"{endpoint}?token={token}"

            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                print("🔗 [Futures] Connected")
                for i in range(0, len(futures_symbols), 50):
                    batch = futures_symbols[i:i+50]
                    symbols_str = ",".join(batch)
                    await ws.send(json.dumps({
                        "id": f"sub_inst_{i}",
                        "type": "subscribe",
                        "topic": f"/contract/instrument:{symbols_str}",
                        "privateChannel": False
                    }))
                for sym in futures_symbols:
                    await ws.send(json.dumps({
                        "id": f"sub_swap_{sym}",
                        "type": "subscribe",
                        "topic": f"/contractMarket/level2Depth5:{sym}",
                        "privateChannel": False
                    }))
                print(f"✅ [Futures] Subscribed to {len(futures_symbols)} symbols")
                reconnect_delay = 1
                last_ping = asyncio.get_event_loop().time()

                while not stop_flag:
                    now = asyncio.get_event_loop().time()
                    if now - last_ping >= 25:
                        await ws.send(json.dumps({"id": "ping", "type": "ping"}))
                        last_ping = now
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                        if msg.get("type") == "pong":
                            continue
                        topic = msg.get("topic", "")
                        if ":" not in topic:
                            continue
                        symbol = topic.split(":")[1]
                        if symbol not in futures_symbols:
                            continue
                        data = msg.get("data", {})
                        subject = msg.get("subject")
                        if subject == "mark.index.price":
                            buffers[symbol]['index_price'].append(float(data['indexPrice']))
                            buffers[symbol]['mark_price'].append(float(data['markPrice']))
                        elif subject == "funding.rate":
                            funding_rate = float(data['fundingRate']) if data.get('fundingRate') else None
                            ts = int(data['timestamp'])
                            if funding_rate is not None:
                                aggregate_on_funding(symbol, funding_rate, ts)
                        elif subject == "level2":
                            bids = data.get("bids", [])
                            asks = data.get("asks", [])
                            if bids and asks:
                                buffers[symbol]['swap_bid'].append(float(bids[0][0]))
                                buffers[symbol]['swap_ask'].append(float(asks[0][0]))
                    except asyncio.TimeoutError:
                        continue
        except Exception as e:
            if stop_flag:
                break
            print(f"⚠️ [Futures] Connection lost: {e}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

async def spot_listener():
    reconnect_delay = 1
    max_reconnect_delay = 60

    while not stop_flag:
        try:
            token_info = get_spot_token()
            endpoint = token_info["instanceServers"][0]["endpoint"]
            token = token_info["token"]
            ws_url = f"{endpoint}?token={token}"

            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                print("🔗 [Spot] Connected")
                for sym in spot_symbols:
                    await ws.send(json.dumps({
                        "id": f"sub_spot_{sym}",
                        "type": "subscribe",
                        "topic": f"/spotMarket/level2Depth5:{sym}",
                        "privateChannel": False
                    }))
                print(f"✅ [Spot] Subscribed to {len(spot_symbols)} symbols")
                reconnect_delay = 1
                last_ping = asyncio.get_event_loop().time()

                while not stop_flag:
                    now = asyncio.get_event_loop().time()
                    if now - last_ping >= 25:
                        await ws.send(json.dumps({"id": "ping", "type": "ping"}))
                        last_ping = now
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                        if msg.get("type") == "pong":
                            continue
                        topic = msg.get("topic", "")
                        if ":" not in topic:
                            continue
                        spot_sym = topic.split(":")[1]
                        if spot_sym not in spot_symbols:
                            continue
                        futures_sym = None
                        for fut, spot in futures_to_spot.items():
                            if spot == spot_sym:
                                futures_sym = fut
                                break
                        if not futures_sym:
                            continue
                        data = msg.get("data", {})
                        bids = data.get("bids", [])
                        asks = data.get("asks", [])
                        if bids and asks:
                            buffers[futures_sym]['spot_bid'].append(float(bids[0][0]))
                            buffers[futures_sym]['spot_ask'].append(float(asks[0][0]))
                    except asyncio.TimeoutError:
                        continue
        except Exception as e:
            if stop_flag:
                break
            print(f"⚠️ [Spot] Connection lost: {e}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

def get_futures_token():
    resp = requests.post("https://api-futures.kucoin.com/api/v1/bullet-public", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]

def get_spot_token():
    resp = requests.post("https://api.kucoin.com/api/v1/bullet-public", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]

# ==================== 信号与主程序 ====================
def signal_handler(sig, frame):
    global stop_flag
    print("\n\n🛑 Shutdown signal received. Finalizing...")
    stop_flag = True
    time.sleep(2)
    # 尝试保存剩余数据（归属到当前窗口）
    now = now_ms()
    window_start, window_end = get_10min_window_label(now)
    with file_write_lock:
        for sym, rows in ten_min_buffers.items():
            if rows:
                filename = f"{sym}_{dt_to_str(window_start)}_{dt_to_str(window_end)}_final.json"
                filepath = os.path.join(DATA_DIR, filename)
                with open(filepath, 'w') as f:
                    json.dump(rows, f, indent=2)
                print(f"💾 Final save: {len(rows)} rows for {sym}")
    sys.exit(0)

async def main():
    global stop_flag
    stop_flag = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🚀 KuCoin Collector (JSON, 10-min windows)")
    print(f"📊 Futures: {len(futures_symbols)} | Spot: {len(spot_symbols)}")
    print(f"📁 Output: {os.path.abspath(DATA_DIR)} (JSON, ~10 points/file)")

    writer_thread = threading.Thread(target=file_writer, daemon=True)
    writer_thread.start()

    await asyncio.gather(futures_listener(), spot_listener())

if __name__ == "__main__":
    asyncio.run(main())