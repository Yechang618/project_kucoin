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
SYMBOL_FILE = 'symbol_2.json'
DATA_DIR = "./kucoin_data/raw_data"
os.makedirs(DATA_DIR, exist_ok=True)

# 全局状态
stop_flag = False
buffers = defaultdict(lambda: {
    'spot_bid': deque(),
    'spot_ask': deque(),
    'swap_bid': deque(),
    'swap_ask': deque(),
    'index_price': deque(),
    'mark_price': deque(),
})
ten_min_buffers = defaultdict(list)
file_write_lock = threading.Lock()

# ==================== 工具函数 ====================
def now_ms():
    return int(time.time() * 1000)

def ms_to_dt(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

def get_10min_window_label(ts_ms):
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    minutes = (dt.hour * 60 + dt.minute) // 10 * 10
    window_start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc) + timedelta(minutes=minutes)
    window_end = window_start + timedelta(minutes=10)
    return window_start, window_end

def dt_to_str(dt):
    return dt.strftime("%Y%m%d_%H%M%S")

def avg_or_nan(dq):
    return sum(dq) / len(dq) if dq else None

# ==================== 聚合逻辑 ====================
def aggregate_on_funding(futures_symbol, funding_rate, funding_ts):
    buf = buffers[futures_symbol]
    row = {
        'timestamp': funding_ts,
        'funding_rate': funding_rate,
        'spot_bid_avg': avg_or_nan(buf['spot_bid']),
        'spot_ask_avg': avg_or_nan(buf['spot_ask']),
        'swap_bid_avg': avg_or_nan(buf['swap_bid']),
        'swap_ask_avg': avg_or_nan(buf['swap_ask']),
        'index_price_avg': avg_or_nan(buf['index_price']),
        'mark_price_avg': avg_or_nan(buf['mark_price']),
    }
    with file_write_lock:
        ten_min_buffers[futures_symbol].append(row)
    for dq in buf.values():
        dq.clear()

# ==================== 分组工具 ====================
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ==================== 加载 symbol ====================
def load_symbols_from_file(symbol_file):
    try:
        with open(symbol_file, 'r', encoding='utf-8') as f:
            symbol_map = json.load(f)
        futures_symbols = []
        spot_symbols = []
        futures_to_spot = {}
        for base_asset, info in symbol_map.items():
            future_sym = info.get('future')
            spot_sym = info.get('spot')
            if future_sym and spot_sym:
                futures_symbols.append(future_sym)
                spot_symbols.append(spot_sym)
                futures_to_spot[future_sym] = spot_sym
        print(f"✅ 从 {symbol_file} 加载 {len(futures_symbols)} 个合约-现货对")
        return futures_symbols, spot_symbols, futures_to_spot
    except FileNotFoundError:
        print(f"❌ 错误: 找不到符号文件 '{symbol_file}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ 错误: 符号文件格式无效: {e}")
        sys.exit(1)

# ==================== Token 获取 ====================
def get_futures_token():
    resp = requests.post("https://api-futures.kucoin.com/api/v1/bullet-public", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]

def get_spot_token():
    resp = requests.post("https://api.kucoin.com/api/v1/bullet-public", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]

# ==================== Futures 监听器（每组） ====================
async def futures_listener_for_group(group_symbols, group_id):
    reconnect_delay = 1
    max_reconnect_delay = 60
    while not stop_flag:
        try:
            token_info = get_futures_token()
            endpoint = token_info["instanceServers"][0]["endpoint"]
            token = token_info["token"]
            ws_url = f"{endpoint}?token={token}"
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                print(f"🔗 [Futures Group {group_id}] Connected")
                # 订阅 instrument（资金费、指数价格等）
                symbols_str = ",".join(group_symbols)
                await ws.send(json.dumps({
                    "id": f"sub_inst_{group_id}",
                    "type": "subscribe",
                    "topic": f"/contract/instrument:{symbols_str}",
                    "privateChannel": False
                }))
                # 订阅 Level2
                for sym in group_symbols:
                    await ws.send(json.dumps({
                        "id": f"sub_swap_{sym}",
                        "type": "subscribe",
                        "topic": f"/contractMarket/level2Depth5:{sym}",
                        "privateChannel": False
                    }))
                print(f"✅ [Futures Group {group_id}] Subscribed to {len(group_symbols)} symbols")
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
                        if symbol not in group_symbols:
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
            print(f"⚠️ [Futures Group {group_id}] Connection lost: {e}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

# ==================== Spot 监听器（每组） ====================
async def spot_listener_for_group(group_symbols, group_id, futures_to_spot):
    reconnect_delay = 1
    max_reconnect_delay = 60
    while not stop_flag:
        try:
            token_info = get_spot_token()
            endpoint = token_info["instanceServers"][0]["endpoint"]
            token = token_info["token"]
            ws_url = f"{endpoint}?token={token}"
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                print(f"🔗 [Spot Group {group_id}] Connected")
                for sym in group_symbols:
                    await ws.send(json.dumps({
                        "id": f"sub_spot_{sym}",
                        "type": "subscribe",
                        "topic": f"/spotMarket/level2Depth5:{sym}",
                        "privateChannel": False
                    }))
                print(f"✅ [Spot Group {group_id}] Subscribed to {len(group_symbols)} symbols")
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
                        if spot_sym not in group_symbols:
                            continue
                        # 找到对应的 futures symbol
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
            print(f"⚠️ [Spot Group {group_id}] Connection lost: {e}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

# ==================== 文件写入线程 ====================
def file_writer():
    while not stop_flag:
        now = now_ms()
        window_start, window_end = get_10min_window_label(now)
        # 在窗口结束前 1.5 秒写入
        if window_end.timestamp() - time.time() <= 1.5:
            with file_write_lock:
                for sym, rows in ten_min_buffers.items():
                    if rows:
                        filename = f"{sym}_{dt_to_str(window_start)}_{dt_to_str(window_end)}.json"
                        filepath = os.path.join(DATA_DIR, filename)
                        with open(filepath, 'w') as f:
                            json.dump(rows, f, indent=2)
                        print(f"📁 Saved {len(rows)} rows for {sym} to {filename}")
                        ten_min_buffers[sym].clear()
            time.sleep(2)  # 避免重复写
        time.sleep(0.5)

# ==================== 信号处理 ====================
def signal_handler(sig, frame):
    global stop_flag
    print("\n\n🛑 Shutdown signal received. Finalizing...")
    stop_flag = True
    time.sleep(2)
    # Final save
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

# ==================== 主程序 ====================
async def main():
    global stop_flag
    stop_flag = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 加载 symbol
    futures_symbols, spot_symbols, futures_to_spot = load_symbols_from_file(SYMBOL_FILE)

    print("🚀 KuCoin Collector (Multi-Connection Mode)")
    print(f"📊 Futures: {len(futures_symbols)} | Spot: {len(spot_symbols)}")
    print(f"📁 Output: {os.path.abspath(DATA_DIR)} (JSON, ~10 points/file)")

    # 启动写入线程
    writer_thread = threading.Thread(target=file_writer, daemon=True)
    writer_thread.start()

    # 分组（每组 20 个）
    futures_groups = list(chunk_list(futures_symbols, 20))
    spot_groups = list(chunk_list(spot_symbols, 20))

    # 创建任务
    futures_tasks = [
        futures_listener_for_group(group, i)
        for i, group in enumerate(futures_groups)
    ]
    spot_tasks = [
        spot_listener_for_group(group, i, futures_to_spot)
        for i, group in enumerate(spot_groups)
    ]

    # 并发运行
    await asyncio.gather(*futures_tasks, *spot_tasks)

if __name__ == "__main__":
    asyncio.run(main())