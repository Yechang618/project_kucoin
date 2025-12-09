import asyncio
import json
import os
import signal
import sys
import datetime
import requests
import websockets
from collections import defaultdict

# ------------------------
# 配置
# ------------------------
spot_symbols = [
    "XBTUSDT", "SOLUSDT", "PNUTUSDT", "XRPUSDT", "FETUSDT", "UNIUSDT",
    "COMPUSDT", "THEUSDT", "AVAXUSDT", "LTCUSDT", "ETCUSDT", "FORMUSDT",
    "TONUSDT", "HFTUSDT", "DOTUSDT", "CHESSUSDT", "ETHUSDT", "BNBUSDT",
    "TRXUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "XLMUSDT", "BCHUSDT",
    "HBARUSDT", "ZECUSDT", "AAVEUSDT", "ENAUSDT", "NEARUSDT", "ONDOUSDT"
]
# spot_symbols = [
#     "AGTUSDT", "DASHUSDT", "XMRUSDT", "BNBUSDT", "WCTUSDT","INJUSDT", 
#     "KAITOUSDT", "HAEDALUSDT", "XPLUSDT", "DOTUSDT", "ONDOUSDT",
#     "ZECUSDT", "SUIUSDT", "GIGGLEUSDT", 
# ]
futures_symbols = [s + "M" for s in spot_symbols]

save_dir = "kucoin_data_combined_2"
os.makedirs(save_dir, exist_ok=True)
data_buffers = defaultdict(list)
stop_flag = False

# ------------------------
# 获取 Token（修复 URL 末尾空格）
# ------------------------
def get_futures_token():
    url = "https://api-futures.kucoin.com/api/v1/bullet-public"  # 移除末尾空格
    resp = requests.post(url, timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]

def get_spot_token():
    url = "https://api.kucoin.com/api/v1/bullet-public"  # 移除末尾空格
    resp = requests.post(url, timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]

# ------------------------
# 保存数据
# ------------------------
def save_data_sync(symbol):
    if not data_buffers[symbol]:
        return
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M")
    filename = f"{date_str}_{time_str}_{symbol}.json"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data_buffers[symbol], f, indent=2)
    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Saved {len(data_buffers[symbol])} records for {symbol}")
    data_buffers[symbol].clear()

# ------------------------
# 带重连的 Futures 监听器
# ------------------------
async def futures_listener_with_reconnect():
    global stop_flag
    reconnect_delay = 1  # 初始重连延迟（秒）
    max_reconnect_delay = 60  # 最大重连延迟

    while not stop_flag:
        try:
            token_info = get_futures_token()
            endpoint = token_info["instanceServers"][0]["endpoint"]
            token = token_info["token"]
            ws_url = f"{endpoint}?token={token}"

            async with websockets.connect(ws_url) as ws:
                print("🔗 [Futures] Connected to KuCoin WebSocket")
                
                # 订阅数据
                batch_size = 50
                for i in range(0, len(futures_symbols), batch_size):
                    batch = futures_symbols[i:i+batch_size]
                    symbols_str = ",".join(batch)
                    await ws.send(json.dumps({
                        "id": f"fut_index_{i}",
                        "type": "subscribe",
                        "topic": f"/contract/instrument:{symbols_str}",
                        "privateChannel": False
                    }))

                for sym in futures_symbols:
                    await ws.send(json.dumps({
                        "id": f"fut_ticker_{sym}",
                        "type": "subscribe",
                        "topic": f"/contractMarket/ticker:{sym}",
                        "privateChannel": False
                    }))
                    await ws.send(json.dumps({
                        "id": f"fut_funding_{sym}",
                        "type": "subscribe",
                        "topic": f"/contractMarket/fundingRate:{sym}",
                        "privateChannel": False
                    }))

                print(f"✅ [Futures] Subscribed to {len(futures_symbols)} symbols")
                reconnect_delay = 1  # 重置重连延迟

                last_ping = asyncio.get_event_loop().time()
                while not stop_flag:
                    # 心跳
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
                        if msg.get("subject") == "mark.index.price":
                            data_buffers[symbol].append({
                                "timestamp": data.get("timestamp"),
                                "index_price": float(data["indexPrice"]) if data.get("indexPrice") else None,
                                "mark_price": float(data["markPrice"]) if data.get("markPrice") else None,
                                "type": "futures_index_mark"
                            })
                        elif msg.get("subject") == "ticker":
                            data_buffers[symbol].append({
                                "timestamp": data.get("ts"),
                                "best_bid": float(data["bestBidPrice"]) if data.get("bestBidPrice") else None,
                                "best_ask": float(data["bestAskPrice"]) if data.get("bestAskPrice") else None,
                                "last_price": float(data["lastPrice"]) if data.get("lastPrice") else None,
                                "type": "futures_ticker"
                            })
                        elif msg.get("subject") == "funding.rate":
                            try:
                                funding_rate = float(data["fundingRate"]) if data.get("fundingRate") else None
                                next_funding = int(data["nextFundingTime"]) if data.get("nextFundingTime") else None
                            except (ValueError, TypeError):
                                funding_rate = next_funding = None
                            data_buffers[symbol].append({
                                "timestamp": data.get("timestamp"),
                                "funding_rate": funding_rate,
                                "next_funding_time": next_funding,
                                "type": "futures_funding"
                            })

                    except asyncio.TimeoutError:
                        continue

        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.InvalidStatusCode,
                requests.RequestException, OSError) as e:
            if stop_flag:
                break
            print(f"⚠️ [Futures] Connection lost: {e}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)  # 指数退避

# ------------------------
# 带重连的 Spot 监听器
# ------------------------
async def spot_listener_with_reconnect():
    global stop_flag
    reconnect_delay = 1
    max_reconnect_delay = 60

    while not stop_flag:
        try:
            token_info = get_spot_token()
            endpoint = token_info["instanceServers"][0]["endpoint"]
            token = token_info["token"]
            ws_url = f"{endpoint}?token={token}"

            async with websockets.connect(ws_url) as ws:
                print("🔗 [Spot] Connected to KuCoin WebSocket")

                for sym in spot_symbols:
                    await ws.send(json.dumps({
                        "id": f"spot_ticker_{sym}",
                        "type": "subscribe",
                        "topic": f"/market/ticker:{sym}",
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
                        symbol = topic.split(":")[1]
                        if symbol not in spot_symbols:
                            continue

                        data = msg.get("data", {})
                        ticker = data.get("ticker", {})
                        data_buffers[symbol].append({
                            "timestamp": int(data.get("ts", 0)) if data.get("ts") else None,
                            "best_bid": float(ticker["bestBid"]) if ticker.get("bestBid") else None,
                            "best_ask": float(ticker["bestAsk"]) if ticker.get("bestAsk") else None,
                            "last_price": float(ticker["last"]) if ticker.get("last") else None,
                            "type": "spot_ticker"
                        })

                    except asyncio.TimeoutError:
                        continue

        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.InvalidStatusCode,
                requests.RequestException, OSError) as e:
            if stop_flag:
                break
            print(f"⚠️ [Spot] Connection lost: {e}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

# ------------------------
# 定时保存
# ------------------------
async def periodic_saver():
    while not stop_flag:
        await asyncio.sleep(600)
        all_symbols = spot_symbols + futures_symbols
        for sym in all_symbols:
            save_data_sync(sym)

# ------------------------
# 优雅退出
# ------------------------
def signal_handler(sig, frame):
    global stop_flag
    print("\n\n🛑 Shutdown signal received. Saving final data...")
    stop_flag = True
    all_symbols = spot_symbols + futures_symbols
    for sym in all_symbols:
        save_data_sync(sym)
    sys.exit(0)

# ------------------------
# 主函数
# ------------------------
async def main():
    global stop_build
    stop_flag = False

    print("🚀 KuCoin Combined Data Collector (with Auto-Reconnect)")
    print(f"📊 Spot symbols: {len(spot_symbols)}")
    print(f"📊 Futures symbols: {len(futures_symbols)}")
    print("📁 Save dir:", save_dir)
    print("🔄 Auto-reconnect enabled (exponential backoff)")
    print("🛑 Press Ctrl+C to stop safely\n")

    tasks = [
        asyncio.create_task(futures_listener_with_reconnect()),
        asyncio.create_task(spot_listener_with_reconnect()),
        asyncio.create_task(periodic_saver())
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        all_symbols = spot_symbols + futures_symbols
        for sym in all_symbols:
            save_data_sync(sym)
        print("✅ All data saved. Goodbye!")

# ------------------------
# 入口
# ------------------------
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    asyncio.run(main())