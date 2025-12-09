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
symbols = ["XBTUSDTM", "ETHUSDTM", "SOLUSDTM"]
save_dir = "kucoin_futures_data"
os.makedirs(save_dir, exist_ok=True)
data_buffers = defaultdict(list)
stop_flag = False

# ------------------------
# 获取 Futures WebSocket Token
# ------------------------
def get_futures_ws_token():
    url = "https://api-futures.kucoin.com/api/v1/bullet-public"
    resp = requests.post(url, timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]

# ------------------------
# 保存数据（同步，用于退出）
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
# 主监听器：同时订阅 index/mark 和 orderbook
# ------------------------
async def kucoin_futures_listener():
    global stop_flag

    # 1. 获取 token
    token_info = get_futures_ws_token()
    endpoint = token_info["instanceServers"][0]["endpoint"]
    token = token_info["token"]
    ws_url = f"{endpoint}?token={token}&acceptUserMessage=true"

    print("🔗 Connecting to KuCoin Futures WebSocket...")
    print("📡 Subscribing symbols:", symbols)

    async with websockets.connect(ws_url) as ws:
        # 2. 订阅 index/mark 价格
        symbols_str = ",".join(symbols)
        index_topic = f"/contract/instrument:{symbols_str}"
        await ws.send(json.dumps({
            "id": "sub_index_mark",
            "type": "subscribe",
            "topic": index_topic,
            "privateChannel": False,
            "response": True
        }))
        print("✅ Subscribed to index/mark prices")

        # 3. 订阅 Level2 Depth5（买一卖一）
        for sym in symbols:
            ob_topic = f"/contractMarket/level2Depth5:{sym}"
            await ws.send(json.dumps({
                "id": f"sub_ob_{sym}",
                "type": "subscribe",
                "topic": ob_topic,
                "privateChannel": False,
                "response": True
            }))
        print("✅ Subscribed to orderbook (top 5)")

        last_ping = asyncio.get_event_loop().time()
        while not stop_flag:
            # 发送 ping 心跳（KuCoin Futures 要求 30 秒内）
            now_time = asyncio.get_event_loop().time()
            if now_time - last_ping >= 25:
                await ws.send(json.dumps({"id": "ping", "type": "ping"}))
                last_ping = now_time

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                msg = json.loads(raw)

                # 忽略 pong
                if msg.get("type") == "pong":
                    continue

                # 处理 index/mark 数据
                if msg.get("subject") == "mark.index.price":
                    data = msg.get("data", {})
                    topic = msg.get("topic", "")
                    symbol = topic.split(":")[1]
                    if symbol in symbols:
                        data_buffers[symbol].append({
                            "timestamp": data.get("timestamp"),
                            "index_price": data.get("indexPrice"),
                            "mark_price": data.get("markPrice"),
                            "type": "index_mark"
                        })

                # 处理订单簿数据
                elif msg.get("subject") == "level2":
                    data = msg.get("data", {})
                    topic = msg.get("topic", "")
                    symbol = topic.split(":")[1]
                    if symbol in symbols:
                        bids = data.get("bids", [])
                        asks = data.get("asks", [])
                        if bids and asks:
                            best_bid = float(bids[0][0])
                            best_ask = float(asks[0][0])
                            data_buffers[symbol].append({
                                "timestamp": data.get("timestamp"),
                                "best_bid": best_bid,
                                "best_ask": best_ask,
                                "type": "orderbook"
                            })

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"⚠️ WebSocket error: {e}")
                break

# ------------------------
# 定时保存任务
# ------------------------
async def periodic_saver():
    while not stop_flag:
        await asyncio.sleep(600)  # 10 minutes
        for sym in symbols:
            save_data_sync(sym)

# ------------------------
# 信号处理器（优雅退出）
# ------------------------
def signal_handler(sig, frame):
    global stop_flag
    print("\n\n🛑 Shutdown signal received. Saving final data...")
    stop_flag = True
    for sym in symbols:
        save_data_sync(sym)
    sys.exit(0)

# ------------------------
# 主函数
# ------------------------
async def main():
    global stop_flag
    stop_flag = False

    print("🚀 KuCoin Futures Real-Time Collector Started")
    print("📊 Symbols:", symbols)
    print("📁 Save dir:", save_dir)
    print("🛑 Press Ctrl+C to stop safely\n")

    tasks = [
        asyncio.create_task(kucoin_futures_listener()),
        asyncio.create_task(periodic_saver())
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        for sym in symbols:
            save_data_sync(sym)
        print("✅ All data saved. Goodbye!")

# ------------------------
# 入口
# ------------------------
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    asyncio.run(main())