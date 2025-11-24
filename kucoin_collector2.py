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
# 配置：现货符号列表（自动转为 Futures 合约）
# ------------------------
spot_symbols = [
    "XBTUSDT", "SOLUSDT", "PNUTUSDT", "XRPUSDT", "FETUSDT", "UNIUSDT",
    "COMPUSDT", "THEUSDT", "AVAXUSDT", "LTCUSDT", "ETCUSDT", "FORMUSDT",
    "TONUSDT", "HFTUSDT", "DOTUSDT", "CHESSUSDT", "ETHUSDT", "BNBUSDT",
    "TRXUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "XLMUSDT", "BCHUSDT",
    "HBARUSDT", "ZECUSDT", "AAVEUSDT", "ENAUSDT", "NEARUSDT", "ONDOUSDT"
]
# spot_symbols = ["XBTUSDTM", "ETHUSDTM", "SOLUSDTM"]
# 转换为 KuCoin Futures 永续合约格式 (添加 "M")
symbols = [s + "M" for s in spot_symbols]

# 可选：手动排除 KuCoin 上不存在的合约（根据实际测试）
# 例如：PNUTUSDTM 可能不存在，可临时移除
# symbols = [s for s in symbols if s not in ["PNUTUSDTM", "CHESSUSDTM", ...]]

save_dir = "kucoin_futures_data_3"
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
# 保存数据（同步）
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
# 主监听器
# ------------------------
async def kucoin_futures_listener():
    global stop_flag

    token_info = get_futures_ws_token()
    endpoint = token_info["instanceServers"][0]["endpoint"]
    token = token_info["token"]
    ws_url = f"{endpoint}?token={token}&acceptUserMessage=true"

    print("🔗 Connecting to KuCoin Futures WebSocket...")
    print(f"📡 Subscribing {len(symbols)} symbols (first 5): {symbols[:5]}...")

    async with websockets.connect(ws_url) as ws:
        # ================================
        # 🔸 分批订阅 index/mark（KuCoin 建议单次 <= 50 symbol）
        # ================================
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            symbols_str = ",".join(batch)
            await ws.send(json.dumps({
                "id": f"sub_index_mark_{i//batch_size}",
                "type": "subscribe",
                "topic": f"/contract/instrument:{symbols_str}",
                "privateChannel": False,
                "response": True
            }))
        print("✅ Subscribed to index/mark prices")

        # ================================
        # 🔸 订阅 orderbook 和 funding rate（逐个订阅，更稳定）
        # ================================
        for sym in symbols:
            # Orderbook spot
            await ws.send(json.dumps({
                "id": f"sub_ob_{sym}",
                "type": "subscribe",
                # "topic": f"/contractMarket/level2Depth5:{sym}",
                "topic": f"/market/ticker:{sym}",
                "privateChannel": False,
                "response": True
            }))
            # Orderbook contractMarket
            await ws.send(json.dumps({
                "id": f"sub_ob_{sym}",
                "type": "subscribe",
                "topic": f"/contractMarket/ticker:{sym}",
                "privateChannel": False,
                "response": True
            }))            
            # Funding rate
            await ws.send(json.dumps({
                "id": f"sub_funding_{sym}",
                "type": "subscribe",
                "topic": f"/contractMarket/fundingRate:{sym}",
                "privateChannel": False,
                "response": True
            }))
        print("✅ Subscribed to orderbook and funding rate for all symbols")

        last_ping = asyncio.get_event_loop().time()
        while not stop_flag:
            # 心跳
            now_time = asyncio.get_event_loop().time()
            if now_time - last_ping >= 25:
                await ws.send(json.dumps({"id": "ping", "type": "ping"}))
                last_ping = now_time

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                msg = json.loads(raw)

                if msg.get("type") == "pong":
                    continue

                topic = msg.get("topic", "")
                if ":" not in topic:
                    continue
                symbol = topic.split(":")[1]
                if symbol not in symbols:
                    continue

                data = msg.get("data", {})

                # 1. Index / Mark Price
                if msg.get("subject") == "mark.index.price":
                    data_buffers[symbol].append({
                        "timestamp": data.get("timestamp"),
                        "index_price": float(data.get("indexPrice")) if data.get("indexPrice") is not None else None,
                        "mark_price": float(data.get("markPrice")) if data.get("markPrice") is not None else None,
                        "type": "index_mark"
                    })

                # 2. Orderbook
                # elif msg.get("subject") == "level2":
                #     bids = data.get("bids", [])
                #     asks = data.get("asks", [])
                #     if bids and asks:
                #         data_buffers[symbol].append({
                #             "timestamp": data.get("timestamp"),
                #             "best_bid": float(bids[0][0]),
                #             "best_ask": float(asks[0][0]),
                #             "type": "orderbook"
                #         })
                # 4. Ticks
                elif msg.get("subject") == "ticker":
                    bids = data.get("bestBidPrice", [])
                    asks = data.get("bestAskPrice", [])
                    if bids and asks:
                        data_buffers[symbol].append({
                            "timestamp": data.get("Time"),
                            "best_bid": float(bids),
                            "best_ask": float(asks),
                            "type": "message"
                        })
                elif msg.get("subject") == "trade.ticker":
                    bids = data.get("bestBidPrice", [])
                    asks = data.get("bestAskPrice", [])
                    if bids and asks:
                        data_buffers[symbol].append({
                            "timestamp": data.get("Time"),
                            "best_bid": float(bids),
                            "best_ask": float(asks),
                            "type": "message"
                        })                        

                # 3. Funding Rate
                elif msg.get("subject") == "funding.rate":
                    funding_rate = data.get("fundingRate")
                    next_funding_time = data.get("nextFundingTime")
                    try:
                        funding_rate = float(funding_rate) if funding_rate is not None else None
                    except (ValueError, TypeError):
                        funding_rate = None
                    try:
                        next_funding_time = int(next_funding_time) if next_funding_time is not None else None
                    except (ValueError, TypeError):
                        next_funding_time = None
                    data_buffers[symbol].append({
                        "timestamp": data.get("timestamp"),
                        "funding_rate": funding_rate,
                        "next_funding_time": next_funding_time,
                        "type": "funding_rate"
                    })

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"⚠️ WebSocket error: {e}")
                break

# ------------------------
# 定时保存（每10分钟）
# ------------------------
async def periodic_saver():
    while not stop_flag:
        await asyncio.sleep(600)
        for sym in symbols:
            save_data_sync(sym)

# ------------------------
# 优雅退出
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

    print("🚀 KuCoin Futures Data Collector (30 Symbols)")
    print(f"📊 Total symbols: {len(symbols)}")
    print("📁 Save dir:", save_dir)
    print("✅ Collecting: index_price, orderbook, funding_rate, next_funding_time")
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