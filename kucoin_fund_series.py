#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import os
import signal
import sys
import datetime
import pandas as pd
import requests
import websockets
from collections import defaultdict

# ------------------------
# 配置
# ------------------------
version = '_4'
SAVE_RAW_JSON = True  # ←←←【新增】是否保存原始中间 JSON 文件

original_symbols = [
    "AGTUSDT", "DASHUSDT", "XMRUSDT", "BNBUSDT", "WCTUSDT", "INJUSDT",
    "KAITOUSDT", "HAEDALUSDT", "XPLUSDT", "DOTUSDT", "ONDOUSDT",
    "ZECUSDT", "SUIUSDT", "GIGGLEUSDT"
]

spot_symbols = [s[:-4] + "-USDT" for s in original_symbols if s.endswith("USDT")]
futures_symbols = [s + "M" for s in original_symbols]

OUTPUT_DIR = f"kucoin_data/kucoin_funding_series{version}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_DATA_DIR = f"kucoin_data/kucoin_data_combined{version}"
if SAVE_RAW_JSON:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Global state
data_buffers = defaultdict(list)
stop_flag = False

# ------------------------
# 工具函数（保持不变）
# ------------------------

def get_base_symbol(symbol):
    if '-USDT' in symbol:
        return symbol.split('-USDT')[0]
    elif symbol.endswith('USDTM'):
        return symbol[:-5]
    elif symbol.endswith('USDT'):
        return symbol[:-4]
    return symbol

def detect_symbol_type(symbol):
    if symbol.endswith('USDTM'):
        return 'futures'
    elif '-USDT' in symbol or symbol.endswith('USDT'):
        return 'spot'
    return 'unknown'

def rename_fields(record, symbol_type):
    mapping = {
        'best_bid': f'{symbol_type}_best_bid',
        'best_ask': f'{symbol_type}_best_ask',
        'last_price': f'{symbol_type}_last_price'
    }
    new_rec = {}
    for k, v in record.items():
        if k in mapping:
            new_rec[mapping[k]] = v
        else:
            new_rec[k] = v
    return new_rec

# ------------------------
# Token / Validation / Listeners（保持不变，略去以节省篇幅）
# ------------------------
# [此处省略 get_futures_token, get_spot_token, validate_spot_symbols,
#  futures_listener_with_reconnect, spot_listener_with_reconnect]
# → 与上一版完全相同，无需修改

# ------------------------
# 每10分钟处理并导出（关键修改在此）
# ------------------------

async def process_and_export():
    global data_buffers, stop_flag

    while not stop_flag:
        now = datetime.datetime.now()
        next_align = now.replace(
            minute=(now.minute // 10) * 10, second=0, microsecond=0
        ) + datetime.timedelta(minutes=10)
        sleep_sec = (next_align - now).total_seconds()
        if sleep_sec <= 0:
            sleep_sec = 600
        await asyncio.sleep(sleep_sec)

        if stop_flag:
            break

        print(f"\n🕒 Processing data up to {next_align.strftime('%Y-%m-%d %H:%M:%S')}")
        cutoff_ts = int(next_align.timestamp() * 1000)

        # Snapshot current buffer
        current_buffers = dict(data_buffers)
        data_buffers.clear()

        # ✅【新增】保存原始 JSON（如果启用）
        if SAVE_RAW_JSON:
            date_str = next_align.strftime("%Y%m%d")
            time_str = next_align.strftime("%H%M")
            for symbol, records in current_buffers.items():
                if not records:
                    continue
                # Filter only records <= cutoff (already done later, but safe)
                filtered_records = []
                for rec in records:
                    ts = rec.get("timestamp")
                    if ts is None:
                        continue
                    try:
                        ts = int(float(ts))
                        if ts > 1893456000000:
                            ts = int(ts / 1000000)
                        if 1577836800000 <= ts <= 1893456000000 and ts <= cutoff_ts:
                            filtered_records.append(rec)
                    except (ValueError, TypeError):
                        continue
                if filtered_records:
                    filename = f"{date_str}_{time_str}_{symbol}.json"
                    filepath = os.path.join(RAW_DATA_DIR, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(filtered_records, f, indent=2, ensure_ascii=False)
                    print(f"💾 Saved raw {len(filtered_records)} records for {symbol}")

        # 继续原有处理逻辑（使用 current_buffers）
        base_symbol_data = defaultdict(list)

        for symbol, records in current_buffers.items():
            sym_type = detect_symbol_type(symbol)
            if sym_type == 'unknown':
                continue
            base_sym = get_base_symbol(symbol)
            for rec in records:
                ts = rec.get("timestamp")
                if ts is None:
                    continue
                try:
                    ts = int(float(ts))
                    if ts > 1893456000000:
                        ts = int(ts / 1000000)
                    if not (1577836800000 <= ts <= 1893456000000):
                        continue
                    if ts > cutoff_ts:
                        data_buffers[symbol].append(rec)  # put back
                        continue
                except (ValueError, TypeError):
                    continue

                clean_rec = {k: v for k, v in rec.items() if k not in ['symbol', 'type'] and v is not None}
                clean_rec = rename_fields(clean_rec, sym_type)
                clean_rec['source_symbol'] = symbol
                clean_rec['timestamp'] = ts
                base_symbol_data[base_sym].append(clean_rec)

        # Process each base symbol → same as before
        for base_symbol, records in base_symbol_data.items():
            if not records:
                continue
            try:
                df = pd.DataFrame(records)
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                df['timestamp'] = df['timestamp'].astype('int64')
                df = df.sort_values('timestamp').drop_duplicates('timestamp', keep='last')

                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('datetime')

                for col in df.columns:
                    if col != 'timestamp' and col != 'source_symbol':
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                if 'funding_rate' not in df.columns:
                    continue
                funding_mask = df['funding_rate'].notna() & (df['funding_rate'] != 0)
                funding_df = df[funding_mask].copy()
                if funding_df.empty:
                    continue

                key_fields = [
                    'index_price', 'mark_price',
                    'spot_best_bid', 'spot_best_ask',
                    'futures_best_bid', 'futures_best_ask'
                ]
                for col in key_fields:
                    if col in funding_df.columns:
                        funding_df[col] = funding_df[col].fillna(method='ffill')

                price_fields = ['index_price', 'mark_price', 'spot_best_bid', 'futures_best_bid']
                available_price_fields = [c for c in price_fields if c in funding_df.columns]
                if available_price_fields:
                    has_price = funding_df[available_price_fields].notna().any(axis=1)
                    final_df = funding_df[has_price]
                else:
                    final_df = funding_df

                if final_df.empty:
                    continue

                result_dict = {}
                for dt, row in final_df.iterrows():
                    d = {k: v for k, v in row.dropna().to_dict().items() if k != 'timestamp'}
                    if d:
                        result_dict[dt.isoformat()] = d

                if not result_dict:
                    continue

                timestamp_str = next_align.strftime("%Y%m%d_%H%M")
                out_file = os.path.join(OUTPUT_DIR, f"{base_symbol}_funding{version}_{timestamp_str}.json")
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
                print(f"✅ Saved {len(result_dict)} events for {base_symbol} → {os.path.basename(out_file)}")

            except Exception as e:
                print(f"❌ Error processing {base_symbol}: {e}")

# ------------------------
# Signal handler & main（保持不变）
# ------------------------
# [略：与上一版完全相同]

def signal_handler(sig, frame):
    global stop_flag
    print("\n\n🛑 Shutdown requested.")
    stop_flag = True
    sys.exit(0)

async def main():
    global stop_flag
    stop_flag = False

    print("🚀 KuCoin Funding Series Exporter (with optional raw JSON saving)")
    print(f"📊 Base symbols: {[get_base_symbol(s) for s in original_symbols]}")
    print(f"📁 Final output: {OUTPUT_DIR}")
    if SAVE_RAW_JSON:
        print(f"📁 Raw JSON output: {RAW_DATA_DIR}")
    else:
        print("📁 Raw JSON saving: DISABLED")
    print("🛑 Press Ctrl+C to exit\n")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    await asyncio.gather(
        futures_listener_with_reconnect(),
        spot_listener_with_reconnect(),
        process_and_export()
    )

# ------------------------
# Token / Validation / Listeners（完整实现）
# ------------------------

def get_futures_token():
    url = "https://api-futures.kucoin.com/api/v1/bullet-public"
    resp = requests.post(url, timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]

def get_spot_token():
    url = "https://api.kucoin.com/api/v1/bullet-public"
    resp = requests.post(url, timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]

def validate_spot_symbols(symbols):
    try:
        resp = requests.get("https://api.kucoin.com/api/v1/symbols", timeout=10)
        if resp.status_code == 200:
            all_symbols = {item["symbol"] for item in resp.json()["data"]}
            valid, invalid = [], []
            for s in symbols:
                (valid if s in all_symbols else invalid).append(s)
            if invalid:
                print(f"⚠️ Skipping invalid spot symbols: {invalid}")
            return valid
    except Exception as e:
        print(f"❌ Failed to validate spot symbols: {e}. Proceeding with all.")
    return symbols

async def futures_listener_with_reconnect():
    global stop_flag
    reconnect_delay = 1
    max_reconnect_delay = 60

    # 合理时间戳范围（毫秒）：2020-01-01 到 2030-12-31
    MIN_TS = 1577836800000
    MAX_TS = 1924992000000

    def normalize_timestamp(ts_val, is_ticker_ts=False):
        """
        Normalize timestamp to milliseconds.
        - For ticker.ts: often in nanoseconds
        - For others: usually in milliseconds
        """
        if ts_val is None:
            return None
        try:
            ts = int(ts_val)
        except (ValueError, TypeError):
            return None

        # Heuristic: detect unit by magnitude
        if ts > 10**16:          # Nanoseconds (e.g., 1765135262823000000)
            ts //= 1_000_000     # → milliseconds
        elif ts > 10**13:        # Microseconds (e.g., 1765135262823000)
            ts //= 1_000         # → milliseconds
        # else: assume already milliseconds

        # Validate range
        if MIN_TS <= ts <= MAX_TS:
            return ts
        return None

    while not stop_flag:
        try:
            token_info = get_futures_token()
            endpoint = token_info["instanceServers"][0]["endpoint"]
            token = token_info["token"]
            ws_url = f"{endpoint}?token={token}"

            async with websockets.connect(ws_url) as ws:
                print("🔗 [Futures] Connected")
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
                        record = {"symbol": symbol}

                        if msg.get("subject") == "mark.index.price":
                            # timestamp is in milliseconds
                            ts = normalize_timestamp(data.get("timestamp"))
                            if ts is not None:
                                record.update({
                                    "timestamp": ts,
                                    "index_price": float(data["indexPrice"]) if data.get("indexPrice") else None,
                                    "mark_price": float(data["markPrice"]) if data.get("markPrice") else None,
                                    "type": "futures_index_mark"
                                })
                                data_buffers[symbol].append(record)

                        elif msg.get("subject") == "ticker":
                            # ts is often in NANOSECONDS!
                            ts = normalize_timestamp(data.get("ts"), is_ticker_ts=True)
                            if ts is not None:
                                record.update({
                                    "timestamp": ts,
                                    "best_bid": float(data["bestBidPrice"]) if data.get("bestBidPrice") else None,
                                    "best_ask": float(data["bestAskPrice"]) if data.get("bestAskPrice") else None,
                                    "last_price": float(data["lastPrice"]) if data.get("lastPrice") else None,
                                    "type": "futures_ticker"
                                })
                                data_buffers[symbol].append(record)

                        elif msg.get("subject") == "funding.rate":
                            # timestamp is in milliseconds
                            ts = normalize_timestamp(data.get("timestamp"))
                            if ts is not None:
                                try:
                                    funding_rate = float(data["fundingRate"]) if data.get("fundingRate") not in (None, "") else None
                                    next_funding = int(data["nextFundingTime"]) if data.get("nextFundingTime") not in (None, "") else None
                                except (ValueError, TypeError):
                                    funding_rate = next_funding = None
                                record.update({
                                    "timestamp": ts,
                                    "funding_rate": funding_rate,
                                    "next_funding_time": next_funding,
                                    "type": "futures_funding"
                                })
                                data_buffers[symbol].append(record)

                    except asyncio.TimeoutError:
                        continue

        except (websockets.exceptions.ConnectionClosed, requests.RequestException, OSError) as e:
            if stop_flag:
                break
            print(f"⚠️ [Futures] Reconnecting in {reconnect_delay}s: {e}")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

async def spot_listener_with_reconnect():
    global stop_flag
    valid_symbols = validate_spot_symbols(spot_symbols)
    if not valid_symbols:
        print("❌ No valid spot symbols. Spot listener disabled.")
        return

    reconnect_delay = 1
    max_reconnect_delay = 60

    while not stop_flag:
        try:
            token_info = get_spot_token()
            endpoint = token_info["instanceServers"][0]["endpoint"]
            token = token_info["token"]
            ws_url = f"{endpoint}?token={token}"

            async with websockets.connect(ws_url) as ws:
                print("🔗 [Spot] Connected")
                for sym in valid_symbols:
                    await ws.send(json.dumps({
                        "id": f"spot_ticker_{sym}",
                        "type": "subscribe",
                        "topic": f"/market/ticker:{sym}",
                        "privateChannel": False
                    }))
                print(f"✅ [Spot] Subscribed to {len(valid_symbols)} symbols")
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
                        if symbol not in valid_symbols:
                            continue

                        data = msg.get("data", {})
                        record = {
                            "symbol": symbol,
                            "timestamp": data.get("time"),
                            "best_bid": float(data["bestBid"]) if data.get("bestBid") else None,
                            "best_ask": float(data["bestAsk"]) if data.get("bestAsk") else None,
                            "last_price": float(data["price"]) if data.get("price") else None,
                            "type": "spot_ticker"
                        }
                        if record["timestamp"] is not None:
                            data_buffers[symbol].append(record)

                    except asyncio.TimeoutError:
                        continue

        except (websockets.exceptions.ConnectionClosed, requests.RequestException, OSError) as e:
            if stop_flag:
                break
            print(f"⚠️ [Spot] Reconnecting in {reconnect_delay}s: {e}")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

# ------------------------
# 入口
# ------------------------

if __name__ == "__main__":
    asyncio.run(main())