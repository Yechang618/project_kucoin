import os
import sys
import json
import traceback
from typing import Dict, List, Optional
from decimal import Decimal
import requests

# 假设你的 util.py 在当前目录（或已加入 sys.path）
try:
    from util import DecimalEncoder
except ImportError:
    # 如果 util.py 在 lib/ 目录下，请取消注释下一行
    # sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
    from util import DecimalEncoder


def safe_decimal(value, default="0") -> Decimal:
    """安全地将值转换为 Decimal，处理 None 和字符串"""
    if value is None:
        return Decimal(default)
    try:
        return Decimal(str(value)).normalize()
    except (ValueError, TypeError):
        return Decimal(default)


def fetch_spot_symbols() -> Dict[str, dict]:
    """从 KuCoin 公开接口获取现货交易对（无认证）"""
    url = "https://api.kucoin.com/api/v1/symbols"  # ✅ 已移除末尾空格
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('code') != '200000':
            raise RuntimeError(f"Spot API returned non-200 code: {data}")
        symbols = {s['symbol']: s for s in data['data']}
        print(f"✅ 成功获取 {len(symbols)} 个现货交易对")
        return symbols
    except Exception as e:
        print(f"❌ 获取现货交易对失败: {e}")
        raise


def fetch_future_contracts() -> Dict[str, dict]:
    """从 KuCoin Futures 公开接口获取活跃永续合约（无认证）"""
    url = "https://api-futures.kucoin.com/api/v1/contracts/active"  # ✅ 已移除末尾空格
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('code') != '200000':
            raise RuntimeError(f"Futures API returned non-200 code: {data}")
        contracts = {c['symbol']: c for c in data['data']}
        print(f"✅ 成功获取 {len(contracts)} 个永续合约")
        return contracts
    except Exception as e:
        print(f"❌ 获取永续合约失败: {e}")
        raise


def update_symbols_public(symbolPath: str, avoid_list: Optional[List[str]] = None) -> Dict:
    if avoid_list is None:
        avoid_list = []

    try:
        spot_info = fetch_spot_symbols()
        future_info = fetch_future_contracts()

        # 筛选可交易的 USDT 现货
        spot_usdt = {
            v['baseCurrency']: v
            for v in spot_info.values()
            if v['quoteCurrency'] == 'USDT' and v.get('enableTrading', False)
        }

        symbols = {}
        special_mapping = {'XBT': 'BTC'}

        for v in future_info.values():
            if v.get('status') != 'Open' or v.get('quoteCurrency') != 'USDT':
                continue

            asset = v.get('baseCurrency', '')
            if not asset or asset in avoid_list:
                continue

            base_asset = None
            mul = 1

            if asset in special_mapping and special_mapping[asset] in spot_usdt:
                base_asset = special_mapping[asset]
            elif asset.startswith('1000') and asset[4:] in spot_usdt:
                base_asset = asset[4:]
                mul = 1000
            elif asset.startswith('10000') and asset[5:] in spot_usdt:
                base_asset = asset[5:]
                mul = 10000
            elif asset in spot_usdt:
                base_asset = asset
            else:
                continue  # 无对应现货

            spot_symbol = spot_usdt[base_asset]['symbol']
            future_symbol = v['symbol']

            # === 安全获取 futures 字段 ===
            multiplier = safe_decimal(v.get('multiplier'), "1")
            lot_size = safe_decimal(v.get('lotSize'), "0")
            tick_size = safe_decimal(v.get('tickSize'), "0")
            funding_interval = v.get('fundingInterval')
            if funding_interval is None:
                funding_interval = 28800  # 默认 8 小时

            # === 安全获取 spot 字段 ===
            spot_data = spot_usdt[base_asset]
            base_inc = safe_decimal(spot_data.get('baseIncrement'), "0")
            base_min = safe_decimal(spot_data.get('baseMinSize'), "0")
            price_inc = safe_decimal(spot_data.get('priceIncrement'), "0")

            # === 跳过无效数据 ===
            if lot_size <= 0 or multiplier <= 0 or base_inc <= 0:
                print(f"⚠️ 跳过无效合约/现货: {future_symbol} / {spot_symbol}")
                continue

            # === 计算 lotSize / minSize ===
            calc_lot = lot_size * multiplier * Decimal(mul)
            lot_result = max(calc_lot, base_inc).normalize()
            min_result = max(base_min, calc_lot).normalize()

            symbols[base_asset] = {
                "spot": spot_symbol,
                "future": future_symbol,
                "feeCategory": spot_data.get('feeCurrency', 'USDT'),
                "isMarginEnabled": spot_data.get('isMarginEnabled', False),
                "mul": mul,
                "spTakerFee": Decimal('0.001'),    # 默认现货手续费
                "fuTakerFee": Decimal('0.0005'),   # 默认合约手续费
                "multiplier": multiplier,
                "lotSize": lot_result,
                "minSize": min_result,
                "spot_tick_size": price_inc,
                "swap_tick_size": tick_size,
                "fundingRateGranularity": int(funding_interval) // 3600,  # 转为小时
            }

        # 保存文件
        os.makedirs(os.path.dirname(symbolPath), exist_ok=True)
        with open(symbolPath, "w") as f:
            json.dump(symbols, f, indent=4, cls=DecimalEncoder)

        print(f"✅ 已更新 symbol 列表，共 {len(symbols)} 个币种，保存至: {symbolPath}")
        return symbols

    except Exception as e:
        exc_str = traceback.format_exc()
        log_file = "kucoin_symbol_update_error.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== 完整错误日志 ===\n")
            f.write(exc_str)
        print(f"❌ 更新失败！完整错误已保存至: {os.path.abspath(log_file)}")
        raise


if __name__ == "__main__":
    symbolPath = os.path.join(os.getcwd(), 'symbol_1.json')
    try:
        updated = update_symbols_public(symbolPath, avoid_list=[])
        print(f"更新完成，共 {len(updated)} 个交易对: {list(updated.keys())}")
    except Exception:
        sys.exit(1)