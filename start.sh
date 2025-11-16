#!/bin/bash
cd "$(dirname "$0")"
echo "[$(date)] Starting KuCoin collector..."
nohup python3.9 kucoin_collector.py > kucoin.log 2>&1 &
echo $! > kucoin.pid
echo "Started with PID $(cat kucoin.pid)"