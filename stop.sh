#!/bin/bash
if [ -f kucoin.pid ]; then
    PID=$(cat kucoin.pid)
    echo "Stopping PID $PID..."
    kill $PID
    rm kucoin.pid
else
    echo "No PID file found. Trying pkill..."
    pkill -f kucoin_collector.py
fi