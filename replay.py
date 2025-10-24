# pip install tardis-client
import asyncio
from tardis_client import TardisClient, Channel
tardis_client = TardisClient(api_key="TD.6xKnyzxAhxsfdXg6.Zp93cW5o9TtOLX4.Rd5-njLq8V9wiUv.qnFS0Qu2F7EXvw0.HKEvrONW1YFqdy8.wcPI")

async def replay():
  # replay method returns Async Generator
  messages = tardis_client.replay(
    exchange="binance",
    from_date="2025-10-01",
    to_date="2025-10-02",
    filters=[Channel(name="depth", symbols=["btcusdt"])]
  )

  # messages as provided by Binance real-time stream
  async for local_timestamp, message in messages:
    print(message)


asyncio.run(replay())