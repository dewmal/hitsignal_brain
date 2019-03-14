import json

import websockets


async def read_from_ws(pair_name, callback, count=10):
    api_url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

    async with websockets.connect(
            api_url) as ws:
        print(f"Web socket start on {pair_name} - {ws}")
        json_data = json.dumps({
            "ticks_history": f"frx{pair_name}",
            "end": "latest",
            "start": 1,
            "style": "candles",
            "adjust_start_time": 1,
            "subscribe": 1,
            "count": count
        })
        await ws.send(json_data)
        async for message in ws:
            await callback(message)
