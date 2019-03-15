import json
import sys
import traceback

import websockets


async def read_from_ws_live(pair_name, callback, count=10):
    api_url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

    async with websockets.connect(
            api_url) as ws:
        print(f"Web socket start on {pair_name} - {ws}")
        json_data = json.dumps({'ticks': f'frx{pair_name}'})
        await ws.send(json_data)
        async for message in ws:
            await callback(message)


async def read_from_ws_history(pair_name, callback, count=10):
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


async def process_message(message):
    fact = json.loads(message)
    # print(message)
    message_type = fact['msg_type']

    if 'error' in fact and fact['error']['code'] == 'AuthorizationRequired':
        # self.login(ws)
        pass

    try:
        if message_type == 'tick':
            _id = fact['tick']['id']
            date = int(fact['tick']['epoch'])
            ask = float(fact['tick']['ask'])
            bid = float(fact['tick']['bid'])
            quote = float(fact['tick']['quote'])
            # data = np.array((ask, bid, quote))
            # print(pair_name,date,data)
            # await context.publish(index=date, data=data)
            return {
                "ask": ask,
                "bid": bid,
                "quote": quote,
                "timestamp": date
            }

        elif message_type == 'proposal':
            # self.buy_contact(ws, fact)
            pass
        elif message_type == 'authorize':
            # self.is_logged_in = True
            # self.account_balance = float(fact['authorize']['balance'])
            pass
        elif message_type == 'buy':
            # self.account_balance = float(fact['buy']['balance_after'])
            pass
        elif message_type == 'sell':
            # self.account_balance = float(fact['sell']['balance_after'])
            pass
        elif message_type == 'candles':
            print(fact)
            for candle in fact['candles']:
                print(candle)
            # self.account_balance = float(fact['sell']['balance_after'])
            pass
        else:
            # logging.info(fact)
            pass

    except Exception as e:
        ex, val, tb = sys.exc_info()
        traceback.print_exception(ex, val, tb)
