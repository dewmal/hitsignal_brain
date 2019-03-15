import asyncio

from app.meth_one.utils import read_from_ws_history, process_message


async def process(message):
    # print(message)
    async def msg_callbac(candle):
        print(candle)

    await process_message(message, call_back=msg_callbac)


async def main():
    await read_from_ws_history("EURUSD", callback=process, count=30)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    # Blocking call which returns when the display_date() coroutine is done
    loop.run_until_complete(main())
    loop.close()
