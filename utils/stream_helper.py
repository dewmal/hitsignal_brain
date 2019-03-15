import queue

from app.meth_one.stream import CandleWindow, Candle


def stream_last_(stream_topic, stapp):
    def _decorator(_fn):
        @stapp.agent(stream_topic)
        async def fx_candle_stream_(stream):
            last_candle_ = 0
            tmp_last_candle = None
            async for candle in stream:
                candle: Candle = candle
                # print(f"{candle.epoch} , {candle.open_time} , {candle.open} {candle.close}")
                if last_candle_ < candle.open_time and tmp_last_candle is not None:
                    await _fn(tmp_last_candle)
                    last_candle_ = candle.open_time
                tmp_last_candle = candle

    return _decorator


def stream_window(window_length, stream_topic, stapp):
    def decorator(_fn):

        @stapp.agent(stream_topic)
        async def fx_candle_stream_prediction(stream):
            window_min = window_length
            candles = queue.Queue(maxsize=window_min)
            async for candle in stream:
                candles.put(candle)
                if candles.full():
                    await _fn(CandleWindow(data=list(candles.queue), window_lenght=window_min,
                                           window_time=candle.epoch))
                    candles.get()

    return decorator
