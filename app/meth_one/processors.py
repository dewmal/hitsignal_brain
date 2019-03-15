# data processors - Pre, Past

from app.meth_one.stream import fx_candle_1min_stream
from application import stapp
from utils.stream_helper import stream_window


@stream_window(window_length=2, stream_topic=fx_candle_1min_stream, stapp=stapp)
async def fx_candle_stream_prediction(candles):
    print(candles)


