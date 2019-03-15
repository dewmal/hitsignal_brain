# data processors - Pre, Past
import datetime

from app.meth_one.stream import fx_candle_1min_stream, fx_price_prediction_stream, PricePrediction, Candle
from application import stapp
from utils.stream_helper import stream_window


@stapp.agent(fx_candle_1min_stream)
async def fx_candle_(stream):
    async for candle in stream:
        candle: Candle = candle
        prediction = PricePrediction(
            time=(datetime.datetime.fromtimestamp(candle.epoch) + datetime.timedelta(seconds=30)).timestamp(),
            price=candle.close * (101) / 100,
            symbol=candle.symbol,
            price_change=candle.high / candle.open,
            originator="random",
            origin_time=candle.epoch
        )
        await fx_price_prediction_stream.send(value=prediction)


@stream_window(window_length=20, stream_topic=fx_candle_1min_stream, stapp=stapp)
async def fx_candle_stream_prediction_(candles):
    window = []
    for candle in candles:
        ar = candle.to_olhc_array()
        window.append(ar)

    print(window)
