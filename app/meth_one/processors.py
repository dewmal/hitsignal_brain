# data processors - Pre, Past
import datetime

from app.meth_one import settings as meth_settings
from app.meth_one.data_processor import predict
from app.meth_one.stream import fx_candle_1min_stream, fx_price_prediction_stream, PricePrediction, Candle

from application import stapp
from utils.stream_helper import stream_window
import pandas as pd
import numpy as np


# @stapp.agent(fx_candle_1min_stream)
# async def fx_candle_(stream):
#     async for candle in stream:
#         candle: Candle = candle
#         prediction = PricePrediction(
#             time=(datetime.datetime.fromtimestamp(candle.epoch) + datetime.timedelta(seconds=120)).timestamp(),
#             price=candle.close * (100.0005) / 100,
#             symbol=candle.symbol,
#             price_change=candle.high / candle.open,
#             originator="random",
#             origin_time=candle.epoch
#         )
#         await fx_price_prediction_stream.send(value=prediction)


def to_date_time(val):
    return datetime.datetime.fromtimestamp(
        int(val)
    ).strftime('%Y-%m-%d %H:%M:%S')


@stream_window(window_length=100, stream_topic=fx_candle_1min_stream, stapp=stapp)
async def fx_candle_stream_prediction_(candles):
    window = []
    # print(len(candles.data))
    for candle in candles.data:
        ar = candle.to_olhc_array()
        window.append(ar)

    window = np.array(window)
    df = pd.DataFrame(window,
                      columns=['date_time', 'open', 'high', 'low', 'close', 'vol'])
    # df.date_time = df.date_time.apply(to_date_time)
    df = df.set_index('date_time')

    prediction, index = predict(df=df, window_size=meth_settings.window_size,
                                consider_value=meth_settings.consider_value,
                                prediction_step=meth_settings.prediction_step,
                                algo_max_window_size=meth_settings.algo_max_window_size)
    print(prediction, index, to_date_time(index))

    prediction = PricePrediction(
        time=index,
        price=float(prediction[0][0]),
        symbol=candle.symbol,
        price_change=float(prediction[0][0]),
        originator="aimodel",
        origin_time=candle.epoch
    )
    await fx_price_prediction_stream.send(value=prediction)

    # print(f"Window Length {len(window)}")
