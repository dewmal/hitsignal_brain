# data processors - Pre, Past
import datetime

from pyti.exponential_moving_average import exponential_moving_average as ema
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.simple_moving_average import simple_moving_average as sma

from app.meth_one.stream import fx_candle_1min_stream, fx_price_prediction_stream, PricePrediction, Candle
from application import stapp
from utils.stream_helper import stream_window
import pandas as pd


def create_panda_data_frame_2min_ready(df, value_name, sma_1, sma_2, sma_3, macd_1, macd_2, macd_3,
                                       new_data_frame=False, drop_na=True):
    if new_data_frame:
        data_frame = pd.DataFrame()
    else:
        data_frame = df

    data_frame[f'sma_{value_name}_{sma_1}'] = sma(data_frame[value_name].values, sma_1)  # 6
    data_frame[f'sma_{value_name}_{sma_2}'] = sma(data_frame[value_name].values, sma_2)  # 14
    data_frame[f'sma_{value_name}_{sma_3}'] = sma(data_frame[value_name].values, sma_3)  # 26
    # 12,26,9
    data_frame[f'mcad_{macd_1}_{macd_2}'] = macd(data_frame.close.values, macd_1, macd_2)
    data_frame[f'mcad_{macd_1}_{macd_2}_signal'] = data_frame[f'mcad_{macd_1}_{macd_2}'].values / ema(
        data_frame.close.values,
        macd_3)  # MACD Signal Line
    if drop_na:
        data_frame.dropna(inplace=True)
    return data_frame


@stapp.agent(fx_candle_1min_stream)
async def fx_candle_(stream):
    async for candle in stream:
        candle: Candle = candle
        prediction = PricePrediction(
            time=(datetime.datetime.fromtimestamp(candle.epoch) + datetime.timedelta(seconds=120)).timestamp(),
            price=candle.close * (100.0005) / 100,
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
