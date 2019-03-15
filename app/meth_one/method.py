import socketio
from mode import Service
from mode.threads import ServiceThread

import settings
from app.meth_one.stream import fx_message_stream_topic_eur_usd, FxStreamMessage, fx_message_live_stream_topic, \
    fx_price_prediction_stream, fx_candle_stream, Candle, fx_candle_1min_stream, PricePrediction
from app.meth_one.utils import read_from_ws_history, read_from_ws_live, process_message
from application import stapp
from utils.stream_helper import stream_last_

m_sio = socketio.AsyncRedisManager(settings.SOCKET_IO_MESSAGE_BROKER, write_only=True)


class MethodOne(ServiceThread):

    async def on_start(self) -> None:
        print('MethodOne STARTING')

    @Service.task
    async def web_stream_live_data_eur_usd(self):
        async def result_reader(message):
            await fx_message_live_stream_topic.send(value=FxStreamMessage(message=message, type="TICK"))

        await read_from_ws_live("EURUSD", result_reader)

    @Service.task
    async def web_steam_data_eur_usd(self):
        async def result_reader(message):
            await fx_message_stream_topic_eur_usd.send(value=FxStreamMessage(message=message, type="CANDLE"))

        await read_from_ws_history("EURUSD", result_reader, count=30)

    @Service.task
    async def web_steam_data_eur_jpy(self):
        async def result_reader(message):
            await fx_message_stream_topic_eur_usd.send(value=FxStreamMessage(message=message, type="CANDLE"))

        await read_from_ws_history("EURUSD", result_reader)


async def call_after_process_candle_live_feed(candle):
    await fx_candle_stream.send(value=candle)


@stream_last_(fx_candle_stream, stapp)
async def candle_stream_fx(stream):
    await fx_candle_1min_stream.send(value=stream)


@stapp.agent(fx_message_stream_topic_eur_usd)
async def process_bar_data(views):
    async for fx_msg in views:
        await process_message(fx_msg.message, call_back=call_after_process_candle_live_feed)


@stapp.agent(fx_candle_1min_stream)
async def live_stream_candle_fx_data(messages):
    async  for candle in messages:
        candle: Candle = candle
        print(candle)
        await m_sio.emit('candle_live_stream_eur_usd',
                         {
                             "open": candle.open,
                             "close": candle.close,
                             "high": candle.high,
                             "low": candle.low,
                             "date": candle.open_time,
                             "close_time": candle.epoch
                         },
                         namespace='/trading'
                         )


@stapp.agent(fx_message_live_stream_topic)
async def live_stream_fx_data(messages):
    async for fx_msg in messages:
        tick = await process_message(fx_msg.message)
        await m_sio.emit('original_data',
                         {'data': [tick['ask'], tick['bid'], tick['quote']], 'time': tick['timestamp']},
                         namespace='/trading'
                         )


@stapp.agent(fx_price_prediction_stream)
async def fx_price_prediction_stream_broad_cast(predictions):
    async for prediction in predictions:
        prediction: PricePrediction = prediction
        await m_sio.emit('price_prediction',
                         {
                             "time": prediction.time,
                             "price": prediction.price,
                             "price_change": prediction.price_change,
                             "origin_time": prediction.origin_time,
                             "originator": prediction.originator,
                         },
                         namespace='/trading'
                         )
