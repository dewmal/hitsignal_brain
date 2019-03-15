import socketio
from mode import Service
from mode.threads import ServiceThread

import settings
from app.meth_one.stream import fx_message_stream_topic, FxStreamMessage, fx_message_live_stream_topic
from app.meth_one.utils import read_from_ws_history, read_from_ws_live

m_sio = socketio.AsyncRedisManager(settings.SOCKET_IO_MESSAGE_BROKER, write_only=True)


class MethodOne(ServiceThread):
    from . import processors, stream

    async def on_start(self) -> None:
        print('MethodOne STARTING')

    # @Service.task
    async def web_stream_live_data_eur_usd(self):
        async def result_reader(message):
            await fx_message_live_stream_topic.send(value=FxStreamMessage(message=message))

        await read_from_ws_live("EURUSD", result_reader)

    @Service.task
    async def web_steam_data_eur_usd(self):
        async def result_reader(message):
            await fx_message_stream_topic.send(value=FxStreamMessage(message=message))

        await read_from_ws_history("EURUSD", result_reader, count=30)

    # @Service.task
    async def web_steam_data_eur_jpy(self):
        async def result_reader(message):
            await fx_message_stream_topic.send(value=FxStreamMessage(message=message))

        await read_from_ws_history("EURJPY", result_reader)
