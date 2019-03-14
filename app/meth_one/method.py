from mode import Service
from mode.threads import ServiceThread

from app.meth_one.stream import fx_message_stream_topic, FxStreamMessage
from app.meth_one.utils import read_from_ws


class MethodOne(ServiceThread):
    from . import processors, stream

    async def on_start(self) -> None:
        print('MethodOne STARTING')

    @Service.task
    async def web_steam_data_eur_usd(self):
        async def result_reader(message):
            await fx_message_stream_topic.send(value=FxStreamMessage(message=message))

        await read_from_ws("EURUSD", result_reader)

    @Service.task
    async def web_steam_data_eur_jpy(self):
        async def result_reader(message):
            await fx_message_stream_topic.send(value=FxStreamMessage(message=message))
        await read_from_ws("EURJPY", result_reader)
