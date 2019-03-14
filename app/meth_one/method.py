from mode import Service
from mode.threads import ServiceThread

from app.meth_one.stream import fx_message_stream_topic, FxStreamMessage
from app.meth_one.utils import read_from_ws


class MethodOne(ServiceThread):
    async def on_start(self) -> None:
        print('MethodOne STARTING')

    async def on_started(self) -> None:
        # from .stream import page_view_topic, PageView
        # page_view_topic.send(PageView(id="One", user="One"))
        pass

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
