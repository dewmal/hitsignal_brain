# data processors - Pre, Past
from app.meth_one.method import m_sio
from app.meth_one.stream import fx_message_stream_topic, fx_message_live_stream_topic
from app.meth_one.utils import process_message
from application import stapp


async def call_after_process_live_feed(candle):
    print(candle)


@stapp.agent(fx_message_stream_topic)
async def count_page_views(views):
    async for fx_msg in views:
        await process_message(fx_msg.message, call_back=call_after_process_live_feed)


@stapp.agent(fx_message_live_stream_topic)
async def live_stream_fx_data(messages):
    async for fx_msg in messages:
        tick = await process_message(fx_msg.message)
        await m_sio.emit('original_data',
                               {'data': [tick['ask'], tick['bid'], tick['quote']], 'time': tick['timestamp']},
                               namespace='/trading'
                               )
