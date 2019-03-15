# data processors - Pre, Past
from app.meth_one.method import m_sio
from app.meth_one.stream import fx_message_stream_topic, fx_message_live_stream_topic
from app.meth_one.utils import process_message
from application import stapp


@stapp.agent(fx_message_stream_topic)
async def count_page_views(views):
    async for message in views:
        print("Method One For Stream Reader")
        print(message)
        print("......")


@stapp.agent(fx_message_live_stream_topic)
async def live_stream_fx_data(messages):
    async for fx_msg in messages:
        tick = await process_message(fx_msg.message)
        print("Live feed")
        print(tick)
        print("....")

        print(m_sio.channel, m_sio.name, m_sio.server)

        res = await m_sio.emit('original_data',
                               {'data': [tick['ask'], tick['bid'], tick['quote']], 'time': tick['timestamp']},
                               namespace='/trading'
                               )
        print(res)
