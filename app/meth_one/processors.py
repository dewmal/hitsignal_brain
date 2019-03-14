# data processors - Pre, Past
from app.meth_one.stream import fx_message_stream_topic
from application import stapp


@stapp.agent(fx_message_stream_topic)
async def count_page_views(views):
    async for message in views:
        print("Method One For Stream Reader")
        print(message)
        print("......")
