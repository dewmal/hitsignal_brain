## Create Faust App here
import faust

import settings

stapp = faust.App(
    f'{settings.FAUST_APP_NAME}',
    broker=settings.FAUST_BROKER
)
'''
Async io socketio
'''
import socketio
from sanic import Sanic

mgr = socketio.AsyncRedisManager(settings.SOCKET_IO_MESSAGE_BROKER)
sio = socketio.AsyncServer(async_mode='sanic', client_manager=mgr)
web_app = Sanic()
sio.attach(web_app)
