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

mgr = socketio.KombuManager(settings.KOMBU_SOCKET_IO_MESSAGE_BROKER)
sio = socketio.AsyncServer(async_mode='sanic')
web_app = Sanic()
sio.attach(web_app)
