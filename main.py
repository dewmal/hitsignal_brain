import asyncio
from typing import Any

from mode import Service
from mode.threads import ServiceThread

import settings
from app.meth_one.method import MethodOne
from application import stapp, web_app


class Webserver(ServiceThread):
    import app.web

    def __init__(self,
                 port: int = 8000,
                 bind: str = None,
                 **kwargs: Any) -> None:
        self._app = web_app
        self.port = port
        self.bind = bind
        self._handler = None
        self._srv = None
        super().__init__(**kwargs)

    async def on_start(self) -> None:
        # handler = self._handler = self._app.create_server()
        # self.loop is the event loop in this thread
        #   self.parent_loop is the loop that created this thread.
        self._srv = self._app.create_server(self.bind, self.port)
        self.log.info('Serving on port %s', self.port)
        # loop = asyncio.get_event_loop()
        await asyncio.ensure_future(self._srv)
        # loop.run_forever()

    async def on_thread_stop(self) -> None:
        # on_thread_stop() executes in the thread.
        # on_stop() executes in parent thread.

        # quite a few steps required to stop the aiohttp server:
        if self._srv is not None:
            self.log.info('Closing server')
            self._srv.close()
            self.log.info('Waiting for server to close handle')
            await self._srv.wait_closed()
        if self._app is not None:
            self.log.info('Shutting down web application')
            await self._app.stop()
        if self._handler is not None:
            self.log.info('Waiting for handler to shut down')
            await self._handler.shutdown(60.0)
        if self._app is not None:
            self.log.info('Cleanup')
            # await self._app.cleanup()


class MainAppService(Service):

    def on_init_dependencies(self):
        return [stapp, MethodOne()]

    async def on_start(self) -> None:
        print('APP STARTING')
        import pydot
        import io
        o = io.StringIO()
        beacon = self.beacon.root or self.beacon
        beacon.as_graph().to_dot(o)
        graph, = pydot.graph_from_dot_data(o.getvalue())
        print('WRITING GRAPH TO image.png')
        with open(f'{settings.TEMP_FOLDER}/image.png', 'wb') as fh:
            fh.write(graph.create_png())


if __name__ == '__main__':
    from mode import Worker

    Worker(
        MainAppService(),
        Webserver(),
        loglevel='INFO',
        daemon=True).execute_from_commandline()
