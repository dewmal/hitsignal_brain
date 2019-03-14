from mode import Service

import settings
from app.meth_one.method import MethodOne
from application import stapp


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

    Worker(MainAppService(), loglevel="info",
           daemon=True).execute_from_commandline()
