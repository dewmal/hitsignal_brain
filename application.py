import faust

import settings

stapp = faust.App(
    f'{settings.FAUST_APP_NAME}',
    broker=settings.FAUST_BROKER
)
