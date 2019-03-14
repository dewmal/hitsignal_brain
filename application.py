import faust

import settings

stapp = faust.App(
    f'{settings.FAUST_APP_NAME}',
    broker=settings.FAUST_BROKER
)


class PageView(faust.Record):
    id: str
    user: str


page_view_topic = stapp.topic('page_views_2', value_type=PageView)
