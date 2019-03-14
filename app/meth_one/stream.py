# Stream Connect
import faust

from application import stapp


class FxStreamMessage(faust.Record, isodates=True):
    message: str


fx_message_stream_topic = stapp.topic('fx_message_stream', value_type=FxStreamMessage)
