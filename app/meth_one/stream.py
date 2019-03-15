# Stream Connect
import faust

from application import stapp


class FxStreamMessage(faust.Record, isodates=True):
    message: str
    type: str


class Candle(faust.Record, isodates=True):
    open_time: float
    epoch: float
    open: float
    low: float
    high: float
    close: float
    symbol: str

    def to_olhc_array(self):
        return [self.open_time, self.open, self.low, self.high, self.close]  # OLHC


fx_message_stream_topic = stapp.topic('fx_message_stream', value_type=FxStreamMessage)
fx_message_live_stream_topic = stapp.topic('fx_message_live_stream', value_type=FxStreamMessage)
