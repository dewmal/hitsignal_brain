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
        return [self.open_time, self.open, self.low, self.high, self.close, 0]  # OLHC


class CandleWindow(faust.Record, isodates=True):
    data: list
    window_lenght: int
    window_time: float


class PricePrediction(faust.Record, isodates=True):
    time: float
    price: float
    symbol: str
    price_change: float
    originator: str
    origin_time: float


fx_message_stream_topic_eur_usd = stapp.topic('fx_message_stream_eur_usd', value_type=FxStreamMessage)
fx_message_stream_topic_eur_jpy = stapp.topic('fx_message_stream_eur_jpy', value_type=FxStreamMessage)
fx_message_stream_topic_eur_chf = stapp.topic('fx_message_stream_eur_chf', value_type=FxStreamMessage)
fx_message_stream_topic_eur_gbp = stapp.topic('fx_message_stream_eur_gbp', value_type=FxStreamMessage)
fx_message_live_stream_topic = stapp.topic('fx_message_live_stream', value_type=FxStreamMessage)

# Manage Candles
fx_candle_stream = stapp.topic('fx_candle_stream', value_type=Candle)
fx_candle_1min_stream = stapp.topic('fx_candle_1min_stream', value_type=Candle)
fx_candle_5min_window_stream = stapp.topic('fx_candle_5min_window_stream', value_type=CandleWindow)
fx_price_prediction_stream = stapp.topic('fx_price_prediction_stream', value_type=PricePrediction)
