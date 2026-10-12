"""Point-in-time view over daily features — one object for live and backtest.

Built once per run from the day's fetch (live), or once over the whole
history (backtest). Either way the engine asks the same questions — the bar
on a date, the bars after a date, the signals on a date, the regime on a
date — so the backtest exercises exactly the code path the live run does.
"""

from datetime import time

import pandas as pd

from ..market import IST
from ..regime import RegimeDial
from ..signals import ReversionSignalEngine


class DailyMarket:

    def __init__(self, universe, index_df, signal_engine=None, regime=None):
        self.engine = signal_engine or ReversionSignalEngine()
        self.feats = {}
        for sym, df in (universe or {}).items():
            if df is None or len(df) < 30:
                continue
            self.feats[sym] = self.engine.features(df).set_index('datetime')
        self.regime = (regime or RegimeDial()).panel(index_df, universe)
        self.dates = pd.DatetimeIndex(pd.to_datetime(index_df['datetime']))

    def bar(self, symbol, date):
        f = self.feats.get(symbol)
        if f is None:
            return None
        date = pd.Timestamp(date)
        if date not in f.index:
            return None
        row = f.loc[date]
        return row.iloc[-1] if isinstance(row, pd.DataFrame) else row

    def bars_after(self, symbol, after, upto):
        f = self.feats.get(symbol)
        if f is None:
            return pd.DataFrame()
        after, upto = pd.Timestamp(after), pd.Timestamp(upto)
        return f[(f.index > after) & (f.index <= upto)]

    def last_close(self, symbol, asof):
        f = self.feats.get(symbol)
        if f is None:
            return None
        s = f.loc[:pd.Timestamp(asof), 'close']
        return float(s.iloc[-1]) if len(s) else None

    def signals(self, asof):
        return self.engine.signals_from_features(self.feats, asof)

    def regime_at(self, asof):
        r = self.regime.loc[:pd.Timestamp(asof)]
        if r.empty:
            return {'score': 0.5, 'size_mult': 0.75, 'max_new_entries': 1, 'breadth': None}
        row = r.iloc[-1]
        return {'score': round(float(row['score']), 3), 'size_mult': round(float(row['size_mult']), 3),
                'max_new_entries': int(row['max_new_entries']),
                'breadth': None if pd.isna(row['breadth']) else round(float(row['breadth']), 3)}

    def turnover(self, symbol, asof):
        b = self.bar(symbol, asof)
        return None if b is None else float(b['turnover'])


def completed_session(index_df, now=None, settle=time(15, 45)):
    """The latest session whose daily bar is FINAL. Before 15:45 IST on a
    trading day, Yahoo's bar for today is still forming — deciding on it would
    mark the session processed and the real close would never be seen."""
    dates = pd.DatetimeIndex(pd.to_datetime(index_df['datetime'])).normalize()
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=IST)
    now = now.tz_localize(IST) if now.tzinfo is None else now.tz_convert(IST)
    today = pd.Timestamp(now.date())
    if dates[-1] == today and now.time() < settle:
        return dates[-2] if len(dates) > 1 else None
    return dates[-1]
