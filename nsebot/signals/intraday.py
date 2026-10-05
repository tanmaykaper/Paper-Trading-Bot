"""Intraday (MIS) opening-range breakout with VWAP and volume confirmation.

The first 15 minutes on NSE set the session's battle lines. A stock that is
"in play" — trading its opening range on well above its usual opening volume —
and then breaks that range on a strong, high-volume bar while on the right
side of VWAP has buyers (or sellers) in control right now. That is the whole
thesis; no oscillators.

Per completed 5-minute bar, between 09:30 and 13:30:

  IN PLAY     opening-range volume >= 1.5x its recent norm, range 0.3%-3%,
              price >= ₹100, median daily value >= ₹50 cr
  LONG        close > opening-range high AND close > session VWAP AND bar
              volume >= 1.3x the typical bar AND close in the upper part of
              the bar (CLV >= +0.3 — buyers won that bar's auction)
  SHORT       the mirror image (MIS shorting is legal intraday on NSE)
  ALIGNMENT   longs only while Nifty trades at or above its session open,
              shorts only at or below it — don't fight the tape you can see

Only the FIRST qualifying bar per symbol, per side, per session counts; a
breakout that already fired is not a fresh signal 20 minutes later.

The stop sits at the nearer of the opening-range midpoint and VWAP — losing
either one means the breakout failed — clamped to 0.3%-1.5% of price so the
stop is neither inside the noise nor too wide to size. Exits (trailing, the
pre-15:15 square-off) belong to the risk layer.

True order-book imbalance needs Kite's quote depth (paid). Until then the
bar's close location x relative volume is the free proxy for it.
"""

import numpy as np
import pandas as pd

from ..config import IntradaySignalConfig
from ..data import completed_bars
from ..indicators import atr, close_location, median_turnover, session_vwap
from . import Signal


class IntradaySignalEngine:

    def __init__(self, cfg=None):
        self.cfg = cfg or IntradaySignalConfig()

    # ── per-symbol, vectorised over every bar of every session ──────────────
    def session_features(self, bars, daily=None, index_bars=None):
        c = self.cfg
        d = bars.copy().reset_index(drop=True)
        d['datetime'] = pd.to_datetime(d['datetime'])
        d['session'] = d['datetime'].dt.date
        tod = d['datetime'].dt.time
        in_or = (tod >= c.session_open) & (tod < c.opening_range_end)
        expected_or_bars = max(int(round((_minutes(c.opening_range_end) - _minutes(c.session_open))
                                         / c.interval_minutes)), 1)

        g = d[in_or].groupby('session')
        or_high, or_low = g['high'].max(), g['low'].min()
        or_vol, or_n = g['volume'].sum(), g.size()
        sessions = pd.Index(sorted(d['session'].unique()))
        or_vol = or_vol.reindex(sessions)

        # Norms come from PRIOR sessions only, so the opening range of today
        # is judged against days that are already over.
        or_norm = or_vol.shift(1).rolling(5, min_periods=2).mean()
        bar_norm = (d[~in_or].groupby('session')['volume'].median()
                    .reindex(sessions).shift(1).rolling(5, min_periods=1).mean())
        session_turnover = ((d['close'] * d['volume']).groupby(d['session']).sum()
                            .reindex(sessions).shift(1).rolling(5, min_periods=1).median())

        if daily is not None and len(daily):
            dd = daily.copy()
            dd['session'] = pd.to_datetime(dd['datetime']).dt.date
            dd = dd.set_index('session')
            avg_vol = dd['volume'].shift(1).rolling(20, min_periods=5).mean()
            fallback = (avg_vol * c.or_volume_share_fallback).reindex(sessions, method='ffill')
            or_norm = or_norm.fillna(fallback)
            turnover = median_turnover(dd.reset_index()).set_axis(dd.index).shift(1)
            session_turnover = turnover.reindex(sessions, method='ffill').fillna(session_turnover)

        d['or_high'] = d['session'].map(or_high)
        d['or_low'] = d['session'].map(or_low)
        d['or_complete'] = d['session'].map(or_n).fillna(0) >= expected_or_bars
        d['or_mid'] = (d['or_high'] + d['or_low']) / 2.0
        d['or_pct'] = (d['or_high'] - d['or_low']) / d['or_mid']
        d['or_rvol'] = d['session'].map(or_vol) / d['session'].map(or_norm)
        d['bar_rvol'] = d['volume'] / d['session'].map(bar_norm)
        d['turnover'] = d['session'].map(session_turnover)
        d['vwap'] = session_vwap(d)
        d['clv'] = close_location(d)
        d['atr'] = atr(d, 14)

        in_window = (tod >= c.entry_start) & (tod < c.entry_end)
        tradeable = (in_window & d['or_complete']
                     & d['or_pct'].between(c.min_or_pct, c.max_or_pct)
                     & (d['or_rvol'] >= c.in_play_rvol)
                     & (d['close'] >= c.min_price)
                     & (d['turnover'] >= c.min_turnover_inr))
        surge = d['bar_rvol'] >= c.bar_rvol

        long_ = (tradeable & surge & (d['close'] > d['or_high']) & (d['close'] > d['vwap'])
                 & (d['clv'] >= c.min_abs_clv))
        short = (tradeable & surge & (d['close'] < d['or_low']) & (d['close'] < d['vwap'])
                 & (d['clv'] <= -c.min_abs_clv)) if c.allow_shorts else pd.Series(False, index=d.index)

        if c.align_with_index and index_bars is not None and len(index_bars):
            bias = self._index_bias(index_bars)
            m = pd.merge_asof(d[['datetime']].sort_values('datetime'), bias, on='datetime',
                              direction='backward')
            b = m['index_bias'].to_numpy()
            long_ &= ~(b < 0)        # unknown bias (NaN) does not block
            short &= ~(b > 0)

        d['long'] = long_.fillna(False).astype(bool)
        d['short'] = short.fillna(False).astype(bool)
        d['first_long'] = d['long'] & (d.groupby('session')['long'].cumsum() == 1)
        d['first_short'] = d['short'] & (d.groupby('session')['short'].cumsum() == 1)

        close = d['close']
        long_raw = np.maximum(d['or_mid'], d['vwap']).where(lambda s: s < close, d['or_mid'])
        short_raw = np.minimum(d['or_mid'], d['vwap']).where(lambda s: s > close, d['or_mid'])
        lo, hi = c.min_stop_pct * close, c.max_stop_pct * close
        d['long_stop'] = close - np.clip(close - long_raw, lo, hi)
        d['short_stop'] = close + np.clip(short_raw - close, lo, hi)
        d['score'] = (np.minimum(d['or_rvol'], 5.0) * 15.0 + np.minimum(d['bar_rvol'], 5.0) * 5.0
                      + d['clv'].abs() * 10.0)
        return d

    @staticmethod
    def _index_bias(index_bars):
        ix = index_bars.copy()
        ix['datetime'] = pd.to_datetime(ix['datetime'])
        ix['session'] = ix['datetime'].dt.date
        open_ = ix.groupby('session')['open'].transform('first')
        ix['index_bias'] = np.sign(ix['close'] - open_)
        return ix[['datetime', 'index_bias']].sort_values('datetime')

    # ── live: what fired on the latest completed bar ────────────────────────
    def scan(self, bars_by_symbol, now=None, daily_by_symbol=None, index_bars=None,
             exclude=()):
        """Signals whose trigger bar is the latest completed bar. `exclude`
        holds symbols already traded today — one trade per name per session."""
        c = self.cfg
        if now is not None and index_bars is not None:
            index_bars = completed_bars(index_bars, now, c.interval_minutes)
        out = []
        for sym, bars in bars_by_symbol.items():
            if sym in exclude or bars is None or len(bars) < 10:
                continue
            if now is not None:
                bars = completed_bars(bars, now, c.interval_minutes)
                if len(bars) < 10:
                    continue
            f = self.session_features(bars, (daily_by_symbol or {}).get(sym), index_bars)
            last = f.iloc[-1]
            for side, flag, stop_col in (('LONG', 'first_long', 'long_stop'),
                                         ('SHORT', 'first_short', 'short_stop')):
                if not bool(last[flag]):
                    continue
                out.append(Signal(
                    symbol=sym, mode='intraday', side=side, trigger='orb_vwap',
                    asof=pd.Timestamp(last['datetime']), ref_price=float(last['close']),
                    stop=float(last[stop_col]), atr=float(last['atr']),
                    score=float(last['score']),
                    features={'or_high': round(float(last['or_high']), 2),
                              'or_low': round(float(last['or_low']), 2),
                              'vwap': round(float(last['vwap']), 2),
                              'or_rvol': round(float(last['or_rvol']), 2),
                              'bar_rvol': round(float(last['bar_rvol']), 2),
                              'clv': round(float(last['clv']), 2)}))
        return sorted(out, key=lambda s: -s.score)


def _minutes(t):
    return t.hour * 60 + t.minute
