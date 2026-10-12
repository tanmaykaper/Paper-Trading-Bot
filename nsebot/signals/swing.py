"""Swing (CNC) momentum continuation.

What V1's own ledger says worked: buying names already in motion and holding
them ~2 weeks (+₹1,245 on 10-20 day holds; stoch_cross and cmf_accum were the
only profitable patterns). What lost: buying pullbacks in a falling tape
(-₹1,508 over 30 rows). So this engine buys strength only, and does it with
five checks instead of V2's twenty-seven:

  1 LIQUID    price >= ₹50, median daily value >= ₹5 cr, enough history
  2 TREND     close > EMA-50 and EMA-20 > EMA-50
  3 TRIGGER   either
                breakout   close above the prior 20-session closing high on
                           >= 1.5x average volume (volume-weighted momentum)
                thrust     takes out yesterday's high, closes in the top 30%
                           of its range on >= 1.2x volume, within 5% of the
                           20-session high (momentum resuming after a pause)
  4 NOT EXTENDED  <= 3 ATR above EMA-20 and not a +8% day (blow-off / circuit)
  5 RANK      0.7 x 3-month relative-strength percentile across the universe
              + 0.3 x relative-volume percentile — used to ORDER candidates,
              never to reject them

There is no regime gate here and no fundamentals, sentiment, MACD, ADX or RSI.
Regime scales size downstream; it does not decide participation.
"""

import numpy as np
import pandas as pd

from ..config import SwingSignalConfig
from ..indicators import (atr, close_location, cross_sectional_percentile, ema,
                          median_turnover, prior_high, rate_of_change, relative_volume)
from . import Signal


class SwingSignalEngine:

    def __init__(self, cfg=None):
        self.cfg = cfg or SwingSignalConfig()

    # ── per-symbol, vectorised over every bar ───────────────────────────────
    def features(self, df):
        c = self.cfg
        d = df.copy()
        d['datetime'] = pd.to_datetime(d['datetime'])
        close = d['close'].astype(float)

        d['ema_fast'] = ema(close, c.ema_fast)
        d['ema_slow'] = ema(close, c.ema_slow)
        d['atr'] = atr(d, c.atr_period)
        d['rvol'] = relative_volume(d['volume'], 20)
        d['clv'] = close_location(d)
        d['turnover'] = median_turnover(d, 20)
        d['close_high_n'] = prior_high(close, c.breakout_lookback)
        d['high_n'] = prior_high(d['high'], c.breakout_lookback)
        d['ret1'] = close.pct_change()
        d['rs'] = rate_of_change(close, c.rs_lookback)
        d['extension'] = (close - d['ema_fast']) / d['atr'].replace(0, np.nan)

        bar_no = np.arange(len(d))
        d['liquid'] = ((bar_no >= c.min_bars - 1) & (close >= c.min_price)
                       & (d['turnover'] >= c.min_turnover_inr))
        d['trend'] = (close > d['ema_slow']) & (d['ema_fast'] > d['ema_slow'])
        breakout = (close > d['close_high_n']) & (d['rvol'] >= c.breakout_rvol)
        thrust = ((close > d['high'].shift(1)) & (d['clv'] >= c.thrust_min_clv)
                  & (d['rvol'] >= c.thrust_rvol)
                  & (close >= d['high_n'] * (1.0 - c.thrust_near_high_pct))
                  & (close > d['ema_fast']))
        d['not_extended'] = (d['extension'] <= c.max_extension_atr) & (d['ret1'] <= c.max_day_return)

        qualifies = d['liquid'] & d['trend'] & d['not_extended']
        d['trigger'] = np.where(qualifies & breakout, 'breakout',
                                np.where(qualifies & thrust, 'thrust', ''))

        dist = np.clip(c.stop_atr_mult * d['atr'], c.min_stop_pct * close, c.max_stop_pct * close)
        d['stop'] = close - dist
        return d

    # ── cross-section ───────────────────────────────────────────────────────
    def panel(self, universe):
        """Every (date, symbol) trigger across history, scored against the
        universe ON THAT DATE. Used by research and by scan() alike, so the
        live decision and the backtest are the same computation."""
        feats = {s: self.features(df) for s, df in universe.items()
                 if df is not None and len(df) >= 30}
        if not feats:
            return pd.DataFrame(), feats

        rs = pd.DataFrame({s: f.set_index('datetime')['rs'] for s, f in feats.items()})
        rv = pd.DataFrame({s: f.set_index('datetime')['rvol'] for s, f in feats.items()})
        liquid = pd.DataFrame({s: f.set_index('datetime')['liquid'] for s, f in feats.items()})
        rs_pct = cross_sectional_percentile(rs.where(liquid.fillna(False).astype(bool)))
        rv_pct = cross_sectional_percentile(rv.where(liquid.fillna(False).astype(bool)))
        score = self.cfg.rs_weight * rs_pct + (1.0 - self.cfg.rs_weight) * rv_pct

        rows = []
        for sym, f in feats.items():
            ev = f[f['trigger'] != '']
            if ev.empty:
                continue
            ev = ev.assign(symbol=sym)
            ev['rs_pct'] = rs_pct[sym].reindex(ev['datetime']).to_numpy()
            ev['rvol_pct'] = rv_pct[sym].reindex(ev['datetime']).to_numpy()
            ev['score'] = score[sym].reindex(ev['datetime']).to_numpy()
            rows.append(ev)
        events = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return events, feats

    def scan(self, universe, asof=None):
        """Today's candidates, best first. A symbol whose last bar is not the
        as-of session (halted, delisted, stale feed) cannot signal."""
        events, feats = self.panel(universe)
        if events.empty:
            return []
        if asof is None:
            asof = max(f['datetime'].iloc[-1] for f in feats.values())
        asof = pd.Timestamp(asof)
        today = events[events['datetime'] == asof].sort_values('score', ascending=False)
        out = []
        for _, r in today.iterrows():
            out.append(Signal(
                symbol=r['symbol'], mode='swing', side='LONG', trigger=r['trigger'],
                asof=asof, ref_price=float(r['close']), stop=float(r['stop']),
                atr=float(r['atr']), score=float(np.nan_to_num(r['score'], nan=0.0)),
                features={'rvol': round(float(r['rvol']), 2), 'rs_63d': round(float(r['rs']), 4),
                          'rs_pct': None if pd.isna(r['rs_pct']) else round(float(r['rs_pct']), 1),
                          'extension_atr': round(float(r['extension']), 2),
                          'clv': round(float(r['clv']), 2),
                          'turnover_cr': round(float(r['turnover']) / 1e7, 1)}))
        return out
