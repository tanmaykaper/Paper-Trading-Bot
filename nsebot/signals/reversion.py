"""Swing (CNC) mean reversion: buy a sharp dip in a stock that is still in a
long-term uptrend, sell the bounce.

Why this and not breakouts: on 774 sessions of real NSE data the Phase 2
breakout/thrust triggers lost money in BOTH halves (IS -0.73%/trade, OOS
-0.87%), with or without a trailing exit. The same data showed the opposite
effect clearly — names that fall hard over a few days tend to recover over
the next few, and the deeper the fall the stronger the bounce (5% drop: OOS
+0.15%/trade; 8% drop: OOS +0.69%/trade). Breakout buyers were paying for
exactly the move that then reverted.

  1 LIQUID     price >= ₹50, median daily value >= ₹5 cr, >= 200 bars
  2 UPTREND    close > EMA-200 (a dip in a long-term winner, not a falling knife
               in a long-term loser)
  3 DIP        close at least 8% below its close 3 sessions ago
  STOP         3 x ATR(14) below the signal close
  EXIT         first close above EMA-5, or 7 sessions, or the stop
               (the exit lives in risk.exits via ExitConfig.exit_above_ema)

Ranking: deepest dip in ATR units first. Rarely matters — the trigger fires
about once every two sessions across 255 names — and it was not separately
tested, so it is a tie-break only.
"""

import numpy as np
import pandas as pd

from ..config import ReversionSignalConfig
from ..indicators import atr, ema, median_turnover
from . import Signal


class ReversionSignalEngine:

    def __init__(self, cfg=None):
        self.cfg = cfg or ReversionSignalConfig()

    def features(self, df):
        c = self.cfg
        d = df.copy()
        d['datetime'] = pd.to_datetime(d['datetime'])
        close = d['close'].astype(float)
        d['ema_trend'] = ema(close, c.trend_ema)
        d['ema_exit'] = ema(close, c.exit_ema)
        d['atr'] = atr(d, c.atr_period)
        d['turnover'] = median_turnover(d, 20)
        d['drop'] = close / close.shift(c.drop_lookback) - 1.0
        bar_no = np.arange(len(d))
        d['liquid'] = ((bar_no >= c.min_bars - 1) & (close >= c.min_price)
                       & (d['turnover'] >= c.min_turnover_inr))
        d['uptrend'] = close > d['ema_trend']
        d['signal'] = d['liquid'] & d['uptrend'] & (d['drop'] <= -c.drop_pct)
        d['stop'] = close - c.stop_atr_mult * d['atr']
        d['depth_atr'] = -(close - close.shift(c.drop_lookback)) / d['atr'].replace(0, np.nan)
        return d

    def scan(self, universe, asof=None):
        """Today's dips, deepest first. Only symbols that printed the as-of
        session can signal."""
        feats = {s: self.features(df).set_index('datetime') for s, df in universe.items()
                 if df is not None and len(df) >= 30}
        if not feats:
            return []
        if asof is None:
            asof = max(f.index[-1] for f in feats.values())
        return self.signals_from_features(feats, asof)

    def signals_from_features(self, feats, asof):
        """feats: {symbol: features() frame indexed by datetime}. The one place a
        Signal is built, shared by live scans and the portfolio backtest."""
        asof = pd.Timestamp(asof)
        out = []
        for sym, f in feats.items():
            if asof not in f.index:
                continue
            last = f.loc[asof]
            if isinstance(last, pd.DataFrame):
                last = last.iloc[-1]
            if not bool(last['signal']):
                continue
            out.append(Signal(
                symbol=sym, mode='swing', side='LONG', trigger='dip_reversion', asof=asof,
                ref_price=float(last['close']), stop=float(last['stop']), atr=float(last['atr']),
                score=float(np.nan_to_num(last['depth_atr'], nan=0.0)),
                features={'drop_3d': round(float(last['drop']), 4),
                          'depth_atr': round(float(last['depth_atr']), 2),
                          'ema_exit': round(float(last['ema_exit']), 2),
                          'pct_above_ema200': round(float(last['close'] / last['ema_trend'] - 1), 4),
                          'turnover_cr': round(float(last['turnover']) / 1e7, 1)}))
        return sorted(out, key=lambda s: -s.score)
