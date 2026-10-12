"""CNC cross-sectional momentum: rank the whole NSE market by its last year,
skipping the last month, and own the top of the list.

  ELIGIBLE  on the decision date: 20-session median traded value >= ₹5 cr and
            close >= ₹50 — both measured that day, never with hindsight
  SCORE     12-1 momentum = close 21 sessions ago / close 252 sessions ago - 1
            (the latest month is skipped: very recent winners tend to give
            some back)
  RANK      eligible names with a score, best first

Sessions are counted on the market calendar, the union of every symbol's
trading days, so a suspended stock's missing days leave its score undefined
instead of silently stretching its window. Everything here is computed the
way the research panel computed it (research.experiments.Panel, used by
research.phase5b); tests/test_momentum.py checks the two agree.
"""

import numpy as np
import pandas as pd

from ..config import MomentumConfig
from ..indicators import median_turnover

COLUMNS = ['close', 'turnover', 'score', 'eligible', 'valid', 'rank']


class MomentumRanker:

    def __init__(self, cfg=None):
        self.cfg = cfg or MomentumConfig()

    def frames(self, universe):
        """Symbol -> bars indexed by date, for symbols with enough history."""
        return {s: df.set_index(pd.to_datetime(df['datetime'])).sort_index() for s, df in universe.items()
                if df is not None and len(df) >= self.cfg.min_history_bars}

    def rank(self, universe, asof, frames=None):
        """One row per symbol (alphabetical): close, turnover, score, eligible,
        valid (eligible and scored) and rank (0 = best; unscored names last)."""
        c = self.cfg
        asof = pd.Timestamp(asof)
        frames = frames if frames is not None else self.frames(universe)
        past = {s: f.loc[:asof] for s, f in frames.items()}
        past = {s: f for s, f in past.items() if len(f)}
        if not past:
            return pd.DataFrame(columns=COLUMNS)
        # Only the last lookback+1 sessions of the calendar matter, and every one
        # of them is within the last lookback+1 bars of a stock that traded that
        # day — so the union of those tails is exact, and far cheaper than the
        # union of whole histories.
        tail = c.lookback + 1
        calendar = pd.DatetimeIndex(sorted(set().union(*[f.index[-tail:] for f in past.values()])))
        pos = len(calendar) - 1
        d_skip = calendar[pos - c.skip] if pos >= c.skip else None
        d_look = calendar[pos - c.lookback] if pos >= c.lookback else None

        syms = sorted(past)
        rows = []
        for s in syms:
            f = past[s]
            today = f.index[-1] == asof
            close = float(f['close'].iloc[-1]) if today else np.nan
            turnover = float(median_turnover(f.iloc[-c.turnover_window:].reset_index(drop=True),
                                             c.turnover_window).iloc[-1]) if today else np.nan
            c_skip = float(f['close'].get(d_skip, np.nan)) if d_skip is not None else np.nan
            c_look = float(f['close'].get(d_look, np.nan)) if d_look is not None else np.nan
            score = c_skip / c_look - 1.0 if np.isfinite(c_skip) and np.isfinite(c_look) and c_look else np.nan
            eligible = bool(turnover >= c.min_turnover_inr and close >= c.min_price)   # NaN -> False
            rows.append((close, turnover, score, eligible, eligible and bool(np.isfinite(score))))
        out = pd.DataFrame(rows, index=syms, columns=COLUMNS[:-1])
        sc = np.where(out['valid'].to_numpy(), out['score'].to_numpy(dtype=float), -np.inf)
        order = np.argsort(-sc)                      # the research tie-break, exactly
        rank = np.empty(len(syms), dtype=int)
        rank[order] = np.arange(len(syms))
        out['rank'] = rank
        return out
