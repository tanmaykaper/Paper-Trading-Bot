"""Market regime as a size dial, never an off switch.

V2 let six discrete triggers drive exposure to exactly zero and required a
0.54 composite to trade at all; with Nifty under its EMA-50 that bar was
unreachable and the book sat in cash for the whole correction (AUTOPSY §1).

Here the regime does exactly two things, both continuous:
  size_mult        0.5x in the weakest tape -> 1.0x in a strong one
  max_new_entries  1 new swing entry/day at worst -> 4 at best

Two inputs, one each for the index and the stocks underneath it:
  index    distance of Nifty from its EMA-50 (±2.5% spans the whole scale)
  breadth  share of the universe above its own EMA-50 (20% -> 70%)

Catastrophe protection is not this module's job; the circuit breakers in the
risk layer halt trading on realised losses, which is evidence about THIS
strategy rather than a forecast about the market.
"""

import numpy as np
import pandas as pd

from .config import RegimeConfig
from .indicators import ema


class RegimeDial:

    def __init__(self, cfg=None):
        self.cfg = cfg or RegimeConfig()

    def panel(self, index_df, universe=None):
        """Per-date regime readings. Point-in-time: row t reads bars <= t only."""
        c = self.cfg
        idx = index_df.set_index(pd.to_datetime(index_df['datetime']))['close'].astype(float)
        index_comp = np.clip((idx / ema(idx, c.ema_period) - 1.0) / 0.05 + 0.5, 0.0, 1.0)

        breadth = None
        if universe:
            above = {}
            for sym, df in universe.items():
                if df is None or len(df) < c.breadth_ema:
                    continue
                s = df.set_index(pd.to_datetime(df['datetime']))['close'].astype(float)
                above[sym] = (s > ema(s, c.breadth_ema)).astype(float)
            if above:
                wide = pd.DataFrame(above).reindex(idx.index)
                counts = wide.notna().sum(axis=1)
                breadth = wide.mean(axis=1).where(counts >= 10)

        out = pd.DataFrame({'index_close': idx, 'index_comp': index_comp})
        if breadth is not None:
            out['breadth'] = breadth
            breadth_comp = np.clip((breadth - 0.20) / 0.50, 0.0, 1.0)
            out['score'] = np.where(breadth.notna(), 0.5 * index_comp + 0.5 * breadth_comp,
                                    index_comp)
        else:
            out['breadth'] = np.nan
            out['score'] = index_comp
        out['size_mult'] = c.size_floor + (c.size_ceiling - c.size_floor) * out['score']
        out['max_new_entries'] = np.round(
            c.entries_floor + (c.entries_ceiling - c.entries_floor) * out['score']).astype(int)
        return out

    def today(self, index_df, universe=None):
        p = self.panel(index_df, universe)
        row = p.iloc[-1]
        return {
            'asof': p.index[-1],
            'score': round(float(row['score']), 3),
            'index_comp': round(float(row['index_comp']), 3),
            'breadth': None if pd.isna(row['breadth']) else round(float(row['breadth']), 3),
            'size_mult': round(float(row['size_mult']), 3),
            'max_new_entries': int(row['max_new_entries']),
        }
