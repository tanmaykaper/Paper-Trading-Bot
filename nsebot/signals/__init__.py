"""Signal engines. Each one answers a single question — "is there a trade on
this bar, and where is it wrong?" — and nothing about size, cost or slots."""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Signal:
    symbol: str
    mode: str            # 'swing' | 'intraday'
    side: str            # 'LONG' | 'SHORT'
    trigger: str
    asof: pd.Timestamp   # timestamp of the bar that produced the signal
    ref_price: float     # that bar's close — the price the decision was made at
    stop: float          # where the thesis is wrong (pre tick-rounding)
    atr: float
    score: float         # ranking only; higher is better
    features: dict = field(default_factory=dict)

    @property
    def risk_per_share(self):
        return abs(self.ref_price - self.stop)

    @property
    def stop_pct(self):
        return self.risk_per_share / self.ref_price if self.ref_price else 0.0

    def to_dict(self):
        d = {k: getattr(self, k) for k in ('symbol', 'mode', 'side', 'trigger', 'ref_price',
                                            'stop', 'atr', 'score')}
        d['asof'] = str(self.asof)
        d['stop_pct'] = round(self.stop_pct, 5)
        d.update({f'f_{k}': v for k, v in self.features.items()})
        return d


from .swing import SwingSignalEngine          # noqa: E402
from .intraday import IntradaySignalEngine    # noqa: E402
from .reversion import ReversionSignalEngine  # noqa: E402

__all__ = ['Signal', 'SwingSignalEngine', 'IntradaySignalEngine', 'ReversionSignalEngine']
