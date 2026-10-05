"""Every tunable in one place.

Dataclasses rather than scattered module constants, so a backtest can run two
configurations side by side and the live runner can log the exact settings it
traded with. Defaults are the deployed values.
"""

from dataclasses import dataclass, field, asdict
from datetime import time


@dataclass
class SwingSignalConfig:
    """CNC momentum continuation on daily bars.

    Five checks, not twenty-seven: liquidity, trend, trigger, extension, rank.
    """
    min_bars: int = 130
    min_price: float = 50.0
    min_turnover_inr: float = 5.0e7          # ₹5 cr median daily value
    ema_fast: int = 20
    ema_slow: int = 50
    breakout_lookback: int = 20              # close above the prior 20-session closing high
    breakout_rvol: float = 1.5               # ...on 1.5x average volume
    thrust_rvol: float = 1.2                 # continuation bar: takes out yesterday's high
    thrust_min_clv: float = 0.40             # ...closing in the top 30% of its range
    thrust_near_high_pct: float = 0.05       # ...within 5% of the 20-session high
    max_extension_atr: float = 3.0           # skip blow-off bars this far above EMA-20
    max_day_return: float = 0.08             # skip +8% days (circuit chasing, exhaustion)
    rs_lookback: int = 63                    # ~3 months of relative strength
    rs_weight: float = 0.7                   # rank = 0.7 RS percentile + 0.3 rvol percentile
    atr_period: int = 14
    stop_atr_mult: float = 2.5               # initial stop distance suggestion
    min_stop_pct: float = 0.025
    max_stop_pct: float = 0.10


@dataclass
class IntradaySignalConfig:
    """MIS opening-range breakout, confirmed by VWAP and volume. Long and short."""
    interval_minutes: int = 5
    session_open: time = time(9, 15)
    opening_range_end: time = time(9, 30)    # first 15 minutes define the range
    entry_start: time = time(9, 30)
    entry_end: time = time(13, 30)           # late breakouts have no session left to run
    min_price: float = 100.0
    min_turnover_inr: float = 5.0e8          # ₹50 cr median daily value
    min_or_pct: float = 0.003                # range narrower than 0.3% is noise
    max_or_pct: float = 0.030                # wider than 3% makes the stop unaffordable
    in_play_rvol: float = 1.5                # opening-range volume vs its own recent norm
    bar_rvol: float = 1.3                    # breakout bar volume vs the typical bar
    min_abs_clv: float = 0.30                # breakout bar must close on the right side
    or_volume_share_fallback: float = 0.12   # share of daily volume NSE trades in 15 min
    allow_shorts: bool = True                # MIS shorts are legal intraday on NSE
    align_with_index: bool = True            # longs only above index VWAP, shorts below
    min_stop_pct: float = 0.003
    max_stop_pct: float = 0.015


@dataclass
class RegimeConfig:
    """One dial, never an off switch. Weak tape = smaller and fewer, not zero."""
    ema_period: int = 50
    breadth_ema: int = 50
    size_floor: float = 0.5                  # weakest tape still trades at half size
    size_ceiling: float = 1.0
    entries_floor: int = 1                   # new swing entries/day in the weakest tape
    entries_ceiling: int = 4


@dataclass
class BotConfig:
    swing: SwingSignalConfig = field(default_factory=SwingSignalConfig)
    intraday: IntradaySignalConfig = field(default_factory=IntradaySignalConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)

    def to_dict(self):
        return asdict(self)
