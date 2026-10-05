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
class ReversionSignalConfig:
    """CNC short-term mean reversion inside a long-term uptrend.

    The only variant that passed the pre-registered out-of-sample test
    (docs/RESEARCH.md, S4b): IS +1.53%/trade (t 6.5, n 597), OOS +0.69%/trade
    (t 1.8, n 168), net of Zerodha CNC costs. Every number below is the
    value that was tested; changing one invalidates that evidence.
    """
    min_bars: int = 200
    min_price: float = 50.0
    min_turnover_inr: float = 5.0e7          # ₹5 cr median daily value
    trend_ema: int = 200                     # long-term uptrend: close above EMA-200
    drop_lookback: int = 3                   # ...that has just fallen
    drop_pct: float = 0.08                   # ...8% or more in 3 sessions
    atr_period: int = 14
    stop_atr_mult: float = 3.0               # wide stop: reversion needs room to work
    exit_ema: int = 5                        # take the bounce: first close above EMA-5
    max_hold_bars: int = 7


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
class SizingConfig:
    """Fractional Kelly on R-multiples, shrunk toward a measured prior.

    risk_pct = clip(kelly_fraction x f*, floor, cap) x regime x breaker scalars
    where f* = p - (1-p)/b is computed from the posterior win rate p and payoff
    b. The prior is the real-data research result for this exact signal; the
    bot's own closed trades pull it toward reality as they accumulate.
    """
    prior_win_rate: float
    prior_payoff: float                      # avg winning R / avg losing R
    prior_strength: int = 40                 # prior counts as this many trades
    kelly_fraction: float = 0.5              # half-Kelly: ~75% of full-Kelly growth, far less ruin
    risk_floor_pct: float = 0.005            # never risk less than this while trading at all
    risk_cap_pct: float = 0.03               # never more than this on one idea
    max_positions: int = 5
    max_position_pct: float = 0.40           # notional cap per name, share of sleeve equity
    leverage: float = 1.0                    # buying power multiple (MIS margin)
    max_adv_participation: float = 0.01      # notional <= 1% of median daily traded value
    max_portfolio_heat_pct: float = 0.12     # sum of open risk across positions
    max_per_sector: int = 2


@dataclass
class ExitConfig:
    breakeven_at_r: float = 1.0              # move stop to entry (+costs) once +1R
    trail_atr_mult: float = 3.0              # chandelier distance from the best price since entry
    trail_tight_after_r: float = 3.0         # ...tightened once the trade is this far in profit
    trail_tight_atr_mult: float = 2.0
    stagnation_bars: int = 0                 # exit if still below stagnation_min_r after N bars (0 = off)
    stagnation_min_r: float = 0.5
    max_hold_bars: int = 0                   # hard time stop (0 = off)
    exit_above_ema: int = 0                  # mean-reversion target: first close above EMA-n (0 = off)
    square_off: time = None                  # MIS: flatten at this time, no exceptions


@dataclass
class BreakerConfig:
    max_consecutive_losses: int = 3          # halt new entries after this many losses in a row
    loss_cooldown_sessions: int = 1          # ...for this many sessions (intraday: rest of day)
    resume_size_mult: float = 0.5            # then trade at half size until the next winner
    daily_loss_limit_pct: float = 0.03       # realised + open P&L today vs sleeve equity
    max_drawdown_pct: float = 0.25           # from peak: halt ALL entries until a human resets
    max_trades_per_day: int = 0              # 0 = no cap
    kill_switch_file: str = 'STOP_TRADING'


def swing_sizing():
    # Prior = the OUT-OF-SAMPLE result of the accepted swing signal (S4b dip
    # reversion, docs/RESEARCH.md): 64% win, payoff 0.89 -> full Kelly +0.24,
    # half-Kelly ~12%, so the 3% cap is what binds. The weaker OOS half is
    # used deliberately — in-sample (70% / Kelly +0.39) is the optimistic one.
    return SizingConfig(prior_win_rate=0.64, prior_payoff=0.89, kelly_fraction=0.5,
                        risk_floor_pct=0.0075, risk_cap_pct=0.03, max_positions=5,
                        max_position_pct=0.40, leverage=1.0, max_portfolio_heat_pct=0.12,
                        max_per_sector=2)


def intraday_sizing():
    # Zerodha MIS on NSE equity: up to 5x intraday leverage (20% margin) on
    # most liquid names. The cap stack below decides how much of it is used.
    # No intraday variant passed out-of-sample (all net-negative after MIS
    # costs, n <= 31 on Yahoo's 59-day window), so the prior is set to that
    # measurement: Kelly < 0 -> the sizer trades the floor until the bot's
    # own trades prove an edge. Aggression is earned, not assumed.
    return SizingConfig(prior_win_rate=0.45, prior_payoff=1.10, kelly_fraction=0.5,
                        risk_floor_pct=0.005, risk_cap_pct=0.02, max_positions=3,
                        max_position_pct=2.50, leverage=5.0, max_portfolio_heat_pct=0.05,
                        max_per_sector=1, max_adv_participation=0.005)


def swing_exits():
    # Exactly the exit the accepted S4b variant was tested with: stop at
    # 3 x ATR, first close above EMA-5, or 7 sessions. No trail, no
    # breakeven — a reversion trade rarely reaches +1R before its target, and
    # a trail the research never ran would be an untested change.
    return ExitConfig(breakeven_at_r=1e9, trail_atr_mult=3.0, trail_tight_after_r=1e9,
                      trail_tight_atr_mult=2.0, stagnation_bars=0, max_hold_bars=7,
                      exit_above_ema=5)


def momentum_exits():
    # Trend-following exits (chandelier trail, stagnation, 40-bar cap). Kept for
    # momentum research variants; not used by the live swing strategy.
    return ExitConfig(breakeven_at_r=1.0, trail_atr_mult=3.0, trail_tight_after_r=3.0,
                      trail_tight_atr_mult=2.0, stagnation_bars=10, stagnation_min_r=0.5,
                      max_hold_bars=40)


def intraday_exits():
    return ExitConfig(breakeven_at_r=1.0, trail_atr_mult=2.0, trail_tight_after_r=2.0,
                      trail_tight_atr_mult=1.2, square_off=time(15, 10))


def swing_breakers():
    return BreakerConfig(max_consecutive_losses=4, loss_cooldown_sessions=3,
                         daily_loss_limit_pct=0.05, max_drawdown_pct=0.25)


def intraday_breakers():
    return BreakerConfig(max_consecutive_losses=3, loss_cooldown_sessions=1,
                         daily_loss_limit_pct=0.03, max_drawdown_pct=0.25,
                         max_trades_per_day=6)


@dataclass
class BotConfig:
    swing: SwingSignalConfig = field(default_factory=SwingSignalConfig)
    reversion: ReversionSignalConfig = field(default_factory=ReversionSignalConfig)
    intraday: IntradaySignalConfig = field(default_factory=IntradaySignalConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    swing_sizing: SizingConfig = field(default_factory=swing_sizing)
    intraday_sizing: SizingConfig = field(default_factory=intraday_sizing)
    swing_exits: ExitConfig = field(default_factory=swing_exits)
    intraday_exits: ExitConfig = field(default_factory=intraday_exits)
    swing_breakers: BreakerConfig = field(default_factory=swing_breakers)
    intraday_breakers: BreakerConfig = field(default_factory=intraday_breakers)

    def to_dict(self):
        return asdict(self)
