"""Exit engine: one pure function, evaluated on every completed bar.

Priority order — the first rule that fires wins:

  1 SQUARE-OFF   MIS only: at the configured time (default 15:10 IST) every
                 intraday position is closed at the bar's close. Zerodha
                 auto-squares MIS from 15:20 with a ₹50+GST penalty per
                 order; we never let it get there.
  2 STOP         price traded through the stop. Filled at the stop, or at the
                 open if the bar GAPPED through it — the honest fill, which is
                 worse than the stop.
  3 TARGET       mean-reversion strategies only: first close back above
                 EMA-n (exit_above_ema) — take the bounce, at the close.
  4 TIME         stagnation (not +0.5R after N bars) or a hard max hold.
  5 TRAIL        otherwise ratchet the stop: breakeven at +1R, then a
                 chandelier at k x ATR from the best price since entry,
                 tightened once the trade is deep in profit. Stops only ever
                 move in the trade's favour.

No fixed profit target. V1's money came from time exits on trades that were
still running (+₹1,557) while its two targets added ₹242; a target caps
exactly the trades a momentum system exists to catch.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Position:
    symbol: str
    mode: str                   # 'swing' | 'intraday'
    side: str                   # 'LONG' | 'SHORT'
    qty: int
    entry_price: float
    initial_stop: float
    stop: float
    entry_time: pd.Timestamp
    atr: float
    bars_held: int = 0
    best_price: float = None    # highest high (long) / lowest low (short) since entry
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.best_price is None:
            self.best_price = self.entry_price

    @property
    def sign(self):
        return 1.0 if self.side == 'LONG' else -1.0

    @property
    def risk_per_share(self):
        return abs(self.entry_price - self.initial_stop)

    def r_multiple(self, price):
        rps = self.risk_per_share
        return self.sign * (price - self.entry_price) / rps if rps > 0 else 0.0


@dataclass
class ExitDecision:
    action: str                 # 'HOLD' | 'TRAIL' | 'EXIT'
    price: float = None
    reason: str = ''
    new_stop: float = None


def evaluate(pos, bar, cfg, atr=None, now=None, exit_level=None):
    """pos: Position (not mutated). bar: mapping with open/high/low/close and
    'datetime'. atr: latest ATR for the trail (defaults to entry ATR).
    exit_level: the strategy's target line on this bar (e.g. EMA-5) when
    cfg.exit_above_ema is set."""
    o, h, l, c = (float(bar[k]) for k in ('open', 'high', 'low', 'close'))
    ts = pd.Timestamp(bar.get('datetime') if hasattr(bar, 'get') else bar['datetime'])
    long_ = pos.side == 'LONG'

    # 1 ── MIS square-off ─────────────────────────────────────────────────────
    if cfg.square_off is not None:
        clock = pd.Timestamp(now) if now is not None else ts
        if clock.time() >= cfg.square_off:
            return ExitDecision('EXIT', c, f'square-off {cfg.square_off.strftime("%H:%M")}')

    # 2 ── stop (gap-aware) ───────────────────────────────────────────────────
    if long_ and l <= pos.stop:
        fill = o if o < pos.stop else pos.stop
        return ExitDecision('EXIT', fill, 'stop (gap)' if o < pos.stop else _stop_label(pos))
    if not long_ and h >= pos.stop:
        fill = o if o > pos.stop else pos.stop
        return ExitDecision('EXIT', fill, 'stop (gap)' if o > pos.stop else _stop_label(pos))

    held = pos.bars_held + 1
    r_now = pos.r_multiple(c)

    # 3 ── reversion target ───────────────────────────────────────────────────
    if cfg.exit_above_ema and exit_level is not None and np.isfinite(exit_level):
        if (long_ and c > exit_level) or (not long_ and c < exit_level):
            return ExitDecision('EXIT', c, f'reversion target: close vs EMA-{cfg.exit_above_ema}')

    # 4 ── time ───────────────────────────────────────────────────────────────
    if cfg.max_hold_bars and held >= cfg.max_hold_bars:
        return ExitDecision('EXIT', c, f'max hold {cfg.max_hold_bars} bars')
    if cfg.stagnation_bars and held >= cfg.stagnation_bars and r_now < cfg.stagnation_min_r:
        return ExitDecision('EXIT', c, f'stagnant: {r_now:+.2f}R after {held} bars')

    # 5 ── trail ──────────────────────────────────────────────────────────────
    best = max(pos.best_price, h) if long_ else min(pos.best_price, l)
    best_r = pos.r_multiple(best)
    a = float(atr if atr is not None and atr > 0 else pos.atr)
    candidates = [pos.stop]
    if best_r >= cfg.breakeven_at_r:
        cost_buffer = 0.001 * pos.entry_price          # breakeven means after costs
        candidates.append(pos.entry_price + pos.sign * cost_buffer)
    if best_r >= cfg.breakeven_at_r and a > 0:
        k = cfg.trail_tight_atr_mult if best_r >= cfg.trail_tight_after_r else cfg.trail_atr_mult
        candidates.append(best - pos.sign * k * a)
    new_stop = max(candidates) if long_ else min(candidates)
    # Never trail a stop through the current close — that is an exit, not a trail.
    if long_:
        new_stop = min(new_stop, c - 1e-6)
    else:
        new_stop = max(new_stop, c + 1e-6)
    improved = new_stop > pos.stop + 1e-9 if long_ else new_stop < pos.stop - 1e-9
    if improved:
        return ExitDecision('TRAIL', None, f'trail to {new_stop:.2f} (best {best_r:+.2f}R)', new_stop)
    return ExitDecision('HOLD')


def _stop_label(pos):
    if pos.stop == pos.initial_stop:
        return 'stop'
    return 'trailing stop' if pos.r_multiple(pos.stop) > 0.05 else 'breakeven stop'


def advance(pos, bar, decision):
    """Apply a non-exit decision to the position for the next bar."""
    pos.bars_held += 1
    if pos.side == 'LONG':
        pos.best_price = max(pos.best_price, float(bar['high']))
    else:
        pos.best_price = min(pos.best_price, float(bar['low']))
    if decision.action == 'TRAIL' and decision.new_stop is not None:
        pos.stop = decision.new_stop
    return pos
