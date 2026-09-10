# exit_manager.py  ── ADAPTIVE EXIT ENGINE  v1
# ═════════════════════════════════════════════════════════════════════════════
# Owns every reason a position stops existing. Shared by live trading
# (paper_trading_manager.py / run_paper_trading.py) and backtesting
# (swing_trading_bot.py), for the reason trailing_stop.py and tranche_manager.py
# already document for themselves: one implementation to test and trust.
# trailing_stop.py v2 is now a thin re-export of this module's
# compute_trailing_stop(), so both call sites move together by construction.
#
# ── Why the exit is where the remaining money is ────────────────────────────
# From paper_trades.csv, 50 closed rows:
#
#   exit_reason    n     total ₹     mean R    mean hold
#   SL Hit         26   -2,099.28     -0.85       18.5
#   Time Exit      22   +2,050.57     +0.76       23.8
#   Target Hit      2     +242.14     +1.50        5.5
#
# Every rupee of profit this system has produced came from the exit it never
# designed — the clock. Targets fired twice in five months. So the exit policy
# is not a detail bolted onto the entry logic; it IS the strategy's P&L
# distribution, and it is currently being set by an arbitrary integer.
#
# Four specific defects this module fixes:
#
# 1. TRAILING FROM THE ENTRY-DAY RISK UNIT, NOT FROM THE TRADE'S OWN PATH.
#    trailing_stop.py v1 ratchets to entry+1R, entry+2R — levels frozen at
#    entry. A position that runs +4R then gives back 1.5R over a week of
#    expanding volatility keeps a stop pinned to a stale reference. The
#    chandelier below trails from the highest high SINCE ENTRY minus a
#    multiple of CURRENT ATR, so it widens when the stock starts breathing
#    harder and tightens when it settles — and it still never moves down.
#
# 2. "BREAKEVEN" THAT LOSES MONEY.
#    v1's first tier moves the stop to entry_price exactly. HAPPSTMNDS closed
#    at 420.20 against an entry of 420.20 — three tranches, gross ₹0.00, net
#    -₹112.44. Breakeven on price is a loss on capital. Every floor here is
#    lifted by the round-trip cost per share, so the first protective tier
#    protects something.
#
# 3. NO CONCEPT OF THE THESIS EXPIRING.
#    A trade entered on ADX expansion and an EMA-20 reclaim should end when
#    ADX rolls over and price loses the EMA-20 — not 15 calendar days later at
#    whatever price happens to print. The momentum-decay check reads the same
#    conditions the entry was built on and exits when they stop being true,
#    which both books gains earlier and returns capital to the scanner sooner.
#
# 4. DEAD CAPITAL HELD TO FULL TERM.
#    Twelve of the closed rows sat for 16-24 days to finish inside ±0.5%.
#    That is a slot, and roughly a third of the book, earning nothing while
#    the scanner has candidates. Stagnation detection marks these for
#    rotation at a far lower replacement bar than a healthy position.
#
# ── The objective function ───────────────────────────────────────────────────
# Everything here optimises expectancy per unit of CAPITAL-TIME, R/day net of
# cost, rather than expectancy per trade. With a small number of economically
# viable slots, the scarce resource is slot-days, not ideas. A +1.2R trade
# held 25 days and a +0.6R trade held 6 days are not close: 0.048 R/day
# against 0.100 R/day. Compounding an account cares about the second number.
# ═════════════════════════════════════════════════════════════════════════════

import logging
import math

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from technical_indicators import TechnicalIndicators as TI
except ImportError:                                              # pragma: no cover
    TI = None

try:
    from trading_costs import PCT_COST_PER_LEG, FLAT_CHARGE_PER_SELL
except ImportError:                                              # pragma: no cover
    PCT_COST_PER_LEG, FLAT_CHARGE_PER_SELL = 0.0013, 20.0


# ═════════════════════════════════════════════════════════════════════════════
# PROFILES
# ═════════════════════════════════════════════════════════════════════════════
# Two calibrations, switched explicitly rather than by editing constants in
# place, so a change of risk appetite is a one-word diff and the alternative
# stays visible next to it.
#
# 'aggressive' gives winners materially more rope before the first tightening
# (a right-skewed momentum distribution pays for patience in the top decile
# and nowhere else), tolerates a deeper give-back from the peak, and rotates
# stagnant capital harder. It does NOT loosen the protective floors — the
# floors are what keep a run of losers survivable, and they are the reason a
# wider trail is affordable at all.
EXIT_PROFILES = {
    'balanced': {
        # (profit_in_R_at_or_above, atr_multiple_for_trail, protective_floor_in_R)
        'chandelier_tiers': [
            (3.0, 1.4, 2.00),
            (2.0, 1.8, 1.00),
            (1.0, 2.2, 0.05),      # cost-adjusted breakeven, see _floor_price
            (0.4, 2.6, None),      # trail only, no floor yet
        ],
        'decay_exit_min_signals_in_profit': 2,
        'decay_exit_min_signals_in_loss':   3,
        'stagnation_progress_r':   0.50,   # R the trade should have reached by the checkpoint
        'stagnation_checkpoint':   0.50,   # ...as a fraction of its planned horizon
        'max_hold_hard_cap':       30,
        'min_hold_before_decay':    3,     # let the trade breathe before judging its health
    },
    'aggressive': {
        'chandelier_tiers': [
            (4.0, 1.6, 2.50),
            (2.5, 2.1, 1.20),
            (1.2, 2.6, 0.05),
            (0.5, 3.1, None),
        ],
        'decay_exit_min_signals_in_profit': 3,   # slower to cut a winner
        'decay_exit_min_signals_in_loss':   3,
        'stagnation_progress_r':   0.60,
        'stagnation_checkpoint':   0.40,   # ...judged sooner: idle capital is the enemy
        'max_hold_hard_cap':       25,
        'min_hold_before_decay':    3,
    },
}

ACTIVE_PROFILE = 'aggressive'


def get_profile(name=None):
    return EXIT_PROFILES[name or ACTIVE_PROFILE]


# ═════════════════════════════════════════════════════════════════════════════
# COST-AWARE BREAKEVEN
# ═════════════════════════════════════════════════════════════════════════════
def round_trip_cost_per_share(entry_price, position_size):
    """
    Per-share round-trip friction, so a protective floor can be set where the
    trade is actually flat on capital rather than flat on price.

    The flat depository charge is divided across the position, which is why
    this number explodes on small rows: a ₹20 charge on 2 shares is ₹10/share,
    on 40 shares it is ₹0.50. That asymmetry is the same one that drove the
    entry-side minimum-notional floor, showing up again on the exit side.
    """
    size = max(int(position_size or 0), 1)
    pct_component = 2.0 * entry_price * PCT_COST_PER_LEG
    flat_component = FLAT_CHARGE_PER_SELL / size
    return pct_component + flat_component


# ═════════════════════════════════════════════════════════════════════════════
# CHANDELIER TRAILING STOP
# ═════════════════════════════════════════════════════════════════════════════
def compute_trailing_stop(entry_price, initial_stop_loss, current_stop_loss,
                          current_price, highest_high=None, current_atr=None,
                          position_size=None, profile=None):
    """
    Backward-compatible with trailing_stop.py v1: called with the original four
    positional arguments it behaves as a monotone R-tier ratchet, so
    paper_trading_manager.apply_trailing_stops() and swing_trading_bot's
    backtester keep working with no change.

    Supplying highest_high and current_atr upgrades it to a true chandelier:

        stop = max( current_stop,
                    highest_high_since_entry - k(profit_tier) * ATR_now,
                    entry + floor_R(profit_tier) * initial_risk + cost )

    Three properties worth stating explicitly, because v1 had none of them:

      • It trails the TRADE'S OWN PATH. A position that reached +4R and pulled
        back is protected relative to where it actually got to, not relative
        to where it started.
      • It breathes with CURRENT volatility. When ATR expands the stop widens,
        which is exactly when a fixed-distance stop gets picked off by noise
        that has nothing to do with the trade failing.
      • It never loosens. Every candidate is max()'d against the existing stop,
        so a widening ATR can slow the ratchet but can never reverse it.

    The 1R reference is always measured from initial_stop_loss, never from the
    current one — the bug trailing_stop.py v1's header documents at length, and
    which is preserved as a fixed invariant here.
    """
    P = get_profile(profile)
    risk = entry_price - initial_stop_loss
    if risk <= 0:
        return current_stop_loss

    profit_r = (current_price - entry_price) / risk
    tier = None
    for trigger_r, atr_mult, floor_r in P['chandelier_tiers']:
        if profit_r >= trigger_r:
            tier = (trigger_r, atr_mult, floor_r)
            break
    if tier is None:
        return current_stop_loss

    _, atr_mult, floor_r = tier
    candidates = [current_stop_loss]

    if floor_r is not None:
        candidates.append(_floor_price(entry_price, risk, floor_r, position_size))

    if highest_high is not None and current_atr is not None and current_atr > 0:
        candidates.append(float(highest_high) - atr_mult * float(current_atr))
    else:
        # Legacy path — no bar history supplied. Reproduce v1's R-tier ladder
        # so behaviour is unchanged for callers that have not been wired up yet.
        legacy_floor = 0.0 if floor_r is None else floor_r
        candidates.append(entry_price + legacy_floor * risk)

    new_stop = max(candidates)
    # A stop parked immediately under the last print gets taken out by the next
    # bar's ordinary range, which converts a live winner into an exit for no
    # informational reason. Hold it at least a fraction of ATR below price —
    # falling back to a few ticks only when no ATR is available.
    breathing_room = 0.30 * float(current_atr) if (current_atr and current_atr > 0) else 0.15
    new_stop = min(new_stop, current_price - max(0.10, breathing_room))
    return round(new_stop, 2) if new_stop > current_stop_loss else current_stop_loss


def _floor_price(entry_price, risk, floor_r, position_size):
    """
    Protective floor, lifted by round-trip friction. floor_r = 0.05 means
    "flat on capital", not "flat on price" — the distinction that turned
    HAPPSTMNDS into a -₹112 breakeven.
    """
    cost_ps = round_trip_cost_per_share(entry_price, position_size) if position_size else 0.0
    return entry_price + floor_r * risk + cost_ps


# ═════════════════════════════════════════════════════════════════════════════
# THESIS HEALTH
# ═════════════════════════════════════════════════════════════════════════════
class PositionHealth:
    """
    Reads the same conditions the entry was built on and reports whether they
    still hold. Deliberately mirrors signal_generator's own gates — trend
    location, directional strength, MACD posture, close quality — because an
    exit rule that measures something unrelated to the entry rule is just a
    second, uncorrelated strategy wearing the first one's positions.

    Each check is a binary "the reason for this trade has stopped being true".
    Count them; act on the count. Signals are reported individually so a live
    log can show WHICH part of the thesis broke, not just that something did.
    """

    @staticmethod
    def evaluate(bars, entry_adx=None):
        """
        bars: OHLCV frame for the held symbol, most recent bar last. Needs
              ~60 rows for the indicators to be warm; fewer degrades to
              whatever can be computed and reports the rest as healthy, so a
              short history never manufactures an exit.

        Returns {'signals': [...], 'n_signals': int, 'detail': {...}}.
        """
        out = {'signals': [], 'n_signals': 0, 'detail': {}}
        if TI is None or bars is None or len(bars) < 30:
            return out

        close, high, low = bars['close'], bars['high'], bars['low']
        ema20 = TI.calculate_ema(close, 20)
        dmi = TI.calculate_dmi(high, low, close, 14)
        macd = TI.calculate_macd(close)
        hist = macd['histogram']
        adx = dmi['adx']

        last = -1
        c = float(close.iloc[last])

        # 1. Trend location lost — two consecutive closes below the EMA-20 the
        #    entry required. One close below is noise; two is a change of side.
        below_ema = bool(c < float(ema20.iloc[last]) and
                         float(close.iloc[-2]) < float(ema20.iloc[-2]))
        if below_ema:
            out['signals'].append('lost EMA-20')

        # 2. Directional conviction reversing — -DI has taken the lead.
        pdi, mdi = float(dmi['plus_di'].iloc[last]), float(dmi['minus_di'].iloc[last])
        if np.isfinite(pdi) and np.isfinite(mdi) and mdi > pdi:
            out['signals'].append('-DI leads')

        # 3. Trend strength decaying — ADX falling three bars and materially
        #    below where it was when the position was opened.
        adx_now = float(adx.iloc[last]) if np.isfinite(adx.iloc[last]) else None
        adx_falling = bool(
            len(adx) >= 4 and
            adx.iloc[last] < adx.iloc[-2] < adx.iloc[-3]
        )
        if adx_falling and entry_adx and adx_now is not None and adx_now < 0.80 * float(entry_adx):
            out['signals'].append('ADX rolling over')

        # 4. Momentum rolling — MACD histogram negative and still falling.
        if len(hist) >= 3 and float(hist.iloc[last]) < 0 and float(hist.iloc[last]) < float(hist.iloc[-2]):
            out['signals'].append('MACD deteriorating')

        # 5. Distribution — three straight closes in the lower half of their
        #    own bar. Sellers are taking the close, which is the single most
        #    reliable tape-reading tell available from daily OHLC.
        if len(bars) >= 3:
            positions = []
            for i in (-1, -2, -3):
                rng = float(high.iloc[i]) - float(low.iloc[i])
                positions.append(((float(close.iloc[i]) - float(low.iloc[i])) / rng) if rng > 1e-9 else 0.5)
            if all(p < 0.45 for p in positions):
                out['signals'].append('closing weak')

        out['n_signals'] = len(out['signals'])
        out['detail'] = {'adx': adx_now, 'plus_di': pdi, 'minus_di': mdi,
                         'ema20': float(ema20.iloc[last]), 'close': c}
        return out


# ═════════════════════════════════════════════════════════════════════════════
# FILL REALISM
# ═════════════════════════════════════════════════════════════════════════════
def resolve_fill(bar, stop_loss, target_price, prev_close=None):
    """
    What a stop or target ACTUALLY fills at on a given bar, on an EOD system.

    Three cases the naive `close <= stop` check gets wrong, all of which
    flatter the reported P&L:

      • GAP THROUGH THE STOP. The scrip opens below the stop; the fill is the
        open, not the stop. A 3% stop on a -6% gap loses 6%. This is the
        single largest source of optimism in an EOD backtest, and it is why
        signal_generator v4 prices p90 down-gap into expected loss.
      • GAP THROUGH THE TARGET. Symmetrically, a gap above the target fills at
        the open, which is better than the target — reported honestly rather
        than clipped.
      • BOTH TOUCHED INTRABAR. High >= target and low <= stop on the same bar.
        Daily OHLC cannot say which came first, so this resolves to the STOP.
        Assuming the favourable ordering is how backtests manufacture edge
        that does not survive contact with a live tape.

    Returns (exit_price, reason) or (None, None) when the bar closes neither.
    """
    o, h, l = float(bar['open']), float(bar['high']), float(bar['low'])

    if o <= stop_loss:
        return round(o, 2), 'SL Hit (gap)'
    if o >= target_price:
        return round(o, 2), 'Target Hit (gap)'

    hit_stop = l <= stop_loss
    hit_target = h >= target_price
    if hit_stop and hit_target:
        return round(stop_loss, 2), 'SL Hit (ambiguous bar, resolved adversely)'
    if hit_stop:
        return round(stop_loss, 2), 'SL Hit'
    if hit_target:
        return round(target_price, 2), 'Target Hit'
    return None, None


# ═════════════════════════════════════════════════════════════════════════════
# EXIT ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class ExitEngine:
    """
    One decision per open position per bar. Precedence is deliberate:

        hard stop / target  >  momentum decay  >  planned horizon  >  trail

    Hard levels first because they are the contract the position was opened
    under. Decay before the clock because the thesis expiring is information
    and the clock is not. The trail is evaluated last and only when nothing
    closed the trade, so a stop raised this bar applies from the next one —
    matching how a real GTT order behaves rather than letting a stop that did
    not exist this morning close a trade this afternoon.
    """

    def __init__(self, profile=None):
        self.P = get_profile(profile)
        self.profile_name = profile or ACTIVE_PROFILE

    def evaluate(self, trade, bars=None, current_price=None, bars_held=None):
        """
        trade: dict/Series with entry_price, stop_loss, initial_stop_loss,
               target_price, position_size, and optionally time_exit_bars,
               entry_adx, highest_high.
        bars:  recent OHLCV for the symbol, newest last. Supplying it enables
               the chandelier and the decay check; without it this degrades to
               the legacy price-only ratchet plus hard levels.

        Returns {'action', 'exit_price', 'exit_reason', 'new_stop',
                 'health', 'stagnant', 'r_multiple', 'r_per_day'}.
        """
        entry = float(trade['entry_price'])
        stop = float(trade['stop_loss'])
        target = float(trade['target_price'])
        size = int(trade.get('position_size', 1) or 1)
        init_stop = _coalesce_float(trade.get('initial_stop_loss'), stop)
        risk = max(entry - init_stop, 1e-9)

        price = float(current_price) if current_price is not None else (
            float(bars['close'].iloc[-1]) if bars is not None and len(bars) else entry)
        held = int(bars_held if bars_held is not None else trade.get('hold_days', 0) or 0)

        r_mult = (price - entry) / risk
        r_per_day = r_mult / max(held, 1)
        result = {'action': 'HOLD', 'exit_price': None, 'exit_reason': None,
                  'new_stop': stop, 'health': None, 'stagnant': False,
                  'r_multiple': round(r_mult, 3), 'r_per_day': round(r_per_day, 4)}

        # ── 1. Hard levels, with realistic fills ─────────────────────────────
        if bars is not None and len(bars):
            fill, reason = resolve_fill(bars.iloc[-1], stop, target)
            if fill is not None:
                return {**result, 'action': 'EXIT', 'exit_price': fill, 'exit_reason': reason}
        else:
            if price <= stop:
                return {**result, 'action': 'EXIT', 'exit_price': round(stop, 2), 'exit_reason': 'SL Hit'}
            if price >= target:
                return {**result, 'action': 'EXIT', 'exit_price': round(target, 2), 'exit_reason': 'Target Hit'}

        # ── 2. Thesis health ─────────────────────────────────────────────────
        health = PositionHealth.evaluate(bars, entry_adx=trade.get('entry_adx'))
        result['health'] = health
        if held >= self.P['min_hold_before_decay'] and health['n_signals']:
            threshold = (self.P['decay_exit_min_signals_in_profit'] if r_mult > 0
                         else self.P['decay_exit_min_signals_in_loss'])
            if health['n_signals'] >= threshold:
                return {**result, 'action': 'EXIT', 'exit_price': round(price, 2),
                        'exit_reason': f"Momentum Decay ({', '.join(health['signals'])})"}

        # ── 3. Planned horizon, per trade rather than per portfolio ──────────
        horizon = int(_coalesce_float(trade.get('time_exit_bars'), self.P['max_hold_hard_cap']))
        horizon = int(min(max(horizon, 4), self.P['max_hold_hard_cap']))
        if held >= horizon:
            return {**result, 'action': 'EXIT', 'exit_price': round(price, 2),
                    'exit_reason': f'Time Exit ({held}/{horizon} bars, own horizon)'}

        # ── 4. Stagnation — not an exit, a rotation flag ─────────────────────
        checkpoint = max(2, int(horizon * self.P['stagnation_checkpoint']))
        if held >= checkpoint and r_mult < self.P['stagnation_progress_r']:
            result['stagnant'] = True

        # ── 5. Trail ─────────────────────────────────────────────────────────
        highest_high, atr_now = self._path_stats(bars, held, trade)
        new_stop = compute_trailing_stop(
            entry, init_stop, stop, price,
            highest_high=highest_high, current_atr=atr_now,
            position_size=size, profile=self.profile_name,
        )
        if new_stop > stop:
            result['action'] = 'TRAIL'
            result['new_stop'] = new_stop
        return result

    @staticmethod
    def _path_stats(bars, held, trade):
        """
        Highest high since entry and current ATR. `highest_high` is carried on
        the trade row when available (cheap, and survives a day where price
        data is missing); otherwise it is recovered from the last `held` bars.
        """
        if bars is None or TI is None or len(bars) < 20:
            return _coalesce_float(trade.get('highest_high'), None), None
        window = bars.tail(max(int(held), 1) + 1)
        hh_from_bars = float(window['high'].max())
        hh_recorded = _coalesce_float(trade.get('highest_high'), None)
        highest_high = max(hh_from_bars, hh_recorded) if hh_recorded else hh_from_bars
        atr = TI.calculate_atr(bars['high'], bars['low'], bars['close'], 14)
        atr_now = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else None
        return highest_high, atr_now

    # ─────────────────────────────────────────────────────────────────────────
    def rank_for_rotation(self, evaluations):
        """
        Order open positions worst-first for slot replacement.

        The score is realised R per slot-day, penalised by broken thesis
        signals and by stagnation. This replaces ranking on the alpha score
        recorded AT ENTRY, which is a stale opinion about a position that has
        since had days to prove or disprove itself — the position's own
        behaviour since entry is strictly better evidence than the forecast
        that opened it.

        evaluations: {symbol: evaluate() output}. Returns [(symbol, score)]
        ascending, so element 0 is the first slot to free.
        """
        scored = []
        for symbol, ev in evaluations.items():
            score = ev.get('r_per_day', 0.0) * 10.0
            health = ev.get('health') or {}
            score -= 0.6 * health.get('n_signals', 0)
            if ev.get('stagnant'):
                score -= 2.0
            scored.append((symbol, round(score, 3)))
        return sorted(scored, key=lambda kv: kv[1])


def _coalesce_float(value, fallback):
    """CSV round-trips turn empty cells into NaN, and NaN is truthy — the
    exact class of bug run_paper_trading.py's header documents for
    trade_group_id. Checked explicitly here for the same reason."""
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return fallback
        if isinstance(value, str) and not value.strip():
            return fallback
        if pd.isna(value):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback
