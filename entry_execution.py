# entry_execution.py  ── ENTRY ROUTING & FILL REALISM  v1
# ═════════════════════════════════════════════════════════════════════════════
# signal_generator emits entry_price = close of the signal bar. Nobody can buy
# that. The bar has closed; the earliest possible transaction is the next
# session. Every position in paper_trades.csv is booked at a price that was
# never available to it — the entry-side twin of the gap-fill optimism
# exit_manager.resolve_fill() already removed from the exit side.
#
# ── The evidence ────────────────────────────────────────────────────────────
# Closed trades bucketed by holding period:
#
#   hold_days     n    mean R      net ₹
#   (0, 2]        5     -1.00     -419.79     <- every one a full stop
#   (2, 5]        1     +1.50     +209.51
#   (5, 10]      10     -0.75   -1,087.05
#   (10, 20]     26     +0.26   +1,193.98
#   (20, 100]     8     +0.25     +296.78
#
# A mean of exactly -1.00 across every trade that died inside two sessions is
# the signature of buying the top of an extended bar and being reverted
# immediately. GRINDWELL and GAIL both opened and stopped within a single day.
# These are not signals that were wrong about direction; they are signals that
# were right about direction and paid the wrong price for it.
#
# ── What this module does ───────────────────────────────────────────────────
# 1. PRICES THE ENTRY WHERE IT CAN ACTUALLY HAPPEN. A signal becomes a plan
#    executed on the NEXT bar, filled at that bar's open (market) or at a
#    limit inside its range (patient), or not filled at all.
#
# 2. ROUTES BY BAR CHARACTER, NOT BY PREFERENCE. A close in the top of a wide,
#    extended bar is chased at a bad price; a close near support after a
#    controlled pullback can be bid patiently. The router reads extension
#    above EMA-20 in sigma, the close's position within its own bar, and the
#    bar's range against ATR, and picks the mode each setup deserves.
#
# 3. RE-ANCHORS THE GEOMETRY AT THE REAL FILL. If the fill lands 1.2% above
#    the signal close, holding the original stop silently widens risk by 1.2%
#    of price and shrinks R:R — which is precisely how a 2.0:1 plan becomes a
#    1.6:1 trade without anyone deciding to accept that. The stop is held at
#    its structural level, the target rebuilt from the fill, and the size
#    recomputed so rupee risk is unchanged. A fill that degrades R:R past
#    min_rr is abandoned rather than downgraded.
#
# 4. EXPIRES UNFILLED PLANS. A limit that has not traded within its window is
#    cancelled. The setup that justified it is stale, and the capital is
#    released to the allocator's next candidate instead of sitting in an order
#    book against a thesis that has aged out.
# ═════════════════════════════════════════════════════════════════════════════

import logging
import math

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from signal_generator import NSEMicrostructure, RISK_PROFILE
    TICK = NSEMicrostructure.TICK
    _round_tick = NSEMicrostructure.round_to_tick
except ImportError:                                              # pragma: no cover
    TICK = 0.05
    RISK_PROFILE = {'min_rr': 1.50}

    def _round_tick(p, mode='nearest'):
        return round(round(p / TICK) * TICK, 2)


EXECUTION_PROFILES = {
    'balanced': {
        'chase_extension_sigma':  1.50,   # above this much extension, do not pay the open
        'chase_close_position':   0.80,   # a close this high in its bar is a bad price to chase
        'wide_bar_atr':           1.60,   # bar range this large vs ATR is an exhaustion tell
        'limit_pullback_sigma':   0.45,   # how far below the signal close to bid
        'limit_valid_bars':       2,
        'max_gap_up_sigma':       1.00,   # abandon if the next open gaps this far past the plan
        'max_slippage_sigma':     0.60,   # abandon a market fill worse than this
    },
    'aggressive': {
        'chase_extension_sigma':  2.40,
        'chase_close_position':   0.90,
        'wide_bar_atr':           2.20,
        'limit_pullback_sigma':   0.35,
        'limit_valid_bars':       3,
        'max_gap_up_sigma':       1.30,
        'max_slippage_sigma':     0.80,
    },
}

ACTIVE_PROFILE = 'aggressive'


def get_profile(name=None):
    return EXECUTION_PROFILES[name or ACTIVE_PROFILE]


# ═════════════════════════════════════════════════════════════════════════════
# ROUTING
# ═════════════════════════════════════════════════════════════════════════════
def route(details, profile=None):
    """
    Decide how this signal should be executed, and at what price.

    Returns {'mode', 'limit_price', 'valid_until_bars', 'rationale'} where mode
    is 'MARKET_OPEN' or 'LIMIT_RETEST'.

    The router is the cheapest accuracy improvement available, because it
    changes nothing about which signals fire — only what is paid for them. The
    three conditions it reads all describe the same thing from different
    angles: a bar that has already travelled most of the move the signal was
    predicting.
    """
    P = get_profile(profile)
    ind = details.get('indicators', {})
    sigma = float(ind.get('sigma_abs') or ind.get('atr') or 0.0)
    close = float(details['entry_price'])
    extension = float(ind.get('extension_atr', 0.0))
    close_pos = float(details.get('close_position', 0.5))
    bar_range_atr = float(details.get('bar_range_atr', 1.0))

    if sigma <= 0:
        return {'mode': 'MARKET_OPEN', 'limit_price': None, 'valid_until_bars': 1,
                'rationale': 'no volatility estimate — taking the open'}

    reasons = []
    if extension >= P['chase_extension_sigma']:
        reasons.append(f'{extension:.1f}σ above EMA-20')
    if close_pos >= P['chase_close_position']:
        reasons.append(f'closed in the top {(1-close_pos)*100:.0f}% of its range')
    if bar_range_atr >= P['wide_bar_atr']:
        reasons.append(f'bar range {bar_range_atr:.1f}x ATR')

    # Two independent tells required, not one. Routing on any single condition
    # sent 73% of signals to a resting bid in testing and forfeited 40% of them
    # unfilled — on a trending tape the entries you skip are disproportionately
    # the ones that ran. Demanding confluence keeps the patient route for bars
    # that are exhausted on more than one reading.
    if len(reasons) >= 2:
        limit = _round_tick(close - P['limit_pullback_sigma'] * sigma, mode='down')
        return {'mode': 'LIMIT_RETEST', 'limit_price': limit,
                'valid_until_bars': P['limit_valid_bars'],
                'rationale': 'bidding the retest — ' + ', '.join(reasons)}

    return {'mode': 'MARKET_OPEN', 'limit_price': None, 'valid_until_bars': 1,
            'rationale': 'controlled bar — taking the next open'}


# ═════════════════════════════════════════════════════════════════════════════
# FILL
# ═════════════════════════════════════════════════════════════════════════════
def attempt_fill(plan, bar, details, profile=None):
    """
    Resolve one execution attempt against the NEXT session's bar.

    Returns {'status', 'fill_price', 'reason'} with status in
    FILLED / PENDING / ABANDONED.

    Two abandonment rules, both guarding the same failure: paying so much more
    than the plan assumed that the trade is no longer the trade that was
    approved.
      • A market order into a large gap up buys the move the signal was
        predicting. The R:R at that price is not what the EV gate cleared.
      • A limit that gaps BELOW its price fills at the open — better than
        asked, which is honest to report and is exactly what a resting bid
        does in reality.
    """
    P = get_profile(profile)
    sigma = float(details.get('indicators', {}).get('sigma_abs') or 0.0) or 1e-9
    ref = float(details['entry_price'])
    o, h, l = float(bar['open']), float(bar['high']), float(bar['low'])

    if plan['mode'] == 'MARKET_OPEN':
        slip_sigma = (o - ref) / sigma
        if slip_sigma > P['max_slippage_sigma']:
            return {'status': 'ABANDONED', 'fill_price': None,
                    'reason': f'open gapped {slip_sigma:.2f}σ above the signal close'}
        return {'status': 'FILLED', 'fill_price': round(o, 2),
                'reason': f'market on open ({slip_sigma:+.2f}σ vs signal close)'}

    limit = float(plan['limit_price'])
    if o <= limit:
        # Gapped through a resting bid — filled at the open, better than asked.
        gap_sigma = (limit - o) / sigma
        if gap_sigma > P['max_gap_up_sigma']:
            return {'status': 'ABANDONED', 'fill_price': None,
                    'reason': f'opened {gap_sigma:.2f}σ below the bid — the setup broke, not retested'}
        return {'status': 'FILLED', 'fill_price': round(o, 2), 'reason': 'bid gapped through at the open'}
    if l <= limit:
        return {'status': 'FILLED', 'fill_price': round(limit, 2), 'reason': 'limit touched intrabar'}
    return {'status': 'PENDING', 'fill_price': None,
            'reason': f'low {l:.2f} did not reach the ₹{limit:.2f} bid'}


# ═════════════════════════════════════════════════════════════════════════════
# RE-ANCHORING
# ═════════════════════════════════════════════════════════════════════════════
def reanchor(details, fill_price, equity=None, min_rr=None):
    """
    Rebuild the trade's geometry around the price actually paid.

    The stop stays where it is — it marks a structural level in the chart, and
    that level does not move because the fill was 40 paise higher. What moves
    is everything derived FROM the gap between fill and stop: risk per share
    rises, so the share count must fall to hold rupee risk constant, and the
    target must be re-measured from the fill rather than from a close that no
    longer features in the trade.

    Skipping this is how a 2.0:1 plan quietly becomes a 1.6:1 trade. Returns
    (revised_details, ok, reason); ok is False when the fill degrades R:R below
    min_rr, in which case the trade is declined rather than accepted on worse
    terms than were approved.
    """
    min_rr = float(min_rr if min_rr is not None else RISK_PROFILE.get('min_rr', 1.50))
    d = dict(details)
    signal_close = float(details['entry_price'])
    stop = float(details['stop_loss'])
    orig_risk = signal_close - stop
    if orig_risk <= 0:
        return d, False, 'original geometry degenerate'

    new_risk = fill_price - stop
    if new_risk <= 0:
        return d, False, 'fill at or below the stop'

    # Target held at its original R-multiple, measured from the fill.
    orig_rr = (float(details['target_price']) - signal_close) / orig_risk
    new_target = _round_tick(fill_price + orig_rr * new_risk, mode='down')
    new_rr = (new_target - fill_price) / new_risk

    # Reachability was solved against the signal close; a higher fill eats into
    # the horizon's remaining travel, so re-test it rather than assume it holds.
    reach_pct = float(details.get('horizon_reachable_pct', 0.0)) / 100.0
    if reach_pct > 0:
        ceiling = signal_close * (1.0 + reach_pct)
        if new_target > ceiling:
            new_target = _round_tick(ceiling, mode='down')
            new_rr = (new_target - fill_price) / new_risk

    if new_rr < min_rr:
        return d, False, (f'fill at ₹{fill_price:.2f} leaves {new_rr:.2f}:1, '
                          f'under the {min_rr:.2f} floor')

    # Hold rupee risk constant: the position was sized for a risk budget, and
    # that budget did not change because the fill did.
    old_size = int(details.get('position_size', 0) or 0)
    new_size = max(int(round(old_size * orig_risk / new_risk)), 1) if old_size else old_size

    d.update({
        'entry_price': round(float(fill_price), 2),
        'signal_close': round(signal_close, 2),
        'target_price': new_target,
        'position_size': new_size,
        'risk_reward_ratio': round(new_rr, 2),
        'risk': round(new_size * new_risk, 2),
        'reward': round(new_size * (new_target - fill_price), 2),
        'risk_pct_of_price': round(new_risk / fill_price * 100, 2),
        'entry_slippage_pct': round((fill_price / signal_close - 1.0) * 100, 3),
    })
    return d, True, ''


# ═════════════════════════════════════════════════════════════════════════════
# PENDING ORDER BOOK
# ═════════════════════════════════════════════════════════════════════════════
class PendingOrders:
    """
    Holds plans awaiting a fill across sessions, and expires them.

    Worth persisting rather than regenerating: a limit bid placed on Monday
    against Monday's setup is not the same order as one the scanner would
    generate fresh on Wednesday, and reserving capital for it is only honest
    if the original plan is what is still working.
    """

    def __init__(self, profile=None):
        self.profile = profile or ACTIVE_PROFILE
        self.book = {}          # symbol -> {'plan','details','bars_waited'}

    def place(self, symbol, details, plan):
        self.book[symbol] = {'plan': plan, 'details': details, 'bars_waited': 0}

    def reserved_capital(self):
        """Cash the allocator must treat as spoken for while bids rest."""
        return sum(float(o['details']['entry_price']) * int(o['details'].get('position_size', 0) or 0)
                   for o in self.book.values())

    def process(self, bars_by_symbol, min_rr=None):
        """
        Work the book against today's bars. Returns
        (filled: [(symbol, revised_details, note)], expired: [(symbol, reason)]).
        """
        filled, expired = [], []
        for symbol, order in list(self.book.items()):
            bar = bars_by_symbol.get(symbol)
            if bar is None:
                order['bars_waited'] += 1
                if order['bars_waited'] > order['plan']['valid_until_bars']:
                    expired.append((symbol, 'no price data within the validity window'))
                    del self.book[symbol]
                continue

            res = attempt_fill(order['plan'], bar, order['details'], self.profile)
            if res['status'] == 'FILLED':
                revised, ok, why = reanchor(order['details'], res['fill_price'], min_rr=min_rr)
                del self.book[symbol]
                if ok:
                    filled.append((symbol, revised, res['reason']))
                else:
                    expired.append((symbol, f"{res['reason']} but {why}"))
            elif res['status'] == 'ABANDONED':
                expired.append((symbol, res['reason']))
                del self.book[symbol]
            else:
                order['bars_waited'] += 1
                if order['bars_waited'] >= order['plan']['valid_until_bars']:
                    expired.append((symbol, f"bid unfilled after "
                                            f"{order['bars_waited']} session(s) — setup stale"))
                    del self.book[symbol]
        return filled, expired


# ═════════════════════════════════════════════════════════════════════════════
def enrich_signal_bar(details, df):
    """
    Adds the two bar-character fields the router needs but signal_generator
    does not currently emit: the close's position within its own bar, and the
    bar's range against ATR. Call once on each BUY before routing.
    """
    last = df.iloc[-1]
    rng = float(last['high']) - float(last['low'])
    atr = float(details.get('indicators', {}).get('atr') or 0.0)
    details['close_position'] = ((float(last['close']) - float(last['low'])) / rng) if rng > 1e-9 else 0.5
    details['bar_range_atr'] = (rng / atr) if atr > 1e-9 else 1.0
    return details
