# profit_engine.py  ── PROFIT MAXIMISATION  v1
# ═════════════════════════════════════════════════════════════════════════════
# Everything built so far protects the downside: cost floors, noise-aware
# stops, regime exposure, gap-honest fills, earnings blackouts. All of it is
# necessary and none of it makes money on its own — it stops money leaking.
# This module is the other half: the three mechanisms that actually compound an
# account, each of which exploits a property of the return distribution this
# strategy produces rather than trying to predict anything new.
#
#   1. EQUITY-CURVE RISK SCALING — trade larger when the system is in sync
#      with the market, smaller when it is not.
#   2. PYRAMIDING — add to winners, because the distribution is right-skewed
#      and a fixed position size caps exactly the trades that pay for the book.
#   3. THE COMPOUNDING LADDER — the account's structural constraints change as
#      it grows, and the parameters should change with them.
#
# ── 1. Equity-curve risk scaling ────────────────────────────────────────────
# Trend-following returns are serially dependent in a specific way: winning and
# losing periods cluster, because the underlying condition the system exploits
# (persistent directional moves) is itself persistent. When the equity curve is
# above its own moving average, the market is paying for what this system does;
# when it is below, it is not.
#
# So the equity curve is treated as a tradeable series in its own right, and
# risk is scaled by where it sits relative to its own average. This is not a
# forecast of the market — it is a measurement of whether the strategy is
# currently in phase with it.
#
# The honest cost: it is a lagging signal and it will cut risk at the bottom of
# a drawdown and restore it after the recovery has begun, giving back some of
# the rebound. It is accepted because the asymmetry favours it — the loss from
# being small during a recovery is bounded, while the loss from being full-size
# through an extended losing streak compounds. Bounds are tight for the same
# reason: 0.60x to 1.30x, never off, never leveraged.
#
# ── 2. Pyramiding ───────────────────────────────────────────────────────────
# This project's own data made the case. Across 50 closed rows the RUNNER
# tranche — the piece with no fixed target, riding only the trailing stop — was
# the best-performing group (n=11, +₹500.83, mean +0.139R) while the whole book
# netted +₹193. The winners run further than a fixed target captures, which is
# the standard right-skew of a momentum system.
#
# A fixed position size cannot express that. Pyramiding does: when a position
# proves itself by reaching +1R, add a smaller second piece and move the stop
# on the WHOLE position to breakeven-plus-costs. The combined risk after the
# add is then lower than the original single-unit risk — the add is funded by
# the profit already banked in the first unit, not by new risk.
#
# That last property is the entire discipline. A pyramid that raises total risk
# is a martingale wearing a trend-following costume, and it is how accounts die
# in exactly the tape where this system otherwise performs best.
#
# ── 3. The compounding ladder ───────────────────────────────────────────────
# At ₹50,000 with a ~₹20 flat depository charge, the minimum economically
# viable position is around ₹9,000, which caps the book at roughly five slots
# and forces 40% concentration in each. At ₹200,000 the same floor is 4.5% of
# capital and the book can hold fifteen genuinely diversified positions. These
# are different portfolios and they should not run identical parameters. The
# ladder makes that explicit rather than leaving it as a constant somebody
# eventually notices is wrong.
# ═════════════════════════════════════════════════════════════════════════════

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from trading_costs import round_trip_commission, FLAT_CHARGE_PER_SELL
except ImportError:                                              # pragma: no cover
    FLAT_CHARGE_PER_SELL = 20.0

    def round_trip_commission(e, x, s, dp_charge=20.0):
        return (e + x) * s * 0.0011 + dp_charge


def _int(value, default=0):
    """CSV round-trips turn empty cells into NaN, which is truthy."""
    try:
        if value is None or (isinstance(value, float) and value != value):
            return default
        if isinstance(value, str) and not value.strip():
            return default
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


PROFIT_PROFILES = {
    'balanced': {
        'curve_ma_bars':        20,
        'curve_scale_max':      1.20,
        'curve_scale_min':      0.65,
        'curve_sensitivity':    6.0,    # how hard distance-from-MA moves the scalar
        'pyramid_enabled':      True,
        'pyramid_trigger_r':    1.20,
        'pyramid_size_pct':     0.50,   # of the original unit
        'pyramid_max_adds':     1,
        'min_trades_for_curve': 15,
    },
    'aggressive': {
        'curve_ma_bars':        20,
        'curve_scale_max':      1.30,
        'curve_scale_min':      0.60,
        'curve_sensitivity':    7.0,
        'pyramid_enabled':      True,
        'pyramid_trigger_r':    1.00,
        'pyramid_size_pct':     0.60,
        'pyramid_max_adds':     2,
        'pyramid_stop_r':       1.50,   # combined stop rides this far under price on each add
        'min_trades_for_curve': 12,
    },
    'growth': {
        # Pyramiding matters more here than in any other profile: a 3-slot book
        # has few positions, so the ones that work have to be made to count.
        # The trigger is early and two adds are permitted, with the stop
        # ratcheting on each so total open risk falls rather than rises.
        'curve_ma_bars':        20,
        'curve_scale_max':      1.30,
        'curve_scale_min':      0.60,
        'curve_sensitivity':    7.0,
        'pyramid_enabled':      True,
        'pyramid_trigger_r':    1.00,
        'pyramid_size_pct':     0.60,
        'pyramid_max_adds':     2,
        'pyramid_stop_r':       1.50,
        'min_trades_for_curve': 12,
    },
}

ACTIVE_PROFILE = 'growth'


def get_profile(name=None):
    """Falls back rather than raising — see market_state.get_profile for why."""
    key = name or ACTIVE_PROFILE
    if key not in PROFIT_PROFILES:
        logger.warning(f"profit_engine: no '{key}' profile — falling back to '{ACTIVE_PROFILE}'")
        key = ACTIVE_PROFILE
    return PROFIT_PROFILES[key]


# ═════════════════════════════════════════════════════════════════════════════
# 1. EQUITY-CURVE RISK SCALING
# ═════════════════════════════════════════════════════════════════════════════
def equity_curve_scalar(equity_series, profile=None):
    """
    Returns (scalar, detail). Multiply the risk fraction by the scalar.

    Distance of equity from its own moving average, in units of the curve's own
    volatility — not in rupees or percent. A ₹500 deviation means something
    entirely different on a smooth curve than on a jagged one, and normalising
    by the curve's own noise is what makes one constant work across both.

    Returns 1.0 — no opinion — until there is enough curve to measure, which
    matters because the first weeks of a fresh account produce a curve that is
    almost all noise.
    """
    P = get_profile(profile)
    detail = {'scalar': 1.0, 'reason': 'insufficient equity history'}
    if equity_series is None or len(equity_series) < P['curve_ma_bars'] + 5:
        return 1.0, detail

    eq = pd.Series(equity_series).astype(float).dropna()
    if len(eq) < P['curve_ma_bars'] + 5:
        return 1.0, detail

    ma = eq.rolling(P['curve_ma_bars']).mean()
    resid = (eq - ma).dropna()
    if len(resid) < 5:
        return 1.0, detail
    sigma = float(resid.std(ddof=1))
    if sigma < 1e-9:
        return 1.0, {'scalar': 1.0, 'reason': 'equity curve is flat'}

    z = float((eq.iloc[-1] - ma.iloc[-1]) / sigma)
    raw = 1.0 + z / P['curve_sensitivity']
    scalar = float(np.clip(raw, P['curve_scale_min'], P['curve_scale_max']))
    return scalar, {
        'scalar': round(scalar, 3), 'z': round(z, 2),
        'equity': round(float(eq.iloc[-1]), 2), 'ma': round(float(ma.iloc[-1]), 2),
        'reason': ('in phase — above the curve average' if z > 0 else
                   'out of phase — below the curve average'),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. PYRAMIDING
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_pyramid(trade, current_price, equity, free_cash, heat_room,
                     evaluation=None, market_state=None, profile=None):
    """
    Should this winning position be added to, and by how much?

    Returns {'add': bool, 'size', 'new_stop', 'reason', 'added_risk'}.

    Five conditions, and every one of them can veto:

      PROOF      the position has reached the trigger R multiple. A position
                 that has not proved itself is not a winner, it is a position.
      THESIS     exit_manager reports no broken health signals and no
                 stagnation. Adding to a position whose reason for existing has
                 broken is the single most expensive mistake available here.
      REGIME     the market state permits new entries. An add IS a new entry;
                 exempting it because it shares a symbol with something already
                 held would be an accounting convenience, not a risk judgement.
      CAPACITY   cash, portfolio heat and the economic floor all clear. An add
                 too small to carry its share of the depository charge is a
                 fee, not a position.
      NET RISK   after moving the stop to breakeven-plus-costs on the COMBINED
                 position, total risk must be no greater than the original
                 unit's risk. This is the condition that separates pyramiding
                 from averaging up, and it is checked arithmetically rather
                 than assumed from the stop's position.
    """
    P = get_profile(profile)
    out = {'add': False, 'size': 0, 'new_stop': None, 'reason': '', 'added_risk': 0.0}
    if not P['pyramid_enabled']:
        out['reason'] = 'pyramiding disabled'
        return out

    entry = float(trade['entry_price'])
    init_stop = float(trade.get('initial_stop_loss') or trade['stop_loss'])
    size = _int(trade.get('position_size'))
    risk_ps = entry - init_stop
    if risk_ps <= 0 or size < 1:
        out['reason'] = 'degenerate geometry'
        return out

    price = float(current_price)
    r_mult = (price - entry) / risk_ps
    # NaN is truthy, so `x or 0` does NOT catch an empty CSV cell — the exact
    # class of bug run_paper_trading.py's header documents for trade_group_id.
    adds_done = _int(trade.get('pyramid_adds'))

    if adds_done >= P['pyramid_max_adds']:
        out['reason'] = f'already added {adds_done}x'
        return out
    if r_mult < P['pyramid_trigger_r']:
        out['reason'] = f'{r_mult:+.2f}R — below the {P["pyramid_trigger_r"]:.2f}R trigger'
        return out

    ev = evaluation or {}
    health = ev.get('health') or {}
    if health.get('n_signals', 0) >= 1:
        out['reason'] = f"thesis showing {health['n_signals']} broken signal(s) — not adding"
        return out
    if ev.get('stagnant'):
        out['reason'] = 'flagged stagnant — not adding'
        return out
    if market_state is not None and not market_state.get('new_entries_allowed', True):
        out['reason'] = f"market state {market_state.get('state')} — an add is a new entry"
        return out

    add_size = max(int(size * P['pyramid_size_pct']), 1)
    notional = add_size * price
    min_notional = FLAT_CHARGE_PER_SELL / 0.0022       # same floor logic as entries
    if notional < min_notional:
        out['reason'] = (f'add of ₹{notional:,.0f} is under the ₹{min_notional:,.0f} '
                         f'economic floor')
        return out
    if notional > max(free_cash, 0.0):
        out['reason'] = f'needs ₹{notional:,.0f}, cash ₹{free_cash:,.0f}'
        return out

    # ── The net-risk test ────────────────────────────────────────────────────
    # New stop = breakeven on the original unit, plus the round-trip cost of the
    # combined position, so "breakeven" means flat on CAPITAL rather than flat
    # on price — the same correction exit_manager applies to its trailing floor.
    combined = size + add_size
    cost_ps = round_trip_commission(entry, price, combined) / combined

    # The stop is trailed to the higher of breakeven-plus-costs and the
    # position's existing trailed stop BEFORE the net-risk test runs. Testing
    # against a stale stop is what blocked every second add in testing: by the
    # time a trade reaches +2.5R its stop has usually ratcheted well above
    # entry, and ignoring that made the add look like it raised risk when it
    # reduced it. Right-skewed distributions pay in the tail, and the tail is
    # exactly where the second add lives.
    # Each add RATCHETS the combined stop up to ride a fixed distance under the
    # current price, rather than sitting at entry. Anchoring it at entry is what
    # blocked every second add in testing: a unit bought at +3R and stopped at
    # entry carries three units of risk on its own, so the net-risk test refused
    # it — correctly, given that stop. The answer is not to relax the test, it
    # is to place the stop where a pyramided position's stop actually belongs.
    # The first unit's locked profit funds the second; the ratchet is what makes
    # that literally true rather than a figure of speech.
    trailed = float(trade.get('stop_loss') or 0.0)
    ratchet = price - P['pyramid_stop_r'] * risk_ps
    new_stop = round(max(entry + cost_ps, trailed, ratchet), 2)
    if new_stop >= price:
        out['reason'] = 'price has not cleared breakeven-plus-costs yet'
        return out

    # Risk at the new stop, across both units, against the original unit's risk.
    risk_original_unit = size * (price - new_stop) if new_stop < entry else 0.0
    risk_after = (size * max(entry - new_stop, 0.0)) + (add_size * (price - new_stop))
    baseline = size * risk_ps
    if risk_after > baseline:
        out['reason'] = (f'add would raise open risk to ₹{risk_after:,.0f} against the '
                         f'₹{baseline:,.0f} original — that is averaging up, not pyramiding')
        return out
    if risk_after - min(baseline, 0.0) > heat_room:
        out['reason'] = f'portfolio heat room ₹{heat_room:,.0f} insufficient'
        return out

    out.update({'add': True, 'size': add_size, 'new_stop': new_stop,
                'added_risk': round(add_size * (price - new_stop), 2),
                'reason': (f'{r_mult:+.2f}R proved — adding {add_size} at ₹{price:,.2f}, '
                           f'stop to ₹{new_stop:,.2f} on the combined {combined}; '
                           f'open risk ₹{risk_after:,.0f} vs ₹{baseline:,.0f} before')})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3. THE COMPOUNDING LADDER
# ═════════════════════════════════════════════════════════════════════════════
def compounding_ladder(equity, flat_charge=None, target_flat_bps=22.0, mode='diversified'):
    """
    What this account size can actually support, derived rather than assumed.

    The binding constraint is the flat depository charge as a share of position
    notional. Everything else — slot count, per-name concentration, whether
    scaled exits are affordable — follows from it, which is why they are
    computed here instead of being independent constants that drift.

    Returns a dict the caller can apply to RISK_PROFILE and the allocator.
    """
    # Read the economic floor from the live risk profile when it is available,
    # so the ladder and signal_generator cannot disagree about what the minimum
    # viable position is — the duplicate-constant failure this project has hit
    # five times.
    try:
        from signal_generator import RISK_PROFILE
        target_flat_bps = float(RISK_PROFILE.get('max_flat_cost_bps', target_flat_bps))
    except Exception:
        pass
    flat = float(flat_charge if flat_charge is not None else FLAT_CHARGE_PER_SELL)
    equity = max(float(equity), 1.0)
    min_notional = flat / (target_flat_bps / 1e4)

    slots = max(int(equity / min_notional), 1)
    slots = min(slots, 15)                      # beyond this, attention is the constraint
    concentration = float(np.clip(1.0 / max(slots - 1, 1), 0.10, 0.45))

    if mode == 'growth':
        # ── Why fewer, larger positions ──────────────────────────────────────
        # Effective risk per trade is concentration x stop width. At 25%
        # concentration and a 4.5% stop that is 1.1% of equity — roughly
        # quarter-Kelly, and a rate that cannot compound an account no matter
        # how good the signals are. The Monte Carlo sweep puts peak median
        # growth near 4.5% risk per trade, which needs concentration in the
        # 40-60% band.
        #
        # The cost of that is real and is not hidden: median drawdown roughly
        # doubles (6% -> 15%) and the probability of finishing a 300-session
        # year down rises from 14% to 24%. That is the trade being made, and it
        # is only worth making if the edge is genuinely positive — at zero edge
        # this configuration loses money considerably faster than the timid one.
        #
        # Slot count is deliberately capped low. Concentration and slot count
        # are the same quantity seen from two sides, and letting both drift up
        # as equity grows would quietly return the book to the diversified
        # configuration this mode exists to leave.
        # Concentration is held CONSTANT rather than derived from slot count.
        # Deriving it (1.6/slots) made risk per trade fall as the account grew —
        # ₹50k gave 3.6% and ₹100k gave 1.9%, which is backwards: the account
        # would de-risk precisely as it earned the capacity to compound. Risk
        # per trade is set once, at the growth-optimal level, and held there;
        # max_capital_pct is a cap for outliers, and signal_generator's
        # risk-based sizing does the actual work.
        # 3 slots at 33% deploys exactly 100% of a cash account. Concentration
        # is held constant as equity grows so risk per trade does not decay the
        # moment the account earns the capacity to compound.
        # Deployment is capped at 100% — a cash delivery account has no
        # leverage, and an earlier draft quietly assumed 104-180% because
        # concentration and slot count were set independently. They are the
        # same quantity seen from two sides and are now derived together.
        slots = int(np.clip(equity / (min_notional * 1.8), 2, 4))
        concentration = float(np.clip(1.0 / slots, 0.25, 0.50))
    tranche_viable = equity >= 3.0 * min_notional * 2     # 3 rows in each of 2 positions

    return {
        'mode': mode,
        'equity': round(equity, 2),
        'min_notional': round(min_notional, 2),
        'recommended_slots': slots,
        'max_capital_pct': round(concentration, 3),
        'scaled_exits_viable': bool(tranche_viable),
        'flat_charge_bps_at_min': round(flat / min_notional * 1e4, 1),
        'note': (f'{slots} economically viable slot(s) at ₹{min_notional:,.0f} minimum; '
                 f'{"tranching affordable" if tranche_viable else "tranching not yet affordable"}'),
    }


# ═════════════════════════════════════════════════════════════════════════════
class ProfitEngine:
    """Thin coordinator so callers touch one object rather than three functions."""

    def __init__(self, profile=None):
        self.profile = profile or ACTIVE_PROFILE
        self.P = get_profile(self.profile)
        logger.info(f"✓ ProfitEngine — {self.profile} "
                    f"(curve scaling {self.P['curve_scale_min']}-{self.P['curve_scale_max']}x, "
                    f"pyramid at {self.P['pyramid_trigger_r']}R)")

    def risk_scalar(self, equity_history, closed_trades=None):
        """Equity-curve scalar, withheld until there are enough CLOSED trades
        for the curve to reflect the strategy rather than a handful of marks."""
        n_closed = 0 if closed_trades is None else len(closed_trades)
        if n_closed < self.P['min_trades_for_curve']:
            return 1.0, {'scalar': 1.0,
                         'reason': f'{n_closed}/{self.P["min_trades_for_curve"]} closed trades '
                                   f'— curve not yet meaningful'}
        return equity_curve_scalar(equity_history, self.profile)

    def pyramid(self, trade, price, equity, free_cash, heat_room,
                evaluation=None, market_state=None):
        return evaluate_pyramid(trade, price, equity, free_cash, heat_room,
                                evaluation, market_state, self.profile)

    @staticmethod
    def ladder(equity, mode='growth'):
        return compounding_ladder(equity, mode=mode)

    def report(self, equity_history, closed_trades, equity):
        scalar, detail = self.risk_scalar(equity_history, closed_trades)
        ladder = self.ladder(equity)
        lines = ["  Profit engine",
                 f"    Risk scalar   : {scalar:.2f}x  ({detail.get('reason')})",
                 f"    Ladder        : {ladder['note']}",
                 f"    Concentration : {ladder['max_capital_pct']*100:.0f}% max per name"]
        return "\n".join(lines)
