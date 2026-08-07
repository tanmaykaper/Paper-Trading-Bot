# tranche_manager.py
#
# Scaled-exit (partial profit-taking) logic — shared by live trading
# (run_paper_trading.py / paper_trading_manager.py) and backtesting
# (swing_trading_bot.py). Extracted into its own module for exactly the
# reason trailing_stop.py already documents for itself: one implementation
# to test and trust, rather than two that can quietly drift apart.
#
# ── Why this used to only exist in run_paper_trading.py ─────────────────────
# TRANCHE_CONFIG/build_tranches() were originally defined directly inside
# run_paper_trading.py. That worked fine for live trading, but it meant
# swing_trading_bot.py's backtester structurally COULD NOT use the same
# tranche logic even if someone tried to wire it in — run_paper_trading.py
# already imports SwingTradingBot FROM swing_trading_bot.py, so the reverse
# import (swing_trading_bot.py importing FROM run_paper_trading.py) would be
# circular. That's not a hypothetical: it's the actual reason the backtester
# went this long simulating a single-exit-per-position policy while live
# trading ran a 3-tranche one — the two had silently diverged, and the ONE
# tool meant to validate live's actual behaviour was quietly testing a
# different, older strategy instead. Moving this here (a module neither
# run_paper_trading.py nor swing_trading_bot.py depends on the other for)
# fixes the structural cause, not just the symptom.
#
# ── The design itself ────────────────────────────────────────────────────────
# A single fixed target either captures the whole move up to that price or
# nothing beyond it — a hard ceiling on every winner regardless of how far
# it runs. Splitting the exit into tranches captures a reliable partial gain
# early (reducing give-back risk, which the trailing stop already helps
# with) while leaving a piece of the position with NO fixed ceiling, riding
# purely on the trailing stop (trailing_stop.py) — the part that can capture
# an outsized move a single target would have capped. This matters
# specifically for a trend/momentum system: the distribution of outcomes
# tends to be right-skewed (most winners are moderate, a few run much
# further), and a fixed target caps exactly the trades where that skew would
# otherwise pay off.
#
#   quick  (35%) — exits at 1.5R, matching signal_generator's own min_rr
#                  (the minimum R:R it requires to take the trade at all) —
#                  locks in "the worst outcome that still justified entry"
#                  early and reliably.
#   core   (35%) — exits at the ORIGINAL target signal_generator already
#                  computes (3-4R depending on pattern) — the planned
#                  outcome, unchanged from a non-tranched trade.
#   runner (30%) — no fixed target at all; rides the trailing stop
#                  (breakeven at 1R, +1R locked at 2R, +2R locked at 3R,
#                  keeps trailing beyond that) plus the existing time-exit
#                  as the ultimate backstop.

ENABLE_SCALED_EXITS    = True
MIN_SIZE_FOR_TRANCHING = 6   # below this, tranches would round to <2 shares each — not worth splitting
TRANCHE_CONFIG = [
    {'label': 'quick',  'size_pct': 0.35, 'r_multiple': 1.5},   # matches RISK_PROFILE['min_rr']
    {'label': 'core',   'size_pct': 0.35, 'r_multiple': None},  # None = use signal_generator's own target
    {'label': 'runner', 'size_pct': 0.30, 'r_multiple': None},  # None here = no fixed target, see below
]


def build_tranches(entry_price, stop_loss, target_price, position_size,
                    enabled=None, min_size=None):
    """
    Splits one BUY signal into quick/core/runner tranches per
    TRANCHE_CONFIG. Returns None if scaled exits are disabled or
    position_size is too small to split meaningfully — callers should fall
    back to a single untranched position in that case.

    enabled/min_size: optional overrides of the module defaults above,
    specifically so callers (e.g. the backtester's A/B harness) can force
    tranching off for a specific run without mutating global state.

    The runner tranche's target is set 1000R away — not infinite, to avoid
    any float/CSV-serialisation edge cases, but far enough beyond any
    realistic 15-day move that it will never actually be the reason a trade
    closes. It's meant to be unreachable BY CONSTRUCTION: the runner exits
    only via the trailing stop or the time-exit, never "Target Hit".
    """
    enabled  = ENABLE_SCALED_EXITS if enabled is None else enabled
    min_size = MIN_SIZE_FOR_TRANCHING if min_size is None else min_size

    if not enabled or position_size < min_size:
        return None

    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return None

    sizes = []
    remaining = position_size
    for i, t in enumerate(TRANCHE_CONFIG):
        if i == len(TRANCHE_CONFIG) - 1:
            sizes.append(remaining)   # last tranche absorbs any rounding remainder
        else:
            s = max(1, int(position_size * t['size_pct']))
            sizes.append(s)
            remaining -= s

    if remaining < 0 or any(s <= 0 for s in sizes):
        return None   # position too small to give every tranche at least 1 share

    tranches = []
    for t, size in zip(TRANCHE_CONFIG, sizes):
        if t['label'] == 'runner':
            tranche_target = entry_price + 1000 * risk_per_share   # effectively unreachable, see docstring
        elif t['r_multiple'] is not None:
            tranche_target = entry_price + t['r_multiple'] * risk_per_share
        else:
            tranche_target = target_price   # 'core' — signal_generator's own computed target
        tranches.append({'label': t['label'], 'size': size, 'target_price': tranche_target})

    return tranches
