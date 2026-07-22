# trailing_stop.py
#
# Shared trailing-stop logic for both live trading (run_paper_trading.py /
# paper_trading_manager.py) and backtesting (swing_trading_bot.py). Kept as
# one standalone module specifically so there's exactly one implementation
# to test and trust — the recurring failure mode throughout this project has
# been the same policy logic existing in two places and quietly drifting
# apart (sector caps, drawdown breaker, capital caps, alpha-gating constants
# all had this happen at some point). Both call sites import THIS function
# rather than each keeping their own copy.
#
# ── The bug this replaces ───────────────────────────────────────────────────
# The original implementation (swing_trading_bot.py's _apply_trailing_stop)
# recomputed "risk" as entry_price − CURRENT stop_loss on every call. That's
# fine on the first ratchet, but wrong on every one after: once the stop has
# already moved up, entry_price − stop_loss no longer equals the original
# 1R distance the tier thresholds are defined in terms of — it can even go
# negative once the stop has passed entry_price. Traced through a concrete
# case: a position ratcheted to its 2R tier on day 1, then reached the
# ORIGINAL 3R level on day 2 — and incorrectly failed to progress to the 3R
# tier, staying stuck at the smaller 2R protection level instead of locking
# in the additional profit tier it had genuinely earned. Fixed here by
# tracking the ORIGINAL stop-loss (the true 1R reference point) separately
# from whatever the stop has since ratcheted to, and always computing risk
# from that fixed reference — never from the current, possibly-already-moved
# stop.
#
# ── The staircase ────────────────────────────────────────────────────────────
# Long positions only (this system doesn't short). At each profit tier
# reached, the stop ratchets up to lock in a portion of that gain:
#   price >= entry + 3R  ->  stop moves to entry + 2R   (lock in 2R, give back at most 1R of the move)
#   price >= entry + 2R  ->  stop moves to entry + 1R   (lock in 1R)
#   price >= entry + 1R  ->  stop moves to entry        (breakeven — can no longer lose on this trade)
# The stop only ever moves up, never down (checked explicitly) — a trailing
# stop that could loosen wouldn't be protecting anything.

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# (trigger_r, trail_r) pairs, checked in descending order — the first tier
# whose trigger is met wins (highest applicable protection level).
TRAILING_TIERS = [(3, 2), (2, 1), (1, 0)]


def compute_trailing_stop(entry_price, initial_stop_loss, current_stop_loss, current_price):
    """
    Returns the new stop-loss value (ratcheted, or unchanged if no tier
    applies or the computed level isn't an improvement).

    entry_price:        trade's entry price (immutable)
    initial_stop_loss:  the ORIGINAL stop-loss set at entry (immutable) —
                         this is what "1R" is measured from, always, even
                         after current_stop_loss has since ratcheted up
    current_stop_loss:  whatever the stop is right now (may already be
                         ratcheted from a previous call)
    current_price:      latest price to evaluate against

    Long positions only. Returns current_stop_loss unchanged if
    initial_stop_loss >= entry_price (malformed input — a stop must be
    below entry for "risk" to be a meaningful positive distance) or if no
    tier's trigger is met, or if the tier that does apply wouldn't actually
    raise the stop above where it already is.
    """
    risk = entry_price - initial_stop_loss
    if risk <= 0:
        return current_stop_loss

    for trigger_r, trail_r in TRAILING_TIERS:
        if current_price >= entry_price + trigger_r * risk:
            candidate = entry_price + trail_r * risk
            if candidate > current_stop_loss:
                return round(candidate, 2)
            return current_stop_loss   # tier applies but doesn't improve on the existing stop
    return current_stop_loss   # no tier's trigger met yet
