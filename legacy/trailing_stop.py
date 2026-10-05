# trailing_stop.py  ── v2  (COMPATIBILITY SHIM OVER exit_manager.py)
# ─────────────────────────────────────────────────────────────────────────────
# The trailing-stop implementation now lives in exit_manager.py, alongside the
# other three reasons a position closes (hard levels, momentum decay, planned
# horizon). Splitting the trail away from the exits it competes with was the
# structural reason the trail could ratchet a stop this bar that the time exit
# then overrode next bar with no coordination between them — precedence has to
# be decided in one place to be decided at all.
#
# This file stays because paper_trading_manager.py and swing_trading_bot.py
# both import `compute_trailing_stop` from it, and that import keeps working
# unchanged. Nothing about the v1 contract is broken:
#
#   compute_trailing_stop(entry_price, initial_stop_loss, current_stop_loss,
#                         current_price)
#
# called with exactly those four positional arguments still returns a monotone
# R-tier ratchet measured from the ORIGINAL stop — the fixed 1R reference that
# v1's own header documents at length, preserved as an invariant.
#
# Passing the optional highest_high / current_atr / position_size arguments
# upgrades the same call to the chandelier trail: it follows the trade's own
# path rather than its entry-day risk unit, breathes with current volatility,
# and lifts its protective floors by round-trip friction so the first defensive
# tier is flat on CAPITAL rather than flat on price. See exit_manager.py for
# the reasoning and for the realised-trade evidence behind each change.
#
# New code should import from exit_manager directly and prefer ExitEngine,
# which evaluates the trail in its correct precedence against the other exits
# instead of in isolation.
# ─────────────────────────────────────────────────────────────────────────────

from exit_manager import (                                    # noqa: F401
    compute_trailing_stop,
    round_trip_cost_per_share,
    EXIT_PROFILES,
    get_profile,
)

# v1 exported this for inspection/logging. Rebuilt from the active profile so
# anything reading it sees the tiers actually in force rather than a stale copy.
TRAILING_TIERS = [(trigger, floor if floor is not None else 0.0)
                  for trigger, _atr_mult, floor in get_profile()['chandelier_tiers']]

__all__ = ['compute_trailing_stop', 'round_trip_cost_per_share',
           'EXIT_PROFILES', 'get_profile', 'TRAILING_TIERS']
