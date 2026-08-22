# trading_costs.py
#
# Realistic Indian retail equity delivery trading costs — shared by live
# trading (paper_trading_manager.py) and backtesting (swing_trading_bot.py),
# for the same reason trailing_stop.py and tranche_manager.py were pulled
# into their own shared modules: one implementation, so live and backtest
# can't silently diverge on what "real" costs look like.
#
# ── Why this replaces a flat ₹0.005/share commission ────────────────────────
# The original placeholder commission (₹0.005/share, ×2 for round trip —
# about ₹0.01/share total) understated real Indian delivery-equity trading
# costs by roughly 100-200x for a typical small retail trade in this
# project. Found while investigating why live paper-trading P&L looked
# thin/inconsistent — this was quietly inflating every closed trade's
# reported result relative to what real capital would actually achieve.
#
# Real costs for NSE equity delivery trades come mostly from sources that
# AREN'T proportional brokerage at all — many discount brokers charge ₹0
# brokerage on delivery trades. The actual costs:
#   • STT (Securities Transaction Tax) — 0.1% on BOTH the buy and sell leg
#     for delivery trades (unlike intraday, where STT is sell-side only and
#     much lower — see the day-trading section of the accompanying
#     architecture notes for why this matters if intraday gets added later).
#   • Exchange transaction charges, SEBI fee, stamp duty, GST on the
#     taxable components — individually small, add up to a modest
#     percentage.
#   • A near-flat depository participant (DP) charge levied per scrip sold
#     per day — commonly ~₹15-20 — which barely moves with trade size, and
#     so is disproportionately expensive for small tranche-sized exits
#     specifically (this project's scaled-exit system produces exactly that:
#     several separate small sells per original signal instead of one).
#
# These are approximate, deliberately conservative-but-realistic figures for
# a typical Indian discount broker — percentages and flat charges do shift
# over time and vary by broker, so treat this as "roughly right," not an
# exact replica of any one broker's current fee schedule. Update
# PCT_COST_PER_LEG / FLAT_CHARGE_PER_SELL here if your actual broker's
# numbers differ meaningfully — this is the only place that needs changing.

# All-in percentage cost (STT + exchange transaction charges + stamp duty +
# GST on the taxable components), applied to trade VALUE on each leg.
# ~0.13%/leg -> ~0.26% round trip, all-in.
PCT_COST_PER_LEG = 0.0013

# Flat charge per SELL transaction (DP/scrip charge) — not proportional to
# size, which is exactly why it matters more for small tranche exits than a
# purely percentage-based model would suggest.
FLAT_CHARGE_PER_SELL = 20.0


def round_trip_commission(entry_price, exit_price, position_size):
    """
    Total realistic cost for one full entry+exit of a position of this
    size. Replaces the old `position_size * commission_per_share * 2`
    formula everywhere it appeared. Returns a rupee amount.

    Called once per CLOSED tranche/trade row — a 3-tranche position that
    closes each tranche separately correctly pays the flat DP-style charge
    THREE times (three separate sell instructions in reality), not once,
    which is part of why realistic costs matter more for a tranched
    strategy than an untranched one — see run_backtest.py's Run A vs Run B
    comparison for a way to actually see this effect in the numbers.
    """
    entry_value = entry_price * position_size
    exit_value  = exit_price * position_size
    pct_cost    = (entry_value + exit_value) * PCT_COST_PER_LEG
    return round(pct_cost + FLAT_CHARGE_PER_SELL, 2)
