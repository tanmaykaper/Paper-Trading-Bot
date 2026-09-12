# run_live_screening.py  ── READ-ONLY SCAN  v4
# ═════════════════════════════════════════════════════════════════════════════
# v3 kept its own hardcoded 40-stock list. run_paper_trading.py's SCAN_UNIVERSE
# has since grown past 100 names across three sub-universes, and the two have
# been diverging silently ever since — so this tool has been reporting on a
# different market than the bot actually trades. That is the same failure this
# project has now hit four separate times (sector caps, the drawdown breaker,
# capital constants, tranche logic), and the fix is the same one: import the
# single definition instead of keeping a second copy.
#
# v4 also stops pretending a raw BUY signal is a decision. Under v11 a signal
# is the FIRST of six gates — market state, entry quality, economics, slot
# competition, sentiment and earnings all follow — so a screen that prints
# every signal as actionable overstates what the bot would do by a wide margin.
# This now reports each candidate WITH the economics that decide its fate, and
# says plainly which ones would survive.
#
# Read-only by construction: it opens nothing, writes nothing, and touches no
# state file. Safe to run at any time alongside the live cron.
# ═════════════════════════════════════════════════════════════════════════════

import logging

from technical_indicators import TechnicalIndicators
from fundamental_screener import FundamentalScreener
from signal_generator import SignalGenerator, RISK_PROFILE
from data_fetcher_free import DataFetcherFree, data_quality
from market_state import MarketState, BreadthPanel, print_state
from run_paper_trading import SCAN_UNIVERSE, SECTOR_MAP, INITIAL_EQUITY, MAX_HOLD_DAYS

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UNIVERSE_LIMIT = None          # None = the full live universe
BARS = 260


def main():
    symbols = SCAN_UNIVERSE[:UNIVERSE_LIMIT] if UNIVERSE_LIMIT else SCAN_UNIVERSE
    fetcher = DataFetcherFree()

    print("\n" + "=" * 72)
    print(f"NSE SWING BOT — LIVE SCREEN v4   |   {len(symbols)} symbols "
          f"(imported from SCAN_UNIVERSE, not a local copy)")
    print("=" * 72)

    index_df = fetcher.get_historical_data('^NSEI', days=BARS + 120, min_bars=200)
    vix_df = fetcher.get_historical_data('^INDIAVIX', days=180, min_bars=30)

    universe, skipped = {}, {}
    for symbol in symbols:
        df = fetcher.get_historical_data(symbol, days=BARS, min_bars=80)
        if df is None:
            skipped[symbol] = 'no data'
            continue
        q = data_quality(df)
        if not q['tradeable']:
            skipped[symbol] = q['reason']
            continue
        universe[symbol] = df

    if not universe:
        print("\nNo usable data — nothing to screen.")
        return

    # ── Market state first, because it can veto everything below it ──────────
    state = MarketState('aggressive').assess(
        index_df, breadth_panel=BreadthPanel(universe), vix_df=vix_df, base_slots=5)
    print_state(state)

    fundamentals = {}
    for symbol in universe:
        try:
            f = fetcher.get_fundamentals(symbol) or {}
        except Exception:
            f = {}
        f['sector'] = SECTOR_MAP.get(symbol, 'UNKNOWN')
        fundamentals[symbol] = f

    screener = FundamentalScreener()
    try:
        screener.calibrate_sector_pe(fundamentals, SECTOR_MAP)
    except AttributeError:
        pass
    sig_gen = SignalGenerator(TechnicalIndicators(), screener)

    candidates = []
    for symbol, df in universe.items():
        try:
            signal, d = sig_gen.generate_signal(
                df, symbol, fundamentals.get(symbol, {}), INITIAL_EQUITY,
                market_regime=state['legacy_regime'], benchmark_df=index_df,
                max_hold_days=MAX_HOLD_DAYS)
        except Exception as e:
            skipped[symbol] = f'signal error: {e}'
            continue
        if signal == 'BUY':
            candidates.append((symbol, d))

    candidates.sort(key=lambda kv: -kv[1].get('quality_score', 0))

    print(f"\n{len(candidates)} candidate(s) from {len(universe)} usable symbols")
    if skipped:
        print(f"{len(skipped)} skipped on data quality — "
              f"{list(skipped.items())[:3]}{' …' if len(skipped) > 3 else ''}")

    for symbol, d in candidates:
        econ = d.get('economics', {})
        print(f"\n  🎯 {symbol}   {d['entry_type']}  "
              f"(quality {d['quality_score']:.2f} vs {d['quality_required']:.2f} required)")
        print(f"     Entry ₹{d['entry_price']:.2f}   Stop ₹{d['stop_loss']:.2f} "
              f"({d['risk_pct_of_price']:.2f}%, {d['stop_sigma_mult']:.1f}σ, {d['stop_basis']})")
        print(f"     Target ₹{d['target_price']:.2f}   R:R 1:{d['risk_reward_ratio']:.2f}   "
              f"plan {d['time_exit_bars']} bars   P(win) {d['p_win_est']:.2f}")
        print(f"     Size {d['position_size']}  ₹{econ.get('notional', 0):,.0f}  "
              f"cost {econ.get('cost_bps', 0):.0f} bps  "
              f"EV ₹{econ.get('expected_value', 0):+,.0f} "
              f"({econ.get('ev_to_cost', 0)}x cost)  tranche {'yes' if d.get('tranche_ok') else 'no'}")
        print(f"     Patterns {', '.join(d['patterns_triggered'])}  |  "
              f"RSI {d['indicators']['rsi']:.0f}  ADX {d['indicators']['adx']:.0f}  "
              f"ER {d['efficiency_ratio']:.2f}")

    # ── What the bot would actually do ───────────────────────────────────────
    # The gap between "signals found" and "trades taken" is the whole point of
    # everything downstream of signal_generator, and printing only the first
    # number is how a screen becomes misleading.
    print("\n" + "-" * 72)
    if not state['new_entries_allowed']:
        print(f"  Entries are SUSPENDED today ({state['state']}). None of the above would be")
        print(f"  taken: {state['note']}")
    else:
        room = state['max_slots']
        print(f"  Market state allows entries at {state['exposure']:.2f}x exposure, "
              f"{room} slot(s).")
        print(f"  Of {len(candidates)} candidate(s), the allocator funds those with the "
              f"highest forward return per slot-day")
        print(f"  that clear the economic floor — and only after the sentiment veto and the "
              f"earnings blackout,")
        print(f"  neither of which runs here. Treat this as the shortlist, not the order book.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
