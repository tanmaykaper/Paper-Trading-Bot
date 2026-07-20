# run_backtest.py  ── PORTFOLIO BACKTEST + A/B ALPHA ENGINE VALIDATION  v4
# ─────────────────────────────────────────────────────────────────────────────
# v4 change — this is now a genuine validation tool, not just a P&L printout:
#
#   1. RUNS TWO BACKTESTS, NOT ONE: identical stock universe, identical
#      date range, identical starting capital — once with the alpha engine
#      active, once without (signal_generator alone, the pre-alpha-engine
#      policy). The question that actually matters isn't "is the number
#      positive," it's "does the alpha layer add anything over the
#      baseline" — a single backtest can't answer that, an A/B can.
#
#   2. FULL PERFORMANCE ANALYTICS (backtest_analytics.py): Sharpe, Sortino,
#      Calmar, max drawdown, CAGR, profit factor, expectancy — not just win
#      rate and total P&L. Also stratified by conviction tier, market
#      regime, and entry pattern, so you can see e.g. whether Tier 1 trades
#      actually show better expectancy than Tier 3 (the real evidence for
#      whether the ranking has predictive value, not just a plausible story).
#
#   3. SAME UNIVERSE AS LIVE: imports SCAN_UNIVERSE directly from
#      run_paper_trading.py instead of maintaining a separate, third stock
#      list that could silently drift out of sync with what's actually
#      trading live (the same class of bug found and fixed twice already
#      in this project — SECTOR_MAP/MAX_SECTOR_EXPOSURE/MAX_DRAWDOWN_DEFAULT
#      existing in the backtester but not live, and a stale 30%-of-equity
#      cap here contradicting signal_generator's already-current 40%).
#
#   4. WALK-FORWARD, POINT-IN-TIME CORRECT: the underlying integration in
#      swing_trading_bot.py's backtest_portfolio() uses only data available
#      up to and including the current simulated day at every step — no
#      vectorized shortcuts that could leak future information. Verified
#      explicitly (see the project's test suite): injecting a deliberately
#      unmistakable future price shock has zero effect on any entry decision
#      made before that shock, and every trade whose full lifecycle
#      completed beforehand is byte-for-byte identical between scenarios.
#
# ── Honest limits ────────────────────────────────────────────────────────────
# This script has not been run against real data — the sandbox this was
# built in has no live yfinance/NSE access. Everything above was validated
# with synthetic data proving the MECHANICS are correct (no lookahead bias,
# correct metric formulas, correct A/B isolation). Whether the alpha engine
# actually adds value, and what the real Sharpe/drawdown/expectancy numbers
# look like, can only be answered by actually running this against real
# history — that's what this script is for. Run it, read the A/B comparison
# at the end, and that tells you something real synthetic testing can't.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import pandas as pd

from swing_trading_bot import SwingTradingBot
from run_paper_trading import SCAN_UNIVERSE   # same universe as live — see note above
from backtest_analytics import compute_performance_report, print_performance_report, compare_reports

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INITIAL_EQUITY  = 10_000    # matches run_paper_trading.py's INITIAL_EQUITY
BACKTEST_DAYS   = 600       # ~2.5 years of daily data
MAX_OPEN_TRADES = 5         # matches run_paper_trading.py's MAX_OPEN_TRADES

# The full live scan universe is 100+ symbols — thorough, but each symbol is
# its own set of API calls plus per-day factor computation across ~540
# simulated days, so this can take a while. Trim BACKTEST_STOCKS below (e.g.
# SCAN_UNIVERSE[:30]) for a faster, smaller-sample run while iterating.
BACKTEST_STOCKS = SCAN_UNIVERSE


def run_one(use_alpha_engine):
    bot = SwingTradingBot(
        send_emails=False,
        initial_equity=INITIAL_EQUITY,
        max_open_trades=MAX_OPEN_TRADES,
        max_hold_days=15,
    )
    trades_df = bot.backtest_portfolio(BACKTEST_STOCKS, days=BACKTEST_DAYS, use_alpha_engine=use_alpha_engine)
    equity_df = getattr(bot, 'last_equity_curve', None)
    report = compute_performance_report(trades_df, equity_df, INITIAL_EQUITY)
    return trades_df, equity_df, report


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NSE SWING TRADING BOT — PORTFOLIO BACKTEST + ALPHA ENGINE A/B  v4")
    print(f"Universe: {len(BACKTEST_STOCKS)} symbols | Days: {BACKTEST_DAYS} | Initial equity: ₹{INITIAL_EQUITY:,}")
    print("=" * 70)

    print("\n\n########## RUN A: ALPHA ENGINE ON (current live policy) ##########")
    trades_on, equity_on, report_on = run_one(use_alpha_engine=True)
    print_performance_report(report_on, title="ALPHA ENGINE ON — PERFORMANCE REPORT")

    print("\n\n########## RUN B: ALPHA ENGINE OFF (baseline — signal_generator alone) ##########")
    trades_off, equity_off, report_off = run_one(use_alpha_engine=False)
    print_performance_report(report_off, title="ALPHA ENGINE OFF — PERFORMANCE REPORT (baseline)")

    compare_reports(report_on, report_off, label_a="Alpha engine ON", label_b="Alpha engine OFF (baseline)")

    if trades_on is not None and len(trades_on) > 0:
        trades_on.to_csv('backtest_results_alpha_on.csv', index=False)
        print("✓ Alpha-engine-ON trades saved to backtest_results_alpha_on.csv")
    if trades_off is not None and len(trades_off) > 0:
        trades_off.to_csv('backtest_results_alpha_off.csv', index=False)
        print("✓ Alpha-engine-OFF trades saved to backtest_results_alpha_off.csv")
    if equity_on is not None and len(equity_on) > 0:
        equity_on.to_csv('backtest_equity_curve_alpha_on.csv', index=False)

    print("\nRead the A/B COMPARISON table above first — that's the answer to \"does the alpha")
    print("engine actually help,\" not either report in isolation. Then check the 'Stratified by")
    print("Conviction Tier' section of the ON report specifically: if Tier 1 doesn't show better")
    print("expectancy than Tier 3, the ranking isn't adding the value it's designed to add, even")
    print("if the overall A/B numbers look fine for other reasons.")
