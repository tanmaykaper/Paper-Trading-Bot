# run_backtest.py  ── PORTFOLIO BACKTEST + A/B/C VALIDATION  v5
# ─────────────────────────────────────────────────────────────────────────────
# v5 change — three-way comparison, not two. swing_trading_bot.py's
# backtest_portfolio() previously had no concept of scaled exits (tranches)
# at all — it simulated a single-exit-per-position policy that stopped being
# what live trading actually runs the moment run_paper_trading.py v10 shipped
# tranching. That's now fixed (see tranche_manager.py), which means this
# script can finally ask the question that actually matters for CURRENT live
# trading, not a stale approximation of it:
#
#   RUN A — Alpha ON,  Tranched     (current live policy, exactly as deployed)
#   RUN B — Alpha ON,  Untranched   (isolates tranching's OWN marginal effect)
#   RUN C — Alpha OFF, Untranched   (original pre-alpha-engine baseline)
#
#   Compare A vs B  ->  does scaled-exit tranching actually help, holding the
#                       alpha layer fixed?
#   Compare B vs C  ->  does the alpha engine actually help, holding the exit
#                       policy fixed (this is what v4 already answered)?
#   Compare A vs C  ->  the full combined effect of both improvements over
#                       the original untouched baseline.
#
# ── Everything from v4 still applies ─────────────────────────────────────────
#   • FULL PERFORMANCE ANALYTICS (backtest_analytics.py): Sharpe, Sortino,
#     Calmar, max drawdown, CAGR, profit factor, expectancy — stratified by
#     conviction tier, market regime, entry pattern, AND now tranche label
#     (quick/core/runner/full) — so you can see e.g. whether the runner
#     tranche is actually earning its complexity or just adding noise.
#   • SAME UNIVERSE, CAPITAL, MAX TRADES, AND SECTOR CAP AS LIVE: imported
#     directly from run_paper_trading.py rather than separately hardcoded
#     values that can silently drift out of sync — which is exactly what
#     happened to INITIAL_EQUITY/MAX_OPEN_TRADES here after a capital
#     increase elsewhere in this project; fixed by importing instead of
#     duplicating, the same fix already applied to SCAN_UNIVERSE in v4.
#   • WALK-FORWARD, POINT-IN-TIME CORRECT: unchanged, still verified by
#     test_lookahead_bias in the project's test suite.
#
# ── Honest limits ────────────────────────────────────────────────────────────
# This script has not been run against real data — the sandbox this was
# built in has no live yfinance/NSE access. The tranche-aware mechanics were
# validated against synthetic, deliberately-constructed price paths (see
# test_tranche_backtest.py): tranches close independently at their own
# targets, a loss closes every tranche together at the shared stop, and
# use_scaled_exits=False reproduces the old untranched behaviour exactly for
# backward compatibility. Whether tranching (or the alpha engine, or both)
# actually add value on REAL history — and by how much — can only be
# answered by running this against real data. Run it, read RUN A vs RUN B
# first (that's the new question), then the rest of the comparisons below it.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import pandas as pd

from swing_trading_bot import SwingTradingBot
from run_paper_trading import SCAN_UNIVERSE, INITIAL_EQUITY, MAX_OPEN_TRADES, LIVE_MAX_SECTOR_EXPOSURE
from backtest_analytics import compute_performance_report, print_performance_report, compare_reports

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BACKTEST_DAYS   = 600       # ~2.5 years of daily data
# INITIAL_EQUITY, MAX_OPEN_TRADES, LIVE_MAX_SECTOR_EXPOSURE imported directly
# above rather than redefined here — see v5 changelog note.

# The full live scan universe is 100+ symbols — thorough, but each symbol is
# its own set of API calls plus per-day factor computation across ~540
# simulated days, so this can take a while. Trim BACKTEST_STOCKS below (e.g.
# SCAN_UNIVERSE[:30]) for a faster, smaller-sample run while iterating.
BACKTEST_STOCKS = SCAN_UNIVERSE


def run_one(use_alpha_engine, use_scaled_exits):
    bot = SwingTradingBot(
        send_emails=False,
        initial_equity=INITIAL_EQUITY,
        max_open_trades=MAX_OPEN_TRADES,
        max_hold_days=15,
    )
    trades_df = bot.backtest_portfolio(
        BACKTEST_STOCKS, days=BACKTEST_DAYS, use_alpha_engine=use_alpha_engine,
        use_scaled_exits=use_scaled_exits, max_sector_exposure=LIVE_MAX_SECTOR_EXPOSURE,
    )
    equity_df = getattr(bot, 'last_equity_curve', None)
    report = compute_performance_report(trades_df, equity_df, INITIAL_EQUITY)
    return trades_df, equity_df, report


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NSE SWING TRADING BOT — PORTFOLIO BACKTEST + ALPHA/TRANCHE A/B/C  v5")
    print(f"Universe: {len(BACKTEST_STOCKS)} symbols | Days: {BACKTEST_DAYS} | "
          f"Initial equity: ₹{INITIAL_EQUITY:,} | Max trades: {MAX_OPEN_TRADES}")
    print("=" * 70)

    print("\n\n########## RUN A: ALPHA ON, TRANCHED (current live policy) ##########")
    trades_a, equity_a, report_a = run_one(use_alpha_engine=True, use_scaled_exits=True)
    print_performance_report(report_a, title="RUN A — ALPHA ON, TRANCHED (current live policy)")

    print("\n\n########## RUN B: ALPHA ON, UNTRANCHED (isolates tranching's own effect) ##########")
    trades_b, equity_b, report_b = run_one(use_alpha_engine=True, use_scaled_exits=False)
    print_performance_report(report_b, title="RUN B — ALPHA ON, UNTRANCHED")

    print("\n\n########## RUN C: ALPHA OFF, UNTRANCHED (original baseline) ##########")
    trades_c, equity_c, report_c = run_one(use_alpha_engine=False, use_scaled_exits=False)
    print_performance_report(report_c, title="RUN C — ALPHA OFF, UNTRANCHED (original baseline)")

    compare_reports(report_a, report_b, label_a="A: Alpha ON + Tranched", label_b="B: Alpha ON + Untranched")
    compare_reports(report_b, report_c, label_a="B: Alpha ON + Untranched", label_b="C: Alpha OFF + Untranched (baseline)")
    compare_reports(report_a, report_c, label_a="A: Current live policy", label_b="C: Original baseline")

    for name, df in [('a', trades_a), ('b', trades_b), ('c', trades_c)]:
        if df is not None and len(df) > 0:
            df.to_csv(f'backtest_results_run_{name}.csv', index=False)
            print(f"✓ Run {name.upper()} trades saved to backtest_results_run_{name}.csv")
    if equity_a is not None and len(equity_a) > 0:
        equity_a.to_csv('backtest_equity_curve_run_a.csv', index=False)

    print("\nRead A vs B FIRST — that's the new question this version answers: does scaled-exit")
    print("tranching (run_paper_trading.py's biggest recent change) actually help, holding the alpha")
    print("layer constant? Then B vs C for the alpha engine's own effect (unchanged from before).")
    print("Then check the 'Stratified by Tranche' section of RUN A specifically: if 'runner' isn't")
    print("outperforming 'quick'/'core' on average, the piece designed to capture outsized moves")
    print("isn't earning its complexity, even if the overall A vs B numbers look fine otherwise.")
