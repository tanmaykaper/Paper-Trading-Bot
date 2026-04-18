# run_paper_trading.py  ── GITHUB ACTIONS / SINGLE-RUN VERSION
# ─────────────────────────────────────────────────────────────────────────────
# Designed for daily scheduled execution via GitHub Actions.
#
# How state is persisted across runs:
#   - paper_trades.csv and daily_equity.csv live in the repo
#   - This script reads them at startup (all open trades, capital, history)
#   - At the end of the run, the GitHub Actions workflow commits any changes
#     back to the repo — so the next run picks up exactly where this one left off
#
# This script does NOT loop or sleep. It runs once, does everything, and exits.
# GitHub Actions cron handles the scheduling.
#
# Workflow:
#   1. Load state from paper_trades.csv (open trades, deployed capital, free cash)
#   2. Fetch latest prices for ALL stocks currently held (not just the scan list)
#   3. Check exits on every open trade (SL hit / target hit / time exit)
#   4. Check market regime via Nifty 50
#   5. Scan for new BUY signals — only if free cash and open slots available
#   6. Print full portfolio summary with unrealised P&L
#   7. Log equity snapshot to daily_equity.csv
#   8. Exit — Actions workflow commits updated CSVs back to repo
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
import os
import pandas as pd
from datetime import datetime

from swing_trading_bot import SwingTradingBot
from paper_trading_manager import PaperTradingManager
from signal_generator import SignalGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),   # visible in Actions logs
    ]
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
INITIAL_EQUITY  = 10_000   # ₹ — must match what the CSV was started with
MAX_OPEN_TRADES = 5
MAX_HOLD_DAYS   = 15

# CSV paths — relative to repo root (where Actions checks out the repo)
TRADES_CSV      = 'paper_trades.csv'
EQUITY_CSV      = 'daily_equity.csv'

# Full scan universe — keep this as the superset of anything you might hold.
# Symbols in open trades that are NOT in this list will still get exit-checked
# because we dynamically add them from the CSV at startup.
SCAN_UNIVERSE = [
    # Core large caps (most liquid, most signals)
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'ASIANPAINT',
    'MARUTI', 'TATASTEEL', 'BAJFINANCE', 'KOTAKBANK', 'LT',
    'AXISBANK', 'TITAN', 'WIPRO', 'ULTRACEMCO', 'NESTLEIND',
    # Additional for signal volume
    'HCLTECH', 'TECHM', 'SUNPHARMA', 'DRREDDY', 'CIPLA',
    'TATAMOTORS', 'BAJAJ-AUTO', 'HINDALCO', 'JSWSTEEL',
    'ONGC', 'BPCL', 'GAIL', 'SIEMENS', 'ABB', 'DLF',
    'INDUSINDBK', 'FEDERALBNK', 'MPHASIS', 'LTIM', 'CHOLAFIN',
]


def get_all_held_symbols(trades_csv):
    """
    Read the CSV and return a set of symbols for all OPEN trades.
    This ensures exit-checking works even for symbols outside SCAN_UNIVERSE.
    """
    if not os.path.exists(trades_csv):
        return set()
    try:
        df = pd.read_csv(trades_csv)
        # Handle legacy 'entry_datr' typo column
        if 'entry_datr' in df.columns and 'entry_date' not in df.columns:
            df = df.rename(columns={'entry_datr': 'entry_date'})
        open_trades = df[df['status'] == 'OPEN']
        return set(open_trades['symbol'].tolist())
    except Exception:
        return set()


def run_eod():
    logger.info("\n" + "=" * 70)
    logger.info(f"📅 NSE PAPER TRADING BOT — {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    logger.info("=" * 70)

    # ── Initialise ─────────────────────────────────────────────────────────────
    # PaperTradingManager reads paper_trades.csv on init and rebuilds state:
    #   - deployed_capital from all OPEN trades
    #   - realised_pnl from all CLOSED trades
    #   - free_cash = initial_equity + realised_pnl - deployed_capital
    paper_mgr = PaperTradingManager(
        initial_equity=INITIAL_EQUITY,
        csv_path=TRADES_CSV,
        equity_csv_path=EQUITY_CSV,
        max_open_trades=MAX_OPEN_TRADES,
    )

    bot = SwingTradingBot(
        send_emails=False,
        initial_equity=INITIAL_EQUITY,
        max_open_trades=MAX_OPEN_TRADES,
        max_hold_days=MAX_HOLD_DAYS,
    )

    # ── Build the price-fetch list ─────────────────────────────────────────────
    # = scan universe + any symbols currently held (catches symbols outside universe)
    held_symbols  = get_all_held_symbols(TRADES_CSV)
    price_symbols = list(set(SCAN_UNIVERSE) | held_symbols)
    logger.info(f"\n  Held positions: {sorted(held_symbols) or 'none'}")
    logger.info(f"  Price fetch list: {len(price_symbols)} symbols")

    # ── Step 1: Fetch latest prices ────────────────────────────────────────────
    logger.info("\n[Step 1] Fetching latest close prices...")
    latest_prices = {}
    for symbol in price_symbols:
        try:
            df = bot.fetcher.get_historical_data(symbol, days=5)
            if df is not None and len(df) > 0:
                latest_prices[symbol] = float(df.iloc[-1]['close'])
        except Exception:
            pass
    logger.info(f"  Got prices for {len(latest_prices)}/{len(price_symbols)} symbols")

    # Verify all held symbols have prices (warn if not — could be a delisted stock)
    missing = held_symbols - set(latest_prices.keys())
    if missing:
        logger.warning(f"  ⚠️ No price found for held symbols: {missing} — exit check will skip these")

    # ── Step 2: Check exits on all open trades ─────────────────────────────────
    logger.info("\n[Step 2] Checking open trades for exits...")
    trades_closed = paper_mgr.update_trades(latest_prices, max_hold_days=MAX_HOLD_DAYS)
    logger.info(f"  Trades closed this run: {trades_closed}")

    # ── Step 3: Market regime ──────────────────────────────────────────────────
    logger.info("\n[Step 3] Checking Nifty 50 market regime...")
    regime = bot._get_market_regime(days=300)
    logger.info(f"  Regime: {regime}")

    # ── Step 4: Scan for new signals ───────────────────────────────────────────
    open_count = len(paper_mgr.get_open_trades())
    slots_free = MAX_OPEN_TRADES - open_count

    logger.info(f"\n[Step 4] Scanning for new signals...")
    logger.info(f"  Open positions: {open_count} / {MAX_OPEN_TRADES} | "
                f"Slots free: {slots_free} | Free cash: ₹{paper_mgr.free_cash:,.2f}")

    new_trades    = 0
    signals_found = []

    if slots_free <= 0:
        logger.info("  No open slots — skipping scan")
    elif paper_mgr.free_cash < 500:
        logger.info(f"  Free cash ₹{paper_mgr.free_cash:.0f} too low — skipping scan")
    else:
        for symbol in SCAN_UNIVERSE:
            if new_trades >= slots_free:
                break   # filled all available slots
            try:
                df = bot.fetcher.get_historical_data(symbol, days=200)
                if df is None or len(df) < 50:
                    continue

                fund = bot.get_fundamentals_safe(symbol)

                sig, details = bot.signal_gen.generate_signal(
                    df, symbol, fund,
                    current_equity=paper_mgr.free_cash,  # size against free cash only
                    market_regime=regime,
                )

                if sig == 'BUY':
                    signals_found.append((symbol, details))
                    opened = paper_mgr.open_trade(
                        symbol=symbol,
                        entry_price=details['entry_price'],
                        stop_loss=details['stop_loss'],
                        target_price=details['target_price'],
                        position_size=details['position_size'],
                        entry_type=details['entry_type'],
                    )
                    if opened:
                        new_trades += 1

            except Exception as e:
                logger.error(f"  Error on {symbol}: {e}")

    if signals_found:
        logger.info(f"\n  BUY signals found ({len(signals_found)}):")
        for sym, det in signals_found:
            logger.info(
                f"    🎯 {sym} | {det['entry_type']} | "
                f"Entry ₹{det['entry_price']:.2f} | SL ₹{det['stop_loss']:.2f} | "
                f"Target ₹{det['target_price']:.2f} | R:R 1:{det['risk_reward_ratio']:.1f} | "
                f"Patterns: {', '.join(det.get('patterns_triggered', [det['entry_type']]))}"
            )
    else:
        logger.info("  No new BUY signals today")
    logger.info(f"  New trades opened: {new_trades}")

    # ── Step 5: Equity snapshot ────────────────────────────────────────────────
    logger.info("\n[Step 5] Logging equity snapshot...")
    paper_mgr.log_daily_equity(latest_prices)

    # ── Step 6: Full portfolio summary ────────────────────────────────────────
    _print_summary(paper_mgr, latest_prices)

    logger.info("\n✅ Run complete — GitHub Actions will now commit updated CSVs to repo")


def _print_summary(paper_mgr, latest_prices):
    summary = paper_mgr.get_summary(latest_prices)

    logger.info("\n" + "=" * 70)
    logger.info("📈 PORTFOLIO SUMMARY")
    logger.info("=" * 70)

    logger.info("\n  ── CAPITAL ──────────────────────────────────────────────────")
    logger.info(f"  Initial Equity        : ₹{summary.get('initial_equity',        0):>10,.2f}")
    logger.info(f"  Deployed Capital      : ₹{summary.get('deployed_capital',      0):>10,.2f}"
                f"  ({summary.get('open_trades', 0)} open positions)")
    logger.info(f"  Free Cash             : ₹{summary.get('free_cash',             0):>10,.2f}"
                f"  (available for new trades)")
    logger.info(f"  Total Portfolio Value : ₹{summary.get('total_portfolio_value', 0):>10,.2f}")

    logger.info("\n  ── P&L ──────────────────────────────────────────────────────")
    logger.info(f"  Realised P&L          : ₹{summary.get('realised_pnl',    0):>+10,.2f}"
                f"  ({summary.get('closed_trades', 0)} closed trades)")
    logger.info(f"  Unrealised P&L        : ₹{summary.get('unrealised_pnl',  0):>+10,.2f}"
                f"  (mark-to-market on open positions)")
    logger.info(f"  Total P&L             : ₹{summary.get('total_pnl',       0):>+10,.2f}")

    if summary.get('closed_trades', 0) > 0:
        logger.info("\n  ── CLOSED TRADE STATS ───────────────────────────────────────")
        logger.info(f"  Win Rate              : {summary.get('win_rate', 0):.1f}%"
                    f"  ({summary.get('wins', 0)}W / {summary.get('losses', 0)}L)")
        logger.info(f"  Avg Win / Avg Loss    : ₹{summary.get('avg_win', 0):+,.2f}"
                    f" / ₹{summary.get('avg_loss', 0):+,.2f}")

    logger.info("\n  ── OPEN POSITIONS ───────────────────────────────────────────")
    paper_mgr.print_open_positions(latest_prices)

    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    run_eod()
