# run_paper_trading.py  ── GITHUB ACTIONS / SINGLE-RUN VERSION
# ─────────────────────────────────────────────────────────────────────────────
# Fixes applied:
#   1. Step 1 now uses get_ltp() — no bar-count minimum, always returns a price
#   2. ^NSEI handled correctly (no .NS suffix) via _to_yf_symbol in fetcher
#   3. TATAMOTORS → TATAMOTOR (yfinance symbol change)
#   4. Fallback: if get_ltp fails, try get_historical_data with min_bars=1
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
INITIAL_EQUITY  = 10_000
MAX_OPEN_TRADES = 5
MAX_HOLD_DAYS   = 15

TRADES_CSV = 'paper_trades.csv'
EQUITY_CSV = 'daily_equity.csv'

SCAN_UNIVERSE = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'ASIANPAINT',
    'MARUTI', 'TATASTEEL', 'BAJFINANCE', 'KOTAKBANK', 'LT',
    'AXISBANK', 'TITAN', 'WIPRO', 'ULTRACEMCO', 'NESTLEIND',
    'HCLTECH', 'TECHM', 'SUNPHARMA', 'DRREDDY', 'CIPLA',
    'TATAMOTOR',           # was TATAMOTORS — yfinance symbol changed
    'BAJAJ-AUTO', 'HINDALCO', 'JSWSTEEL',
    'ONGC', 'BPCL', 'GAIL', 'SIEMENS', 'ABB', 'DLF',
    'INDUSINDBK', 'FEDERALBNK', 'MPHASIS', 'LTIM', 'CHOLAFIN',
]


def get_all_held_symbols(trades_csv):
    if not os.path.exists(trades_csv):
        return set()
    try:
        df = pd.read_csv(trades_csv)
        if 'entry_datr' in df.columns and 'entry_date' not in df.columns:
            df = df.rename(columns={'entry_datr': 'entry_date'})
        return set(df[df['status'] == 'OPEN']['symbol'].tolist())
    except Exception:
        return set()


def _fetch_ltp(fetcher, symbol):
    """
    Best-effort price fetch:
      1. get_ltp()  — fast, single-row, no bar-count check
      2. Fallback: get_historical_data with min_bars=1, take last close
    Returns float or None.
    """
    price = fetcher.get_ltp(symbol)
    if price is not None:
        return price

    # Fallback — fetch a week of data, no minimum bar requirement
    try:
        df = fetcher.get_historical_data(symbol, days=10, min_bars=1)
        if df is not None and len(df) > 0:
            return float(df['close'].iloc[-1])
    except Exception:
        pass

    return None


def run_eod():
    logger.info("\n" + "=" * 70)
    logger.info(f"📅 NSE PAPER TRADING BOT — {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    logger.info("=" * 70)

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

    held_symbols  = get_all_held_symbols(TRADES_CSV)
    price_symbols = list(set(SCAN_UNIVERSE) | held_symbols)
    logger.info(f"\n  Held positions : {sorted(held_symbols) or 'none'}")
    logger.info(f"  Price fetch    : {len(price_symbols)} symbols")

    # ── Step 1: Fetch latest prices via get_ltp ───────────────────────────────
    logger.info("\n[Step 1] Fetching latest close prices...")
    latest_prices = {}
    for symbol in price_symbols:
        price = _fetch_ltp(bot.fetcher, symbol)
        if price is not None:
            latest_prices[symbol] = price

    logger.info(f"  Got prices for {len(latest_prices)}/{len(price_symbols)} symbols")

    missing_held = held_symbols - set(latest_prices.keys())
    if missing_held:
        logger.warning(f"  ⚠️ No price for held symbols: {missing_held} — exit check skipped for these")

    # ── Step 2: Exit checks ───────────────────────────────────────────────────
    logger.info("\n[Step 2] Checking open trades for exits...")
    trades_closed = paper_mgr.update_trades(latest_prices, max_hold_days=MAX_HOLD_DAYS)
    logger.info(f"  Trades closed this run: {trades_closed}")

    # ── Step 3: Market regime ─────────────────────────────────────────────────
    logger.info("\n[Step 3] Checking Nifty 50 market regime...")
    regime = bot._get_market_regime(days=300)
    logger.info(f"  Regime: {regime}")

    # ── Step 4: Scan for new signals ──────────────────────────────────────────
    open_count = len(paper_mgr.get_open_trades())
    slots_free = MAX_OPEN_TRADES - open_count

    logger.info(f"\n[Step 4] Scanning for new signals...")
    logger.info(
        f"  Open: {open_count}/{MAX_OPEN_TRADES} | "
        f"Slots free: {slots_free} | Free cash: ₹{paper_mgr.free_cash:,.2f}"
    )

    new_trades    = 0
    signals_found = []

    if slots_free <= 0:
        logger.info("  No open slots — skipping scan")
    elif paper_mgr.free_cash < 500:
        logger.info(f"  Free cash ₹{paper_mgr.free_cash:.0f} too low — skipping scan")
    else:
        for symbol in SCAN_UNIVERSE:
            if new_trades >= slots_free:
                break
            try:
                # Full history needed for signal generation — standard 200-bar fetch
                df = bot.fetcher.get_historical_data(symbol, days=200, min_bars=50)
                if df is None:
                    continue

                fund = bot.get_fundamentals_safe(symbol)
                sig, details = bot.signal_gen.generate_signal(
                    df, symbol, fund,
                    current_equity=paper_mgr.free_cash,
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
        logger.info(f"\n  BUY signals ({len(signals_found)}):")
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

    # ── Step 5: Equity snapshot ───────────────────────────────────────────────
    logger.info("\n[Step 5] Logging equity snapshot...")
    paper_mgr.log_daily_equity(latest_prices)

    # ── Step 6: Portfolio summary ─────────────────────────────────────────────
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
    logger.info(f"  Free Cash             : ₹{summary.get('free_cash',             0):>10,.2f}")
    logger.info(f"  Total Portfolio Value : ₹{summary.get('total_portfolio_value', 0):>10,.2f}")

    logger.info("\n  ── P&L ──────────────────────────────────────────────────────")
    logger.info(f"  Realised P&L          : ₹{summary.get('realised_pnl',    0):>+10,.2f}"
                f"  ({summary.get('closed_trades', 0)} closed trades)")
    logger.info(f"  Unrealised P&L        : ₹{summary.get('unrealised_pnl',  0):>+10,.2f}")
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
