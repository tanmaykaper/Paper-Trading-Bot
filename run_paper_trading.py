# run_paper_trading.py  ── FIXED + ENHANCED v2
# ─────────────────────────────────────────────────────────────────────────────
# Fixes vs original:
#   1. Passes latest prices to PaperTradingManager for unrealised P&L
#   2. EOD summary shows capital breakdown: initial / deployed / free cash / total P&L
#   3. Prints every open position with live CMP, unrealised P&L, hold days
#   4. Signal generation passes free_cash (not full equity) for position sizing
#   5. market_regime passed into signal generator
#   6. max_hold_days passed correctly into update_trades()
#   7. Added a stocks list that covers all symbols in existing paper_trades.csv
#      so exit checking works for legacy open positions too
# ─────────────────────────────────────────────────────────────────────────────

import logging
import pandas as pd
from datetime import datetime, time as dt_time
import time
from swing_trading_bot import SwingTradingBot
from paper_trading_manager import PaperTradingManager
from signal_generator import SignalGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DailyPaperTradingRunner:
    def __init__(self, initial_equity=10000, max_open_trades=5, max_hold_days=15):
        self.max_hold_days = max_hold_days

        self.bot = SwingTradingBot(
            send_emails=False,
            initial_equity=initial_equity,
            max_open_trades=max_open_trades,
            max_hold_days=max_hold_days,
        )
        self.paper_mgr = PaperTradingManager(
            initial_equity=initial_equity,
            max_open_trades=max_open_trades,
        )

        # ── Expanded stock universe ────────────────────────────────────────
        # Includes all symbols that appear in paper_trades.csv (for exit checking)
        # plus additional candidates for new signals
        self.stocks = [
            # Legacy positions (must be included so exits are checked)
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
            'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'ASIANPAINT',
            'MARUTI', 'TATASTEEL', 'BAJFINANCE', 'KOTAKBANK', 'LT',
            'AXISBANK', 'TITAN', 'WIPRO', 'ULTRACEMCO', 'NESTLEIND',
            # Additional for new signals
            'HCLTECH', 'TECHM', 'SUNPHARMA', 'DRREDDY', 'CIPLA',
            'TATAMOTORS', 'BAJAJ-AUTO', 'HINDALCO', 'JSWSTEEL',
            'ONGC', 'BPCL', 'GAIL', 'SIEMENS', 'ABB', 'DLF',
        ]

        logger.info(f"✓ DailyPaperTradingRunner | Equity: ₹{initial_equity:,} | "
                    f"Max trades: {max_open_trades} | Max hold: {max_hold_days}d")

    # ─────────────────────────────────────────────────────────────────────────
    def is_market_closed(self):
        now = datetime.now().time()
        return now >= dt_time(16, 0)

    def _fetch_prices(self, days=5):
        """
        Fetch latest close price for every stock.
        Uses a short window (5 days) — only need the last close for exit checks.
        Warnings about 'only N candles fetched' for days=5 are EXPECTED and fine.
        """
        prices = {}
        for symbol in self.stocks:
            try:
                df = self.bot.fetcher.get_historical_data(symbol, days=days)
                if df is not None and len(df) > 0:
                    prices[symbol] = float(df.iloc[-1]['close'])
            except Exception:
                pass
        logger.info(f"  Fetched prices for {len(prices)}/{len(self.stocks)} stocks")
        return prices

    # ─────────────────────────────────────────────────────────────────────────
    def run_eod_process(self):
        logger.info("\n" + "="*70)
        logger.info("📊 EOD PAPER TRADING PROCESS")
        logger.info("="*70)

        # ── Step 1: Fetch latest prices ────────────────────────────────────
        logger.info("\n[Step 1] Fetching latest prices...")
        latest_prices = self._fetch_prices(days=5)

        # ── Step 2: Check exits ────────────────────────────────────────────
        logger.info("\n[Step 2] Checking open trades for exits...")
        trades_closed = self.paper_mgr.update_trades(
            latest_prices, max_hold_days=self.max_hold_days)
        logger.info(f"  Trades closed today: {trades_closed}")

        # ── Step 3: Get market regime ──────────────────────────────────────
        logger.info("\n[Step 3] Checking market regime...")
        regime = self.bot._get_market_regime(days=300)

        # ── Step 4: Scan for new signals ───────────────────────────────────
        logger.info("\n[Step 4] Scanning for new signals...")
        new_trades = 0
        signals_found = []

        for symbol in self.stocks:
            try:
                df = self.bot.fetcher.get_historical_data(symbol, days=200)
                if df is None or len(df) < 50:
                    continue

                fundamentals = self.bot.get_fundamentals_safe(symbol)

                # Use FREE CASH for position sizing, not full initial equity
                signal_type, signal_details = self.bot.signal_gen.generate_signal(
                    df, symbol, fundamentals,
                    current_equity=self.paper_mgr.free_cash,
                    market_regime=regime,
                )

                if signal_type == 'BUY':
                    signals_found.append((symbol, signal_details))
                    opened = self.paper_mgr.open_trade(
                        symbol=symbol,
                        entry_price=signal_details['entry_price'],
                        stop_loss=signal_details['stop_loss'],
                        target_price=signal_details['target_price'],
                        position_size=signal_details['position_size'],
                        entry_type=signal_details['entry_type'],
                    )
                    if opened:
                        new_trades += 1

            except Exception as e:
                logger.error(f"  Error processing {symbol}: {e}")

        if signals_found:
            logger.info(f"\n  Found {len(signals_found)} BUY signal(s):")
            for sym, det in signals_found:
                logger.info(
                    f"    🎯 {sym} | {det['entry_type']} | "
                    f"Entry ₹{det['entry_price']:.2f} | "
                    f"SL ₹{det['stop_loss']:.2f} | "
                    f"Target ₹{det['target_price']:.2f} | "
                    f"R:R 1:{det['risk_reward_ratio']:.1f}"
                )
        else:
            logger.info("  No new BUY signals today")

        logger.info(f"  New trades opened: {new_trades}")

        # ── Step 5: Log equity snapshot ────────────────────────────────────
        logger.info("\n[Step 5] Logging equity snapshot...")
        self.paper_mgr.log_daily_equity(latest_prices)

        # ── Step 6: Full summary ───────────────────────────────────────────
        self._print_full_summary(latest_prices)

    # ─────────────────────────────────────────────────────────────────────────
    def _print_full_summary(self, latest_prices):
        summary = self.paper_mgr.get_summary(latest_prices)

        logger.info("\n" + "="*70)
        logger.info("📈 PORTFOLIO SUMMARY")
        logger.info("="*70)

        # Capital breakdown
        logger.info("\n  ── CAPITAL ──────────────────────────────────────────")
        logger.info(f"  Initial Equity       : ₹{summary.get('initial_equity', 0):>10,.2f}")
        logger.info(f"  Deployed Capital     : ₹{summary.get('deployed_capital', 0):>10,.2f}  "
                    f"({summary.get('open_trades', 0)} open positions)")
        logger.info(f"  Free Cash            : ₹{summary.get('free_cash', 0):>10,.2f}  "
                    f"(available for new trades)")
        logger.info(f"  Total Portfolio Value: ₹{summary.get('total_portfolio_value', 0):>10,.2f}")

        # P&L breakdown
        logger.info("\n  ── P&L ──────────────────────────────────────────────")
        logger.info(f"  Realised P&L         : ₹{summary.get('realised_pnl', 0):>+10,.2f}  "
                    f"(from {summary.get('closed_trades', 0)} closed trades)")
        logger.info(f"  Unrealised P&L       : ₹{summary.get('unrealised_pnl', 0):>+10,.2f}  "
                    f"(mark-to-market on open positions)")
        logger.info(f"  Total P&L            : ₹{summary.get('total_pnl', 0):>+10,.2f}")

        # Closed trade stats
        if summary.get('closed_trades', 0) > 0:
            logger.info("\n  ── CLOSED TRADE STATS ───────────────────────────────")
            logger.info(f"  Win Rate             : {summary.get('win_rate', 0):.1f}%  "
                        f"({summary.get('wins', 0)}W / {summary.get('losses', 0)}L)")
            logger.info(f"  Avg Win / Avg Loss   : ₹{summary.get('avg_win', 0):+,.2f} / "
                        f"₹{summary.get('avg_loss', 0):+,.2f}")

        # Open positions table
        logger.info("\n  ── OPEN POSITIONS ───────────────────────────────────")
        self.paper_mgr.print_open_positions(latest_prices)

        logger.info("\n" + "="*70)
        logger.info("✓ EOD PROCESS COMPLETE")
        logger.info("="*70 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    def run_continuous(self, hours=24):
        logger.info(f"🚀 Starting continuous paper trading for {hours} hours...")
        start = datetime.now()

        while (datetime.now() - start).total_seconds() < hours * 3600:
            try:
                if self.is_market_closed():
                    logger.info("Market closed. Running EOD process...")
                    self.run_eod_process()
                    logger.info("Waiting for next market open (9:15 AM IST)...")
                    time.sleep(60)
                else:
                    logger.info(f"Market open. Next EOD check at 4:00 PM IST...")
                    time.sleep(300)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in continuous run: {e}")
                time.sleep(60)


if __name__ == "__main__":
    runner = DailyPaperTradingRunner(
        initial_equity=10000,
        max_open_trades=5,
        max_hold_days=15,
    )

    # Option 1: Run continuously (recommended for live use)
    runner.run_continuous(hours=24)

    # Option 2: Run EOD once manually (for testing)
    # runner.run_eod_process()
