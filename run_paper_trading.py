# run_paper_trading.py  ── GITHUB ACTIONS / SINGLE-RUN VERSION  v5
# ─────────────────────────────────────────────────────────────────────────────
# This is now the ONE canonical runner. (run_paper_tradingjbkb.py has been
# removed — it imported LARGECAP_SYMBOLS / SignalGenerator.score_signal, which
# don't exist anywhere in this codebase, so it could never have run.)
#
# What changed vs the previous version, and why — this directly targets the
# root causes behind "same 5 stocks held for months, ₹0 realised P&L":
#
#   1. ROOT CAUSE FIX — dtype crash on close (paper_trading_manager.py):
#      When every OPEN row has an empty exit_date/exit_reason, pandas infers
#      those columns as float64. Newer pandas then refuses to assign a string
#      into a float64 column and raises. That exception was silently caught
#      by the old try/except in update_trades(), so trades_closed was ALWAYS
#      0 — regardless of whether prices were available. This was the primary
#      cause and is now fixed at the source in _load_csv().
#
#   2. Bulk, retrying price fetch (get_ltp_bulk): one batched request for the
#      whole universe instead of one HTTP call per symbol in a loop — far
#      less likely to get rate-limited by Yahoo Finance from a shared IP
#      (e.g. GitHub Actions), and much faster.
#
#   3. Exit checks can no longer fail silently: a held position that's still
#      missing a price after bulk + individual retries gets a loud, explicit
#      HEALTH CHECK error (and an email alert if credentials are configured)
#      instead of just being skipped with a debug-level log line nobody sees.
#
#   4. force_close_stale() safety net: any OPEN trade older than
#      max_hold_days gets closed even in the rare case a live price still
#      isn't available (falls back to entry price rather than staying open
#      indefinitely). This is what unsticks the current backlog on first run.
#
#   5. Expanded scan universe (large-cap + mid-cap) — no longer hardcoded to
#      ~15 names, so signal generation isn't the bottleneck either.
#
#   6. Position replacement: when all slots are full, a materially stronger
#      new signal can swap out the weakest open position — subject to
#      profit-protection and near-target guards — instead of being dropped
#      on the floor while a mediocre position sits untouched for months.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
import os
import pandas as pd
from datetime import datetime

from swing_trading_bot import SwingTradingBot
from paper_trading_manager import PaperTradingManager
from notification_handler import NotificationHandler

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

# Replacement gate — a new signal must clear ALL of these to bump an
# existing open position out of its slot:
REPLACE_SCORE_MULTIPLE = 1.40   # new composite score must beat the weakest by 40%+
PROTECT_PROFIT_PCT     = 0.03   # never replace a position up >3% unrealised
PROTECT_PROGRESS_PCT   = 0.80   # never replace a position >80% of the way to target

LARGECAP_UNIVERSE = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'ASIANPAINT',
    'MARUTI', 'TATASTEEL', 'BAJFINANCE', 'KOTAKBANK', 'LT',
    'AXISBANK', 'TITAN', 'WIPRO', 'ULTRACEMCO', 'NESTLEIND',
    'HCLTECH', 'TECHM', 'SUNPHARMA', 'DRREDDY', 'CIPLA',
    'TATAMOTOR', 'BAJAJ-AUTO', 'HINDALCO', 'JSWSTEEL',
    'ONGC', 'BPCL', 'GAIL', 'SIEMENS', 'ABB', 'DLF',
    'INDUSINDBK', 'FEDERALBNK', 'MPHASIS', 'LTIM', 'CHOLAFIN',
]

# NOTE: I can't reach yfinance/NSE from this sandbox to verify every ticker
# below trades under exactly this symbol today. That's fine by design — any
# symbol yfinance doesn't recognise just returns None from get_historical_data
# and is skipped (existing, already-safe behaviour) — but you should spot
# check this list once you run it live and prune anything that never resolves.
MIDCAP_UNIVERSE = [
    'PERSISTENT', 'COFORGE', 'KPITTECH', 'TATAELXSI', 'INTELLECT',
    'ALKEM', 'TORNTPHARM', 'AUROPHARMA', 'GRANULES', 'IPCALAB',
    'AUBANK', 'RBLBANK', 'CREDITACC',
    'MOTHERSON', 'BALKRISIND', 'SUPRAJIT',
    'TATACONSUM', 'RADICO', 'VSTIND',
    'DEEPAKNTR', 'AARTIIND', 'VINATIORGA', 'NAVINFLUOR',
    'KAJARIACER', 'APLAPOLLO', 'GRINDWELL', 'RATNAMANI',
    'SOBHA', 'PHOENIXLTD',
    'HAPPSTMNDS', 'DIXON', 'AMBER',
]

SCAN_UNIVERSE = LARGECAP_UNIVERSE + MIDCAP_UNIVERSE


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


def _composite_score(details):
    """Score a candidate BUY signal for replacement comparisons."""
    return float(details.get('confidence', 1)) * float(details.get('risk_reward_ratio', 1.0))


def _existing_position_score(trade):
    """
    Approximate the same composite score for an already-open position,
    using the confidence/risk_reward_ratio recorded at entry time. Trades
    opened before this field existed fall back to a neutral estimate so
    they aren't unfairly favoured or penalised by missing data.
    """
    conf = trade.get('confidence')
    rr   = trade.get('risk_reward_ratio')
    conf = float(conf) if pd.notna(conf) and conf != '' else 3.0   # neutral mid-range
    rr   = float(rr)   if pd.notna(rr)   and rr   != '' else 2.0   # neutral mid-range
    return conf * rr


def find_replaceable_position(open_trades, new_details, latest_prices):
    """
    Return the weakest open trade eligible for replacement by new_details,
    or None if nothing qualifies. ALL of these must hold:
      1. new signal's composite score > weakest existing score * REPLACE_SCORE_MULTIPLE
      2. that position's unrealised gain < PROTECT_PROFIT_PCT (don't cut winners)
      3. that position's progress toward its own target < PROTECT_PROGRESS_PCT
    """
    new_score = _composite_score(new_details)
    candidates = []

    for t in open_trades:
        sym = t['symbol']
        if sym not in latest_prices:
            continue  # can't safely evaluate without a current price

        ep, sl, tp = float(t['entry_price']), float(t['stop_loss']), float(t['target_price'])
        cmp        = float(latest_prices[sym])
        unreal_pct = (cmp - ep) / ep if ep else 0.0
        progress   = max(0.0, (cmp - ep) / (tp - ep)) if tp > ep else 0.0

        if unreal_pct >= PROTECT_PROFIT_PCT:
            continue
        if progress >= PROTECT_PROGRESS_PCT:
            continue

        candidates.append((t, _existing_position_score(t)))

    if not candidates:
        return None

    weakest_trade, weakest_score = min(candidates, key=lambda x: x[1])
    if new_score >= weakest_score * REPLACE_SCORE_MULTIPLE:
        return weakest_trade
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

    # Separate from bot.notifier (which is only for trade-signal emails) —
    # this fires regardless of that setting whenever email creds exist,
    # because system-health failures matter even if you don't want signal spam.
    alert_notifier = NotificationHandler(use_email=True, use_sms=False)

    held_symbols  = get_all_held_symbols(TRADES_CSV)
    price_symbols = list(set(SCAN_UNIVERSE) | held_symbols)
    logger.info(f"\n  Held positions : {sorted(held_symbols) or 'none'}")
    logger.info(f"  Price fetch    : {len(price_symbols)} symbols")

    # ── Step 1: Bulk price fetch ──────────────────────────────────────────────
    logger.info("\n[Step 1] Fetching latest prices (bulk, with retries)...")
    latest_prices = bot.fetcher.get_ltp_bulk(price_symbols)
    logger.info(f"  Got prices for {len(latest_prices)}/{len(price_symbols)} symbols")

    missing_held = held_symbols - set(latest_prices.keys())
    if missing_held:
        logger.warning(f"  ⚠️ Still no price for held symbols after retries: {missing_held}")

    # ── Step 2: Exit checks ───────────────────────────────────────────────────
    logger.info("\n[Step 2] Checking open trades for exits...")
    trades_closed = paper_mgr.update_trades(latest_prices, max_hold_days=MAX_HOLD_DAYS)
    logger.info(f"  Trades closed this run: {trades_closed}")

    # Safety net: anything still open past its hold window despite the above
    # (i.e. price genuinely unavailable even after retries) gets force-closed
    # rather than left to rot silently for months, as happened before.
    stale_closed = paper_mgr.force_close_stale(latest_prices, max_hold_days=MAX_HOLD_DAYS)
    if stale_closed:
        logger.warning(f"  ⚠️ force_close_stale cleared {len(stale_closed)} position(s) that update_trades missed")
        trades_closed += len(stale_closed)

    # ── Step 3: Health check — never let a bad run pass silently again ───────
    logger.info("\n[Step 3] Price-fetch health check...")
    health = paper_mgr.price_fetch_health_check(latest_prices)
    if not health['healthy']:
        alert_notifier.send_alert(
            subject="Price fetch incomplete — exit checks may be skipped",
            body=(
                f"{len(health['missing_symbols'])}/{health['held_positions']} open positions "
                f"had no live price this run: {health['missing_symbols']}.\n\n"
                f"If this repeats for several consecutive runs, price fetching is broken "
                f"and positions can silently stay open indefinitely — check yfinance/network "
                f"status and this bot's logs."
            ),
        )

    # ── Step 4: Market regime ─────────────────────────────────────────────────
    logger.info("\n[Step 4] Checking Nifty 50 market regime...")
    regime = bot._get_market_regime(days=300)
    logger.info(f"  Regime: {regime}")

    # ── Step 5: Scan for new signals (with position replacement) ─────────────
    open_trades = paper_mgr.get_open_trades()
    open_count  = len(open_trades)
    slots_free  = MAX_OPEN_TRADES - open_count

    logger.info(f"\n[Step 5] Scanning {len(SCAN_UNIVERSE)} symbols for new signals...")
    logger.info(
        f"  Open: {open_count}/{MAX_OPEN_TRADES} | "
        f"Slots free: {slots_free} | Free cash: ₹{paper_mgr.free_cash:,.2f}"
    )

    new_trades      = 0
    replacements     = 0
    signals_found   = []

    if paper_mgr.free_cash < 500:
        logger.info(f"  Free cash ₹{paper_mgr.free_cash:.0f} too low — skipping scan")
    else:
        for symbol in SCAN_UNIVERSE:
            if symbol in held_symbols:
                continue
            try:
                df = bot.fetcher.get_historical_data(symbol, days=200, min_bars=50)
                if df is None:
                    continue

                fund = bot.get_fundamentals_safe(symbol)
                sig, details = bot.signal_gen.generate_signal(
                    df, symbol, fund,
                    current_equity=paper_mgr.free_cash,
                    market_regime=regime,
                )

                if sig != 'BUY':
                    continue

                signals_found.append((symbol, details))

                if slots_free > 0:
                    opened = paper_mgr.open_trade(
                        symbol=symbol,
                        entry_price=details['entry_price'],
                        stop_loss=details['stop_loss'],
                        target_price=details['target_price'],
                        position_size=details['position_size'],
                        entry_type=details['entry_type'],
                        confidence=details.get('confidence'),
                        risk_reward_ratio=details.get('risk_reward_ratio'),
                    )
                    if opened:
                        new_trades  += 1
                        slots_free  -= 1
                        held_symbols.add(symbol)
                else:
                    # No free slots — see if this signal is strong enough to
                    # replace the weakest eligible open position.
                    weak = find_replaceable_position(
                        paper_mgr.get_open_trades(), details, latest_prices
                    )
                    if weak is not None:
                        exit_price = float(latest_prices.get(weak['symbol'], weak['entry_price']))
                        closed_ok = paper_mgr.close_position(
                            weak['trade_id'], exit_price,
                            exit_reason=f'Replaced by stronger signal ({symbol})',
                        )
                        if closed_ok:
                            opened = paper_mgr.open_trade(
                                symbol=symbol,
                                entry_price=details['entry_price'],
                                stop_loss=details['stop_loss'],
                                target_price=details['target_price'],
                                position_size=details['position_size'],
                                entry_type=details['entry_type'],
                                confidence=details.get('confidence'),
                                risk_reward_ratio=details.get('risk_reward_ratio'),
                            )
                            if opened:
                                new_trades   += 1
                                replacements += 1
                                held_symbols.discard(weak['symbol'])
                                held_symbols.add(symbol)

            except Exception as e:
                logger.error(f"  Error on {symbol}: {e}")

    if signals_found:
        logger.info(f"\n  BUY signals found ({len(signals_found)}):")
        for sym, det in signals_found:
            logger.info(
                f"    🎯 {sym} | {det['entry_type']} | "
                f"Entry ₹{det['entry_price']:.2f} | SL ₹{det['stop_loss']:.2f} | "
                f"Target ₹{det['target_price']:.2f} | R:R 1:{det['risk_reward_ratio']:.1f} | "
                f"conf={det.get('confidence')}"
            )
    else:
        logger.info("  No new BUY signals today")
    logger.info(f"  New trades opened: {new_trades}  (of which replacements: {replacements})")

    # ── Step 6: Equity snapshot ───────────────────────────────────────────────
    logger.info("\n[Step 6] Logging equity snapshot...")
    paper_mgr.log_daily_equity(latest_prices)

    # ── Step 7: Portfolio summary ─────────────────────────────────────────────
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
