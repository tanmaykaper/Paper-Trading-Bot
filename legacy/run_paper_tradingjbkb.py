# run_paper_trading.py  ── v4  FIXED + MIDCAP + POSITION REPLACEMENT
# ─────────────────────────────────────────────────────────────────────────────
# New in v4 vs v3:
#   1. Expanded SCAN_UNIVERSE — 40 largecaps + 35 midcaps
#   2. Position replacement logic:
#        When all slots are full, a new BUY signal is evaluated against the
#        weakest open position. Replacement happens only if:
#          a) new signal score > weakest score × REPLACE_THRESHOLD (default 1.4×)
#          b) the existing position has no meaningful unrealised profit
#             (unrealised_pct < PROTECT_PROFIT_PCT, default 3%)
#          c) the existing position is not near its target (< 80% of the way)
#        Replacement = close the weak position at market, open the new one.
#   3. get_ltp_bulk() for all price fetching (fixes the 0/40 price bug)
#   4. ^NSEI handled correctly (no .NS suffix)
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
import os
import time
import pandas as pd
from datetime import datetime

from swing_trading_bot import SwingTradingBot
from paper_trading_manager import PaperTradingManager
from signal_generator import SignalGenerator, LARGECAP_SYMBOLS

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

# Replacement gate:
#   new_score must be >= weak_score × REPLACE_THRESHOLD
#   existing position must have unrealised gain < PROTECT_PROFIT_PCT
#   existing position must not be > PROTECT_PROGRESS_PCT toward target
REPLACE_THRESHOLD      = 1.40   # new signal must be 40% better in composite score
PROTECT_PROFIT_PCT     = 0.03   # don't replace if position is up >3% unrealised
PROTECT_PROGRESS_PCT   = 0.80   # don't replace if position is >80% to target

# ── Scan universe ─────────────────────────────────────────────────────────────
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

MIDCAP_UNIVERSE = [
    # IT / Software
    'PERSISTENT', 'COFORGE', 'KPITTECH', 'TATAELXSI', 'INTELLECT',
    # Pharma / Healthcare
    'ALKEM', 'TORNTPHARM', 'AUROPHARMA', 'GRANULES', 'IPCALAB',
    # Banking / NBFC
    'AUBANK', 'RBLBANK', 'UJJIVANSFB', 'AAVAS', 'CREDITACC',
    # Auto ancillaries
    'MOTHERSON', 'BALKRISIND', 'ENDURANCE', 'SUPRAJIT',
    # Consumer / FMCG
    'TATACONSUM', 'RADICO', 'VSTIND', 'SULA',
    # Chemicals
    'DEEPAKNITR', 'AARTI', 'VINATI', 'NAVINFLUOR',
    # Infra / Capital goods
    'KAJARIACER', 'APLAPOLLO', 'GRINDWELL', 'RATNAMANI',
    # Realty
    'SOBHA', 'PHOENIXLTD',
    # Others
    'HAPPSTMNDS', 'DIXON', 'AMBER',
]

SCAN_UNIVERSE = LARGECAP_UNIVERSE + MIDCAP_UNIVERSE


def get_all_held_symbols(trades_csv: str) -> set:
    if not os.path.exists(trades_csv):
        return set()
    try:
        df = pd.read_csv(trades_csv)
        if 'entry_datr' in df.columns and 'entry_date' not in df.columns:
            df = df.rename(columns={'entry_datr': 'entry_date'})
        return set(df[df['status'] == 'OPEN']['symbol'].tolist())
    except Exception:
        return set()


def _position_unrealised_pct(trade: dict, latest_prices: dict) -> float:
    """Return unrealised % gain for an open trade. Returns 0 if price unavailable."""
    sym = trade['symbol']
    if sym not in latest_prices:
        return 0.0
    ep  = float(trade['entry_price'])
    cmp = float(latest_prices[sym])
    return (cmp - ep) / ep


def _position_progress_to_target(trade: dict, latest_prices: dict) -> float:
    """Return how far (0–1) the position has moved toward its target."""
    sym = trade['symbol']
    if sym not in latest_prices:
        return 0.0
    ep  = float(trade['entry_price'])
    tp  = float(trade['target_price'])
    cmp = float(latest_prices[sym])
    total_move = tp - ep
    if total_move <= 0:
        return 0.0
    return max(0.0, (cmp - ep) / total_move)


def _existing_signal_score(trade: dict) -> float:
    """
    Approximate composite score for an existing open position.
    Uses original R:R and entry_type as a proxy for confidence.
    """
    rr = float(trade.get('risk_reward_ratio', 2.0)) if 'risk_reward_ratio' in trade else 2.0
    # Infer confidence from entry_type: midcap patterns get 0.85 cap_bonus
    cap_bonus = 0.85 if trade.get('symbol') not in LARGECAP_SYMBOLS else 1.0
    # We don't have original confidence stored; use 0.4 (middle of range) as default
    return rr * 0.4 * cap_bonus


def find_replaceable_position(open_trades: list, new_signal_details: dict,
                              latest_prices: dict) -> dict | None:
    """
    Return the open trade that should be replaced by new_signal_details,
    or None if no replacement is warranted.

    Replacement criteria (ALL must be true):
      1. New signal's composite score > existing score × REPLACE_THRESHOLD
      2. Existing position unrealised gain < PROTECT_PROFIT_PCT
      3. Existing position progress toward target < PROTECT_PROGRESS_PCT
    """
    new_score = SignalGenerator.score_signal(new_signal_details)

    candidates = []
    for t in open_trades:
        unreal_pct = _position_unrealised_pct(t, latest_prices)
        progress   = _position_progress_to_target(t, latest_prices)
        score      = _existing_signal_score(t)

        # Hard protections
        if unreal_pct >= PROTECT_PROFIT_PCT:
            continue
        if progress >= PROTECT_PROGRESS_PCT:
            continue

        # Must be meaningfully better
        if new_score >= score * REPLACE_THRESHOLD:
            candidates.append((score, unreal_pct, t))

    if not candidates:
        return None

    # Pick the weakest candidate (lowest score; if tie, most negative unrealised)
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def run_eod():
    t0 = time.time()
    logger.info("\n" + "=" * 70)
    logger.info(f"📅 NSE PAPER TRADING BOT — {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    logger.info("=" * 70)

    # ── Init ──────────────────────────────────────────────────────────────────
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

    # ── Build price-fetch list ─────────────────────────────────────────────────
    held_symbols  = get_all_held_symbols(TRADES_CSV)
    price_symbols = sorted(set(SCAN_UNIVERSE) | held_symbols)
    logger.info(f"\n  Held positions  : {sorted(held_symbols) or 'none'}")
    logger.info(f"  Price fetch list: {len(price_symbols)} symbols "
                f"({len(LARGECAP_UNIVERSE)} largecap + {len(MIDCAP_UNIVERSE)} midcap)")

    # ── Step 1: Batch price fetch ──────────────────────────────────────────────
    logger.info("\n[Step 1] Batch-fetching latest close prices...")
    t1 = time.time()
    latest_prices = bot.fetcher.get_ltp_bulk(price_symbols, period='5d')
    logger.info(f"  Got prices for {len(latest_prices)}/{len(price_symbols)} symbols "
                f"(took {time.time()-t1:.1f}s)")

    missing_held = held_symbols - set(latest_prices.keys())
    if missing_held:
        logger.warning(f"  ⚠️ No price for held: {missing_held} (possibly delisted)")

    # ── Step 2: Exit checks ────────────────────────────────────────────────────
    logger.info("\n[Step 2] Checking open trades for exits...")
    trades_closed = paper_mgr.update_trades(latest_prices, max_hold_days=MAX_HOLD_DAYS)
    logger.info(f"  Trades closed this run: {trades_closed}")

    # ── Step 3: Market regime ──────────────────────────────────────────────────
    logger.info("\n[Step 3] Checking Nifty 50 market regime...")
    regime = bot._get_market_regime(days=300)
    logger.info(f"  Regime: {regime}")

    # ── Step 4: Scan + replacement logic ──────────────────────────────────────
    open_count = len(paper_mgr.get_open_trades())
    slots_free = MAX_OPEN_TRADES - open_count

    logger.info(f"\n[Step 4] Scanning for signals + evaluating replacements...")
    logger.info(f"  Open: {open_count}/{MAX_OPEN_TRADES} | Slots free: {slots_free} "
                f"| Free cash: ₹{paper_mgr.free_cash:,.2f}")

    new_trades      = 0
    replacements    = 0
    signals_found   = []

    if paper_mgr.free_cash < 200 and slots_free <= 0:
        logger.info("  No cash and no slots — skipping scan entirely")
    else:
        for symbol in SCAN_UNIVERSE:
            try:
                df = bot.fetcher.get_historical_data(symbol, days=200)
                if df is None or len(df) < 50:
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

                # Case A: free slot available
                if slots_free > 0 and paper_mgr.free_cash >= details['entry_price']:
                    opened = paper_mgr.open_trade(
                        symbol        = symbol,
                        entry_price   = details['entry_price'],
                        stop_loss     = details['stop_loss'],
                        target_price  = details['target_price'],
                        position_size = details['position_size'],
                        entry_type    = details['entry_type'],
                    )
                    if opened:
                        new_trades += 1
                        slots_free -= 1

                # Case B: all slots full — evaluate replacement
                elif slots_free <= 0:
                    current_open = paper_mgr.get_open_trades()
                    victim = find_replaceable_position(current_open, details, latest_prices)

                    if victim is None:
                        continue

                    # Execute replacement: close victim at current market price
                    victim_sym = victim['symbol']
                    exit_price = latest_prices.get(victim_sym)
                    if exit_price is None:
                        logger.warning(f"  ⚠️ Can't replace {victim_sym} — no market price")
                        continue

                    victim_score = _existing_signal_score(victim)
                    new_score    = SignalGenerator.score_signal(details)
                    unreal_pct   = _position_unrealised_pct(victim, latest_prices) * 100

                    logger.info(
                        f"  🔄 REPLACING {victim_sym} (score={victim_score:.2f}, "
                        f"unrealised={unreal_pct:+.1f}%) "
                        f"→ {symbol} (score={new_score:.2f}, "
                        f"R:R={details['risk_reward_ratio']:.1f}, "
                        f"type={details['entry_type']}, "
                        f"cap={details.get('cap_type','?')})"
                    )

                    # Force-close the victim by injecting a price that triggers exit
                    # We do this by temporarily setting victim's SL to current price
                    closed = paper_mgr.force_close_trade(
                        trade_id   = victim['trade_id'],
                        exit_price = float(exit_price),
                        exit_reason= f"Replaced by {symbol}",
                    )
                    if closed:
                        opened = paper_mgr.open_trade(
                            symbol        = symbol,
                            entry_price   = details['entry_price'],
                            stop_loss     = details['stop_loss'],
                            target_price  = details['target_price'],
                            position_size = details['position_size'],
                            entry_type    = details['entry_type'],
                        )
                        if opened:
                            replacements += 1

            except Exception as e:
                logger.error(f"  Error scanning {symbol}: {e}")

    if signals_found:
        logger.info(f"\n  BUY signals found ({len(signals_found)}):")
        for sym, det in signals_found:
            logger.info(
                f"    🎯 {sym} [{det.get('cap_type','?')}] | {det['entry_type']} | "
                f"Entry ₹{det['entry_price']:.2f} | SL ₹{det['stop_loss']:.2f} | "
                f"Target ₹{det['target_price']:.2f} | R:R 1:{det['risk_reward_ratio']:.1f} | "
                f"Patterns: {', '.join(det.get('patterns_triggered', [det['entry_type']]))}"
            )
    else:
        logger.info("  No new BUY signals today")
    logger.info(f"  New trades opened: {new_trades} | Replacements: {replacements}")

    # ── Step 5: Equity snapshot ────────────────────────────────────────────────
    logger.info("\n[Step 5] Logging equity snapshot...")
    paper_mgr.log_daily_equity(latest_prices)

    # ── Step 6: Summary ────────────────────────────────────────────────────────
    _print_summary(paper_mgr, latest_prices)

    logger.info(f"\n✅ Run complete in {time.time()-t0:.0f}s — "
                "GitHub Actions will commit updated CSVs to repo")


def _print_summary(paper_mgr, latest_prices):
    summary = paper_mgr.get_summary(latest_prices)
    ie      = summary.get('initial_equity', INITIAL_EQUITY)
    tv      = summary.get('total_portfolio_value', ie)
    ret_pct = (tv - ie) / ie * 100

    logger.info("\n" + "=" * 70)
    logger.info("📈 PORTFOLIO SUMMARY")
    logger.info("=" * 70)

    logger.info("\n  ── CAPITAL ──────────────────────────────────────────────────")
    logger.info(f"  Initial Equity        : ₹{ie:>10,.2f}")
    logger.info(f"  Deployed Capital      : ₹{summary.get('deployed_capital', 0):>10,.2f}"
                f"  ({summary.get('open_trades', 0)} positions)")
    logger.info(f"  Free Cash             : ₹{summary.get('free_cash', 0):>10,.2f}")
    logger.info(f"  Total Portfolio Value : ₹{tv:>10,.2f}  ({ret_pct:+.2f}%)")

    logger.info("\n  ── P&L ──────────────────────────────────────────────────────")
    logger.info(f"  Realised P&L          : ₹{summary.get('realised_pnl', 0):>+10,.2f}"
                f"  ({summary.get('closed_trades', 0)} closed trades)")
    logger.info(f"  Unrealised P&L        : ₹{summary.get('unrealised_pnl', 0):>+10,.2f}"
                f"  (mark-to-market)")
    logger.info(f"  Total P&L             : ₹{summary.get('total_pnl', 0):>+10,.2f}")

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
