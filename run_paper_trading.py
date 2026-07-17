# run_paper_trading.py  ── GITHUB ACTIONS / SINGLE-RUN VERSION  v7
# ─────────────────────────────────────────────────────────────────────────────
# v7 change — calibrated for an explicitly stated high risk capacity and a
# preference for high-growth/momentum names, with reinvestment/compounding
# made explicit. Three things changed:
#
#   1. RISK DIAL RAISED, NOT REMOVED: risk_pct_per_trade (2.5%→4%) and
#      max_capital_pct (30%→40%) in signal_generator.py, plus
#      MAX_PORTFOLIO_RISK_PCT (10%→16%), LIVE_MAX_SECTOR_EXPOSURE (3→4), and
#      LIVE_MAX_DRAWDOWN (30%→35%) here — see the "Risk tolerance" block
#      below for the full before/after table and reasoning. Every one of
#      these is still an active cap; none were deleted. The sector/drawdown
#      overrides live HERE rather than in swing_trading_bot.py on purpose,
#      so a backtest run doesn't silently inherit a live-only risk setting.
#
#   2. HIGH-GROWTH/MOMENTUM UNIVERSE ADDED: LARGECAP/MIDCAP skew toward
#      established, comparatively stable businesses — not where "high risk,
#      high growth, momentum" exposure actually lives. Added
#      HIGH_GROWTH_MOMENTUM_UNIVERSE (new-age tech/internet, defence,
#      renewable energy/EV — themes confirmed live via search, July 2026,
#      not just historically notable) and gave each theme a proper
#      SECTOR_MAP grouping in swing_trading_bot.py, so the concentration cap
#      actually treats correlated theme clusters as one sector instead of
#      each stock silently counting as its own separate "sector."
#
#   3. COMPOUNDING MADE VISIBLE: the sizing-basis fix in v6 (total equity,
#      not free_cash) already meant realised + unrealised gains flow into
#      the next trade's size automatically — that mechanism doesn't need to
#      change again, it needed to be checked and shown. Added a "Growth
#      Multiple" line to the summary so reinvestment is a visible, auditable
#      number every run instead of an implicit side-effect.
#
# Carried over from v6/v5 (still true — see CHANGES_step1.md / step2.md):
# resilient bulk price fetching, dtype-crash fix for closing trades, health
# checks + alerting, stale-position safety net, position replacement for
# full slots, aggregate risk budget, sector cap, drawdown circuit breaker.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
import os
import pandas as pd
from datetime import datetime

from swing_trading_bot import SwingTradingBot, SECTOR_MAP, MAX_SECTOR_EXPOSURE, MAX_DRAWDOWN_DEFAULT
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

# ── Risk tolerance ───────────────────────────────────────────────────────────
# Raised across the board to reflect an explicitly stated high risk capacity —
# willing to size up on high-growth/momentum names for higher upside, in
# exchange for a wider (but still real) downside band. Nothing here removes
# a safeguard; every cap is still active, just calibrated looser. If this
# turns out to be too aggressive (or not aggressive enough) once you've
# watched it run, these five numbers are the whole risk dial — no code
# changes needed to retune.
#
#                          before  →  now      reasoning
#   risk_pct_per_trade      2.5%   →  4%       (signal_generator.py) bigger bet per high-conviction idea
#   max_capital_pct         30%    →  40%      (signal_generator.py) allows more concentrated single-name bets
#   MAX_PORTFOLIO_RISK_PCT   10%   →  16%      raised in proportion to the per-trade increase
#   MAX_SECTOR_EXPOSURE       3    →  4        allows heavier weighting into one high-conviction theme
#   MAX_DRAWDOWN (circuit breaker) 30% → 35%   tolerates a deeper drawdown before pausing new entries
#
# MAX_SECTOR_EXPOSURE and MAX_DRAWDOWN are deliberately overridden HERE
# rather than edited in swing_trading_bot.py — that file's constants are
# also used by the internal backtester, and there's no reason a backtest
# calibration run should silently inherit a live-trading-specific risk
# preference. Live trading and backtesting can reasonably run different
# risk settings; this keeps them decoupled on purpose.
MAX_PORTFOLIO_RISK_PCT   = 0.16   # % of total equity, worst case, across the whole book
LIVE_MAX_SECTOR_EXPOSURE = 4      # overrides swing_trading_bot.MAX_SECTOR_EXPOSURE (3) for live trading
LIVE_MAX_DRAWDOWN        = 0.35   # overrides swing_trading_bot.MAX_DRAWDOWN_DEFAULT (0.30) for live trading

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

# ── High-growth / momentum universe ─────────────────────────────────────────
# Added to reflect an explicitly stated high risk capacity and preference for
# high-growth, high-momentum names — the LARGECAP/MIDCAP lists above skew
# toward established, comparatively stable businesses, which isn't where
# that kind of exposure lives. These three themes were confirmed live (web
# search, July 2026) as currently active, not just historically notable:
#
#   • New-age tech/internet — high growth, high volatility, sentiment-driven.
#     Zomato's parent renamed to Eternal Ltd in 2025 and was added to the
#     Nifty 50; ticker remains ZOMATO on NSE/yfinance.
#   • Defence — genuinely in a live momentum phase as of mid-2026: multiple
#     consecutive rally sessions in June/July on record defence production
#     figures and large DAC procurement approvals (₹52,000cr+ tranches).
#   • Renewable energy / EV — an active, high-beta theme through 2026 (solar
#     manufacturing capacity buildout, wind order momentum), though names
#     here swing both ways day to day, consistent with genuinely higher risk.
#
# Same caveat as MIDCAP_UNIVERSE: I don't have live yfinance/NSE access from
# this sandbox to individually confirm every ticker still resolves — a few
# were spot-checked via search (ZOMATO, WAAREEENER, ACMESOLAR), the rest are
# good-faith based on current sourcing. Unresolvable tickers are skipped
# automatically (existing safe behaviour) — prune anything that never hits.
HIGH_GROWTH_MOMENTUM_UNIVERSE = [
    # New-age tech / internet
    'ZOMATO', 'NYKAA', 'PAYTM', 'POLICYBZR', 'DELHIVERY', 'IRCTC',
    'NAUKRI', 'INDIAMART', 'CARTRADE', 'MAPMYINDIA', 'EASEMYTRIP', 'NAZARA',
    # Defence — live momentum theme as of mid-2026, see note above
    'HAL', 'BEL', 'BDL', 'MAZDOCK', 'COCHINSHIP', 'SOLARINDS',
    'ASTRAMICRO', 'MTARTECH', 'PARAS', 'ZENTEC', 'DATAPATTNS', 'BEML', 'GRSE',
    # Renewable energy / EV — high-beta, both-directions theme
    'SUZLON', 'WAAREEENER', 'ADANIGREEN', 'NTPCGREEN', 'ACMESOLAR',
    'PREMIERENE', 'JSWENERGY', 'TATAPOWER', 'INOXWIND',
]

SCAN_UNIVERSE = LARGECAP_UNIVERSE + MIDCAP_UNIVERSE + HIGH_GROWTH_MOMENTUM_UNIVERSE


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


def get_peak_equity(equity_csv_path, floor):
    """
    Highest total_portfolio_value ever recorded in the equity log, used as
    the reference point for the drawdown circuit breaker. Falls back to
    `floor` (INITIAL_EQUITY) if there's no usable history yet.
    """
    if not os.path.exists(equity_csv_path):
        return floor
    try:
        df = pd.read_csv(equity_csv_path)
        if 'total_portfolio_value' not in df.columns:
            return floor
        vals = pd.to_numeric(df['total_portfolio_value'], errors='coerce').dropna()
        if len(vals) == 0:
            return floor
        return max(floor, float(vals.max()))
    except Exception:
        return floor


def find_replaceable_position(open_trades, new_details, latest_prices, sector_filter=None):
    """
    Return the weakest open trade eligible for replacement by new_details,
    or None if nothing qualifies. ALL of these must hold:
      1. new signal's composite score > weakest existing score * REPLACE_SCORE_MULTIPLE
      2. that position's unrealised gain < PROTECT_PROFIT_PCT (don't cut winners)
      3. that position's progress toward its own target < PROTECT_PROGRESS_PCT
      4. if sector_filter is given, only positions in that sector are considered
         (used when the new signal's own sector is already at its exposure cap —
         it may only swap in by replacing a position in the SAME sector, so the
         swap is sector-neutral rather than adding new concentration)
    """
    new_score = _composite_score(new_details)
    candidates = []

    for t in open_trades:
        sym = t['symbol']
        if sym not in latest_prices:
            continue  # can't safely evaluate without a current price
        if sector_filter is not None and SECTOR_MAP.get(sym, sym) != sector_filter:
            continue

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

    # ── Step 4: Portfolio equity & risk state ─────────────────────────────────
    # Computed AFTER exits so it reflects today's true state, and used as the
    # basis for position sizing instead of paper_mgr.free_cash (see header
    # comment — sizing off free_cash under-sizes later trades in a run by up
    # to ~90% as slots fill, purely as an accounting artifact).
    logger.info("\n[Step 4] Computing portfolio equity & risk state...")
    summary      = paper_mgr.get_summary(latest_prices)
    total_equity = summary.get('total_portfolio_value', INITIAL_EQUITY)
    peak_equity  = max(get_peak_equity(EQUITY_CSV, floor=INITIAL_EQUITY), total_equity)
    drawdown_pct = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0

    current_agg_risk   = paper_mgr.get_aggregate_open_risk()
    portfolio_risk_pct = (current_agg_risk / total_equity) if total_equity > 0 else 0.0
    sector_counts       = {s: len(syms) for s, syms in
                            paper_mgr.get_open_positions_by_sector(SECTOR_MAP).items()}

    logger.info(f"  Total equity          : ₹{total_equity:,.2f}  (peak: ₹{peak_equity:,.2f})")
    logger.info(f"  Drawdown from peak     : {drawdown_pct*100:.1f}%  (circuit breaker at {LIVE_MAX_DRAWDOWN*100:.0f}%)")
    logger.info(f"  Aggregate open risk    : ₹{current_agg_risk:,.2f}  ({portfolio_risk_pct*100:.1f}% of equity, cap {MAX_PORTFOLIO_RISK_PCT*100:.0f}%)")
    logger.info(f"  Sector exposure        : {sector_counts or 'none'}  (cap {LIVE_MAX_SECTOR_EXPOSURE}/sector)")

    circuit_breaker_active = drawdown_pct >= LIVE_MAX_DRAWDOWN
    if circuit_breaker_active:
        logger.warning(
            f"  🛑 DRAWDOWN CIRCUIT BREAKER ACTIVE: equity is down {drawdown_pct*100:.1f}% "
            f"from its peak (₹{peak_equity:,.2f} → ₹{total_equity:,.2f}), ≥ the "
            f"{LIVE_MAX_DRAWDOWN*100:.0f}% halt threshold. Skipping new entries this run "
            f"— existing positions still get their normal exit checks."
        )
        alert_notifier.send_alert(
            subject="Drawdown circuit breaker active — new entries paused",
            body=(
                f"Portfolio equity is down {drawdown_pct*100:.1f}% from its peak "
                f"(₹{peak_equity:,.2f} → ₹{total_equity:,.2f}). New trade entries are "
                f"paused until this recovers. Existing positions continue to be "
                f"monitored and will still exit normally on stop-loss/target/time."
            ),
        )

    # ── Step 5: Market regime ─────────────────────────────────────────────────
    logger.info("\n[Step 5] Checking Nifty 50 market regime...")
    regime = bot._get_market_regime(days=300)
    logger.info(f"  Regime: {regime}")

    # ── Step 6: Scan for new signals (risk-budgeted, sector-capped, with
    #            position replacement) ────────────────────────────────────────
    open_trades = paper_mgr.get_open_trades()
    open_count  = len(open_trades)
    slots_free  = MAX_OPEN_TRADES - open_count

    logger.info(f"\n[Step 6] Scanning {len(SCAN_UNIVERSE)} symbols for new signals...")
    logger.info(f"  Open: {open_count}/{MAX_OPEN_TRADES} | Slots free: {slots_free} | Free cash: ₹{paper_mgr.free_cash:,.2f}")

    new_trades    = 0
    replacements  = 0
    skipped_risk  = 0
    skipped_sector = 0
    signals_found = []

    if circuit_breaker_active:
        logger.info("  Skipping scan entirely — drawdown circuit breaker is active")
    elif paper_mgr.free_cash < 500:
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
                    current_equity=total_equity,   # true equity, not free_cash — see header
                    market_regime=regime,
                )

                if sig != 'BUY':
                    continue

                # Defensive: compute risk directly from entry/stop/size rather
                # than trusting a pre-baked 'risk' key in details — keeps this
                # robust even if the signal generator's return schema changes.
                risk_per_share = details['entry_price'] - details['stop_loss']
                details['risk'] = round(details.get('position_size', 0) * risk_per_share, 2)
                details.setdefault('reward', round(
                    details.get('position_size', 0) * (details['target_price'] - details['entry_price']), 2))

                signals_found.append((symbol, details))
                sector = SECTOR_MAP.get(symbol, symbol)

                # ── Portfolio risk budget: shrink or skip to fit what's left ──
                risk_budget_left = MAX_PORTFOLIO_RISK_PCT * total_equity - current_agg_risk
                live_risk_pct = (current_agg_risk / total_equity) if total_equity > 0 else 0.0
                if risk_budget_left <= 0:
                    if skipped_risk < 3:
                        logger.info(f"    {symbol}: skipped — portfolio risk budget exhausted "
                                    f"({live_risk_pct*100:.1f}% ≥ {MAX_PORTFOLIO_RISK_PCT*100:.0f}% cap)")
                    elif skipped_risk == 3:
                        logger.info("    ... further risk-budget skips suppressed (see summary count below)")
                    skipped_risk += 1
                    continue
                if details['risk'] > risk_budget_left:
                    shrunk_size = max(0, int(risk_budget_left / risk_per_share)) if risk_per_share > 0 else 0
                    if shrunk_size < 1:
                        logger.info(f"    {symbol}: skipped — no room left in portfolio risk budget")
                        skipped_risk += 1
                        continue
                    details['position_size'] = shrunk_size
                    details['risk']   = round(shrunk_size * risk_per_share, 2)
                    details['reward'] = round(shrunk_size * (details['target_price'] - details['entry_price']), 2)
                    logger.info(f"    {symbol}: position size reduced to fit remaining risk budget "
                                f"(₹{risk_budget_left:.0f} left)")

                # ── Sector cap: at limit → only allowed via same-sector swap ──
                sector_at_cap = sector_counts.get(sector, 0) >= LIVE_MAX_SECTOR_EXPOSURE

                if slots_free > 0 and not sector_at_cap:
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
                        current_agg_risk += details['risk']
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                else:
                    if sector_at_cap and slots_free > 0:
                        logger.info(f"    {symbol}: sector '{sector}' at cap ({LIVE_MAX_SECTOR_EXPOSURE}) "
                                    f"— can only swap in via same-sector replacement")
                        skipped_sector += 1

                    # No free slots (or sector capped): look for a weak position
                    # to replace. If the sector itself is capped, the swap MUST
                    # come from within that same sector (sector-neutral), so it
                    # never increases concentration beyond the cap.
                    weak = find_replaceable_position(
                        paper_mgr.get_open_trades(), details, latest_prices,
                        sector_filter=sector if sector_at_cap else None,
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
                                weak_risk = (float(weak['entry_price']) - float(weak['stop_loss'])) * int(weak['position_size'])
                                current_agg_risk += details['risk'] - max(0.0, weak_risk)
                                weak_sector = SECTOR_MAP.get(weak['symbol'], weak['symbol'])
                                sector_counts[weak_sector] = max(0, sector_counts.get(weak_sector, 1) - 1)
                                sector_counts[sector] = sector_counts.get(sector, 0) + 1

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
    logger.info(f"  New trades opened: {new_trades}  (replacements: {replacements}, "
                f"skipped on risk budget: {skipped_risk}, skipped on sector cap: {skipped_sector})")

    # ── Step 7: Equity snapshot ───────────────────────────────────────────────
    logger.info("\n[Step 7] Logging equity snapshot...")
    paper_mgr.log_daily_equity(latest_prices)

    # ── Step 8: Portfolio summary ─────────────────────────────────────────────
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

    # Makes compounding tangible: every rupee of realised + unrealised P&L is
    # already inside total_portfolio_value, which is what position sizing is
    # based on (see RISK_PROFILE header note) — so this multiple is a direct
    # readout of how much bigger your NEXT trade's sizing basis has become as
    # a result of past gains, not just a vanity stat.
    initial_eq = summary.get('initial_equity', 0) or INITIAL_EQUITY
    if initial_eq > 0:
        growth_multiple = summary.get('total_portfolio_value', 0) / initial_eq
        logger.info(f"  Growth Multiple       : {growth_multiple:.3f}x initial equity "
                    f"(this is what your next trade's position size scales off)")

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
