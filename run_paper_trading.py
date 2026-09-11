# run_paper_trading.py  ── GITHUB ACTIONS / SINGLE-RUN VERSION  v11
# ─────────────────────────────────────────────────────────────────────────────
# v11 change — tranche logic moved to tranche_manager.py. Purely a relocation
# (byte-identical TRANCHE_CONFIG/build_tranches, imported not redefined) so
# swing_trading_bot.py's backtester can finally simulate the same scaled-exit
# policy live trading actually runs, which it structurally could not do
# before (would have required a circular import — see tranche_manager.py's
# header for the full story). No behaviour change for live trading itself.
#
# v10 change — scaled exits (partial profit-taking). Every BUY signal large
# enough to split (position_size >= MIN_SIZE_FOR_TRANCHING) now opens as
# THREE tranches instead of one position with one target:
#   quick  (35%) — exits at 1.5R (signal_generator's own min_rr)
#   core   (35%) — exits at the original planned target (3-4R)
#   runner (30%) — no fixed target; rides the trailing stop (v9) all the
#                  way, capturing whatever a fixed target would have capped
#
# Why: a single fixed target caps every winner at the same price regardless
# of how far it runs — exactly the trades a trend/momentum system should
# want to let ride further. Splitting the exit locks in a reliable partial
# gain early while leaving a piece of the position genuinely open-ended.
#
# Architecturally, one trading decision now becomes multiple CSV rows
# sharing one trade_group_id, each tracked and exited independently through
# the existing update_trades()/trailing-stop machinery — no new exit logic
# needed, tranches are just ordinary trades with different targets. What
# DID need to change: slot counting (get_open_symbol_count(), not row
# count — a 3-tranche position is still one symbol's worth of exposure),
# sector counting (same fix, get_open_positions_by_sector() now dedupes by
# symbol), and position-replacement logic (now operates on whole groups via
# get_open_position_groups()/close_position_group(), so replacing a
# tranched position closes all its tranches together, not just one).
#
# Caught and fixed while building this: a `trade_group_id or trade_id`
# fallback for legacy (pre-this-feature) positions looked safe but wasn't —
# NaN is truthy in Python, so a legacy row's genuinely-empty trade_group_id
# cell (read back from CSV as NaN, not an empty string) would have made the
# fallback silently keep the NaN instead of using trade_id. Fixed with an
# explicit pd.notna() check; the exact same class of bug already found and
# fixed elsewhere in this project (paper_trading_manager.py's dtype-crash
# fix from the very first pass) — same lesson, different column.
#
# v9 change — trailing stops, ported from the backtest engine to live
# trading for the first time (trailing_stop.py, new shared module). Every
# open position's stop-loss now ratchets up as price moves favorably —
# breakeven at 1R profit, entry+1R locked in at 2R, entry+2R locked in at
# 3R — instead of sitting at its original fixed level for the whole trade.
# Applied in Step 2, before exit checks, so a position that closes today
# closes against its current (possibly just-ratcheted) stop, not a stale one.
#
# This existed in swing_trading_bot.py's backtest already
# (_apply_trailing_stop) but had never been connected to live trading — the
# same "built for backtest, missing in live" gap already found and fixed
# for sector caps and the drawdown circuit breaker earlier in this project.
# It also had a real bug, caught while porting it rather than carried
# forward: it recomputed "risk" from entry_price minus the CURRENT
# stop-loss on every call, which is only correct before the first ratchet —
# after that, entry_price minus stop_loss no longer equals the original 1R
# distance the tiers are defined in terms of. Traced through a concrete
# case: a position correctly ratcheted to its 2R tier, then — despite
# reaching the genuine 3R price level on a later day — incorrectly stayed
# stuck at the 2R protection level instead of progressing further, because
# risk was being measured from the wrong, already-moved reference point.
# Fixed by tracking initial_stop_loss (new column, set once at entry, never
# modified) as the permanent 1R reference, in trailing_stop.py — the ONE
# implementation both live trading and the backtest engine now import,
# rather than two copies that could drift apart the way this one already had.
#
# v8 change — alpha_engine.py (built and independently tested against
# synthetic data in a prior step — see ALPHA_ENGINE_DESIGN.md) is now wired
# in as a conviction-scoring layer ON TOP OF signal_generator.py, not in
# place of it:
#
#   WHAT DIDN'T CHANGE: signal_generator.py still owns entry-pattern
#   detection and the exact entry/stop-loss/target price levels for every
#   trade — that logic, and its own simple BULL/NEUTRAL/BEAR regime input,
#   are untouched. alpha_engine has no opinion on price levels; it only
#   scores conviction in a signal signal_generator has already produced.
#
#   WHAT'S NEW: every technical BUY signal now also gets a 0-100
#   cross-sectional conviction score — ranked against the rest of today's
#   scan universe, adjusted for a richer 5-state market regime (not just
#   BULL/BEAR), and weighted by this bot's own realized track record per
#   entry pattern (persisted across runs in pattern_weights.csv). That
#   score now:
#     1. GATES entries — a technical signal below MIN_ALPHA_SCORE_TO_TRADE
#        (Tier 3/"Marginal") doesn't get taken, even if signal_generator
#        liked it.
#     2. SIZES entries — TIER_SIZE_MULTIPLIER scales position size by
#        conviction tier, layered BEFORE the existing portfolio risk-budget
#        check (Step 2's hard cap), so higher conviction means a bigger bet
#        within the same risk limits, not outside them.
#     3. RANKS entries — replaces the old confidence×risk_reward_ratio
#        heuristic in position-replacement decisions (Step 1) with the
#        richer alpha score, recorded per-trade so existing positions can
#        be fairly re-evaluated later, not just new candidates.
#
#   RELIABILITY: the whole alpha-scoring step is wrapped in try/except — if
#   it fails for any reason (e.g. no Nifty data this run), trading falls
#   back to exactly the pre-integration behaviour (signal_generator's
#   BUY/HOLD alone, no gating) rather than stopping entirely. The universe
#   data fetch that both signal_generator AND the alpha engine need is
#   deliberately OUTSIDE that try/except, so a failure in the new layer
#   can never silently stop the old one from working — see Step 5.
#
# Carried over from v7/v6/v5 (still true — see CHANGES_step1/2.md,
# ALPHA_ENGINE_DESIGN.md): resilient bulk price fetching, dtype-crash fix,
# health checks + alerting, stale-position safety net, position replacement,
# aggregate risk budget, sector cap, drawdown circuit breaker, raised risk
# tolerance, high-growth/momentum universe, visible compounding.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
import os
import pandas as pd
from datetime import datetime

from swing_trading_bot import SwingTradingBot, SECTOR_MAP, MAX_SECTOR_EXPOSURE, MAX_DRAWDOWN_DEFAULT
from paper_trading_manager import PaperTradingManager
from notification_handler import NotificationHandler
from alpha_engine import CompositeAlphaScore
from orchestrator import TradingOrchestrator
from market_state import MarketState
from signal_generator import apply_earnings_constraint
from signal_generator import SignalGenerator, RISK_PROFILE
from sentiment_engine import SentimentEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
INITIAL_EQUITY  = 50_000
# Raised from 5 -> 10 (confirmed explicitly) alongside the 10k -> 50k capital
# increase, specifically to observe the bot's decision-making over a higher
# volume of concurrent trades — capital alone wouldn't have done this; it
# would've just made the same 5 slots bigger. Average capital per slot goes
# from ~₹2,000 to ~₹5,000 (still a real, risk-managed position, not noise).
# LIVE_MAX_SECTOR_EXPOSURE (4/sector, below) doesn't need to move with this:
# filling all 10 slots now requires spreading across at least 3 sectors
# (4+4+2) instead of effectively 2 (4+1 out of 5) — slightly MORE forced
# diversification than before, not less, so no separate tuning needed there.
MAX_OPEN_TRADES = 10
# Raised from 15. The barrier geometry in signal_generator v4 bounds achievable
# R:R at reach_fraction*sqrt(H)/k_stop_max; at H=15 that ceiling is 1.64 against
# a 1.50 floor, which squeezes structural stops toward the volatility floor and
# leaves almost no room for a 2:1 trade. At 18 the ceiling is 1.80, at 20 it is
# 1.89. Realised time-exits in paper_trades.csv averaged 23.8 days, so the old
# clock was also truncating trades that were still working. exit_manager now
# enforces a PER-TRADE horizon (time_exit_bars) inside this outer cap.
MAX_HOLD_DAYS   = 18

# Alpha is a ranking input, not an admission gate. Its own tier breakdown shows
# no separation on the sample so far (Tier 2 mean -0.086R vs Tier 3 -0.072R,
# Tier 3 winning more often, n=41), so portfolio_allocator applies it as a
# bounded +/-7.5% tie-break and the economics decide participation. Set False to
# drop it entirely; the allocator ranks on forward return per slot-day either way.
USE_ALPHA_ENGINE = True

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

# Slot-aware capital cap — see the detailed comment at its use site in Step
# 6c for the full story (a real concentration bug this fixes: 3 symbols
# consuming 98% of equity while 7 of 10 slots sat empty). A trade can be
# sized up to FAIR_SHARE_FLEX times an equal split of current free cash
# across remaining open slots — 2.0x allows genuine conviction-based size-up
# without letting early signals starve every slot that comes after them.
FAIR_SHARE_FLEX = 2.0
LIVE_MAX_SECTOR_EXPOSURE = 4      # overrides swing_trading_bot.MAX_SECTOR_EXPOSURE (3) for live trading
LIVE_MAX_DRAWDOWN        = 0.35   # overrides swing_trading_bot.MAX_DRAWDOWN_DEFAULT (0.30) for live trading

# Replacement gate — a new signal must clear ALL of these to bump an
# existing open position out of its slot:
REPLACE_SCORE_MULTIPLE = 1.40   # new composite score must beat the weakest by 40%+
PROTECT_PROFIT_PCT     = 0.03   # never replace a position up >3% unrealised
PROTECT_PROGRESS_PCT   = 0.80   # never replace a position >80% of the way to target

# ── Alpha engine integration ────────────────────────────────────────────────
# alpha_engine.py (built and independently tested against synthetic data
# with known-correct answers — see ALPHA_ENGINE_DESIGN.md) sits ON TOP of
# signal_generator.py, not in place of it. signal_generator.py still owns
# entry-pattern detection and the exact entry/stop/target price levels —
# that logic is unchanged. What alpha_engine adds: a cross-sectional
# conviction score (0-100) for every BUY candidate, ranked against the
# CURRENT scan universe and adjusted for the current market regime and this
# bot's own realized track record per entry pattern. That score now decides
# (a) whether a technically-valid signal is actually worth taking, (b) how
# large a bet it gets within the existing risk-budget system, and (c) which
# position wins when two signals compete for a limited slot — replacing the
# old crude confidence×risk_reward_ratio heuristic used for all three.
PATTERN_WEIGHTS_CSV = 'pattern_weights.csv'

# MIN_ALPHA_SCORE_TO_TRADE and TIER_SIZE_MULTIPLIER now live as class
# constants on alpha_engine.CompositeAlphaScore (moved there so the backtest
# engine references the exact same values instead of a second, separately
# maintained copy — see alpha_engine.py for the full reasoning). Read here
# via the already-instantiated alpha_scorer further down, not redefined.

# ── Sentiment engine integration (sentiment_engine.py) ──────────────────────
# Deliberately fetched LAZILY — only for symbols that already produced a
# technical BUY signal in Step 6a below, NOT for the whole ~100-symbol scan
# universe up front. Unlike the price-based alpha factors (computed for free
# from OHLCV data already bulk-fetched in Step 5), sentiment needs a fresh
# network call PER symbol (news fetch), and this project has already been
# bitten once by GitHub Actions IP rate-limiting under too much fetch volume
# (see data_fetcher_free.py's own history) — no reason to invite a repeat
# for symbols that were never going to trade today anyway. In practice this
# means sentiment only ever fetches for a handful of symbols per run, not
# the whole universe.
#
# Two independent effects, deliberately kept separate (see sentiment_engine.
# SentimentEngine's own docstring for the full reasoning):
#   • SOFT — cross-sectional sentiment percentile merged into factor_ranks
#     as alpha_engine's optional 6th 'sentiment' factor (10% weight).
#   • HARD — an absolute-threshold veto that can reject a trade outright on
#     fresh, corroborated bad news (fraud/litigation cluster, or a pile-up
#     of strongly negative headlines), independent of how the other factors
#     score. This can't be a percentile effect — "worst sentiment among
#     today's 3 candidates" doesn't mean "actually bad news" — so it uses
#     its own fixed thresholds instead.
#
# NOT wired into swing_trading_bot.py's backtester: backtesting sentiment's
# actual contribution would need a free, dated historical news archive that
# doesn't exist — fetching TODAY's news while replaying a HISTORICAL price
# bar would be lookahead bias, not a real backtest. This can only be
# evaluated prospectively, from here forward, the same way Tanmay already
# plans to validate the rest of this system empirically — not retrofitted
# onto historical price replay.

# ── Scaled exits (partial profit-taking) ────────────────────────────────────
# Logic itself now lives in tranche_manager.py (shared with the backtester —
# see that module's header for why this moved and what it fixes). Imported
# here, not redefined, so live trading and backtesting can never quietly
# diverge on this again.
from tranche_manager import (
    ENABLE_SCALED_EXITS, MIN_SIZE_FOR_TRANCHING, TRANCHE_CONFIG, build_tranches,
)


LARGECAP_UNIVERSE = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'ASIANPAINT',
    'MARUTI', 'TATASTEEL', 'BAJFINANCE', 'KOTAKBANK', 'LT',
    'AXISBANK', 'TITAN', 'WIPRO', 'ULTRACEMCO', 'NESTLEIND',
    'HCLTECH', 'TECHM', 'SUNPHARMA', 'DRREDDY', 'CIPLA',
    'TMPV', 'BAJAJ-AUTO', 'HINDALCO', 'JSWSTEEL',
    'ONGC', 'BPCL', 'GAIL', 'SIEMENS', 'ABB', 'DLF',
    'INDUSINDBK', 'FEDERALBNK', 'MPHASIS', 'LTM', 'CHOLAFIN',
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
#     Zomato's parent renamed to Eternal Ltd in 2025; the NSE/BSE ticker
#     changed from ZOMATO to ETERNAL effective 9 April 2025 (confirmed via
#     search while auditing a live run's logs — an earlier version of this
#     comment incorrectly assumed the old ticker still worked; it hadn't for
#     well over a year, and every fetch for it was silently erroring out).
#   • Defence — genuinely in a live momentum phase as of mid-2026: multiple
#     consecutive rally sessions in June/July on record defence production
#     figures and large DAC procurement approvals (₹52,000cr+ tranches).
#   • Renewable energy / EV — an active, high-beta theme through 2026 (solar
#     manufacturing capacity buildout, wind order momentum), though names
#     here swing both ways day to day, consistent with genuinely higher risk.
#
# Same caveat as MIDCAP_UNIVERSE: I don't have live yfinance/NSE access from
# this sandbox to individually confirm every ticker still resolves — a few
# were spot-checked via search (ETERNAL, WAAREEENER, ACMESOLAR), the rest are
# good-faith based on current sourcing. Unresolvable tickers are skipped
# automatically (existing safe behaviour) — prune anything that never hits.
HIGH_GROWTH_MOMENTUM_UNIVERSE = [
    # New-age tech / internet
    'ETERNAL', 'NYKAA', 'PAYTM', 'POLICYBZR', 'DELHIVERY', 'IRCTC',
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


# Neutral placeholder for a position with no alpha_score on record, used
# ONLY when alpha scoring is otherwise active this run (i.e. the new
# candidate being compared against DOES have a real 0-100 alpha_score).
# Deliberately sits right at the Tier 2/Tier 3 boundary ("assume roughly
# average until shown otherwise") rather than trying to convert
# confidence×risk_reward_ratio onto a 0-100 scale — that conversion has no
# principled basis (the two metrics aren't measuring the same thing), and
# this only matters for a short, self-resolving transition window: every
# position open when the alpha engine was integrated fully cycles out
# within MAX_HOLD_DAYS regardless, after which every trade has a real score.
NEUTRAL_ALPHA_PLACEHOLDER = 55.0


def _composite_score(details, alpha_active=True):
    """
    Score a candidate BUY signal for replacement comparisons.

    alpha_active: whether alpha_engine scoring succeeded THIS RUN (see the
    try/except around it in run_eod). This must be threaded through rather
    than inferred per-candidate, so that every comparison in a given run
    uses ONE consistent scale — mixing a 0-100 alpha_score against a raw
    confidence×risk_reward_ratio value (typically ~1.5-45) would make new
    candidates look systematically stronger than old ones purely from
    scale, not genuine quality.
    """
    if alpha_active and details.get('alpha_score') is not None:
        return float(details['alpha_score'])
    return float(details.get('confidence', 1)) * float(details.get('risk_reward_ratio', 1.0))


def _existing_position_score(trade, alpha_active=True):
    """
    Same idea for an already-open position, read back from the CSV.

    If alpha scoring is active this run: prefer the trade's own recorded
    alpha_score; a legacy trade with none gets NEUTRAL_ALPHA_PLACEHOLDER
    (comparable 0-100 scale) rather than a confidence×risk_reward_ratio
    number that isn't on the same scale as what it's being compared to.

    If alpha scoring is NOT active this run (engine failed, see run_eod):
    every candidate falls back to confidence×risk_reward_ratio uniformly,
    including this one — consistent scale maintained either way.
    """
    alpha = trade.get('alpha_score')
    if alpha_active:
        if pd.notna(alpha) and alpha != '':
            return float(alpha)
        return NEUTRAL_ALPHA_PLACEHOLDER
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


def find_replaceable_position(open_trades, new_details, latest_prices, sector_filter=None, alpha_active=True):
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

    alpha_active: passed straight through to the scoring functions so every
    comparison in this call uses one consistent scale (see _composite_score).
    """
    new_score = _composite_score(new_details, alpha_active=alpha_active)
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

        candidates.append((t, _existing_position_score(t, alpha_active=alpha_active)))

    if not candidates:
        return None

    weakest_trade, weakest_score = min(candidates, key=lambda x: x[1])
    if new_score >= weakest_score * REPLACE_SCORE_MULTIPLE:
        return weakest_trade
    return None


def build_candidate_enricher(logger_, earnings_bars_for):
    """
    Returns a callable the orchestrator applies to its candidate list, or None
    when the sentiment engine is unavailable.

    ── Why this runs on candidates only ────────────────────────────────────────
    Every other factor in this system is computed from OHLCV already in memory.
    Sentiment is the one layer that needs a fresh HTTP round trip per symbol,
    so scoring the full 100+ scan universe would be minutes of network for data
    discarded on all but a handful of names. By the time the orchestrator calls
    this, signal_generator has already reduced the universe to the few setups
    that qualified on price — which is precisely the set worth asking the news
    about.

    ── Two effects, deliberately separate ──────────────────────────────────────
    HARD VETO (absolute thresholds, SentimentEngine.check_veto): a pending
    regulatory action, a pileup of strongly negative headlines, or a decisively
    negative tone on good confidence removes the candidate outright. This
    cannot be a percentile effect — "least bad news in today's universe" is not
    the same statement as "no bad news", and on a day when every name is clean
    the bottom-ranked one is still clean.

    SOFT TILT (cross-sectional percentile): ranked against the other candidates
    and handed to portfolio_allocator as sentiment_percentile, where it moves
    the ordering by at most ±6%. Ranking rather than thresholding self-calibrates
    to the day's news environment, which on a broad risk-off session skews
    negative for everything.

    A symbol with no usable headlines is left untouched — no veto, no tilt. That
    is the honest reading of an absent signal, and it is why the percentile is
    attached only when the ranker actually produced one.
    """
    try:
        engine = SentimentEngine()
    except Exception as e:
        logger_.warning(f"  Sentiment engine unavailable ({e}) — candidates proceed unscored")
        engine = None

    def enrich(candidates):
        symbols = [c['symbol'] for c in candidates]
        if not symbols:
            return candidates

        # ── Earnings first: it is a hard, dated constraint and it is cheap ────
        # Checked before sentiment so a candidate already disqualified by an
        # imminent results date never costs a news fetch. Same reason sentiment
        # runs on candidates rather than the universe — the expensive lookups
        # belong behind the cheap filters.
        survivors = []
        for c in candidates:
            bars = earnings_bars_for(c['symbol'])
            details, ok, why = apply_earnings_constraint(c['details'], bars)
            if not ok:
                logger_.info(f"    ✗ {c['symbol']}: {why}")
                continue
            c['details'] = details
            if details.get('earnings_note'):
                logger_.info(f"    ~ {c['symbol']}: {details['earnings_note']}")
            survivors.append(c)
        candidates = survivors

        if engine is None or not candidates:
            return candidates
        symbols = [c['symbol'] for c in candidates]
        logger_.info(f"  📰 Sentiment: scoring {len(symbols)} candidates...")
        readings = engine.score_universe(symbols)
        ranks = engine.rank_universe(readings)

        survivors = []
        for c in candidates:
            reading = readings.get(c['symbol'])
            if reading is None:
                survivors.append(c)
                continue
            blocked, reason = engine.check_veto(reading)
            if blocked:
                logger_.info(f"    ✗ {c['symbol']}: {reason}")
                continue
            pct = ranks.get(c['symbol'])
            if pct is not None:
                c['sentiment_percentile'] = pct
                c['sentiment_tier'] = engine.tier(pct)
            survivors.append(c)

        scored = sum(1 for c in survivors if c.get('sentiment_percentile') is not None)
        logger_.info(f"    {len(survivors)}/{len(candidates)} cleared the veto, "
                     f"{scored} carry a percentile")
        return survivors

    return enrich


def run_eod():
    """
    v11 — DAILY RUN, DELEGATED TO orchestrator.TradingOrchestrator
    ─────────────────────────────────────────────────────────────────────────
    v10's pipeline made every decision inline: it scanned, scored, sized,
    sector-capped, replaced positions, applied trailing stops and closed
    trades across ~500 lines in one function. Each of those responsibilities
    now lives in a module that can be tested on its own, and this function's
    job is reduced to the two things a runner should actually do — assemble
    the data, and report what happened.
    
    What moved, and where:
    
      market_state.MarketState      today's exposure, slot count and the
                                    defensive clamp. Replaces the static
                                    BULL/NEUTRAL/BEAR call, which changed
                                    state roughly four times a year and could
                                    not see a five-day risk-off event — the
                                    condition that produced three separate
                                    weeks at a 0% win rate.
      signal_generator v4           volatility- and structure-adaptive stops,
                                    horizon-feasible targets, the cost floor.
      entry_execution               plans rest overnight and fill against the
                                    NEXT session, at the price the market
                                    actually offered. v10 booked entries at a
                                    close nobody could transact at.
      portfolio_allocator           slot competition on forward return per
                                    slot-day, fractional Kelly sizing, and a
                                    switching hurdle in rupees. Replaces
                                    find_replaceable_position, REPLACE_SCORE_
                                    MULTIPLE, PROTECT_PROFIT_PCT,
                                    PROTECT_PROGRESS_PCT and FAIR_SHARE_FLEX.
      exit_manager.ExitEngine       chandelier trail, momentum-decay exit,
                                    per-trade horizon, stagnation flag, and
                                    realistic fills. Replaces the separate
                                    apply_trailing_stops / update_trades pair,
                                    whose split meant the trail could ratchet
                                    a stop the time exit then overrode with no
                                    coordination between them.
      calibration                   learns P(win | quality) from closed trades
                                    and feeds it back to the EV gate and Kelly.
    
    The one data change this requires: universe_dfs must now include HELD
    symbols. v10 skipped them (`if symbol in held_symbols: continue`), leaving
    open positions with a last price and nothing else — the decay check and the
    chandelier both need bars.
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"📅 NSE PAPER TRADING BOT v11 — {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    logger.info("=" * 70)

    paper_mgr = PaperTradingManager(
        initial_equity=INITIAL_EQUITY,
        csv_path=TRADES_CSV,
        equity_csv_path=EQUITY_CSV,
        max_open_trades=MAX_OPEN_TRADES,
    )
    bot = SwingTradingBot(send_emails=False, initial_equity=INITIAL_EQUITY,
                          max_open_trades=MAX_OPEN_TRADES, max_hold_days=MAX_HOLD_DAYS)

    # ── Step 1: index, volatility and the scan universe ──────────────────────
    # Held symbols are unioned into the fetch list rather than skipped. This is
    # the single wiring change the new exit logic depends on.
    held_symbols = set()
    try:
        open_df = paper_mgr.get_open_trades()
        if open_df is not None and len(open_df):
            held_symbols = set(open_df['symbol'].astype(str))
    except Exception:
        pass

    fetch_list = list(dict.fromkeys(list(SCAN_UNIVERSE) + sorted(held_symbols)))
    logger.info(f"\n📡 Fetching {len(fetch_list)} symbols "
                f"({len(held_symbols)} held, {len(SCAN_UNIVERSE)} scanned)...")

    nifty_df = bot.fetcher.get_historical_data('^NSEI', days=400, min_bars=200)
    vix_df = bot.fetcher.get_historical_data('^INDIAVIX', days=180, min_bars=30)
    if vix_df is None:
        logger.info("  India VIX unavailable this run — volatility scores from index realised vol alone")

    universe_dfs = {}
    for symbol in fetch_list:
        try:
            df = bot.fetcher.get_historical_data(symbol, days=260, min_bars=80)
            if df is not None:
                universe_dfs[symbol] = df
        except Exception as e:
            logger.warning(f"  {symbol}: fetch failed ({e})")

    missing_held = held_symbols - set(universe_dfs)
    if missing_held:
        # Named explicitly rather than counted: a held position with no bars is
        # a position the exit engine cannot evaluate today, which is worth
        # seeing in the log rather than inferring from a silent gap.
        logger.warning(f"  ⚠ no data for held positions: {sorted(missing_held)} — "
                       f"their exits are deferred to the next run")

    if len(universe_dfs) < 20:
        logger.error(f"✗ Only {len(universe_dfs)} symbols resolved — aborting rather than "
                     f"trading on a universe too thin to measure breadth against")
        return

    # ── Step 2: fundamentals ─────────────────────────────────────────────────
    # data_fetcher_free v2 parses screener.in correctly and caches for 7 days,
    # so this is now real per-company data rather than the identical default
    # profile every symbol used to receive. Each dict is stamped with its
    # sector so FundamentalScreener v4 can compare P/E against the sector
    # median it computes below, instead of against the placeholder 25.
    fundamentals = {}
    for symbol in universe_dfs:
        try:
            f = bot.get_fundamentals_safe(symbol) or {}
        except Exception:
            f = {}
        f['sector'] = SECTOR_MAP.get(symbol, 'UNKNOWN')
        fundamentals[symbol] = f

    try:
        bot.signal_gen.fund.calibrate_sector_pe(fundamentals, SECTOR_MAP)
        measured = sum(1 for f in fundamentals.values() if f.get('fundamentals_measured'))
        logger.info(f"  Fundamentals measured for {measured}/{len(fundamentals)} symbols "
                    f"(the rest score neutral rather than inheriting healthy defaults)")
    except AttributeError:
        logger.info("  Screener predates v4 — sector P/E calibration skipped")

    # ── Step 3: alpha scores, as a ranking tilt rather than a gate ───────────
    # The alpha composite is passed to the allocator, which applies it as a
    # bounded tie-break (ALPHA_TILT, +/-7.5%) instead of the hard
    # MIN_ALPHA_SCORE_TO_TRADE cut v10 used. That cut was spending trades on a
    # score whose own tier breakdown showed no separation: Tier 2 mean -0.086R
    # against Tier 3 -0.072R, with Tier 3 winning more often (n=41). Until a
    # larger sample says otherwise, it informs ordering and does not decide
    # participation.
    alpha_scores = {}
    if USE_ALPHA_ENGINE:
        try:
            alpha_scorer = CompositeAlphaScore()
            regime_result = alpha_scorer.regime_detector.classify(nifty_df)
            factor_values = {s: alpha_scorer.factor_engine.compute_all(d)
                             for s, d in universe_dfs.items()}
            factor_ranks = alpha_scorer.ranker.rank_universe(factor_values, sector_map=SECTOR_MAP)
            for s in universe_dfs:
                r = alpha_scorer.score_symbol(s, factor_ranks.get(s, {}), regime_result)
                if r.get('composite_score') is not None:
                    alpha_scores[s] = r['composite_score']
            logger.info(f"  Alpha scored {len(alpha_scores)} symbols "
                        f"(regime: {regime_result.get('regime')})")
        except Exception as e:
            logger.warning(f"  Alpha engine unavailable this run ({e}) — allocator ranks on economics alone")

    # ── Step 3b: earnings calendar ───────────────────────────────────────────
    # Fetched lazily and only where it can change a decision: for the symbols
    # already held (so a position can be closed on our terms ahead of results)
    # and, through the enricher, for the few candidates that survived the price
    # filters. A calendar lookup for every scanned symbol would be 100+ extra
    # HTTP calls a day for data that matters to perhaps three of them.
    _earnings_cache = {}

    def bars_to_earnings(symbol):
        """Trading sessions until the next results date, or None when unknown.
        None means no constraint — an absent calendar must never become a
        blackout that empties the universe."""
        if symbol in _earnings_cache:
            return _earnings_cache[symbol]
        sessions = None
        try:
            date = bot.fetcher.get_next_earnings_date(symbol)
            if date is not None:
                sessions = max(int(np.busday_count(datetime.now().date(), date)), 0)
        except Exception:
            sessions = None
        _earnings_cache[symbol] = sessions
        return sessions

    held_earnings = {s: bars_to_earnings(s) for s in sorted(held_symbols)}
    imminent = {s: b for s, b in held_earnings.items() if b is not None and b <= 3}
    if imminent:
        logger.info(f"  📆 Results imminent for held positions: {imminent} — "
                    f"these are closed ahead of the announcement")

    # ── Step 4: one orchestrated cycle ───────────────────────────────────────
    orchestrator = TradingOrchestrator(
        paper_mgr, bot.signal_gen, SECTOR_MAP, profile='aggressive',
        trades_csv=TRADES_CSV,
    )
    report = orchestrator.run(
        universe_dfs, nifty_df, vix_df=vix_df, fundamentals=fundamentals,
        alpha_scores=alpha_scores, base_slots=MAX_OPEN_TRADES,
        max_hold_days=MAX_HOLD_DAYS,
        candidate_enricher=build_candidate_enricher(logger, bars_to_earnings),
        earnings_bars=held_earnings,
    )

    # ── Step 5: equity log and report ────────────────────────────────────────
    latest_prices = {s: float(d['close'].iloc[-1]) for s, d in universe_dfs.items()}
    try:
        paper_mgr.log_daily_equity(latest_prices)
    except Exception as e:
        logger.warning(f"  equity log failed ({e})")

    ms = report['market_state']
    logger.info("\n" + "=" * 70)
    logger.info("  DAILY SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Market state          : {ms['state']}  (exposure {ms['exposure']:.2f}x, "
                f"{ms['max_slots']} slots, risk score {ms['risk_score']:.2f})")
    if ms['triggers']:
        for t in ms['triggers']:
            logger.info(f"    ⚠ {t}")
    logger.info(f"  Candidates            : {report['candidates']}")
    logger.info(f"  Plans placed (fill T+1): {report['placed'] or 'none'}")
    logger.info(f"  Filled today          : "
                f"{[f'{s} @ ₹{p:.2f}' for s, p, _ in report['fills']['opened']] or 'none'}")
    for sym, why in report['fills']['expired']:
        logger.info(f"    ✗ {sym}: {why}")
    logger.info(f"  Closed today          : "
                f"{[f'{s} ({r}, {rm:+.2f}R)' for s, r, rm in report['exits']['closed']] or 'none'}")
    logger.info(f"  Stops trailed         : {len(report['exits']['trailed'])}")

    summary = paper_mgr.get_summary(latest_prices)
    logger.info("\n  ── CAPITAL ──────────────────────────────────────────────────")
    logger.info(f"  Equity                : ₹{summary.get('current_equity', 0):>12,.2f}")
    logger.info(f"  Free Cash             : ₹{summary.get('free_cash',      0):>12,.2f}")
    logger.info(f"  Total P&L             : ₹{summary.get('total_pnl',      0):>+12,.2f}")
    if summary.get('closed_trades', 0) > 0:
        logger.info(f"  Win Rate              : {summary.get('win_rate', 0):.1f}%"
                    f"  ({summary.get('wins', 0)}W / {summary.get('losses', 0)}L)")
        logger.info(f"  Avg Win / Avg Loss    : ₹{summary.get('avg_win', 0):+,.2f}"
                    f" / ₹{summary.get('avg_loss', 0):+,.2f}")

    logger.info("\n  ── OPEN POSITIONS ───────────────────────────────────────────")
    paper_mgr.print_open_positions(latest_prices)
    logger.info("=" * 70 + "\n")
    return report


if __name__ == "__main__":
    run_eod()
