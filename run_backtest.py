# run_backtest.py  ── WALK-FORWARD VALIDATION OF THE DEPLOYED STACK  v6
# ═════════════════════════════════════════════════════════════════════════════
# v5 drove swing_trading_bot.backtest_portfolio() and compared three policies
# (alpha on/off, tranched/untranched). That harness is now measuring a strategy
# that no longer exists. Since it was written, live trading gained adaptive
# barrier geometry, a chandelier trail, momentum-decay exits, per-trade
# horizons, slot competition on forward return per slot-day, fractional Kelly
# sizing, a regime exposure controller, next-bar fills, a cost floor, an
# earnings blackout and T+1 settlement. backtest_portfolio models none of them.
#
# This is the exact failure tranche_manager.py's header already documents for
# the tranching case — "the ONE tool meant to validate live's actual behaviour
# was quietly testing a different, older strategy instead" — and it has
# happened again, at a much larger scale. Running v5 today would produce
# confident numbers about a system nobody is trading, which is worse than
# having no backtest: it invites abandoning a change that works, or keeping one
# that does not.
#
# v6 drives the REAL orchestrator (backtest_engine.WalkForwardBacktest), bar by
# bar, against an in-memory book. Whatever comes out is what the deployed stack
# would have done, because it is the deployed stack.
#
#   RUN D   full stack, exactly as run_paper_trading.py runs it
#   RUN C'  identical SIGNALS, pre-upgrade POLICY: filled at the signal close,
#           flat risk-fraction sizing, first-come-first-served slots, fixed
#           stop and target, hard time exit, no trail, no regime gate
#
# Holding the signal source constant across both is deliberate: running the old
# signal generator in C' as well would confound entry changes with policy
# changes and make the delta uninterpretable. D minus C' isolates what the
# execution, exit and allocation work actually bought.
#
# ── Runtime, honestly ───────────────────────────────────────────────────────
# The orchestrator re-scans the universe on every simulated bar, and each scan
# builds a full indicator frame per symbol. That is the price of testing the
# real thing rather than an approximation of it, and it is not cheap:
# roughly UNIVERSE x BARS signal evaluations, twice.
#
#   30 symbols x 120 bars   ~10 min    — iterate here
#   60 symbols x 250 bars   ~1.5 hrs   — a real read
#   100+ symbols x 350 bars ~5 hrs+    — run it overnight, once
#
# Start small. A clean lookahead check and a sane D-vs-C' delta on 30 symbols
# tells you the plumbing is sound; only then spend the overnight run.
# ═════════════════════════════════════════════════════════════════════════════

import logging
import os

import numpy as np
import pandas as pd

from technical_indicators import TechnicalIndicators
from fundamental_screener import FundamentalScreener
from signal_generator import SignalGenerator
from data_fetcher_free import DataFetcherFree
from profit_engine import ProfitEngine
from backtest_engine import (WalkForwardBacktest, summarise, print_comparison,
                             print_sensitivity)
from backtest_analytics import compute_performance_report, print_performance_report
from run_paper_trading import (SCAN_UNIVERSE, SECTOR_MAP, INITIAL_EQUITY,
                               MAX_OPEN_TRADES, MAX_HOLD_DAYS)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Knobs ────────────────────────────────────────────────────────────────────
UNIVERSE_LIMIT = 30          # raise once a small run looks sane; None = full universe

# get_historical_data(days=N) starts N+60 CALENDAR days back, so a request for
# 420 returns only ~330 trading bars. The first full-universe run asked for 420,
# received 328, and with a 300-bar warmup simulated 28 sessions — a window far
# too short to conclude anything from. Trading days are ~69% of calendar days,
# so the request is now sized in calendar terms and the warmup is set from what
# the indicators actually need: momentum_rank is the hungriest at ~157 bars.
HISTORY_FETCH_DAYS = 700     # ~480 trading bars
WARMUP_BARS        = 200     # leaves ~280 test sessions
PROFILE        = 'growth'    # must match ACTIVE_PROFILE in the six modules
RUN_SENSITIVITY = False      # True multiplies runtime by ~15; an overnight job
CACHE_PATH     = 'backtest_universe.pkl'   # so a re-run does not re-fetch


def load_universe(symbols, bars=HISTORY_FETCH_DAYS, use_cache=True):
    """
    Fetch once, cache to disk. Re-fetching 30-100 symbols on every iteration is
    the slowest part of a short run and the most likely to be rate-limited —
    and the data does not change between two runs on the same evening.
    """
    if use_cache and os.path.exists(CACHE_PATH):
        try:
            cached = pd.read_pickle(CACHE_PATH)
            if set(symbols).issubset(cached['universe'].keys()):
                logger.info(f"✓ Universe loaded from {CACHE_PATH} "
                            f"({len(cached['universe'])} symbols)")
                return cached['universe'], cached['index'], cached.get('vix')
        except Exception as e:
            logger.warning(f"  cache unreadable ({e}) — refetching")

    fetcher = DataFetcherFree()
    logger.info(f"📥 Fetching {len(symbols)} symbols x {bars} bars — this is the slow part")
    index_df = fetcher.get_historical_data('^NSEI', days=bars + 120, min_bars=200)
    vix_df = fetcher.get_historical_data('^INDIAVIX', days=bars + 60, min_bars=30)

    universe = {}
    for i, symbol in enumerate(symbols, 1):
        df = fetcher.get_historical_data(symbol, days=bars, min_bars=WARMUP_BARS + 30)
        if df is not None:
            universe[symbol] = df
        if i % 10 == 0:
            logger.info(f"  {i}/{len(symbols)} fetched, {len(universe)} usable")

    try:
        pd.to_pickle({'universe': universe, 'index': index_df, 'vix': vix_df}, CACHE_PATH)
    except Exception as e:
        logger.warning(f"  could not cache universe ({e})")
    return universe, index_df, vix_df



REQUIRED_BUILD = '2026-09-14.4'


def preflight():
    """
    Refuse to run against stale or inconsistent modules.

    Two full-universe backtests in a row reported zero trades from a pipeline
    that was working. Neither output said why — the first because a wall-clock
    staleness check silently rejected every sliced frame, the second because
    the fixed files had not reached the machine at all. A backtest that quietly
    validates the wrong code is worse than one that refuses to start, because
    its numbers look like findings.

    Every check below corresponds to a defect that actually produced a void
    run, and names the file to replace rather than reporting that something is
    wrong.
    """
    import inspect
    problems = []

    # 1. Staleness must be measured against the SIMULATED date. Without this,
    #    every frame in a walk-forward reads as weeks old and the entire
    #    universe is rejected before generate_signal is ever called.
    try:
        from data_fetcher_free import data_quality
        if 'as_of' not in inspect.signature(data_quality).parameters:
            problems.append("data_quality() has no 'as_of' parameter — staleness will use "
                            "the wall clock and reject every symbol. Replace data_fetcher_free.py.")
        else:
            idx = pd.bdate_range(end=pd.Timestamp.today() - pd.Timedelta(days=60), periods=40)
            hist = pd.DataFrame({'datetime': idx, 'open': 100.0, 'high': 101.0, 'low': 99.0,
                                 'close': np.linspace(100, 110, len(idx)), 'volume': 1e5})
            if not data_quality(hist, as_of=idx[-1])['tradeable']:
                problems.append("data_quality() rejects a frame fresh relative to its own "
                                "simulated date. Replace data_fetcher_free.py.")
    except ImportError as e:
        problems.append(f"cannot import data_quality: {e}")

    # 2. The orchestrator must derive and pass that date.
    try:
        import orchestrator
        if 'as_of' not in inspect.getsource(orchestrator.TradingOrchestrator._scan):
            problems.append("orchestrator._scan() does not pass as_of to data_quality(). "
                            "Replace orchestrator.py.")
    except Exception as e:
        problems.append(f"cannot inspect orchestrator: {e}")

    # 3. Breadth must survive symbols whose histories end on different dates,
    #    or the regime engine loses its only leading sensor.
    try:
        import market_state
        if 'n_symbols' not in inspect.getsource(market_state.BreadthPanel.at):
            problems.append("BreadthPanel.at() lacks the quorum walk-back — breadth will "
                            "read n/a on uneven histories. Replace market_state.py.")
    except Exception as e:
        problems.append(f"cannot inspect market_state: {e}")

    # 4. One profile name, understood by every module that receives it. A
    #    module missing it raised KeyError mid-run once already.
    try:
        import market_state as ms, profit_engine as pe, exit_manager as em
        import portfolio_allocator as pa, entry_execution as ee, signal_generator as sg
        registries = [('market_state', ms.STATE_PROFILES), ('profit_engine', pe.PROFIT_PROFILES),
                      ('exit_manager', em.EXIT_PROFILES),
                      ('portfolio_allocator', pa.ALLOCATOR_PROFILES),
                      ('entry_execution', ee.EXECUTION_PROFILES)]
        for label, registry in registries:
            if PROFILE not in registry:
                problems.append(f"{label} has no '{PROFILE}' profile. Replace {label}.py.")
        if sg.ACTIVE_CALIBRATION != PROFILE:
            problems.append(f"signal_generator is calibrated '{sg.ACTIVE_CALIBRATION}' but this "
                            f"backtest runs '{PROFILE}' — the entry geometry and the risk "
                            f"settings would come from different configurations.")
    except Exception as e:
        problems.append(f"cannot verify profile coherence: {e}")

    # 5. Build stamps.
    for mod_name in ('orchestrator', 'market_state', 'data_fetcher_free', 'backtest_engine'):
        try:
            mod = __import__(mod_name)
            if getattr(mod, 'BUILD', None) != REQUIRED_BUILD:
                problems.append(f"{mod_name}.py BUILD is {getattr(mod, 'BUILD', 'absent')}, "
                                f"expected {REQUIRED_BUILD}. Replace {mod_name}.py.")
        except Exception as e:
            problems.append(f"cannot import {mod_name}: {e}")

    # 6. Every name used in this file must resolve. A NameError 260 lines in —
    #    after the universe has been fetched — is the cheapest possible bug to
    #    catch and the most expensive one to hit, because the failure arrives
    #    twenty minutes into a run that has already done all its slow work.
    try:
        import ast as _ast, builtins as _bi
        tree = _ast.parse(open(__file__).read())
        defined = {n.id for x in _ast.walk(tree) if isinstance(x, _ast.Assign)
                   for n in _ast.walk(x.targets[0]) if isinstance(n, _ast.Name)}
        defined |= {x.name for x in _ast.walk(tree)
                    if isinstance(x, (_ast.FunctionDef, _ast.ClassDef))}
        defined |= {a.asname or a.name.split('.')[0] for x in _ast.walk(tree)
                    if isinstance(x, (_ast.Import, _ast.ImportFrom)) for a in x.names}
        defined |= {t.id for x in _ast.walk(tree) if isinstance(x, (_ast.For, _ast.comprehension))
                    for t in _ast.walk(x.target) if isinstance(t, _ast.Name)}
        defined |= {a.arg for x in _ast.walk(tree)
                    if isinstance(x, _ast.FunctionDef) for a in x.args.args}
        defined |= {h.name for h in _ast.walk(tree)
                    if isinstance(h, _ast.ExceptHandler) and h.name}
        defined |= {a.arg for x in _ast.walk(tree)
                    if isinstance(x, _ast.Lambda) for a in x.args.args}
        # Module dunders are always bound at runtime; f-string format specs
        # ({value:s}) parse as Name nodes and are not identifiers at all.
        defined |= {'__file__', '__name__', '__doc__', '__package__', 's', 'd', 'f', 'g'}
        used = {n.id for n in _ast.walk(tree)
                if isinstance(n, _ast.Name) and isinstance(n.ctx, _ast.Load)}
        unresolved = sorted(used - defined - set(dir(_bi)))
        if unresolved:
            problems.append(f"names used but never imported or defined in run_backtest.py: "
                            f"{unresolved} — add the missing import(s).")
    except Exception:
        pass

    if problems:
        print("\n" + "!" * 78)
        print("  PREFLIGHT FAILED — not running. Fix these first:")
        print("!" * 78)
        for p in problems:
            print(f"  x {p}")
        print("!" * 78 + "\n")
        raise SystemExit(1)
    print(f"Preflight: OK (build {REQUIRED_BUILD}, profile '{PROFILE}')")


if __name__ == "__main__":
    preflight()
    symbols = SCAN_UNIVERSE[:UNIVERSE_LIMIT] if UNIVERSE_LIMIT else SCAN_UNIVERSE
    universe, index_df, vix_df = load_universe(symbols)

    if index_df is None or len(universe) < 10:
        logger.error(f"✗ Only {len(universe)} symbols and "
                     f"{'no' if index_df is None else 'an'} index — too thin to measure breadth "
                     f"against, let alone draw a conclusion from. Aborting.")
        raise SystemExit(1)

    if len(universe) < 20:
        logger.warning(f"  ⚠ {len(universe)} symbols is below BreadthPanel's 20-symbol minimum: "
                       f"breadth is withheld, the regime engine runs on index and volatility "
                       f"alone, and its conclusions are correspondingly weaker. Raise "
                       f"UNIVERSE_LIMIT before reading the comparison as a verdict.")

    fundamentals = {}
    fetcher = DataFetcherFree()
    for symbol in universe:
        try:
            f = fetcher.get_fundamentals(symbol) or {}
        except Exception:
            f = {}
        f['sector'] = SECTOR_MAP.get(symbol, 'UNKNOWN')
        fundamentals[symbol] = f

    def make_signal_gen():
        screener = FundamentalScreener()
        # Sector P/E is calibrated once from the same fundamentals the run will
        # see, so pe_check compares against the peer group rather than the
        # placeholder 25 — matching what run_paper_trading does each morning.
        try:
            screener.calibrate_sector_pe(fundamentals, SECTOR_MAP)
        except AttributeError:
            pass
        return SignalGenerator(TechnicalIndicators(), screener)

    wf = WalkForwardBacktest(make_signal_gen, SECTOR_MAP, INITIAL_EQUITY, PROFILE)

    print("\n" + "=" * 78)
    print("NSE SWING TRADING BOT — WALK-FORWARD VALIDATION  v6")
    n_bars = min(len(index_df), min((len(d) for d in universe.values()), default=0))
    slots = ProfitEngine.ladder(INITIAL_EQUITY)['recommended_slots']
    print(f"Universe: {len(universe)} symbols | {n_bars} bars, simulating "
          f"{max(n_bars - WARMUP_BARS, 0)} sessions after a {WARMUP_BARS}-bar warmup")
    print(f"Equity ₹{INITIAL_EQUITY:,} | slots {slots} (from the growth ladder) | "
          f"hold {MAX_HOLD_DAYS} | profile {PROFILE}")
    if n_bars - WARMUP_BARS < 100:
        print(f"  ⚠ only {n_bars - WARMUP_BARS} test sessions — too short to conclude from. "
              f"Raise HISTORY_FETCH_DAYS or lower WARMUP_BARS.")
    print("=" * 78)

    # ── Lookahead first ──────────────────────────────────────────────────────
    # Before any number is worth reading. A harness that cannot prove this is
    # measuring its own bugs, and every conclusion below inherits them.
    check = wf.assert_no_lookahead(universe, index_df, fundamentals,
                                   bar=WARMUP_BARS, sample=10)
    print(f"\nLookahead check: {'CLEAN' if check['clean'] else 'FAILED'} "
          f"({check['checked']} symbols)")
    if not check['clean']:
        for sym, a, b in check['mismatches']:
            print(f"   ✗ {sym}: sliced {a} vs truncated {b}")
        print("   Decisions differ when future bars are physically removed — something "
              "downstream reads past its slice. Fix that before trusting anything below.")
        raise SystemExit(1)

    # ── Run D ────────────────────────────────────────────────────────────────
    print("\n\n########## RUN D — FULL STACK (current live policy) ##########")
    trades_d, equity_d = wf.run_stack(universe, index_df, vix_df=vix_df,
                                      fundamentals=fundamentals, start=WARMUP_BARS,
                                      base_slots=slots,
                                      max_hold_days=MAX_HOLD_DAYS, tag='D')

    # ── Run C' ───────────────────────────────────────────────────────────────
    print("\n\n########## RUN C' — SAME SIGNALS, PRE-UPGRADE POLICY ##########")
    trades_c, equity_c = wf.run_legacy(universe, index_df, fundamentals=fundamentals,
                                       start=WARMUP_BARS, max_slots=MAX_OPEN_TRADES,
                                       max_hold_days=15, tag='C')

    # ── Reports ──────────────────────────────────────────────────────────────
    print_performance_report(
        compute_performance_report(trades_d, equity_d, INITIAL_EQUITY),
        title="RUN D — FULL STACK")
    print_performance_report(
        compute_performance_report(trades_c, equity_c, INITIAL_EQUITY),
        title="RUN C' — PRE-UPGRADE POLICY, SAME SIGNALS")
    print_comparison(summarise(trades_d, equity_d, INITIAL_EQUITY, "RUN D full stack", slots),
                     summarise(trades_c, equity_c, INITIAL_EQUITY, "RUN C' legacy policy",
                               MAX_OPEN_TRADES))

    util = summarise(trades_d, equity_d, INITIAL_EQUITY, 'd', slots).get('utilisation_pct')
    if util is not None:
        print(f"  Slot utilisation: {util:.0f}% of available capital-time deployed.")
        if util < 45:
            print(f"  -> Below ~45% the constraint is CANDIDATE FLOW, not edge: expectancy")
            print(f"     per trade cannot lift the return if the book is mostly in cash.")
            print(f"     Widen the universe before loosening any filter.")
        elif util > 80:
            print(f"  -> Above ~80% the book is saturated; more candidates will not help.")
            print(f"     Further gains have to come from expectancy or from more slots.")

    # ── The breakdowns that say WHY, not just how much ───────────────────────
    if trades_d is not None and len(trades_d):
        print("  Exit reasons (D):", trades_d['exit_reason'].value_counts().to_dict())
        if 'market_state_at_entry' in trades_d:
            by_state = trades_d.groupby('market_state_at_entry', dropna=False)['net_pnl'].agg(
                ['count', 'sum', 'mean']).round(2)
            print("\n  P&L by market state at entry — does the regime gate earn its keep?")
            print(by_state.to_string())
        if 'quality_score' in trades_d and trades_d['quality_score'].notna().any():
            q = trades_d.dropna(subset=['quality_score']).copy()
            q['band'] = pd.qcut(q['quality_score'].astype(float), 3,
                                labels=['low', 'mid', 'high'], duplicates='drop')
            print("\n  P&L by entry-quality band — is quality_score predictive at all?")
            print(q.groupby('band', observed=True)['net_pnl'].agg(
                ['count', 'mean', lambda s: (s > 0).mean()]).round(3).to_string())

    for tag, df in (('d', trades_d), ('c', trades_c)):
        if df is not None and len(df):
            df.to_csv(f'backtest_results_run_{tag}.csv', index=False)
    if equity_d is not None and len(equity_d):
        equity_d.to_csv('backtest_equity_curve_run_d.csv', index=False)

    # ── Sensitivity ──────────────────────────────────────────────────────────
    # Off by default: it multiplies runtime by the number of values swept. Turn
    # it on for the one overnight run that decides a configuration, because a
    # single backtest number cannot distinguish an edge from a setting.
    if RUN_SENSITIVITY:
        sens = wf.sensitivity(universe, index_df, vix_df=vix_df, fundamentals=fundamentals,
                              start=WARMUP_BARS, base_slots=MAX_OPEN_TRADES,
                              max_hold_days=MAX_HOLD_DAYS)
        sens.to_csv('backtest_sensitivity.csv', index=False)
        print_sensitivity(sens)

    print("\nRead D vs C' first — that is the whole question this file exists to answer:")
    print("did the execution, exit and allocation work pay for itself on real history?")
    print("Then the quality-band table: if expectancy does not rise with quality_score,")
    print("the EV gate and Kelly sizing are both being driven by a number with no signal,")
    print("and calibration.py has nothing to learn from. That is the single most")
    print("important diagnostic in this output.")
