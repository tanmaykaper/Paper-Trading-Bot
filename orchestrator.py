# orchestrator.py  ── DAILY RUN COMPOSITION  v1
# ═════════════════════════════════════════════════════════════════════════════
# Wires the six modules built over the last five sessions into one daily cycle
# and drives them through PaperTradingManager's existing public API. Nothing in
# paper_trading_manager.py is rewritten: open_trade / close_position /
# free_cash / current_equity / get_open_trades are called exactly as they
# stand, so persistence, capital accounting and the CSV schema keep working.
#
#   market_state.MarketState          how much risk the book carries today
#   signal_generator.SignalGenerator  which setups qualify
#   entry_execution                   what price they can actually be had at
#   portfolio_allocator               which of them get the scarce slots
#   exit_manager.ExitEngine           when positions end
#   calibration.WinCalibrator         what the probabilities really are
#
# ── The ordering is the design ──────────────────────────────────────────────
# Exits run before entries, because a slot freed this morning is a slot the
# allocator can fill this afternoon, and running them the other way round
# leaves the book a day behind itself. Pending fills run before the scan,
# because yesterday's approved plan has a prior claim on capital over a
# candidate the scanner has not evaluated yet. Calibration runs last, on the
# day's closed trades, so tomorrow starts better informed than today did.
#
# ── Why entries are placed today and filled tomorrow ────────────────────────
# The scan reads the closing bar. The earliest transactable moment is the next
# session. So allocation produces PLANS, not positions: they rest in a pending
# book, consume reserved capital, and become trades on the next run at the
# price the market actually offered. This is the structural fix behind
# entry_execution.py — booking a position at a close that was never available
# to it overstated results by ~₹47/trade and 1.7pp of win rate in testing.
#
# ── State that must survive between runs ────────────────────────────────────
# The defensive clamp, its cooldown, and the pending order book. A cron process
# that wakes up with no memory re-risks every morning regardless of what
# yesterday's tape did, and re-places bids it already decided to cancel. Both
# live in orchestrator_state.json.
# ═════════════════════════════════════════════════════════════════════════════

import json
import logging
import os

import numpy as np
import pandas as pd

from market_state import MarketState, BreadthPanel, print_state
from portfolio_allocator import PortfolioAllocator, print_plan
from exit_manager import ExitEngine
from entry_execution import PendingOrders, route, enrich_signal_bar
from calibration import WinCalibrator, refresh_from_csv
from data_fetcher_free import data_quality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATE_JSON = 'orchestrator_state.json'

# Columns the new stack needs on each trade row. PaperTradingManager.open_trade
# does not accept them, so they are written straight after the row is created
# rather than by changing its signature — keeping the manager untouched means
# the existing backtester, replacement logic and CSV migration keep working
# against it unchanged.
EXTRA_COLUMNS = ['quality_score', 'p_win_est', 'time_exit_bars', 'entry_adx',
                 'highest_high', 'health_signals_at_exit', 'entry_slippage_pct',
                 'market_state_at_entry']


def _assign(df, mask, key, value):
    """
    Write a value into a column that may still be an all-NaN float64 block.

    pandas raises rather than silently upcasting when a string lands in a float
    column, and an exception here aborts the whole daily run — which is exactly
    what happened in the first walk-forward: every bar that produced a fill
    threw on the market_state_at_entry write, so no extra column was ever
    persisted and the sample collapsed. Coercing to object first is the fix,
    and it is applied in one place so both patch paths share it.
    """
    if key not in df.columns:
        df[key] = pd.NA
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
        df[key] = df[key].astype(object)
    df.loc[mask, key] = value


class TradingOrchestrator:

    def __init__(self, manager, signal_gen, sector_map=None, profile='aggressive',
                 trades_csv='paper_trades.csv', state_path=STATE_JSON):
        self.mgr = manager
        self.sig = signal_gen
        self.sector_map = sector_map or {}
        self.trades_csv = trades_csv
        self.state_path = state_path

        state = self._load_state()
        self.market = MarketState.from_dict(state['market']) if state.get('market') else MarketState(profile)
        self.pending = PendingOrders(profile)
        self.pending.book = state.get('pending', {})
        self.exits = ExitEngine(profile)
        self.alloc = PortfolioAllocator(profile, self.sector_map)
        self.calibrator = WinCalibrator()
        self.funnel_extra = {}
        # Hand the calibrated map to the signal generator, so the EV gate and
        # Kelly both read measured probability rather than the asserted tilt.
        self.sig.calibrator = self.calibrator

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, universe_dfs, index_df, vix_df=None, fundamentals=None,
            alpha_scores=None, base_slots=5, max_hold_days=18,
            candidate_enricher=None, earnings_bars=None, entries_allowed=True):
        """
        One complete daily cycle. universe_dfs is the {symbol: OHLCV} dict the
        scanner already builds — it must now include HELD symbols, which the
        current runner skips; the exit engine and the chandelier both need bars
        for open positions, not just a last price.
        """
        fundamentals = fundamentals or {}
        report = {}

        prices = {s: float(df['close'].iloc[-1]) for s, df in universe_dfs.items()
                  if df is not None and len(df)}

        # ── 1. Market state ──────────────────────────────────────────────────
        panel = BreadthPanel(universe_dfs)
        ms = self.market.assess(index_df, vix_df=vix_df, breadth_panel=panel,
                                base_slots=base_slots)
        print_state(ms)
        report['market_state'] = ms

        # ── 2. Exits, before anything competes for the slots they free ───────
        report['exits'] = self._process_exits(universe_dfs, prices, earnings_bars or {})

        # ── 3. Yesterday's approved plans, before today's candidates ─────────
        report['fills'] = self._process_pending(universe_dfs, ms, entries_allowed)

        # ── 4. Scan ──────────────────────────────────────────────────────────
        open_trades = self._open_trades()
        held = set(open_trades['symbol']) if len(open_trades) else set()
        # Two independent authorities can suspend entries and both must agree
        # to allow them: the market state (is this a day worth trading?) and the
        # caller's safety layer (is the system in a fit state to trade?). Kept
        # separate because they fail for unrelated reasons and a single flag
        # would make a data outage indistinguishable from a risk-off tape.
        if ms['new_entries_allowed'] and entries_allowed:
            candidates = self._scan(universe_dfs, index_df, fundamentals, held,
                                    ms, alpha_scores, max_hold_days)
        else:
            candidates = []
            logger.info(f"  Entries suspended — {ms['state']}: {ms['note']}")
        # ── 4b. Candidate enrichment ─────────────────────────────────────────
        # A hook rather than a hard dependency, because the expensive
        # per-symbol work that belongs here — news sentiment above all — is
        # only affordable once the universe has been narrowed to a handful of
        # candidates. Running it across 100+ symbols daily would be minutes of
        # HTTP for data that gets discarded on 98 of them.
        #
        # The enricher receives the candidate list and returns the surviving
        # subset, free to drop entries (a hard veto) and to attach fields the
        # allocator reads (a soft tilt). Keeping it a callback means the
        # orchestrator does not import the sentiment engine, and a failure in
        # an optional layer cannot take the daily run down with it.
        if candidate_enricher is not None and candidates:
            before = len(candidates)
            try:
                candidates = candidate_enricher(candidates) or []
                if before != len(candidates):
                    logger.info(f"  Enricher: {before} → {len(candidates)} candidates")
            except Exception as e:
                logger.warning(f"  Candidate enricher failed ({e}) — proceeding unenriched")

        report['candidates'] = len(candidates)

        # ── 5. Allocate ──────────────────────────────────────────────────────
        incumbents = self._incumbents(open_trades, universe_dfs, prices)
        cash = float(self.mgr.free_cash()) - self.pending.reserved_capital()
        equity = float(self.mgr.current_equity())
        plan = self.alloc.plan(
            candidates, incumbents, equity=equity, cash_available=max(cash, 0.0),
            peak_equity=self._peak_equity(equity), max_slots=ms['max_slots'],
            returns_frame=self._returns_frame(universe_dfs), exposure=ms['exposure'],
        )
        plan['diagnostics']['exposure'] = ms['exposure']
        print_plan(plan)

        for ev in plan['evictions']:
            tid = ev['trade'].get('trade_id')
            if tid:
                self.mgr.close_position(tid, ev['price'], 'Slot Rotation')

        # ── 6. Plans rest until tomorrow's open ──────────────────────────────
        placed = []
        for entry in plan['entries']:
            d = dict(entry['details'])
            d['position_size'] = entry['size']
            d['market_state_at_entry'] = ms['state']
            self.pending.place(entry['symbol'], d, route(d))
            placed.append(entry['symbol'])
        report['placed'] = placed
        report['plan'] = plan

        # ── 7. Learn, then persist ───────────────────────────────────────────
        self.calibrator = refresh_from_csv(self.trades_csv)
        self.sig.calibrator = self.calibrator
        self._save_state()
        return report

    # ═════════════════════════════════════════════════════════════════════════
    def _process_exits(self, universe_dfs, prices, earnings_bars=None):
        out = {'closed': [], 'trailed': []}
        df = self._open_trades()
        if len(df) == 0:
            return out
        updates = {}
        for _, row in df.iterrows():
            sym = row['symbol']
            if sym not in prices:
                continue                     # no price this run: skip, never assume one
            ev = self.exits.evaluate(row, bars=universe_dfs.get(sym),
                                     current_price=prices[sym],
                                     bars_held=int(float(row.get('hold_days') or 0)),
                                     bars_to_earnings=(earnings_bars or {}).get(sym))
            if ev['action'] == 'EXIT':
                self.mgr.close_position(row['trade_id'], ev['exit_price'], ev['exit_reason'])
                updates[row['trade_id']] = {
                    'health_signals_at_exit': (ev.get('health') or {}).get('n_signals')}
                out['closed'].append((sym, ev['exit_reason'], ev['r_multiple']))
            elif ev['action'] == 'TRAIL':
                updates[row['trade_id']] = {'stop_loss': ev['new_stop']}
                out['trailed'].append((sym, ev['new_stop']))
            bars = universe_dfs.get(sym)
            if bars is not None and len(bars):
                hh = max(float(row.get('highest_high') or 0), float(bars['high'].iloc[-1]))
                updates.setdefault(row['trade_id'], {})['highest_high'] = hh
        self._patch_rows(updates)
        return out

    def _process_pending(self, universe_dfs, ms, entries_allowed=True):
        bars = {s: df.iloc[-1] for s, df in universe_dfs.items() if df is not None and len(df)}
        filled, expired = self.pending.process(bars)
        for sym, reason in expired:
            logger.info(f"  ✗ {sym}: {reason}")
        opened = []
        for sym, d, note in filled:
            # A plan approved under one market state can fill into another. The
            # exposure decision is made fresh at the moment capital is actually
            # committed, not at the moment the plan was drafted.
            if not (ms['new_entries_allowed'] and entries_allowed):
                logger.info(f"  ✗ {sym}: filled but entries now suspended ({ms['state']})")
                continue
            # Re-anchoring shrinks the share count to hold rupee risk constant
            # when the fill lands above the signal close. That can drop a plan
            # approved at ₹9,100 under the economic floor — at which point it
            # is a position too small to pay its own depository charge, and the
            # right answer is to let it go rather than book it.
            ok_econ, why_econ, _ = self.sig.validate_economics(d)
            if not ok_econ:
                logger.info(f"  ✗ {sym}: filled but no longer economic — {why_econ}")
                continue
            ok = self.mgr.open_trade(
                sym, d['entry_price'], d['stop_loss'], d['target_price'],
                d['position_size'], d['entry_type'], confidence=d.get('confidence'),
                risk_reward_ratio=d.get('risk_reward_ratio'),
                alpha_score=d.get('alpha_score'), alpha_tier=d.get('alpha_tier'),
            )
            if ok:
                self._patch_latest(sym, {
                    'quality_score': d.get('quality_score'),
                    'p_win_est': d.get('p_win_est'),
                    'time_exit_bars': d.get('time_exit_bars'),
                    'entry_adx': (d.get('indicators') or {}).get('adx'),
                    'highest_high': d['entry_price'],
                    'entry_slippage_pct': d.get('entry_slippage_pct'),
                    'market_state_at_entry': d.get('market_state_at_entry'),
                })
                opened.append((sym, d['entry_price'], note))
                logger.info(f"  ▲ {sym} filled ₹{d['entry_price']:.2f} — {note}")
        return {'opened': opened, 'expired': expired}

    def _scan(self, universe_dfs, index_df, fundamentals, held, ms, alpha_scores, max_hold_days):
        candidates = []
        skip = held | set(self.pending.book)
        for sym, df in universe_dfs.items():
            if sym in skip or df is None:
                continue
            # A stale, halted or circuit-frozen frame computes cleanly and
            # produces a confident signal about a stock that is not trading.
            # Checked here rather than in the fetcher because a held position
            # still needs its bars for the exit engine even when the symbol is
            # no longer enterable.
            quality = data_quality(df)
            if not quality['tradeable']:
                self.funnel_extra[quality['reason'][:60]] = \
                    self.funnel_extra.get(quality['reason'][:60], 0) + 1
                continue
            try:
                sig, d = self.sig.generate_signal(
                    df, sym, fundamentals.get(sym, {}), self.mgr.current_equity(),
                    market_regime=ms['legacy_regime'], benchmark_df=index_df,
                    max_hold_days=max_hold_days)
            except Exception as e:                        # one bad symbol never ends a scan
                logger.warning(f"  {sym}: signal error {e}")
                continue
            if sig != 'BUY':
                continue
            # A hostile tape raises the bar every setup must clear, on top of
            # whatever signal_generator already required of it.
            if ms['quality_add'] > 0 and d['quality_score'] < d['quality_required'] + ms['quality_add']:
                continue
            d = enrich_signal_bar(d, df)
            candidates.append({'symbol': sym, 'details': d,
                               'alpha_score': (alpha_scores or {}).get(sym)})
        if self.funnel_extra:
            logger.info(f"  Skipped on data quality: {self.funnel_extra}")
            self.funnel_extra = {}
        logger.info(f"  Scan: {len(candidates)} candidates | funnel {self.sig.funnel_summary(reset=True)}")
        return candidates

    def _incumbents(self, open_trades, universe_dfs, prices):
        out = []
        for _, row in open_trades.iterrows():
            sym = row['symbol']
            if sym not in prices:
                continue
            ev = self.exits.evaluate(row, bars=universe_dfs.get(sym), current_price=prices[sym],
                                     bars_held=int(float(row.get('hold_days') or 0)))
            out.append({'symbol': sym, 'trade': row, 'evaluation': ev, 'price': prices[sym]})
        return out

    # ═════════════════════════════════════════════════════════════════════════
    def _open_trades(self):
        try:
            df = self.mgr.get_open_trades()
            return df if df is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _patch_rows(self, updates):
        """
        Write the new stack's columns onto existing rows. Done through the CSV
        rather than by extending open_trade's signature, so paper_trading_manager
        stays byte-identical and its own callers are unaffected.
        """
        if not updates or not os.path.exists(self.trades_csv):
            return
        df = pd.read_csv(self.trades_csv)
        for col in EXTRA_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        for tid, fields in updates.items():
            m = df['trade_id'] == tid
            for k, v in fields.items():
                if v is not None:
                    _assign(df, m, k, v)
        df.to_csv(self.trades_csv, index=False)

    def _patch_latest(self, symbol, fields):
        """Apply fields to the most recently created OPEN rows for a symbol —
        plural, because a tranched entry writes one row per tranche."""
        if not os.path.exists(self.trades_csv):
            return
        df = pd.read_csv(self.trades_csv)
        for col in EXTRA_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        m = (df['symbol'] == symbol) & (df['status'] == 'OPEN')
        if not m.any():
            return
        if 'trade_group_id' in df.columns and pd.notna(df.loc[m, 'trade_group_id']).any():
            latest_group = df.loc[m, 'trade_group_id'].dropna().iloc[-1]
            m = m & (df['trade_group_id'] == latest_group)
        for k, v in fields.items():
            if v is not None:
                _assign(df, m, k, v)
        df.to_csv(self.trades_csv, index=False)

    def _returns_frame(self, universe_dfs):
        cols = {}
        for s, df in universe_dfs.items():
            if df is not None and len(df) >= 60:
                cols[s] = pd.Series(df['close'].astype(float).pct_change().tail(60).to_numpy())
        return pd.DataFrame(cols) if cols else None

    def _peak_equity(self, equity, path='daily_equity.csv'):
        try:
            eq = pd.read_csv(path)
            col = 'equity' if 'equity' in eq.columns else eq.columns[-1]
            return max(float(eq[col].astype(float).max()), equity)
        except (OSError, ValueError, KeyError):
            return equity

    # ═════════════════════════════════════════════════════════════════════════
    def _load_state(self):
        if not os.path.exists(self.state_path):
            return {}
        try:
            return json.load(open(self.state_path))
        except (ValueError, OSError):
            return {}

    def _save_state(self):
        payload = {'market': self.market.to_dict(),
                   'pending': {s: {'plan': o['plan'],
                                   'details': _jsonable(o['details']),
                                   'bars_waited': o['bars_waited']}
                               for s, o in self.pending.book.items()}}
        json.dump(payload, open(self.state_path, 'w'), indent=1, default=str)


def _jsonable(d):
    """Timestamps and numpy scalars round-trip badly through JSON; coerce the
    few fields that carry them rather than losing the whole pending book to a
    serialisation error on one key."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (pd.Timestamp,)):
            out[k] = str(v)
        elif hasattr(v, 'item'):
            try:
                out[k] = v.item()
            except (ValueError, AttributeError):
                out[k] = str(v)
        else:
            out[k] = v
    return out
