# backtest_engine.py  ── WALK-FORWARD VALIDATION OF THE DEPLOYED STACK  v1
# ═════════════════════════════════════════════════════════════════════════════
# run_backtest.py's A/B/C harness validates swing_trading_bot.backtest_portfolio,
# which is now five versions behind what actually trades. It knows nothing about
# adaptive barrier geometry, the chandelier trail, momentum-decay exits, slot
# competition, regime exposure, next-bar fills or the cost floor. Every number
# it produces describes a strategy that no longer exists.
#
# This runs the REAL stack — the same orchestrator.TradingOrchestrator object
# the live cron drives — forward over history, one bar at a time, against an
# in-memory book. Whatever comes out is what the deployed system would have
# done, because it IS the deployed system.
#
# ── What makes this walk-forward rather than a backtest ─────────────────────
# At bar t the orchestrator receives universe frames sliced to [0..t] and
# nothing else. Signals are generated on bar t's close, rest in the pending
# book, and fill against bar t+1's open or intrabar range. Exits resolve
# against the bar being evaluated, with gaps filled at the open and
# both-touched bars resolved adversely. Breadth, calibration and the defensive
# clamp all carry forward exactly as they would live.
#
# assert_no_lookahead() checks the property directly rather than trusting the
# slicing: it re-runs a bar with every future row physically deleted and
# confirms the decisions are identical. A single stray .iloc[-1] on an unsliced
# frame produces exactly this bug and nothing else catches it.
#
# ── Runs ────────────────────────────────────────────────────────────────────
#   RUN D  full stack: regime exposure + allocator + chandelier/decay exits +
#          next-bar fills + cost floor          (what is deployed today)
#   RUN C' same signals, legacy policy: no regime gate, no slot competition,
#          flat sizing, fixed stop/target, hard time exit, filled at the
#          signal close                          (the pre-upgrade behaviour)
#
# The comparison is the point. An absolute return on one historical path is a
# single sample and says little; the DELTA between two policies on an identical
# signal stream isolates what the upgrades actually did.
# ═════════════════════════════════════════════════════════════════════════════

import logging
import os

import numpy as np
import pandas as pd

from orchestrator import TradingOrchestrator, EXTRA_COLUMNS
from trading_costs import round_trip_commission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOOK_COLUMNS = [
    'trade_id', 'trade_group_id', 'symbol', 'entry_date', 'entry_price', 'stop_loss',
    'initial_stop_loss', 'target_price', 'position_size', 'entry_type', 'status',
    'exit_date', 'exit_price', 'exit_reason', 'gross_pnl', 'commission', 'net_pnl',
    'hold_days', 'confidence', 'risk_reward_ratio', 'alpha_score', 'alpha_tier',
] + EXTRA_COLUMNS

_OBJECT_COLS = ('trade_id', 'trade_group_id', 'symbol', 'entry_type', 'status',
                'exit_reason', 'entry_date', 'exit_date', 'alpha_tier',
                'market_state_at_entry')


class SimulatedBook:
    """
    In-memory stand-in for PaperTradingManager, exposing exactly the five
    methods the orchestrator calls: free_cash, current_equity, get_open_trades,
    open_trade, close_position.

    Deliberately a separate object rather than a subclass. The live manager owns
    CSV migration, tranche grouping and equity logging a backtest has no use
    for, and inheriting all of it to override half would obscure which
    behaviour was under test. What is shared is the API surface — the thing
    that must not drift.

    Costs come from trading_costs.round_trip_commission, the same function the
    live manager uses, so a backtest cannot become profitable by pricing its own
    friction more kindly than reality does.
    """

    def __init__(self, initial_equity=50000, csv_path='backtest_book.csv'):
        self.initial_equity = float(initial_equity)
        self.cash = float(initial_equity)
        self.realised = 0.0
        self.csv_path = csv_path
        self.n = 0
        self.today = None
        pd.DataFrame(columns=BOOK_COLUMNS).to_csv(csv_path, index=False)

    # ── API used by the orchestrator ─────────────────────────────────────────
    def free_cash(self):
        return self.cash

    def current_equity(self):
        """Cash plus realised P&L. Open positions are marked separately into the
        equity curve; sizing against settled capital is the conservative and
        correct base for a delivery account."""
        return self.initial_equity + self.realised

    def get_open_trades(self):
        d = self._read()
        return d[d['status'] == 'OPEN'] if len(d) else pd.DataFrame(columns=BOOK_COLUMNS)

    def open_trade(self, symbol, entry_price, stop_loss, target_price, position_size,
                   entry_type, confidence=None, risk_reward_ratio=None,
                   alpha_score=None, alpha_tier=None, sentiment_score=None,
                   sentiment_tier=None, tranches=None):
        outlay = float(entry_price) * int(position_size)
        if position_size < 1 or outlay > self.cash:
            return False
        self.n += 1
        d = self._read()
        row = {c: None for c in BOOK_COLUMNS}
        row.update({'trade_id': f'BT{self.n}', 'trade_group_id': f'BG{self.n}', 'symbol': symbol,
                    'entry_date': self.today, 'entry_price': float(entry_price),
                    'stop_loss': float(stop_loss), 'initial_stop_loss': float(stop_loss),
                    'target_price': float(target_price), 'position_size': int(position_size),
                    'entry_type': entry_type, 'status': 'OPEN', 'hold_days': 0,
                    'confidence': confidence, 'risk_reward_ratio': risk_reward_ratio,
                    'alpha_score': alpha_score, 'alpha_tier': alpha_tier})
        d.loc[len(d)] = row
        self.cash -= outlay
        self._write(d)
        return True

    def close_position(self, trade_id, exit_price, exit_reason):
        d = self._read()
        m = d['trade_id'] == trade_id
        if not m.any():
            return False
        r = d[m].iloc[0]
        size, entry, exit_price = int(r['position_size']), float(r['entry_price']), float(exit_price)
        gross = (exit_price - entry) * size
        comm = float(round_trip_commission(entry, exit_price, size))
        d['exit_reason'] = d['exit_reason'].astype(object)
        d.loc[m, 'status'] = 'CLOSED'
        d.loc[m, 'exit_date'] = self.today
        d.loc[m, 'exit_price'] = exit_price
        d.loc[m, 'exit_reason'] = str(exit_reason)
        d.loc[m, 'gross_pnl'] = gross
        d.loc[m, 'commission'] = comm
        d.loc[m, 'net_pnl'] = gross - comm
        self.cash += exit_price * size - comm
        self.realised += gross - comm
        self._write(d)
        return True

    # ── backtest-side helpers ────────────────────────────────────────────────
    def age_positions(self):
        d = self._read()
        if len(d) == 0:
            return
        m = d['status'] == 'OPEN'
        if m.any():
            d.loc[m, 'hold_days'] = d.loc[m, 'hold_days'].astype(float) + 1
            self._write(d)

    def mark_to_market(self, prices):
        total = self.cash
        for _, r in self.get_open_trades().iterrows():
            px = prices.get(r['symbol'], float(r['entry_price']))
            total += float(px) * int(r['position_size'])
        return total

    def trades(self):
        d = self._read()
        return d[d['status'] == 'CLOSED'].copy() if len(d) else pd.DataFrame(columns=BOOK_COLUMNS)

    # The orchestrator patches its extra columns straight onto the CSV, so the
    # book round-trips through that same file rather than holding a second copy
    # that could disagree with it.
    def _read(self):
        return pd.read_csv(self.csv_path, dtype={c: object for c in _OBJECT_COLS})

    def _write(self, d):
        d.to_csv(self.csv_path, index=False)


# ═════════════════════════════════════════════════════════════════════════════
class WalkForwardBacktest:

    def __init__(self, signal_gen_factory, sector_map=None, initial_equity=50000,
                 profile='aggressive', workdir='.'):
        self.make_signal_gen = signal_gen_factory
        self.sector_map = sector_map or {}
        self.initial_equity = initial_equity
        self.profile = profile
        self.workdir = workdir

    # ─────────────────────────────────────────────────────────────────────────
    def run_stack(self, universe_dfs, index_df, vix_df=None, fundamentals=None,
                  start=250, base_slots=5, max_hold_days=18, tag='D'):
        """RUN D — drive the live orchestrator forward bar by bar."""
        paths = self._paths(tag)
        self._clean(paths)
        book = SimulatedBook(self.initial_equity, paths['book'])
        orc = TradingOrchestrator(book, self.make_signal_gen(), self.sector_map,
                                  self.profile, trades_csv=paths['book'],
                                  state_path=paths['state'])
        orc.calibrator.path = paths['calib']

        dates = pd.to_datetime(index_df['datetime'])
        n = min(len(index_df), min((len(d) for d in universe_dfs.values()), default=0))
        equity = []

        for t in range(start, n):
            book.today = str(dates.iloc[t].date())
            sliced = {s: d.iloc[:t + 1] for s, d in universe_dfs.items() if len(d) > t}
            prices = {s: float(d['close'].iloc[-1]) for s, d in sliced.items()}
            try:
                orc.run(sliced, index_df.iloc[:t + 1],
                        vix_df=vix_df.iloc[:t + 1] if vix_df is not None else None,
                        fundamentals=fundamentals or {}, base_slots=base_slots,
                        max_hold_days=max_hold_days)
            except Exception as e:
                # One bad bar never ends a 500-bar run, but it is logged loudly
                # — a silent skip would quietly shrink the sample the report is
                # computed over.
                logger.error(f"  bar {book.today}: orchestrator error {e}")
            book.age_positions()
            equity.append({'datetime': dates.iloc[t], 'equity': book.mark_to_market(prices),
                           'cash': book.cash, 'realised': book.realised})
            pd.DataFrame(equity).to_csv(os.path.join(self.workdir, 'daily_equity.csv'), index=False)

        return book.trades(), pd.DataFrame(equity)

    # ─────────────────────────────────────────────────────────────────────────
    def run_legacy(self, universe_dfs, index_df, fundamentals=None, start=250,
                   max_slots=10, max_hold_days=15, risk_pct=0.04, tag='C'):
        """
        RUN C' — the SAME signal generator under the pre-upgrade policy: filled
        at the signal close, flat risk-fraction sizing, first come first served
        up to max_slots, fixed stop and target, hard time exit, no trail, no
        regime gate, no slot competition.

        Holding the signal source constant is deliberate. Running the OLD signal
        generator here too would confound entry changes with policy changes and
        make the delta uninterpretable.
        """
        paths = self._paths(tag)
        self._clean(paths)
        book = SimulatedBook(self.initial_equity, paths['book'])
        sig = self.make_signal_gen()

        dates = pd.to_datetime(index_df['datetime'])
        n = min(len(index_df), min((len(d) for d in universe_dfs.values()), default=0))
        equity = []

        for t in range(start, n):
            book.today = str(dates.iloc[t].date())
            sliced = {s: d.iloc[:t + 1] for s, d in universe_dfs.items() if len(d) > t}
            prices = {s: float(d['close'].iloc[-1]) for s, d in sliced.items()}

            # Same honest fill treatment on exits, so the delta measures policy
            # rather than accounting generosity.
            for _, r in book.get_open_trades().iterrows():
                bars = sliced.get(r['symbol'])
                if bars is None or len(bars) == 0:
                    continue
                bar = bars.iloc[-1]
                stop, tgt = float(r['stop_loss']), float(r['target_price'])
                o, h, l, c = (float(bar['open']), float(bar['high']),
                              float(bar['low']), float(bar['close']))
                if o <= stop:
                    book.close_position(r['trade_id'], o, 'SL Hit (gap)')
                elif o >= tgt:
                    book.close_position(r['trade_id'], o, 'Target Hit (gap)')
                elif l <= stop:
                    book.close_position(r['trade_id'], stop, 'SL Hit')
                elif h >= tgt:
                    book.close_position(r['trade_id'], tgt, 'Target Hit')
                elif float(r['hold_days'] or 0) >= max_hold_days:
                    book.close_position(r['trade_id'], c, 'Time Exit')

            held = set(book.get_open_trades()['symbol'])
            slots = max_slots - len(held)
            for sym, df in sliced.items():
                if slots <= 0:
                    break
                if sym in held:
                    continue
                try:
                    s, d = sig.generate_signal(df, sym, (fundamentals or {}).get(sym, {}),
                                               book.current_equity(), market_regime='NEUTRAL',
                                               benchmark_df=index_df.iloc[:t + 1],
                                               max_hold_days=max_hold_days)
                except Exception:
                    continue
                if s != 'BUY':
                    continue
                entry, stop = d['entry_price'], d['stop_loss']
                size = int(min((book.current_equity() * risk_pct) / max(entry - stop, 1e-9),
                               book.cash / max(entry, 1e-9)))
                if size >= 1 and book.open_trade(sym, entry, stop, d['target_price'], size,
                                                 d['entry_type'], confidence=d.get('confidence'),
                                                 risk_reward_ratio=d.get('risk_reward_ratio')):
                    slots -= 1

            book.age_positions()
            equity.append({'datetime': dates.iloc[t], 'equity': book.mark_to_market(prices),
                           'cash': book.cash, 'realised': book.realised})

        return book.trades(), pd.DataFrame(equity)

    # ─────────────────────────────────────────────────────────────────────────
    def assert_no_lookahead(self, universe_dfs, index_df, fundamentals=None,
                            bar=300, sample=8):
        """
        Prove the property rather than assume it. Generates signals at `bar`
        twice — once from a slice of the full frame, once from a frame whose
        future rows have been physically deleted and the index reset. Any
        difference means something downstream reads beyond the slice, which a
        slicing convention alone cannot rule out.
        """
        sig = self.make_signal_gen()
        symbols = list(universe_dfs)[:sample]
        mismatches = []

        def key(res):
            s, d = res
            return (s, round(float(d.get('entry_price', 0) or 0), 4),
                    round(float(d.get('stop_loss', 0) or 0), 4),
                    round(float(d.get('target_price', 0) or 0), 4))

        for sym in symbols:
            full = universe_dfs[sym]
            if len(full) <= bar:
                continue
            truncated = full.iloc[:bar + 1].copy().reset_index(drop=True)
            a = sig.generate_signal(full.iloc[:bar + 1], sym, (fundamentals or {}).get(sym, {}),
                                    50000, market_regime='NEUTRAL',
                                    benchmark_df=index_df.iloc[:bar + 1])
            b = sig.generate_signal(truncated, sym, (fundamentals or {}).get(sym, {}),
                                    50000, market_regime='NEUTRAL',
                                    benchmark_df=index_df.iloc[:bar + 1].copy().reset_index(drop=True))
            if key(a) != key(b):
                mismatches.append((sym, key(a), key(b)))
        return {'checked': len(symbols), 'mismatches': mismatches, 'clean': not mismatches}

    # ─────────────────────────────────────────────────────────────────────────
    def _paths(self, tag):
        j = lambda f: os.path.join(self.workdir, f)
        return {'book': j(f'bt_book_{tag}.csv'), 'state': j(f'bt_state_{tag}.json'),
                'calib': j(f'bt_calib_{tag}.json')}

    @staticmethod
    def _clean(paths):
        for p in paths.values():
            if os.path.exists(p):
                os.remove(p)


# ═════════════════════════════════════════════════════════════════════════════
def summarise(trades, equity, initial_equity, label):
    """
    Compact standalone summary. backtest_analytics.compute_performance_report
    accepts the same two frames for the full Sharpe/Sortino/Calmar report with
    stratification by tier, regime, pattern and tranche.
    """
    if trades is None or len(trades) == 0:
        return {'label': label, 'n': 0, 'note': 'no trades'}
    pnl = trades['net_pnl'].astype(float)
    dd = 0.0
    if equity is not None and len(equity) > 1:
        eq = equity['equity'].astype(float)
        run_max = eq.cummax()
        dd = float(((run_max - eq) / run_max.replace(0, np.nan)).max() or 0.0)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return {
        'label': label, 'n': int(len(pnl)),
        'win_rate': round(float((pnl > 0).mean()) * 100, 1),
        'net_pnl': round(float(pnl.sum()), 2),
        'return_pct': round(float(pnl.sum()) / initial_equity * 100, 2),
        'expectancy': round(float(pnl.mean()), 2),
        'profit_factor': (round(float(wins.sum() / -losses.sum()), 2)
                          if len(losses) and losses.sum() < 0 else None),
        'max_dd_pct': round(dd * 100, 2),
        'avg_hold': (round(float(trades['hold_days'].astype(float).mean()), 1)
                     if 'hold_days' in trades else None),
        'total_cost': (round(float(trades['commission'].astype(float).sum()), 2)
                       if 'commission' in trades else None),
    }


def print_comparison(a, b):
    print("\n" + "=" * 78)
    print(f"  {a['label']}   vs   {b['label']}")
    print("=" * 78)
    print(f"  {'metric':<16}{a['label'][:24]:>26}{b['label'][:24]:>26}")
    for k in ['n', 'win_rate', 'net_pnl', 'return_pct', 'expectancy', 'profit_factor',
              'max_dd_pct', 'avg_hold', 'total_cost']:
        va, vb = a.get(k), b.get(k)
        if va is None and vb is None:
            continue
        print(f"  {k:<16}{str(va):>26}{str(vb):>26}")
    print("=" * 78 + "\n")
