"""Swing (CNC) end-of-day cycle.

One call per session, after the close:

  1 FILL      yesterday's plans at the open of the first bar after their
              signal (paper), or reconcile the real AMO fill (Kite). An open
              already through the stop cancels the plan — the research rule.
  2 EXITS     every bar each position has not yet seen, oldest first: stop
              (gap-aware) > EMA-5 reversion target > 7-session time stop.
              Catching up bar-by-bar means a missed run never skips a stop.
  3 BREAKERS  mark equity, check loss streak / daily loss / drawdown / kill file.
  4 ENTRIES   today's dip signals -> half-Kelly sizing under the cap stack ->
              plans that fill at tomorrow's open.
  5 PERSIST   state.json, trades.csv, equity.csv — committed by the workflow.

Idempotent: a session that has already been processed is skipped, so a
re-run (or a GitHub Actions retry) cannot double-fill or double-exit.
"""

import logging

import pandas as pd

from ..broker.base import OrderSpec
from ..market import TickTable, round_stop
from ..risk import CircuitBreakers, allocate, estimate_edge, evaluate
from ..risk.exits import advance
from ..universe import sector_of
from .book import Book

logger = logging.getLogger(__name__)

PENDING_EXPIRY_SESSIONS = 3


class SwingEngine:
    mode = 'swing'

    def __init__(self, cfg, broker, ledger, workdir='.', log=None):
        self.cfg = cfg
        self.sizing, self.exits = cfg.swing_sizing, cfg.swing_exits
        self.broker, self.L = broker, ledger
        self.breakers = CircuitBreakers(cfg.swing_breakers, self.mode,
                                        state=ledger.state.get('breakers'), workdir=workdir)
        self.log = log or logger.info
        self.book = Book(ledger, broker, self.breakers, self.mode, self.sizing.leverage, self.log)
        self.ticks = getattr(broker, 'ticks', None) or TickTable()

    def run(self, market, asof=None):
        L, st = self.L, self.L.state
        asof = pd.Timestamp(asof if asof is not None else market.dates[-1])
        report = {'mode': self.mode, 'asof': str(asof.date()), 'filled': [], 'cancelled': [],
                  'closed': [], 'placed': [], 'declined': [], 'status': 'ok'}
        if st.get('last_processed') and pd.Timestamp(st['last_processed']) >= asof.normalize():
            report['status'] = f"session {asof.date()} already processed — nothing to do"
            return report

        self.breakers.start_session(asof.date())
        self.book.closed_today = []

        # 1 ── pending entries ───────────────────────────────────────────────
        keep = []
        for p in st.get('pending', []):
            spec = p['spec']
            bars = market.bars_after(spec['symbol'], spec['signal_time'], asof)
            bar = None
            if len(bars):
                bar = bars.iloc[0].copy()
                bar['datetime'] = bars.index[0]
            res = self.broker.resolve_entry(p, bar)
            if res.status == 'FILLED':
                pos = self.book.open(res.fill, spec, res.fill.time, spec.get('meta', {}).get('atr', 0.0))
                report['filled'].append((pos.symbol, pos.qty, pos.entry_price))
                self.log(f'  ▲ {pos.symbol} filled {pos.qty} @ ₹{pos.entry_price:.2f} '
                         f'(stop ₹{pos.stop:.2f})')
            elif res.status == 'CANCELLED':
                report['cancelled'].append((spec['symbol'], res.reason))
                self.log(f'  ✗ {spec["symbol"]}: {res.reason}')
            else:
                p['sessions_waited'] = int(p.get('sessions_waited', 0)) + 1
                if p['sessions_waited'] > PENDING_EXPIRY_SESSIONS:
                    report['cancelled'].append((spec['symbol'], 'no tradeable bar — expired'))
                else:
                    keep.append(p)
        st['pending'] = keep

        # 2 ── exits, bar by bar ─────────────────────────────────────────────
        for pos in list(L.positions):
            bars = market.bars_after(pos.symbol, pos.meta.get('last_bar', pos.entry_time), asof)
            for when, row in bars.iterrows():
                bar = {'open': row['open'], 'high': row['high'], 'low': row['low'],
                       'close': row['close'], 'datetime': when}
                d = evaluate(pos, bar, self.exits, atr=row.get('atr'), exit_level=row.get('ema_exit'))
                if d.action == 'EXIT':
                    fill = self.broker.exit(pos, d.price, d.reason, when)
                    t = self.book.close(pos, fill, when)
                    report['closed'].append((pos.symbol, t['exit_reason'], t['net_pnl'], t['net_r']))
                    self.log(f'  ▼ {pos.symbol} {t["exit_reason"]} @ ₹{t["exit_price"]:.2f} '
                             f'net ₹{t["net_pnl"]:+,.0f} ({t["net_r"]:+.2f}R)')
                    break
                advance(pos, bar, d)
                if d.action == 'TRAIL':
                    self.broker.update_protection(pos)
                pos.meta['last_bar'] = str(when)

        # 3 ── equity and breakers ───────────────────────────────────────────
        price_of = lambda s: market.last_close(s, asof)                       # noqa: E731
        equity = self.book.equity(price_of)
        verdict = self.breakers.check(equity)
        regime = market.regime_at(asof)
        report.update({'equity': round(equity, 2), 'cash': round(L.cash, 2), 'regime': regime,
                       'breakers': verdict.reasons, 'size_mult': verdict.size_mult})

        # 4 ── edge monitor and entries ──────────────────────────────────────
        # Kelly on the bot's own net results is reported every run. In notional
        # mode it does not size trades (that fed a fee death-spiral); when it
        # turns negative on a real sample it raises a warning for a human to
        # act on (STOP_TRADING), rather than an automatic state the book could
        # never trade its way out of.
        edge = estimate_edge(self.sizing, L.realised_r())
        report['edge'] = edge.__dict__
        report['warnings'] = []
        if edge.n_realised >= self.sizing.edge_warn_min_trades and not edge.edge_positive:
            report['warnings'].append(
                f'measured edge negative after {edge.n_realised} trades (Kelly {edge.kelly_full:+.3f}) '
                f'— review the strategy; create STOP_TRADING to pause entries')
        signals = market.signals(asof) if verdict.entries_allowed else []
        report['signals'] = len(signals)
        if signals:
            occupied = list(L.positions) + [self._pending_as_position(p) for p in st['pending']]
            cash = self.broker.available_cash(L.cash) - self.book.reserved_for_pending()
            plan = allocate(signals, occupied, self.sizing, equity=equity, cash=cash, edge=edge,
                            regime_mult=regime['size_mult'] if self.sizing.use_regime_size else 1.0,
                            breaker_mult=verdict.size_mult,
                            max_new=regime['max_new_entries'], sector_of=sector_of,
                            turnover_of=lambda s: market.turnover(s, asof),
                            lot_size_of=self.ticks.lot, product='CNC')
            report['declined'] = plan.declined
            for o in plan.orders:
                tick = self.ticks.tick(o.symbol, o.stop)
                spec = OrderSpec(o.symbol, o.side, o.qty, 'CNC', o.ref_price,
                                 round_stop(o.stop, o.side, tick), str(asof),
                                 tag=f'nbs{asof:%m%d}{o.symbol[:12]}',
                                 meta={'atr': o.signal.atr, 'trigger': o.signal.trigger,
                                       'score': round(o.signal.score, 3),
                                       'risk_pct': o.sizing.risk_pct, 'binding': o.sizing.binding})
                try:
                    oid = self.broker.place_entry(spec)
                except Exception as e:                    # explicit, logged, never silent
                    report['declined'].append((o.symbol, f'order failed: {e}'))
                    self.log(f'  ✗ {o.symbol}: order failed — {e}')
                    continue
                st['pending'].append({'spec': spec.to_dict(), 'order_id': oid, 'placed': str(asof)})
                report['placed'].append((o.symbol, o.qty, o.ref_price, spec.stop, o.sizing.binding))
                self.log(f'  ◆ plan {o.symbol} {o.qty} sh (ref ₹{o.ref_price:.2f}, stop ₹{spec.stop:.2f}, '
                         f'risk {o.sizing.risk_pct * 100:.2f}%, binding: {o.sizing.binding})')

        # 5 ── persist ───────────────────────────────────────────────────────
        st['last_processed'] = str(asof.date())
        st['breakers'] = self.breakers.state()
        peak = float(self.breakers.state().get('peak_equity') or equity)
        L.append_equity({'session': str(asof.date()), 'equity': round(equity, 2),
                         'cash': round(L.cash, 2), 'open_positions': len(L.positions),
                         'open_value': round(equity - L.cash, 2),
                         'realised_today': round(sum(t['net_pnl'] for t in self.book.closed_today), 2),
                         'peak_equity': round(peak, 2),
                         'drawdown_pct': round((peak - equity) / peak * 100, 2) if peak else 0.0})
        L.save()
        report['open_positions'] = [(p.symbol, p.qty, p.entry_price, p.stop) for p in L.positions]
        report['pending'] = [(p['spec']['symbol'], p['spec']['qty']) for p in st['pending']]
        return report

    @staticmethod
    def _pending_as_position(p):
        from ..risk.exits import Position
        s = p['spec']
        return Position(s['symbol'], 'swing', s['side'], int(s['qty']), float(s['ref_price']),
                        float(s['stop']), float(s['stop']), pd.Timestamp(s['signal_time']), 0.0)
