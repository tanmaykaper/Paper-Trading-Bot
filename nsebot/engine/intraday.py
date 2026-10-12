"""Intraday (MIS) session loop.

`step()` is one poll: it is given every symbol's 5-minute bars as of `now`
and does, in order:

  1 NEW SESSION  reset the day's trade list and breaker counters; anything
                 left over from a previous session (a crashed run) is closed
                 at its last price immediately — MIS never carries.
  2 FILL         pending entries at the open of the first completed bar after
                 their signal bar (paper) / reconcile the real fill (Kite).
  3 EXITS        every unseen completed bar per position: square-off at
                 15:10 > stop (gap-aware) > trail. The square-off also fires
                 on the wall clock, so a late data feed cannot carry a
                 position past 15:10.
  4 BREAKERS     daily loss limit -> flatten everything now; loss streak,
                 trades/day, kill file -> no new entries.
  5 ENTRIES      fresh ORB signals on the latest completed bar -> sized under
                 5x MIS margin and the cap stack -> pending for the next bar.
  6 PERSIST      state.json after every step, so a killed runner resumes.

`run_session()` wraps it in a clock-aligned loop with a data provider.
"""

import logging
import time as _time

import pandas as pd

from ..broker.base import OrderSpec
from ..data import completed_bars
from ..indicators import atr as atr_fn, median_turnover
from ..market import IST, TickTable, at_time, round_stop
from ..risk import CircuitBreakers, allocate, estimate_edge, evaluate
from ..risk.exits import advance
from ..signals import IntradaySignalEngine
from ..universe import INDEX_SYMBOL, INTRADAY_UNIVERSE, sector_of
from .book import Book

logger = logging.getLogger(__name__)


class IntradayEngine:
    mode = 'intraday'

    def __init__(self, cfg, broker, ledger, workdir='.', log=None, universe=None, latch_dir=None):
        self.cfg = cfg
        self.sig = IntradaySignalEngine(cfg.intraday)
        self.sizing, self.exits = cfg.intraday_sizing, cfg.intraday_exits
        self.broker, self.L = broker, ledger
        self.universe = list(universe or INTRADAY_UNIVERSE)
        self.breakers = CircuitBreakers(cfg.intraday_breakers, self.mode,
                                        state=ledger.state.get('breakers'), workdir=workdir,
                                        latch_dir=latch_dir)
        self.log = log or logger.info
        self.book = Book(ledger, broker, self.breakers, self.mode, self.sizing.leverage, self.log)
        self.ticks = getattr(broker, 'ticks', None) or TickTable()
        self.interval = cfg.intraday.interval_minutes

    # ─────────────────────────────────────────────────────────────────────────
    def step(self, bars_by_symbol, now, daily_by_symbol=None, index_bars=None):
        st = self.L.state
        now = pd.Timestamp(now)
        now = now.tz_localize(IST) if now.tzinfo is None else now.tz_convert(IST)
        rep = {'now': str(now), 'filled': [], 'closed': [], 'placed': [], 'cancelled': [],
               'breakers': [], 'flattened': False}
        done = {s: completed_bars(b, now, self.interval) for s, b in bars_by_symbol.items()
                if b is not None and len(b)}
        price_of = lambda s: (float(done[s]['close'].iloc[-1])                 # noqa: E731
                              if s in done and len(done[s]) else None)

        # 1 ── session rollover ──────────────────────────────────────────────
        session = str(now.date())
        if st.get('session') != session:
            for pos in list(self.L.positions):
                px = price_of(pos.symbol) or pos.entry_price
                self._exit(pos, px, 'stale MIS carry-over closed', now, rep)
            if st.get('pending'):
                rep['cancelled'] += [(p['spec']['symbol'], 'previous session') for p in st['pending']]
            st.update({'session': session, 'traded_today': [], 'pending': []})
            self.breakers.start_session(session)
            self.book.closed_today = []

        square_off = self.exits.square_off
        past_square_off = square_off is not None and now.time() >= square_off

        # 2 ── pending entries ───────────────────────────────────────────────
        keep = []
        for p in st.get('pending', []):
            spec = p['spec']
            bars = done.get(spec['symbol'])
            bar = None
            if bars is not None and len(bars):
                after = bars[pd.to_datetime(bars['datetime']) > pd.Timestamp(spec['signal_time'])]
                if len(after):
                    bar = after.iloc[0]
            res = self.broker.resolve_entry(p, bar)
            if res.status == 'FILLED':
                # A live fill that lands after the square-off still becomes a
                # position — the clock square-off below closes it at once.
                pos = self.book.open(res.fill, spec, res.fill.time, spec.get('meta', {}).get('atr', 0.0))
                rep['filled'].append((pos.symbol, pos.side, pos.qty, pos.entry_price))
                self.log(f'  ▲ {pos.side} {pos.symbol} {pos.qty} @ ₹{pos.entry_price:.2f} (stop ₹{pos.stop:.2f})')
            elif res.status == 'WAIT' and not past_square_off:
                keep.append(p)
            else:
                rep['cancelled'].append((spec['symbol'], res.reason or 'square-off reached'))
        st['pending'] = keep

        # 3 ── exits ─────────────────────────────────────────────────────────
        for pos in list(self.L.positions):
            bars = done.get(pos.symbol)
            if bars is None or not len(bars):
                continue
            atr_series = atr_fn(bars, 14)
            ts = pd.to_datetime(bars['datetime'])
            new = bars[ts > pd.Timestamp(pos.meta.get('last_bar', pos.entry_time))]
            for i, row in new.iterrows():
                bar_end = pd.Timestamp(row['datetime']) + pd.Timedelta(minutes=self.interval)
                d = evaluate(pos, row, self.exits, atr=float(atr_series.loc[i]), now=bar_end)
                if d.action == 'EXIT':
                    self._exit(pos, d.price, d.reason, bar_end, rep)
                    break
                advance(pos, row, d)
                if d.action == 'TRAIL':
                    self.broker.update_protection(pos)
                pos.meta['last_bar'] = str(row['datetime'])
        if past_square_off:
            for pos in list(self.L.positions):
                self._exit(pos, price_of(pos.symbol) or pos.entry_price,
                           f'square-off {square_off:%H:%M} (clock)', now, rep)

        # 4 ── breakers ──────────────────────────────────────────────────────
        equity = self.book.equity(price_of)
        verdict = self.breakers.check(equity, open_pnl=self.book.open_pnl(price_of))
        rep['breakers'] = verdict.reasons
        if verdict.flatten and self.L.positions:
            for pos in list(self.L.positions):
                self._exit(pos, price_of(pos.symbol) or pos.entry_price, 'daily loss limit — flatten',
                           now, rep)
            rep['flattened'] = True

        # 5 ── entries ───────────────────────────────────────────────────────
        entry_open = (verdict.entries_allowed and not past_square_off
                      and now.time() < self.cfg.intraday.entry_end)
        if entry_open:
            exclude = set(st['traded_today']) | {p.symbol for p in self.L.positions} \
                | {p['spec']['symbol'] for p in st['pending']}
            sigs = self.sig.scan(bars_by_symbol, now=now, daily_by_symbol=daily_by_symbol,
                                 index_bars=index_bars, exclude=exclude)
            rep['signals'] = len(sigs)
            if sigs:
                self._enter(sigs, equity, verdict, daily_by_symbol or {}, now, rep)

        # 6 ── persist ───────────────────────────────────────────────────────
        rep['equity'] = round(equity, 2)
        rep['open'] = [(p.symbol, p.side, p.qty, p.entry_price, round(p.stop, 2)) for p in self.L.positions]
        st['breakers'] = self.breakers.state()
        self.L.save()
        return rep

    # ─────────────────────────────────────────────────────────────────────────
    def _enter(self, sigs, equity, verdict, daily, now, rep):
        st = self.L.state
        edge = estimate_edge(self.sizing, self.L.realised_r())

        def turnover_of(sym):
            d = daily.get(sym)
            return float(median_turnover(d).iloc[-1]) if d is not None and len(d) >= 5 else None

        cash = self.broker.available_cash(self.L.cash) - self.book.reserved_for_pending()
        plan = allocate(sigs, list(self.L.positions), self.sizing, equity=equity, cash=cash, edge=edge,
                        breaker_mult=verdict.size_mult, sector_of=sector_of, turnover_of=turnover_of,
                        lot_size_of=self.ticks.lot, product='MIS')
        for o in plan.orders:
            tick = self.ticks.tick(o.symbol, o.stop)
            spec = OrderSpec(o.symbol, o.side, o.qty, 'MIS', o.ref_price,
                             round_stop(o.stop, o.side, tick), str(o.signal.asof),
                             tag=f'nbi{now:%H%M}{o.symbol[:12]}',
                             meta={'atr': o.signal.atr, 'trigger': o.signal.trigger,
                                   'score': round(o.signal.score, 2), 'risk_pct': o.sizing.risk_pct,
                                   'binding': o.sizing.binding})
            try:
                oid = self.broker.place_entry(spec)
            except Exception as e:                        # explicit, logged, never silent
                rep['cancelled'].append((o.symbol, f'order failed: {e}'))
                self.log(f'  ✗ {o.symbol}: order failed — {e}')
                continue
            st['pending'].append({'spec': spec.to_dict(), 'order_id': oid, 'placed': str(now)})
            st['traded_today'].append(o.symbol)
            rep['placed'].append((o.symbol, o.side, o.qty, o.ref_price, spec.stop))
            self.log(f'  ◆ {o.side} {o.symbol} {o.qty} (ref ₹{o.ref_price:.2f}, stop ₹{spec.stop:.2f}, '
                     f'risk {o.sizing.risk_pct * 100:.2f}%, binding: {o.sizing.binding})')

    def _exit(self, pos, price, reason, when, rep):
        fill = self.broker.exit(pos, price, reason, when)
        t = self.book.close(pos, fill, when)
        rep['closed'].append((pos.symbol, t['exit_reason'], t['net_pnl'], t['net_r']))
        self.log(f'  ▼ {pos.side} {pos.symbol} {t["exit_reason"]} @ ₹{t["exit_price"]:.2f} '
                 f'net ₹{t["net_pnl"]:+,.0f} ({(t["net_r"] or 0):+.2f}R)')

    def finish(self, now, last_prices=None):
        """End of session: nothing may remain open; write the day's equity row."""
        now = pd.Timestamp(now)
        rep = {'closed': []}
        for pos in list(self.L.positions):
            px = (last_prices or {}).get(pos.symbol) or pos.entry_price
            self._exit(pos, px, 'end-of-session square-off', now, rep)
        self.L.state['pending'] = []
        equity = self.L.cash
        peak = float(self.breakers.state().get('peak_equity') or equity)
        self.L.append_equity({'session': str(now.date()), 'equity': round(equity, 2),
                              'cash': round(equity, 2), 'open_positions': 0, 'open_value': 0.0,
                              'realised_today': round(sum(t['net_pnl'] for t in self.book.closed_today), 2),
                              'peak_equity': round(peak, 2),
                              'drawdown_pct': round((peak - equity) / peak * 100, 2) if peak else 0.0})
        self.L.state['breakers'] = self.breakers.state()
        self.L.save()
        return rep

    # ─────────────────────────────────────────────────────────────────────────
    def run_session(self, provider, clock=None, sleep=_time.sleep, latency_s=25, end=None):
        """Live loop: poll every completed 5-minute bar until the square-off."""
        clock = clock or (lambda: pd.Timestamp.now(tz=IST))
        today = clock()
        end = end or at_time(today, self.exits.square_off) + pd.Timedelta(minutes=2)
        daily = provider.daily(self.universe, lookback_days=60)
        reports = []
        had_session = False
        while True:
            now = clock()
            if now >= end:
                break
            symbols = list(dict.fromkeys(self.universe + [p.symbol for p in self.L.positions]))
            bars = provider.intraday(symbols, self.interval, lookback_days=5)
            index_bars = provider.intraday([INDEX_SYMBOL], self.interval, lookback_days=5).get(INDEX_SYMBOL)
            if self._has_session(index_bars, now):
                had_session = True
            elif now.time() >= self.cfg.intraday.entry_start:
                self.log(f'  No NSE session data for {now.date()} — holiday or feed outage; stopping.')
                break
            reports.append(self.step(bars, now, daily, index_bars))
            nxt = now.floor(f'{self.interval}min') + pd.Timedelta(minutes=self.interval, seconds=latency_s)
            sleep(max((min(nxt, end) - clock()).total_seconds(), 1.0))
        if not had_session and not self.L.positions:
            self.L.save()
            return reports                       # holiday: no trades, no equity row
        last = {}
        try:
            bars = provider.intraday([p.symbol for p in self.L.positions], self.interval, lookback_days=1)
            last = {s: float(b['close'].iloc[-1]) for s, b in bars.items() if len(b)}
        except Exception as e:
            self.log(f'  could not fetch final prices ({e}) — closing at entry price')
        reports.append(self.finish(clock(), last))
        return reports

    @staticmethod
    def _has_session(index_bars, now):
        if index_bars is None or not len(index_bars):
            return False
        return (pd.to_datetime(index_bars['datetime']).dt.date == now.date()).any()
