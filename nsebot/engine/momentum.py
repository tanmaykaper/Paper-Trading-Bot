"""Momentum (CNC) sleeve: own the ten strongest stocks in the NSE market,
re-ranked every five sessions — research design B4 (docs/RESEARCH.md, round 6).

One call per session, after the close:

  1 FILL      the last rebalance's orders, at the open of the next session:
              sales first (their cash pays for the buys), then buys in rank
              order. A paper fill follows the research rules: that open,
              Zerodha CNC costs per leg, and no fill on a bar frozen at its
              circuit limit (no sellers at the upper band, no buyers at the
              lower) or if the stock did not trade that session. A sale that
              cannot fill waits for the next rebalance, as tested.
  2 MARK      equity at today's closes, then the breakers: a 55% drawdown
              latch and the kill switch. Breakers stop buys, never sales.
  3 REBALANCE every 5 sessions: rank every eligible stock by 12-1 momentum;
              sell holdings that dropped out of the top 30 or stopped being
              eligible; fill empty slots from the top, each sized at a tenth
              of equity at the close.
  4 PERSIST   state.json, trades.csv, equity.csv under state/momentum/.

Live-data guards the backtest never needed:
  * Splits and bonuses. Yahoo's history is split-adjusted but the ledger's
    share count is not, so each holding keeps an anchor (a date and the open
    and close Yahoo showed for it). When Yahoo later shows both moved by the
    same factor, more than 15% and a clean split/bonus ratio (2:1, 3:2,
    10:1...), the share count and entry price are rescaled, cost basis
    unchanged. A change that fails those checks is a data revision: flagged,
    not applied. Smaller moves are dividend adjustments and are ignored
    (dividends are not credited to cash).
  * Late data. A stock whose bars stop before the fill session is waiting on
    Yahoo, not suspended: its order stays queued for up to 3 sessions. A
    holding with no bar on a rebalance day is kept rather than sold.
  * A rebalance that sees implausibly few eligible stocks (a data fault, not
    a market) is skipped with a warning rather than allowed to sell the book.
Idempotent like the swing engine: a processed session is never redone.

Paper only for now. A live rebalance would have to queue its sales and buys
as after-market orders, and the Kite adapter's exit path sends immediate
market orders; trading this sleeve live needs that path first.
"""

import logging
from fractions import Fraction

import numpy as np
import pandas as pd

from ..broker.base import OrderSpec
from ..costs import DEFAULT_CHARGES
from ..risk import CircuitBreakers
from ..signals.momentum import MomentumRanker
from .book import Book

logger = logging.getLogger(__name__)

STARVED_ALERT_REBALANCES = 2        # free slots + candidates + nothing bought, this many rebalances running
PENDING_EXPIRY_SESSIONS = 3         # an order whose fill-session data never arrives gives up after this
SPLIT_RATIO = 1.15                  # an anchor close moving more than this (either way) is a split/bonus
STALE_PRICE_SESSIONS = 5            # warn when a holding has had no bar for this many sessions
MIN_ELIGIBLE_SHARE = 0.5            # fewer than half last rebalance's eligible count = a data fault...
LOW_ELIGIBLE_ACCEPT_AFTER = 3       # ...unless it persists this many runs: then it is the market


def clean_split_ratio(r):
    """The split/bonus factor r is, or None. Indian corporate actions are
    small-number ratios: a bonus of a for every b held multiplies the share
    count by (a + b) / b with b <= 5, and face-value splits give 2, 5, 10 or
    5:2. A reverse split is the reciprocal. Within 0.5%, since a daily anchor
    leaves only a sliver of dividend adjustment in r."""
    if r < 1:
        f = clean_split_ratio(1 / r)
        return None if f is None else 1 / f
    f = Fraction(r).limit_denominator(5)
    return f if f.numerator <= 20 and abs(float(f) - r) / r <= 0.005 else None


class MomentumMarket:
    """Point-in-time view for the momentum sleeve: raw daily bars by symbol,
    and the index's sessions as the calendar for fills and the schedule."""

    def __init__(self, universe, index_df, cfg=None):
        self.ranker = MomentumRanker(cfg)
        self.raw = {s: df for s, df in (universe or {}).items() if df is not None and len(df)}
        self.frames = {s: df.set_index(pd.to_datetime(df['datetime'])) for s, df in self.raw.items()}
        self._rank_frames = self.ranker.frames(self.raw)
        self.calendar = pd.DatetimeIndex(pd.to_datetime(index_df['datetime']))

    def bar(self, symbol, date):
        f = self.frames.get(symbol)
        date = pd.Timestamp(date)
        if f is None or date not in f.index:
            return None
        row = f.loc[date]
        return row.iloc[-1] if isinstance(row, pd.DataFrame) else row

    def close_on(self, symbol, date):
        b = self.bar(symbol, date)
        return None if b is None else float(b['close'])

    def data_through(self, symbol, date):
        """True when this symbol's bars reach `date` — so a missing bar on that
        date means it did not trade, not that Yahoo is late."""
        f = self.frames.get(symbol)
        return f is not None and len(f) and f.index[-1] >= pd.Timestamp(date)

    def last_close(self, symbol, asof):
        f = self.frames.get(symbol)
        if f is None:
            return None
        s = f.loc[:pd.Timestamp(asof), 'close']
        return float(s.iloc[-1]) if len(s) else None

    def next_session(self, date):
        later = self.calendar[self.calendar > pd.Timestamp(date)]
        return later[0] if len(later) else None

    def prev_close(self, symbol, date):
        """Close on the session before `date` (None if the stock didn't trade then)."""
        earlier = self.calendar[self.calendar < pd.Timestamp(date)]
        if not len(earlier):
            return None
        b = self.bar(symbol, earlier[-1])
        return None if b is None else float(b['close'])

    def sessions_between(self, after, upto):
        after, upto = pd.Timestamp(after), pd.Timestamp(upto)
        return int(((self.calendar > after) & (self.calendar <= upto)).sum())

    def rank(self, asof):
        return self.ranker.rank(self.raw, asof, frames=self._rank_frames)


class MomentumEngine:
    mode = 'momentum'

    def __init__(self, cfg, broker, ledger, workdir='.', log=None, latch_dir=None):
        if getattr(broker, 'is_live', False):
            raise ValueError('the momentum sleeve is paper-only: live rebalance orders need an '
                             'after-market order path the Kite adapter does not have yet')
        self.cfg, self.m = cfg, cfg.momentum
        self.broker, self.L = broker, ledger
        self.charges = getattr(broker, 'charges', DEFAULT_CHARGES)
        self.breakers = CircuitBreakers(cfg.momentum_breakers, self.mode,
                                        state=ledger.state.get('breakers'), workdir=workdir,
                                        latch_dir=latch_dir)
        self.log = log or logger.info
        self.book = Book(ledger, broker, self.breakers, self.mode, 1.0, self.log)

    # ── schedule ────────────────────────────────────────────────────────────
    def rebalance_due(self, market, asof):
        """True on the first run, then once `rebalance_every` sessions have
        passed since the last rebalance (late, never skipped, after a gap)."""
        st = self.L.state
        if st.get('sessions_since_rebalance') is None:
            return True
        elapsed = market.sessions_between(st['last_processed'], asof) if st.get('last_processed') else 0
        return int(st['sessions_since_rebalance']) + elapsed >= self.m.rebalance_every

    def symbols_needed(self):
        """Held and pending symbols: all a non-rebalance session needs."""
        st = self.L.state
        return sorted({p.symbol for p in self.L.positions}
                      | {(p.get('symbol') or p['spec']['symbol']) for p in st.get('pending', [])})

    # ── the cycle ───────────────────────────────────────────────────────────
    def run(self, market, asof=None):
        L, st = self.L, self.L.state
        if market.ranker.cfg != self.m:
            raise ValueError('market and engine momentum configs differ — build the market with cfg.momentum')
        asof = pd.Timestamp(asof if asof is not None else market.calendar[-1])
        report = {'mode': self.mode, 'asof': str(asof.date()), 'filled': [], 'cancelled': [],
                  'closed': [], 'kept': [], 'placed': [], 'selling': [], 'warnings': [],
                  'adjusted': [], 'status': 'ok', 'rebalanced': False}
        if st.get('last_processed') and pd.Timestamp(st['last_processed']) >= asof.normalize():
            report['status'] = f"session {asof.date()} already processed — nothing to do"
            return report

        due = self.rebalance_due(market, asof)
        elapsed = market.sessions_between(st['last_processed'], asof) if st.get('last_processed') else 0
        self.breakers.start_session(asof.date())
        self.book.closed_today = []
        st.setdefault('pending', [])

        # 0 ── splits and bonuses on what is already held ──────────────────
        self._rescale_for_splits(market, report)

        # 1 ── fills ─────────────────────────────────────────────────────────
        self._fill_pending(market, asof, report)

        # 2 ── mark and breakers ─────────────────────────────────────────────
        equity = self.book.equity(lambda s: market.last_close(s, asof))
        verdict = self.breakers.check(equity)
        report.update({'equity': round(equity, 2), 'cash': round(L.cash, 2), 'breakers': verdict.reasons})

        # 3 ── rebalance ─────────────────────────────────────────────────────
        if st.get('sessions_since_rebalance') is not None:
            st['sessions_since_rebalance'] = int(st['sessions_since_rebalance']) + elapsed
        if due and self._rebalance(market, asof, equity, verdict, report):
            st['sessions_since_rebalance'] = 0
        since = st.get('sessions_since_rebalance')
        report['next_rebalance_in'] = 0 if since is None else max(self.m.rebalance_every - since, 0)

        # 4 ── persist ───────────────────────────────────────────────────────
        self._reanchor(market, asof, report)
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
        report['holdings'] = [(p.symbol, p.qty, round(p.entry_price, 2), market.last_close(p.symbol, asof))
                              for p in sorted(L.positions, key=lambda p: p.symbol)]
        report['pending'] = [((p.get('symbol') or p['spec']['symbol']), p['action']) for p in st['pending']]
        return report

    # ── 1: fills ────────────────────────────────────────────────────────────
    def _fill_pending(self, market, asof, report):
        st, L = self.L.state, self.L
        waiting = []
        sells = sorted((p for p in st['pending'] if p['action'] == 'SELL'), key=lambda p: p['symbol'])
        buys = [p for p in st['pending'] if p['action'] == 'BUY']
        for p in sells:
            sym, when = p['symbol'], market.next_session(p['signal_time'])
            if self._wait_for_data(market, sym, when, asof, p, waiting, report):
                continue
            pos = next((q for q in L.positions if q.symbol == sym), None)
            bar = market.bar(sym, when)
            if pos is None:
                report['cancelled'].append((sym, 'sell: no longer held'))
            elif bar is None or not np.isfinite(float(bar['open'])):
                report['kept'].append((sym, f'did not trade on {when.date()} — re-checked at the next rebalance'))
            elif self._locked(market, sym, when, bar, 'SELL'):
                report['kept'].append((sym, f'frozen at its lower circuit on {when.date()} (no buyers) — '
                                            f're-checked at the next rebalance'))
            else:
                fill = self.broker.exit(pos, float(bar['open']), f"rebalance: {p['reason']}", when)
                t = self.book.close(pos, fill, when)
                report['closed'].append((sym, t['exit_reason'], t['net_pnl'], t['net_r']))
                self.log(f'  ▼ {sym} sold {t["qty"]} @ ₹{t["exit_price"]:.2f} net ₹{t["net_pnl"]:+,.0f}')
        for p in buys:
            spec = p['spec']
            sym, when = spec['symbol'], market.next_session(spec['signal_time'])
            if self._wait_for_data(market, sym, when, asof, p, waiting, report):
                continue
            bar = market.bar(sym, when)
            if any(q.symbol == sym for q in L.positions):
                report['cancelled'].append((sym, 'buy: already held'))
                continue
            if bar is None or not np.isfinite(float(bar['open'])):
                report['cancelled'].append((sym, f'did not trade on {when.date()}'))
                continue
            if self._locked(market, sym, when, bar, 'BUY'):
                report['cancelled'].append((sym, f'frozen at its upper circuit on {when.date()} (no sellers)'))
                continue
            b = {'open': float(bar['open']), 'high': float(bar['high']), 'low': float(bar['low']),
                 'close': float(bar['close']), 'datetime': when}
            res = self.broker.resolve_entry(p, b)
            if res.status != 'FILLED':
                report['cancelled'].append((sym, res.reason or res.status))
                continue
            if res.fill.qty * res.fill.price > L.cash:        # the open gapped up: buy what cash allows
                qty = int(max(L.cash, 0.0) // res.fill.price)
                if qty < 1:
                    report['cancelled'].append((sym, f'not enough cash at the {when.date()} open'))
                    continue
                p = dict(p, spec=dict(spec, qty=qty))
                spec, res = p['spec'], self.broker.resolve_entry(p, b)
            pos = self.book.open(res.fill, spec, when, 0.0)
            pos.meta.update({'anchor_date': str(when.date()), 'anchor_open': float(bar['open']),
                             'anchor_close': float(bar['close'])})
            report['filled'].append((sym, pos.qty, pos.entry_price))
            self.log(f'  ▲ {sym} bought {pos.qty} @ ₹{pos.entry_price:.2f}')
        st['pending'] = waiting

    @staticmethod
    def _wait_for_data(market, sym, when, asof, p, waiting, report):
        """Keep an order queued while its fill session hasn't happened or
        Yahoo hasn't delivered it yet; give up after PENDING_EXPIRY_SESSIONS."""
        if when is None or when > asof:
            waiting.append(p)
            return True
        if market.data_through(sym, when):
            return False
        if market.sessions_between(when, asof) < PENDING_EXPIRY_SESSIONS:
            waiting.append(p)
            report['kept'].append((sym, f'no data for {when.date()} yet — order kept queued'))
            return True
        return False                                  # long enough: treat as not traded

    def _rescale_for_splits(self, market, report):
        for pos in self.L.positions:
            anchor = pos.meta.get('anchor_date')
            bar = market.bar(pos.symbol, anchor) if anchor else None
            then_c, then_o = pos.meta.get('anchor_close'), pos.meta.get('anchor_open')
            if bar is None or not then_c or not then_o:
                continue
            r_close, r_open = float(then_c) / float(bar['close']), float(then_o) / float(bar['open'])
            if 1 / SPLIT_RATIO <= r_close <= SPLIT_RATIO:
                continue
            ratio = clean_split_ratio(r_close)
            if ratio is None or abs(r_open / r_close - 1) > 0.005:
                report['warnings'].append(
                    f'{pos.symbol}: Yahoo revised {anchor} prices (close x{r_close:.3f}, open x{r_open:.3f}) '
                    f'— not a clean split, holding left unadjusted; please verify')
                continue
            cost = pos.qty * pos.entry_price
            qty = max(int(pos.qty * ratio), 1)                # fractional entitlements are paid in cash
            pos.qty, pos.entry_price = qty, cost / qty
            pos.meta.update({'anchor_close': float(bar['close']), 'anchor_open': float(bar['open'])})
            report['adjusted'].append((pos.symbol, f'{ratio.numerator}:{ratio.denominator}', qty))
            report['warnings'].append(f'{pos.symbol}: split/bonus {ratio.numerator}:{ratio.denominator} '
                                      f'detected — now {qty} shares at ₹{pos.entry_price:,.2f} cost each; '
                                      f'please verify')

    def _reanchor(self, market, asof, report):
        """Move each holding's anchor to today (when it traded), so the next
        run compares against a date its short fetch still covers."""
        for pos in self.L.positions:
            bar = market.bar(pos.symbol, asof)
            if bar is not None:
                pos.meta.update({'anchor_date': str(pd.Timestamp(asof).date()),
                                 'anchor_open': float(bar['open']), 'anchor_close': float(bar['close'])})
                continue
            f = market.frames.get(pos.symbol)
            last = f.index[-1] if f is not None and len(f) else None
            if last is None or market.sessions_between(last, asof) >= STALE_PRICE_SESSIONS:
                report['warnings'].append(f'{pos.symbol}: no price since {last.date() if last is not None else "—"} '
                                          f'— valued at its last close; check for a suspension or delisting')

    @staticmethod
    def _locked(market, sym, when, bar, action):
        """One price all day = frozen at the circuit band (research.experiments.locked_bars)."""
        h, l, c = float(bar['high']), float(bar['low']), float(bar['close'])
        if not abs(h - l) <= 1e-9 * max(abs(h), 1.0):
            return False
        prev = market.prev_close(sym, when)
        if prev is None or not np.isfinite(prev):
            return False
        return c >= prev if action == 'BUY' else c <= prev

    # ── 3: rebalance ────────────────────────────────────────────────────────
    def _rebalance(self, market, asof, equity, verdict, report):
        """Queue the sales and buys. False when the rebalance was skipped as
        a data fault (it then stays due for the next run)."""
        m, st, L = self.m, self.L.state, self.L
        table = market.rank(asof)
        eligible = int(table['valid'].sum()) if len(table) else 0
        last = int(st.get('last_eligible') or 0)
        floor = max(2 * m.top_n, int(MIN_ELIGIBLE_SHARE * last))
        report['eligible'] = eligible
        if eligible < floor:
            low = int(st.get('low_eligible_runs', 0)) + 1
            st['low_eligible_runs'] = low
            if eligible < 2 * m.top_n or low < LOW_ELIGIBLE_ACCEPT_AFTER:
                report['warnings'].append(
                    f'DATA: only {eligible} eligible stocks (need at least {floor}) — rebalance skipped, '
                    f'retried next session. Usually short or missing price history from Yahoo.')
                return False
            report['warnings'].append(
                f'DATA: eligible stocks fell from {last} to {eligible} and stayed there {low} runs — '
                f'accepted as the market; rebalancing.')
        st['low_eligible_runs'] = 0
        st['last_eligible'] = eligible
        report['rebalanced'] = True
        band = m.band_mult * m.top_n
        held = {p.symbol: p for p in L.positions}
        cash = L.cash

        # Sales, in symbol order (the research loop's order, so the cash
        # arithmetic matches it to the rupee).
        selling = set()
        for sym in sorted(held):
            row = table.loc[sym] if sym in table.index else None
            if market.bar(sym, asof) is None:
                report['kept'].append((sym, f'no bar for {asof.date()} — held, re-checked next rebalance'))
                continue
            if row is None or not row['valid']:
                reason = 'no longer eligible'
            elif row['rank'] >= band:
                reason = f'fell out of the top {band} (rank {int(row["rank"]) + 1})'
            else:
                continue
            pos = held[sym]
            ref = market.last_close(sym, asof) or pos.entry_price
            value = pos.qty * ref
            cash += value - self.charges.cnc(0.0, value)
            selling.add(sym)
            st['pending'].append({'action': 'SELL', 'symbol': sym, 'qty': pos.qty, 'ref_price': ref,
                                  'signal_time': str(asof), 'reason': reason})
            report['selling'].append((sym, pos.qty, reason))

        # Buys: empty slots from the top, a tenth of equity each.
        slots = m.top_n - (len(held) - len(selling))
        report['slots_free'] = slots
        if not verdict.entries_allowed:
            report['buys_blocked'] = verdict.reasons
            return True
        target = equity / m.top_n
        candidates = 0
        for sym, row in table.sort_values('rank', kind='stable').iterrows():
            if slots <= 0 or not row['valid']:
                break
            if sym in held:
                continue
            candidates += 1
            ref = float(row['close'])
            qty = int(min(target, cash) // ref)
            if qty < 1:
                continue
            cash -= qty * ref + self.charges.cnc(qty * ref, 0.0, dp_charged=False)
            spec = OrderSpec(sym, 'LONG', qty, 'CNC', ref, 0.0, str(asof), tag=f'nbm{asof:%m%d}{sym[:12]}',
                             meta={'trigger': 'momentum_12_1', 'score': round(float(row['score']), 4),
                                   'rank': int(row['rank']) + 1})
            oid = self.broker.place_entry(spec)
            st['pending'].append({'action': 'BUY', 'spec': spec.to_dict(), 'order_id': oid, 'placed': str(asof)})
            report['placed'].append((sym, qty, round(ref, 2), int(row['rank']) + 1, round(float(row['score']) * 100, 1)))
            slots -= 1

        # Paralysis watch: free slots and candidates, but nothing bought.
        n = int(st.get('starved_rebalances', 0))
        if report['placed'] or report['slots_free'] <= 0:
            n = 0
        elif candidates:
            n += 1
        st['starved_rebalances'] = n
        if n >= STARVED_ALERT_REBALANCES:
            report['warnings'].append(
                f'NOT TRADING: {n} rebalances in a row with {report["slots_free"]} free slot(s) and '
                f'candidates, but nothing could be bought (cash ₹{L.cash:,.0f}, target ₹{target:,.0f}) '
                f'— check cash and prices')
        return True
