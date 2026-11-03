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
  * Corporate actions Yahoo doesn't adjust. A holding that opens AND closes
    30%+ under its previous close has almost certainly gone through a
    demerger, split or bonus (NSE's circuit bands rule out a market move in
    all but a few names). It is flagged, alerted on every run, and never
    sold by a rebalance until a human records what happened in
    state/momentum/corporate_actions.json: a demerger credits the new
    company's shares at the value the gap took out; a split rescales the
    holding; "market_move" releases it to the normal rules. BHAGYANGR's
    demerger into Tieramet (8 Oct 2026) was booked as an 89% loss before
    this guard existed (docs/DIAGNOSIS.md).
  * Split anchors. Besides the rolling anchor, each holding keeps the bar it
    was bought on, so a split Yahoo adjusts days late is still caught on the
    next rebalance day (whose fetch covers the entry date).
  * Money moved in. Cash swept in from a retired sleeve (receive()) is put
    to work at the next rebalance: every holding still in the band is topped
    up to an equal share of equity, and empty slots are filled at that size.
Idempotent like the swing engine: a processed session is never redone.

Paper only for now. A live rebalance would have to queue its sales and buys
as after-market orders, and the Kite adapter's exit path sends immediate
market orders; trading this sleeve live needs that path first.
"""

import json
import logging
import os
from fractions import Fraction

import numpy as np
import pandas as pd

from ..broker.base import OrderSpec
from ..costs import DEFAULT_CHARGES
from ..risk import CircuitBreakers
from ..risk.exits import Position
from ..signals.momentum import MomentumRanker
from .book import Book

logger = logging.getLogger(__name__)

STARVED_ALERT_REBALANCES = 2        # free slots + candidates + nothing bought, this many rebalances running
PENDING_EXPIRY_SESSIONS = 3         # an order whose fill-session data never arrives gives up after this
SPLIT_RATIO = 1.15                  # an anchor close moving more than this (either way) is a split/bonus
STALE_PRICE_SESSIONS = 5            # warn when a holding has had no bar for this many sessions
MIN_ELIGIBLE_SHARE = 0.5            # fewer than half last rebalance's eligible count = a data fault...
LOW_ELIGIBLE_ACCEPT_AFTER = 3       # ...unless it persists this many runs: then it is the market
CA_GAP = 0.30                       # open AND close this far under the previous close = a corporate action
ACTIONS_FILE = 'corporate_actions.json'


def clean_split_ratio(r, tol=0.005):
    """The split/bonus factor r is, or None. Indian corporate actions are
    small-number ratios: a bonus of a for every b held multiplies the share
    count by (a + b) / b with b <= 5, and face-value splits give 2, 5, 10 or
    5:2. A reverse split is the reciprocal. Within 0.5% by default, since a
    daily anchor leaves only a sliver of dividend adjustment in r; an anchor
    weeks old carries more, so it is checked with a wider `tol`."""
    if r < 1:
        f = clean_split_ratio(1 / r, tol)
        return None if f is None else 1 / f
    f = Fraction(r).limit_denominator(5)
    return f if f.numerator <= 20 and abs(float(f) - r) / r <= tol else None


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

    def receive(self, amount, source, asof):
        """Cash swept in from a retired sleeve. Recorded as a flow (returns stay
        time-weighted), lifts the drawdown peak by the same rupees so the
        latch measures the strategy and not the deposit, and is put to work
        at the next rebalance."""
        amount = round(float(amount), 2)
        if amount <= 0:
            return 0.0
        self.L.move_cash(amount, pd.Timestamp(asof).date(), source)
        if self.breakers.s.get('peak_equity') is not None:
            self.breakers.s['peak_equity'] = float(self.breakers.s['peak_equity']) + amount
        self.L.state['deploy_inflow'] = True
        return amount

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
                  'closed': [], 'kept': [], 'placed': [], 'topups': [], 'selling': [], 'warnings': [],
                  'adjusted': [], 'actions': [], 'status': 'ok', 'rebalanced': False}
        if st.get('last_processed') and pd.Timestamp(st['last_processed']) >= asof.normalize():
            report['status'] = f"session {asof.date()} already processed — nothing to do"
            return report

        due = self.rebalance_due(market, asof)
        elapsed = market.sessions_between(st['last_processed'], asof) if st.get('last_processed') else 0
        self.breakers.start_session(asof.date())
        self.book.closed_today = []
        st.setdefault('pending', [])

        # 0 ── corporate actions on what is already held ───────────────────
        self._rescale_for_splits(market, report)
        self._apply_actions(market, asof, report)
        self._flag_gaps(market, asof, report)

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
        buys = [p for p in st['pending'] if p['action'] in ('BUY', 'TOPUP')]          # in rank order
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
            held = next((q for q in L.positions if q.symbol == sym), None)
            topup = p['action'] == 'TOPUP'
            if held is not None and not topup:
                report['cancelled'].append((sym, 'buy: already held'))
                continue
            if held is None and topup:
                report['cancelled'].append((sym, 'top-up: no longer held'))
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
            if topup:
                self.book.add(held, res.fill)
                report['filled'].append((f'{sym} (top-up)', res.fill.qty, res.fill.price))
                self.log(f'  ▲ {sym} topped up {res.fill.qty} @ ₹{res.fill.price:.2f}')
                continue
            pos = self.book.open(res.fill, spec, when, 0.0)
            day = str(when.date())
            pos.meta.update({'anchor_date': day, 'anchor_open': float(bar['open']),
                             'anchor_close': float(bar['close']), 'gap_checked': day,
                             'entry_anchor': {'date': day, 'open': float(bar['open']),
                                              'close': float(bar['close'])}})
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
        """Compare each holding's anchors (the latest bar it was marked on, and
        the bar it was bought on) with what Yahoo shows for those dates now."""
        for pos in self.L.positions:
            m = pos.meta
            ea = m.get('entry_anchor') or {}
            checks = [('rolling', m.get('anchor_date'), m.get('anchor_open'), m.get('anchor_close'), 0.005)]
            if ea.get('date') and ea.get('date') != m.get('anchor_date'):
                checks.append(('entry', ea['date'], ea.get('open'), ea.get('close'), 0.03))
            for which, when, then_o, then_c, tol in checks:
                bar = market.bar(pos.symbol, when) if when else None
                if bar is None or not then_c or not then_o:
                    continue
                r_close, r_open = float(then_c) / float(bar['close']), float(then_o) / float(bar['open'])
                if 1 / SPLIT_RATIO <= r_close <= SPLIT_RATIO:
                    continue
                ratio = clean_split_ratio(r_close, tol)
                if ratio is None or abs(r_open / r_close - 1) > tol:
                    report['warnings'].append(
                        f'{pos.symbol}: Yahoo revised {when} prices (close x{r_close:.3f}, open x{r_open:.3f}) '
                        f'— not a clean split, holding left unadjusted; please verify')
                    break
                self._rescale(pos, ratio, market, which, report)
                break

    def _rescale(self, pos, ratio, market, which, report):
        """A split or bonus: more shares at a lower price, cost basis unchanged."""
        cost = pos.qty * pos.entry_price
        qty = max(int(pos.qty * ratio), 1)                # fractional entitlements are paid in cash
        pos.qty, pos.entry_price = qty, cost / qty
        m = pos.meta
        if which == 'rolling':                            # the entry anchor's trigger leaves it as it is
            bar = market.bar(pos.symbol, m.get('anchor_date'))
            if bar is not None:
                m.update({'anchor_open': float(bar['open']), 'anchor_close': float(bar['close'])})
        a = m.get('entry_anchor')
        if a:
            bar = market.bar(pos.symbol, a.get('date'))
            if bar is not None:
                a.update({'open': float(bar['open']), 'close': float(bar['close'])})
            else:                                         # still in pre-split units
                a.update({'open': float(a['open']) / float(ratio), 'close': float(a['close']) / float(ratio)})
        m.pop('ca_suspect', None)
        report['adjusted'].append((pos.symbol, f'{ratio.numerator}:{ratio.denominator}', qty))
        report['warnings'].append(f'{pos.symbol}: split/bonus {ratio.numerator}:{ratio.denominator} '
                                  f'detected — now {qty} shares at ₹{pos.entry_price:,.2f} cost each; '
                                  f'please verify')

    def _flag_gaps(self, market, asof, report):
        """Flag a holding whose session opened and closed CA_GAP under the
        previous close: a corporate action or a data fault, not a market."""
        for pos in self.L.positions:
            f = market.frames.get(pos.symbol)
            since = pos.meta.get('gap_checked') or pos.meta.get('anchor_date') or pos.entry_time
            if f is not None and len(f) and not pos.meta.get('ca_suspect'):
                closes = f['close']
                for i in np.flatnonzero((f.index > pd.Timestamp(since)) & (f.index <= asof)):
                    if i == 0:
                        continue
                    prev = float(closes.iloc[i - 1])
                    o, c = float(f['open'].iloc[i]), float(closes.iloc[i])
                    if prev > 0 and o <= prev * (1 - CA_GAP) and c <= prev * (1 - CA_GAP):
                        pos.meta['ca_suspect'] = {'date': str(f.index[i].date()), 'prev_close': prev,
                                                  'open': o, 'close': c}
                        break
            if f is not None and len(f):                  # only as far as Yahoo has delivered: a late
                seen = min(f.index[-1], pd.Timestamp(asof))   # bar is still checked when it arrives
                if seen > pd.Timestamp(since):
                    pos.meta['gap_checked'] = str(seen.date())
            g = pos.meta.get('ca_suspect')
            if g:
                report['warnings'].append(
                    f"CORPORATE ACTION? {pos.symbol} opened and closed {1 - g['close'] / g['prev_close']:.0%} under "
                    f"its previous close on {g['date']} (₹{g['prev_close']:,.2f} → ₹{g['close']:,.2f}). That is a "
                    f"demerger, split or bonus, or a data fault — not a market move. It is held, and no "
                    f"rebalance sells it, until state/momentum/{ACTIONS_FILE} records what happened "
                    f"(docs/DIAGNOSIS.md §2.1).")

    # ── corporate actions recorded by a human ───────────────────────────────
    def _apply_actions(self, market, asof, report):
        path = os.path.join(self.L.dir, ACTIONS_FILE)
        if not os.path.exists(path):
            return
        with open(path) as fh:
            actions = json.load(fh)
        st = self.L.state
        done = st.setdefault('actions_applied', [])
        for a in actions:
            if not isinstance(a, dict) or 'symbol' not in a or 'type' not in a:
                continue
            aid = a.get('id') or f"{a['symbol']}-{a.get('ex_date')}-{a['type']}"
            if aid in done:
                continue
            kind, sym = a['type'], a['symbol']
            if kind == 'demerger':
                note = self._credit_demerger(a, aid, asof)
            elif kind == 'split':
                note = self._manual_split(a)
            elif kind == 'market_move':
                note = 'recorded as a real price move: normal rules apply again'
            else:
                report['warnings'].append(f'{ACTIONS_FILE}: unknown type {kind!r} in {aid} — ignored')
                continue
            for p in self.L.positions:
                if p.symbol == sym:
                    p.meta.pop('ca_suspect', None)
            done.append(aid)
            report['actions'].append((aid, note))
            self.log(f'  ✱ {aid}: {note}')

    def _held_on(self, symbol, ex_date):
        """Shares of `symbol` held into `ex_date` (bought before it, sold on or after it)."""
        n = sum(p.qty for p in self.L.positions
                if p.symbol == symbol and pd.Timestamp(p.entry_time).normalize() < ex_date)
        t = self.L.closed_trades()
        if len(t):
            for r in t[(t['symbol'] == symbol) & (t['trigger'] != 'corporate_action')].itertuples():
                if pd.Timestamp(r.entry_time).tz_localize(None).normalize() < ex_date <= \
                        pd.Timestamp(r.exit_time).tz_localize(None).normalize():
                    n += int(r.qty)
        return n

    def _credit_demerger(self, a, aid, asof):
        """The resulting company's shares, at the value the demerger took out
        of the parent's price: a holding (valued at that until it lists) and
        a journal row that puts the same rupees back in the realised P&L."""
        sym, new, ex = a['symbol'], a['new_symbol'], pd.Timestamp(a['ex_date'])
        held = self._held_on(sym, ex)
        qty = int(held * float(a.get('ratio', 1.0)))
        if qty < 1:
            return f'no {sym} shares were held into {ex.date()} — nothing to credit'
        value = float(a['value_per_share'])
        day = str(ex.date())
        pos = Position(symbol=new, mode=self.mode, side='LONG', qty=qty, entry_price=value, initial_stop=0.0,
                       stop=0.0, entry_time=ex, atr=0.0,
                       meta={'trade_id': self.L.next_trade_id(new, ex), 'margin': qty * value,
                             'entry_costs': 0.0, 'trigger': f'demerger of {sym}', 'score': None,
                             'last_bar': day, 'order_id': '', 'corporate_action': aid,
                             'awaiting_listing': True, 'gap_checked': str(asof.date())})
        self.L.positions.append(pos)
        t = self.L.closed_trades()
        if len(t) and f'{aid}-credit' in set(t['trade_id'].astype(str)):    # a re-run after a failed save
            return f'{qty} {new} credited at ₹{value:,.2f} each (journal row already written)'
        self.L.append_trade({'trade_id': f'{aid}-credit', 'mode': self.mode, 'symbol': sym, 'side': 'LONG',
                             'qty': held, 'entry_time': str(ex), 'entry_price': 0.0, 'initial_stop': 0.0,
                             'exit_time': str(pd.Timestamp(asof)), 'exit_price': round(qty * value / held, 2),
                             'exit_reason': f'corporate action: {qty} {new} credited at ₹{value:,.2f} each '
                                            f'(the value the demerger took out of {sym})',
                             'bars_held': 0, 'gross_pnl': round(qty * value, 2), 'costs': 0.0,
                             'net_pnl': round(qty * value, 2), 'net_r': None, 'trigger': 'corporate_action',
                             'score': None})
        return f'{qty} {new} credited at ₹{value:,.2f} each (₹{qty * value:,.0f}); valued at that until it lists'

    def _manual_split(self, a):
        ratio = Fraction(float(a['ratio'])).limit_denominator(20)
        out = []
        for pos in self.L.positions:
            if pos.symbol == a['symbol']:
                cost = pos.qty * pos.entry_price
                pos.qty = max(int(pos.qty * ratio), 1)
                pos.entry_price = cost / pos.qty
                for k in ('anchor_date', 'anchor_open', 'anchor_close', 'entry_anchor'):
                    pos.meta.pop(k, None)                 # re-anchored on today's bar
                out.append(f'now {pos.qty} shares at ₹{pos.entry_price:,.2f}')
        return '; '.join(out) or 'not held — nothing to rescale'

    def _reanchor(self, market, asof, report):
        """Move each holding's anchor to today (when it traded), so the next
        run compares against a date its short fetch still covers."""
        for pos in self.L.positions:
            bar = market.bar(pos.symbol, asof)
            if bar is not None:
                pos.meta.update({'anchor_date': str(pd.Timestamp(asof).date()),
                                 'anchor_open': float(bar['open']), 'anchor_close': float(bar['close'])})
                pos.meta.pop('awaiting_listing', None)
                continue
            f = market.frames.get(pos.symbol)
            last = f.index[-1] if f is not None and len(f) else None
            if pos.meta.get('awaiting_listing') and last is None:
                report['warnings'].append(f'{pos.symbol} ({pos.meta.get("trigger")}): not trading yet — valued '
                                          f'at ₹{pos.entry_price:,.2f} a share until it lists')
                continue
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
    def _queue(self, asof, sym, qty, ref, row, action, into):
        tag = 'nbt' if action == 'TOPUP' else 'nbm'
        spec = OrderSpec(sym, 'LONG', qty, 'CNC', ref, 0.0, str(asof), tag=f'{tag}{asof:%m%d}{sym[:12]}',
                         meta={'trigger': 'momentum_12_1', 'score': round(float(row['score']), 4),
                               'rank': int(row['rank']) + 1})
        oid = self.broker.place_entry(spec)
        self.L.state['pending'].append({'action': action, 'spec': spec.to_dict(), 'order_id': oid,
                                        'placed': str(asof)})
        into.append((sym, qty, round(ref, 2), int(row['rank']) + 1, round(float(row['score']) * 100, 1)))

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
            if held[sym].meta.get('ca_suspect'):
                report['kept'].append((sym, f"probable corporate action on {held[sym].meta['ca_suspect']['date']} "
                                            f"— held until {ACTIONS_FILE} records it"))
                continue
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

        # Buys: empty slots from the top, a tenth of equity each. Only names
        # that pass the V4 filter are bought; after money has moved in, every
        # holding still in the band is also topped up to that tenth.
        slots = m.top_n - (len(held) - len(selling))
        report['slots_free'] = slots
        if not verdict.entries_allowed:
            report['buys_blocked'] = verdict.reasons
            return True
        target = equity / m.top_n
        deploy = bool(st.get('deploy_inflow'))
        candidates = 0
        for sym, row in table.sort_values('rank', kind='stable').iterrows():
            if not row['valid'] or (slots <= 0 and not deploy):
                break
            ref = float(row['close'])
            if sym in held:
                pos = held[sym]
                if deploy and sym not in selling and not pos.meta.get('ca_suspect'):
                    qty = int(min(target - pos.qty * ref, cash) // ref)
                    if qty >= 1:
                        cash -= qty * ref + self.charges.cnc(qty * ref, 0.0, dp_charged=False)
                        self._queue(asof, sym, qty, ref, row, 'TOPUP', report['topups'])
                continue
            if slots <= 0 or not row.get('buyable', True):
                continue
            candidates += 1
            qty = int(min(target, cash) // ref)
            if qty < 1:
                continue
            cash -= qty * ref + self.charges.cnc(qty * ref, 0.0, dp_charged=False)
            self._queue(asof, sym, qty, ref, row, 'BUY', report['placed'])
            slots -= 1
        if deploy:
            st['deploy_inflow'] = False

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
