"""Cash and position accounting shared by both engines.

One model covers CNC and MIS: opening a position blocks margin
(notional / leverage — the full notional for CNC) plus entry costs; closing
it releases the margin plus gross P&L minus exit costs. Equity is cash plus,
for every open position, its blocked margin and unrealised P&L.

Every closed trade is journalled with its NET R (net P&L over the rupees
initially at risk) — the number the Kelly sizer learns from, so costs are
never invisible to sizing.
"""

import pandas as pd

from ..risk.exits import Position


class Book:

    def __init__(self, ledger, broker, breakers, mode, leverage=1.0, log=None):
        self.L, self.broker, self.breakers = ledger, broker, breakers
        self.mode, self.leverage = mode, float(leverage)
        self.log = log
        self.closed_today = []

    # ── entries ─────────────────────────────────────────────────────────────
    def open(self, fill, spec, when, atr):
        margin = fill.qty * fill.price / self.leverage
        self.L.state['cash'] = self.L.cash - margin - fill.costs
        pos = Position(symbol=spec['symbol'], mode=self.mode, side=spec['side'], qty=int(fill.qty),
                       entry_price=float(fill.price), initial_stop=float(spec['stop']),
                       stop=float(spec['stop']), entry_time=pd.Timestamp(when), atr=float(atr),
                       meta={'trade_id': self.L.next_trade_id(spec['symbol'], when),
                             'margin': margin, 'entry_costs': fill.costs,
                             'trigger': spec.get('meta', {}).get('trigger', ''),
                             'score': spec.get('meta', {}).get('score'),
                             'last_bar': str(spec['signal_time']), 'order_id': fill.order_id})
        self.L.positions.append(pos)
        self.breakers.on_entry()
        try:
            self.broker.protect(pos)
        except Exception as e:                      # live only; surfaced, never silent
            self._log(f'  ⚠ could not place protective stop for {pos.symbol}: {e}')
        return pos

    # ── exits ───────────────────────────────────────────────────────────────
    def close(self, pos, fill, when):
        gross = pos.sign * (fill.price - pos.entry_price) * pos.qty
        costs = float(pos.meta.get('entry_costs', 0.0)) + fill.costs
        net = gross - costs
        risk = pos.risk_per_share * pos.qty
        self.L.state['cash'] = self.L.cash + float(pos.meta.get('margin', 0.0)) + gross - fill.costs
        self.L.positions.remove(pos)
        trade = {'trade_id': pos.meta.get('trade_id'), 'mode': self.mode, 'symbol': pos.symbol,
                 'side': pos.side, 'qty': pos.qty, 'entry_time': str(pos.entry_time),
                 'entry_price': round(pos.entry_price, 2), 'initial_stop': round(pos.initial_stop, 2),
                 'exit_time': str(pd.Timestamp(when)), 'exit_price': round(fill.price, 2),
                 'exit_reason': fill.note, 'bars_held': pos.bars_held,
                 'gross_pnl': round(gross, 2), 'costs': round(costs, 2), 'net_pnl': round(net, 2),
                 'net_r': round(net / risk, 4) if risk > 0 else None,
                 'trigger': pos.meta.get('trigger'), 'score': pos.meta.get('score')}
        self.L.append_trade(trade)
        self.breakers.on_exit(net)
        self.closed_today.append(trade)
        return trade

    # ── valuation ───────────────────────────────────────────────────────────
    def open_pnl(self, price_of):
        total = 0.0
        for p in self.L.positions:
            px = price_of(p.symbol)
            if px is not None:
                total += p.sign * (px - p.entry_price) * p.qty
        return total

    def equity(self, price_of):
        blocked = sum(float(p.meta.get('margin', 0.0)) for p in self.L.positions)
        return self.L.cash + blocked + self.open_pnl(price_of)

    def reserved_for_pending(self):
        return sum(float(p['spec']['qty']) * float(p['spec']['ref_price']) / self.leverage
                   for p in self.L.state.get('pending', []))

    def _log(self, msg):
        if self.log:
            self.log(msg)
