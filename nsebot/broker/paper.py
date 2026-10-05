"""Paper execution that fills exactly the way the research measured.

Entries fill at the open of the first bar AFTER the signal bar, plus a small
slippage haircut; a bar that opens beyond the stop cancels the entry (the
setup broke before it could be bought). Exits book at the price the exit
engine decided. Prices are snapped to the instrument's tick in the adverse
direction, so a paper fill is always a price the exchange could print.
Costs are Zerodha's itemised schedule, charged per leg.
"""

import pandas as pd

from ..costs import DEFAULT_CHARGES
from ..market import TickTable, round_to_tick
from .base import Broker, EntryResolution, Fill, entry_action, exit_action


class PaperBroker(Broker):
    name = 'paper'
    is_live = False

    def __init__(self, slippage_bps=5.0, charges=DEFAULT_CHARGES, ticks=None):
        self.slip = float(slippage_bps) / 1e4
        self.charges = charges
        self.ticks = ticks or TickTable()
        self._seq = 0

    def place_entry(self, spec):
        self._seq += 1
        return f'PAPER-{spec.symbol}-{self._seq}'

    def resolve_entry(self, pending, bar):
        spec = pending['spec']
        if bar is None:
            return EntryResolution('WAIT', reason='no bar after the signal yet')
        o = float(bar['open'])
        long_ = spec['side'] == 'LONG'
        if (long_ and o <= spec['stop']) or (not long_ and o >= spec['stop']):
            return EntryResolution('CANCELLED', reason=f'opened {o:.2f}, already through the '
                                                       f'{spec["stop"]:.2f} stop')
        tick = self.ticks.tick(spec['symbol'], o)
        px = o * (1 + self.slip) if long_ else o * (1 - self.slip)
        px = round_to_tick(px, tick, 'up' if long_ else 'down')
        value = px * spec['qty']
        costs = self._leg_cost(spec['product'], value, 'buy' if long_ else 'sell', dp=False)
        return EntryResolution('FILLED', Fill(spec['symbol'], entry_action(spec['side']), spec['qty'],
                                              px, costs, pd.Timestamp(bar['datetime']),
                                              pending.get('order_id', ''), 'next-bar open'))

    def exit(self, position, price, reason, when):
        long_ = position.side == 'LONG'
        tick = self.ticks.tick(position.symbol, price)
        # Stop and gap fills are already pessimistic; market exits pay slippage.
        if 'stop' not in reason:
            price = price * (1 - self.slip) if long_ else price * (1 + self.slip)
        px = round_to_tick(price, tick, 'down' if long_ else 'up')
        value = px * position.qty
        product = 'MIS' if position.mode == 'intraday' else 'CNC'
        costs = self._leg_cost(product, value, 'sell' if long_ else 'buy', dp=True)
        return Fill(position.symbol, exit_action(position.side), position.qty, px, costs,
                    pd.Timestamp(when), '', reason)

    def _leg_cost(self, product, value, leg, dp):
        c = self.charges
        if product == 'MIS':
            return c.mis(value, 0.0) if leg == 'buy' else c.mis(0.0, value)
        if leg == 'buy':
            return c.cnc(value, 0.0, dp_charged=False)
        return c.cnc(0.0, value, dp_charged=dp)
