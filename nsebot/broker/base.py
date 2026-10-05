"""Broker interface shared by paper and live execution.

The engine never knows which one it is driving. Entries follow a two-step
protocol that is identical for both:

  1. place_entry(spec)        at decision time (after a bar closes)
                               paper: records the intent
                               kite:  sends the order (AMO for swing, market
                                      for intraday) and returns its id
  2. resolve_entry(pending, bar)  on the next cycle
                               paper: fills at that bar's open (+ slippage), or
                                      cancels if the open is already through
                                      the stop — the research rule
                               kite:  reads the order's real status/fill price

Exits are immediate: the exit engine has already decided the price (a stop,
a gap open, a close, the square-off), so paper books it there and live sends
a market order.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class OrderSpec:
    symbol: str
    side: str                 # 'LONG' | 'SHORT'
    qty: int
    product: str              # 'CNC' | 'MIS'
    ref_price: float          # signal close
    stop: float
    signal_time: str
    tag: str = ''
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return {k: getattr(self, k) for k in ('symbol', 'side', 'qty', 'product', 'ref_price',
                                              'stop', 'signal_time', 'tag', 'meta')}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ('symbol', 'side', 'qty', 'product', 'ref_price', 'stop',
                                        'signal_time')}, tag=d.get('tag', ''), meta=d.get('meta', {}))


@dataclass
class Fill:
    symbol: str
    action: str               # 'BUY' | 'SELL'
    qty: int
    price: float
    costs: float
    time: pd.Timestamp
    order_id: str = ''
    note: str = ''


@dataclass
class EntryResolution:
    status: str               # 'FILLED' | 'WAIT' | 'CANCELLED'
    fill: Fill = None
    reason: str = ''


class Broker(ABC):
    name = 'base'
    is_live = False

    @abstractmethod
    def place_entry(self, spec: OrderSpec) -> str:
        ...

    @abstractmethod
    def resolve_entry(self, pending: dict, bar) -> EntryResolution:
        ...

    @abstractmethod
    def exit(self, position, price, reason, when) -> Fill:
        ...

    def protect(self, position):            # live: place a resting stop (GTT / SL-M)
        return None

    def update_protection(self, position):  # live: move it when the stop trails
        return None

    def cancel_protection(self, position):
        return None

    def available_cash(self, ledger_cash):
        return float(ledger_cash)


def entry_action(side):
    return 'BUY' if side == 'LONG' else 'SELL'


def exit_action(side):
    return 'SELL' if side == 'LONG' else 'BUY'
