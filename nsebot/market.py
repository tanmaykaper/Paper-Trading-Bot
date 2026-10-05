"""NSE cash-market structure: tick sizes, price rounding, session clock.

Tick size is NOT a constant ₹0.05 (the V2 landmine — AUTOPSY §3). NSE sets
it per instrument, by price band. The authoritative value is the
`tick_size` field of Kite's instrument master; when a KiteInstruments table
is loaded it wins. Without one, the price-band schedule below is used. NSE
revises these bands periodically — treat the table as a fallback and verify
it against the current exchange circular before trading real money.

Rounding is always in the CONSERVATIVE direction for the order's purpose:
  buy limit   down      sell limit   up
  long stop   down      short stop   up       (further from price, never closer)
"""

import math
from datetime import datetime, time

import pandas as pd

IST = 'Asia/Kolkata'

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)

# (upper price bound exclusive, tick) — fallback schedule for NSE equities.
PRICE_BAND_TICKS = [
    (250.0, 0.01),
    (1_000.0, 0.05),
    (5_000.0, 0.10),
    (10_000.0, 0.50),
    (20_000.0, 1.00),
    (float('inf'), 5.00),
]


def band_tick(price):
    for upper, tick in PRICE_BAND_TICKS:
        if price < upper:
            return tick
    return PRICE_BAND_TICKS[-1][1]


class TickTable:
    """symbol -> tick size. Loaded from Kite's instrument master when
    available; otherwise every lookup falls back to the price band."""

    def __init__(self, ticks=None, lots=None):
        self.ticks = dict(ticks or {})
        self.lots = dict(lots or {})

    @classmethod
    def from_kite_instruments(cls, instruments):
        ticks, lots = {}, {}
        for ins in instruments:
            if ins.get('segment') not in (None, 'NSE') and ins.get('exchange') not in (None, 'NSE'):
                continue
            sym = ins.get('tradingsymbol')
            if not sym:
                continue
            ticks[sym] = float(ins.get('tick_size') or 0.0) or None
            lots[sym] = int(ins.get('lot_size') or 1)
        return cls({k: v for k, v in ticks.items() if v}, lots)

    def tick(self, symbol, price):
        return self.ticks.get(symbol) or band_tick(float(price))

    def lot(self, symbol):
        return max(int(self.lots.get(symbol, 1)), 1)


def round_to_tick(price, tick, mode='nearest'):
    if price is None or not math.isfinite(price) or tick <= 0:
        return price
    units = price / tick
    if mode == 'down':
        n = math.floor(round(units, 6))
    elif mode == 'up':
        n = math.ceil(round(units, 6))
    else:
        n = round(units)
    decimals = max(0, -int(math.floor(math.log10(tick)))) if tick < 1 else 0
    return round(n * tick, decimals + 2)


def round_stop(price, side, tick):
    """Long stops round down, short stops round up — never tighter than planned."""
    return round_to_tick(price, tick, 'down' if side == 'LONG' else 'up')


def round_limit(price, side, tick):
    """Buy limits round down, sell limits round up — never pay more than planned."""
    return round_to_tick(price, tick, 'down' if side in ('LONG', 'BUY') else 'up')


def now_ist():
    return pd.Timestamp.now(tz=IST)


def is_weekday(ts=None):
    ts = pd.Timestamp(ts) if ts is not None else now_ist()
    return ts.weekday() < 5


def session_bounds(day=None):
    day = pd.Timestamp(day if day is not None else now_ist())
    if day.tzinfo is None:
        day = day.tz_localize(IST)
    d = day.tz_convert(IST).normalize()
    return (d + pd.Timedelta(hours=SESSION_OPEN.hour, minutes=SESSION_OPEN.minute),
            d + pd.Timedelta(hours=SESSION_CLOSE.hour, minutes=SESSION_CLOSE.minute))


def at_time(day, t):
    d = pd.Timestamp(day)
    if d.tzinfo is None:
        d = d.tz_localize(IST)
    return d.tz_convert(IST).normalize() + pd.Timedelta(hours=t.hour, minutes=t.minute)


def is_valid_tick_price(price, tick):
    return abs(round(price / tick) * tick - price) < 1e-6
