import json

import pandas as pd
import pytest

from nsebot.broker.paper import PaperBroker
from nsebot.ledger import Ledger
from nsebot.market import TickTable, band_tick, is_valid_tick_price, round_limit, round_stop
from nsebot.ratelimit import RateLimitExceeded, TokenBucket, WindowCounter
from nsebot.risk.exits import Position


# ── market structure ─────────────────────────────────────────────────────────
@pytest.mark.parametrize('price,tick', [(120, 0.01), (600, 0.05), (1443, 0.10), (7000, 0.50),
                                        (15000, 1.00), (30000, 5.00)])
def test_price_band_ticks(price, tick):
    assert band_tick(price) == tick


def test_rounding_is_always_conservative():
    assert round_stop(1443.07, 'LONG', 0.10) == 1443.0     # long stop: further down
    assert round_stop(1443.03, 'SHORT', 0.10) == 1443.1    # short stop: further up
    assert round_limit(612.34, 'BUY', 0.05) == 612.30      # never pay more
    assert round_limit(612.31, 'SELL', 0.05) == 612.35     # never sell for less
    assert is_valid_tick_price(round_stop(98.7654, 'LONG', 0.01), 0.01)


def test_instrument_master_beats_the_band_table():
    t = TickTable.from_kite_instruments([{'tradingsymbol': 'XYZ', 'tick_size': 0.05, 'lot_size': 1,
                                          'segment': 'NSE'}])
    assert t.tick('XYZ', 1500) == 0.05 and t.tick('OTHER', 1500) == 0.10


# ── rate limiting ────────────────────────────────────────────────────────────
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_token_bucket_paces_to_the_rate():
    c = FakeClock()
    b = TokenBucket(3.0, burst=1, clock=c, sleep=c.sleep)
    for _ in range(7):
        b.acquire()
    assert c.t == pytest.approx(2.0, abs=1e-6)           # 6 waits of 1/3 s


def test_daily_order_cap_raises_instead_of_waiting_a_day():
    c = FakeClock()
    w = WindowCounter(2, 86_400, clock=c, sleep=c.sleep, block=False)
    w.acquire()
    w.acquire()
    with pytest.raises(RateLimitExceeded):
        w.acquire()


# ── ledger ───────────────────────────────────────────────────────────────────
def test_ledger_round_trips_positions_and_is_atomic(tmp_path):
    L = Ledger(str(tmp_path), 'swing', 50_000)
    L.positions.append(Position('INFY', 'swing', 'LONG', 5, 1500.0, 1440.0, 1450.0,
                                pd.Timestamp('2026-10-05'), 20.0, meta={'trade_id': 'S1'}))
    L.state['pending'].append({'spec': {'symbol': 'TCS'}, 'order_id': 'P1'})
    L.save()
    assert not any(p.name.startswith('.tmp_') for p in (tmp_path / 'swing').iterdir())
    L2 = Ledger(str(tmp_path), 'swing', 99_999)               # fresh process, committed state
    assert L2.cash == 50_000 and L2.positions[0].stop == 1450.0
    assert L2.positions[0].meta['trade_id'] == 'S1' and L2.state['pending'][0]['order_id'] == 'P1'


def test_realised_r_reads_net_r(tmp_path):
    L = Ledger(str(tmp_path), 'swing', 50_000)
    L.append_trade({'trade_id': 'a', 'net_pnl': 100, 'net_r': 0.5})
    L.append_trade({'trade_id': 'b', 'net_pnl': -50, 'net_r': -0.25})
    assert L.realised_r() == [0.5, -0.25]


# ── paper broker ─────────────────────────────────────────────────────────────
def _pending(side='LONG', stop=95.0, product='CNC'):
    return {'order_id': 'X', 'spec': {'symbol': 'ABC', 'side': side, 'qty': 10, 'product': product,
                                      'ref_price': 100.0, 'stop': stop, 'signal_time': '2026-10-05'}}


def test_paper_entry_fills_next_open_with_slippage_on_a_valid_tick():
    b = PaperBroker(slippage_bps=10)
    r = b.resolve_entry(_pending(), {'open': 101.0, 'datetime': '2026-10-06'})
    assert r.status == 'FILLED' and r.fill.price == pytest.approx(101.11, abs=1e-9)   # 101.101 -> up to 0.01 tick
    assert r.fill.costs > 0


def test_paper_entry_cancels_when_open_is_through_the_stop():
    r = PaperBroker().resolve_entry(_pending(stop=95.0), {'open': 94.0, 'datetime': '2026-10-06'})
    assert r.status == 'CANCELLED'
    s = PaperBroker().resolve_entry(_pending(side='SHORT', stop=105.0), {'open': 106.0, 'datetime': 'x'})
    assert s.status == 'CANCELLED'


def test_paper_stop_exit_gets_no_extra_slippage_but_market_exit_does():
    b = PaperBroker(slippage_bps=10)
    p = Position('ABC', 'swing', 'LONG', 10, 100.0, 95.0, 95.0, pd.Timestamp('2026-10-05'), 2.0)
    assert b.exit(p, 95.0, 'stop', '2026-10-06').price == 95.0
    assert b.exit(p, 110.0, 'reversion target', '2026-10-06').price < 110.0
    mis = Position('ABC', 'intraday', 'LONG', 10, 100.0, 99.0, 99.0, pd.Timestamp('2026-10-05'), 1.0)
    cnc_cost = b.exit(p, 110.0, 'x', '2026-10-06').costs
    mis_cost = b.exit(mis, 110.0, 'x', '2026-10-06').costs
    assert mis_cost < cnc_cost                                   # no DP charge, lower STT on MIS
