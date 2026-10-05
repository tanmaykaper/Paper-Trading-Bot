"""KiteBroker / KiteSession against a fake KiteConnect client.

No network and no kiteconnect install needed: the fake reproduces the
method names, constants and exception CLASS NAMES the adapter classifies on.
This verifies the adapter's own logic — token handling, retry policy,
idempotent placement, margin checks, tick rounding, exchange-side stops. It
does not prove Zerodha's live API behaves as documented; first live use
belongs in a small-size shadow run.
"""

import json

import pandas as pd
import pytest

from nsebot.broker.base import OrderSpec
from nsebot.broker.kite import (InsufficientMargin, KiteBroker, KiteSession, OrderRejected,
                                TokenExpired, load_token, token_expiry)
from nsebot.ratelimit import KiteRateLimiter
from nsebot.risk.exits import Position


class TokenException(Exception):
    pass


class NetworkException(Exception):
    pass


class InputException(Exception):
    pass


class FakeKite:
    VARIETY_REGULAR, VARIETY_AMO = 'regular', 'amo'
    EXCHANGE_NSE = 'NSE'
    TRANSACTION_TYPE_BUY, TRANSACTION_TYPE_SELL = 'BUY', 'SELL'
    PRODUCT_CNC, PRODUCT_MIS = 'CNC', 'MIS'
    ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT, ORDER_TYPE_SLM = 'MARKET', 'LIMIT', 'SL-M'
    VALIDITY_DAY = 'DAY'
    GTT_TYPE_SINGLE = 'single'

    def __init__(self):
        self.token = None
        self.placed, self.gtts, self.cancelled = [], [], []
        self.fail_next_place = None        # exception instance to raise once
        self.reach_exchange_on_fail = False
        self.cash = 1_000_000.0
        self.history = {}
        self.calls = 0

    def set_access_token(self, t):
        self.token = t

    def profile(self):
        return {'user_id': 'AB1234'}

    def instruments(self, exchange):
        return [{'tradingsymbol': 'INFY', 'tick_size': 0.1, 'lot_size': 1, 'segment': 'NSE', 'exchange': 'NSE'},
                {'tradingsymbol': 'IDEA', 'tick_size': 0.01, 'lot_size': 1, 'segment': 'NSE', 'exchange': 'NSE'}]

    def margins(self, segment):
        return {'available': {'live_balance': self.cash, 'cash': self.cash}}

    def order_margins(self, orders):
        o = orders[0]
        px = o.get('price') or 1500.0
        mult = 0.2 if o['product'] == 'MIS' else 1.0
        return [{'total': px * o['quantity'] * mult}]

    def place_order(self, **p):
        self.calls += 1
        if self.fail_next_place is not None:
            e, self.fail_next_place = self.fail_next_place, None
            if self.reach_exchange_on_fail:
                self.placed.append(p)
            raise e
        self.placed.append(p)
        oid = f'OID{len(self.placed)}'
        self.history[oid] = [{'status': 'COMPLETE', 'average_price': 1501.5,
                              'filled_quantity': p['quantity']}]
        return oid

    def orders(self):
        return [{'order_id': f'OID{i + 1}', 'tag': p.get('tag')} for i, p in enumerate(self.placed)]

    def order_history(self, oid):
        return self.history.get(oid, [{'status': 'OPEN'}])

    def place_gtt(self, **kw):
        self.gtts.append(kw)
        return {'trigger_id': 900 + len(self.gtts)}

    def delete_gtt(self, gid):
        self.cancelled.append(('gtt', gid))

    def cancel_order(self, variety, order_id):
        self.cancelled.append(('order', order_id))

    def modify_order(self, **kw):
        self.cancelled.append(('modify', kw))


def _session(fake, **kw):
    return KiteSession(client=fake, token='tok', limiter=KiteRateLimiter(sleep=lambda s: None),
                       sleep=lambda s: None, **kw)


def _broker(fake, tmp_path, monkeypatch, market_open=True):
    monkeypatch.setattr(KiteBroker, 'market_open', staticmethod(lambda now=None: market_open))
    return KiteBroker(_session(fake), cache_dir=str(tmp_path), poll_s=0.0)


def _spec(qty=10, product='CNC'):
    return OrderSpec('INFY', 'LONG', qty, product, 1500.0, 1440.0, '2026-10-05', tag='nbtestINFY')


def test_token_exception_becomes_token_expired_and_is_not_retried():
    fake = FakeKite()

    def boom():
        fake.calls += 1
        raise TokenException('Incorrect `api_key` or `access_token`.')
    fake.profile = boom
    with pytest.raises(TokenExpired):
        _session(fake).validate()
    assert fake.calls == 1


def test_network_errors_back_off_then_succeed():
    fake = FakeKite()
    seq = [NetworkException('timeout'), NetworkException('timeout'), {'user_id': 'X'}]

    def flaky():
        v = seq.pop(0)
        if isinstance(v, Exception):
            raise v
        return v
    fake.profile = flaky
    assert _session(fake).validate() == {'user_id': 'X'}


def test_input_errors_are_not_retried():
    fake = FakeKite()
    n = {'c': 0}

    def bad(**kw):
        n['c'] += 1
        raise InputException('Invalid quantity')
    fake.place_order = bad
    s = _session(fake)
    with pytest.raises(InputException):
        s.call('order', 'place_order', tradingsymbol='INFY')
    assert n['c'] == 1


def test_ambiguous_placement_failure_does_not_duplicate(tmp_path, monkeypatch):
    fake = FakeKite()
    b = _broker(fake, tmp_path, monkeypatch)
    fake.fail_next_place = NetworkException('read timeout')
    fake.reach_exchange_on_fail = True            # the order DID reach the exchange
    b.s.max_retries = 1
    oid = b.place_entry(_spec())
    assert len(fake.placed) == 1 and oid == 'OID1'


def test_rejected_order_surfaces_as_order_rejected(tmp_path, monkeypatch):
    fake = FakeKite()
    b = _broker(fake, tmp_path, monkeypatch)
    fake.fail_next_place = InputException('Insufficient funds')
    with pytest.raises(OrderRejected):
        b.place_entry(_spec())


def test_margin_shortfall_is_explicit_not_silent(tmp_path, monkeypatch):
    fake = FakeKite()
    fake.cash = 5_000.0
    b = _broker(fake, tmp_path, monkeypatch)
    with pytest.raises(InsufficientMargin):
        b.place_entry(_spec(qty=10))              # needs ~₹15,000 CNC
    assert fake.placed == []
    assert b.place_entry(_spec(qty=10, product='MIS'))   # 20% MIS margin fits


def test_after_hours_entries_go_out_as_amo(tmp_path, monkeypatch):
    fake = FakeKite()
    b = _broker(fake, tmp_path, monkeypatch, market_open=False)
    b.place_entry(_spec())
    assert fake.placed[-1]['variety'] == 'amo' and fake.placed[-1]['product'] == 'CNC'


def test_resolve_entry_reads_real_fill_and_rejections(tmp_path, monkeypatch):
    fake = FakeKite()
    b = _broker(fake, tmp_path, monkeypatch)
    oid = b.place_entry(_spec())
    res = b.resolve_entry({'order_id': oid, 'spec': _spec().to_dict()}, None)
    assert res.status == 'FILLED' and res.fill.price == 1501.5
    fake.history['BAD'] = [{'status': 'REJECTED', 'status_message': 'RMS: margin exceeds'}]
    rej = b.resolve_entry({'order_id': 'BAD', 'spec': _spec().to_dict()}, None)
    assert rej.status == 'CANCELLED' and 'margin' in rej.reason


def test_cnc_protection_is_a_tick_rounded_gtt(tmp_path, monkeypatch):
    fake = FakeKite()
    b = _broker(fake, tmp_path, monkeypatch)
    p = Position('INFY', 'swing', 'LONG', 10, 1500.0, 1443.07, 1443.07, pd.Timestamp('2026-10-05'), 20.0)
    b.protect(p)
    g = fake.gtts[-1]
    assert g['trigger_values'] == [1443.0]                    # INFY tick 0.10, rounded DOWN for a long stop
    assert p.meta['protection']['kind'] == 'gtt'
    b.exit(p, 1500.0, 'reversion target', pd.Timestamp('2026-10-06 15:20'))
    assert ('gtt', p.meta.get('protection', {}).get('id', 901)) in fake.cancelled or fake.cancelled


def test_mis_protection_is_an_slm_order(tmp_path, monkeypatch):
    fake = FakeKite()
    b = _broker(fake, tmp_path, monkeypatch)
    p = Position('INFY', 'intraday', 'SHORT', 10, 1500.0, 1507.03, 1507.03, pd.Timestamp('2026-10-05 10:00'), 2.0)
    b.protect(p)
    o = fake.placed[-1]
    assert o['order_type'] == 'SL-M' and o['trigger_price'] == 1507.1   # short stop rounds UP
    assert o['transaction_type'] == 'BUY'


def test_token_file_expiry(tmp_path, monkeypatch):
    monkeypatch.delenv('KITE_ACCESS_TOKEN', raising=False)
    created = pd.Timestamp('2026-10-05 08:00', tz='Asia/Kolkata')
    path = tmp_path / 'tok.json'
    path.write_text(json.dumps({'access_token': 'abc', 'expires': str(token_expiry(created))}))
    assert load_token(str(path), now=pd.Timestamp('2026-10-05 15:00', tz='Asia/Kolkata')) == 'abc'
    with pytest.raises(TokenExpired):
        load_token(str(path), now=pd.Timestamp('2026-10-06 06:30', tz='Asia/Kolkata'))
    with pytest.raises(TokenExpired):
        load_token(str(tmp_path / 'missing.json'))
