import numpy as np

from nsebot.regime import RegimeDial
from conftest import daily_frame


def _universe(drift, n=20, bars=200):
    rng = np.random.default_rng(7)
    return {f'S{i}': daily_frame(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, bars))))
            for i in range(n)}


def test_strong_tape_trades_full_size():
    idx = daily_frame(20000 * np.exp(np.cumsum(np.full(200, 0.002))))
    r = RegimeDial().today(idx, _universe(0.003))
    assert r['size_mult'] > 0.9 and r['max_new_entries'] >= 3


def test_weak_tape_shrinks_but_never_stops():
    idx = daily_frame(20000 * np.exp(np.cumsum(np.full(200, -0.003))))
    r = RegimeDial().today(idx, _universe(-0.004))
    assert r['size_mult'] == 0.5
    assert r['max_new_entries'] == 1          # V2 would have said 0 here


def test_works_without_breadth():
    idx = daily_frame(20000 * np.exp(np.cumsum(np.full(200, 0.001))))
    r = RegimeDial().today(idx)
    assert r['breadth'] is None and 0.5 <= r['size_mult'] <= 1.0
