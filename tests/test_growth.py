import numpy as np
import pandas as pd

from nsebot.research import growth
from conftest import daily_frame


def _res(r3, maxdd=-0.40, untouched=(0.2,) * 5):
    return {'r3': np.asarray(r3, dtype=float), 'r5': np.asarray(r3, dtype=float),
            'untouched': np.asarray(untouched, dtype=float), 'maxdd': maxdd, 'full': 0.2}


def test_rule_one_drops_a_design_deeper_than_the_latch():
    results = {('B4', 10): _res([0.20]), ('V4', 5): _res([0.60], maxdd=-0.62), ('V4', 7): _res([0.21])}
    win, deploy, _ = growth.decide(results)
    assert win == ('V4', 7) and deploy == ('B4', 10)          # +1 point: rule 3 keeps the baseline


def test_rule_three_needs_two_points_and_three_untouched_phases():
    base = _res([0.20], untouched=(0.2,) * 5)
    better = _res([0.25], untouched=(0.3, 0.3, 0.3, 0.1, 0.1))
    assert growth.decide({('B4', 10): base, ('V4', 10): better})[1] == ('V4', 10)
    narrow = _res([0.25], untouched=(0.3, 0.3, 0.1, 0.1, 0.1))
    assert growth.decide({('B4', 10): base, ('V4', 10): narrow})[1] == ('B4', 10)


def test_rolling_cagr_reads_every_window_once():
    idx = pd.bdate_range('2018-07-02', periods=252 * 4)
    e = pd.Series(100 * 1.1 ** (np.arange(len(idx)) / 252), index=idx)
    r = growth.rolling_cagr([e], 3, start=idx[0], step=21)
    assert len(r) == len(range(0, len(idx) - 756, 21)) and np.allclose(r, 0.1)


def test_run_end_to_end_on_fake_sources(tmp_path):
    rng = np.random.default_rng(2)
    dates = pd.bdate_range('2016-07-01', pd.Timestamp.today())
    T = len(dates)
    syms = [f'S{i:03d}' for i in range(40)]

    def frame(sym):
        drift = 0.0002 + 0.00004 * (sum(map(ord, sym)) % 20)
        return daily_frame(300 * np.exp(np.cumsum(rng.normal(drift, 0.018, T))), 400_000.0, start='2016-07-01')

    universe = {s: frame(s) for s in syms}

    def fetch(symbols, start, end):
        out = {s: universe[s] for s in symbols if s in universe}
        if '^NSEI' in symbols:
            out['^NSEI'] = daily_frame(10_000 * np.exp(np.cumsum(np.full(T, 0.0004))), start='2016-07-01')
        return out

    report = growth.run(str(tmp_path), fetch=fetch, get_list=lambda: (syms, 'fake list'))
    assert '| B4 top 10 |' in report and '| V4 top 5 |' in report and '**Deploy: ' in report
    assert '| 3 years |' in report and (tmp_path / 'GROWTH.md').exists()
