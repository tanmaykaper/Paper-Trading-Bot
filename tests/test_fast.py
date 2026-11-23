import numpy as np
import pandas as pd

from nsebot.research import fast
from conftest import daily_frame


def _res(r2, maxdd=-0.40, untouched=(0.2,) * 5, tail=None):
    r2 = np.asarray(r2, dtype=float)
    if tail is not None:                       # a spread of windows with a chosen 10th percentile
        r2 = np.r_[np.full(20, tail), np.full(80, r2.max())]
    return {'r2': r2, 'r3': r2, 'untouched': np.asarray(untouched, dtype=float), 'maxdd': maxdd,
            'full': 0.2, 'ahead6m': 0.6, 'ir': 1.0, 'years_t2': 4.0}


B = fast.BASELINE


def test_rule_one_drops_a_design_deeper_than_the_latch():
    results = {B: _res([0.20]), 'F6 constant 1.3x margin': _res([0.60], maxdd=-0.70),
               'F1 6-1 momentum': _res([0.21])}
    win, deploy, _ = fast.decide(results)
    assert win == 'F1 6-1 momentum' and deploy == B          # +1 point: rule 3 keeps the baseline


def test_rule_three_needs_three_points_a_no_worse_tail_and_three_phases():
    base = _res([0.20], tail=0.05)
    good = _res([0.25], tail=0.06, untouched=(0.3, 0.3, 0.3, 0.1, 0.1))
    assert fast.decide({B: base, 'F2 x': good})[1] == 'F2 x'
    fat_tail = _res([0.25], tail=0.01, untouched=(0.3, 0.3, 0.3, 0.1, 0.1))
    assert fast.decide({B: base, 'F2 x': fat_tail})[1] == B
    narrow = _res([0.25], tail=0.06, untouched=(0.3, 0.3, 0.1, 0.1, 0.1))
    assert fast.decide({B: base, 'F2 x': narrow})[1] == B
    small = _res([0.22], tail=0.06, untouched=(0.3,) * 5)
    assert fast.decide({B: base, 'F2 x': small})[1] == B


def test_rule_four_keeps_a_margin_winner_off_the_paper_book():
    base = _res([0.20], tail=0.05)
    margin = _res([0.30], tail=0.08, untouched=(0.3,) * 5)
    win, deploy, why = fast.decide({B: base, 'F5 volatility-targeted, up to 1.5x margin': margin})
    assert win.startswith('F5') and deploy == B and 'MTF' in why[-1]


def test_overlay_scales_returns_and_charges_margin_by_calendar_day():
    idx = pd.bdate_range('2020-01-01', periods=200)
    e = pd.Series(100.0 * 1.01 ** np.maximum(np.arange(200) - 4, 0), index=idx)   # cash, then +1% a session
    same = fast.overlay(e, 'lev', 1.0)
    assert np.allclose(same.to_numpy(), e.to_numpy())
    double = fast.overlay(e, 'lev', 2.0, rate=0.0, cost=0.0)
    assert np.isclose(double.iloc[-1] / double.iloc[4], 1.02 ** (len(e) - 5))
    charged = fast.overlay(e, 'lev', 2.0, rate=0.15, cost=0.0)
    assert charged.iloc[-1] < double.iloc[-1]


def test_volatility_target_cuts_exposure_when_the_book_swings_harder():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2019-01-01', periods=400)
    r = np.r_[rng.normal(0.001, 0.01, 200), rng.normal(0.001, 0.04, 200)]
    e = pd.Series(100 * np.cumprod(1 + r), index=idx)
    target = float(fast.realised_vol(e).iloc[:200].median())
    scaled = fast.overlay(e, 'vol', 1.0, target=target, cost=0.0)
    late = scaled.pct_change().iloc[300:].std() / e.pct_change().iloc[300:].std()
    assert late < 0.5                                          # the 4x-volatile half runs at ~1/4 exposure


def test_run_end_to_end_on_fake_sources(tmp_path):
    rng = np.random.default_rng(3)
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

    report = fast.run(str(tmp_path), fetch=fetch, get_list=lambda: (syms, 'fake list'))
    for name, *_ in fast.DESIGNS:
        assert f'| {name}' in report
    assert '**Deploy: ' in report and '| 2 years |' in report and (tmp_path / 'FAST.md').exists()
