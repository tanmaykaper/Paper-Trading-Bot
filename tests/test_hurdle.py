import numpy as np
import pandas as pd

from nsebot.research.hurdle import bot_variants, curve, smallcap_funds
from conftest import daily_frame


def _fake_fetch(url, params=None):
    if url.endswith('/mf/search'):
        return [{'schemeCode': 1, 'schemeName': 'Alpha Small Cap Fund - Direct Plan - Growth'},
                {'schemeCode': 2, 'schemeName': 'Alpha Small Cap Fund - Regular Plan - Growth'},
                {'schemeCode': 3, 'schemeName': 'Beta Smallcap Fund Direct Growth IDCW'},
                {'schemeCode': 4, 'schemeName': 'Gamma Nifty Smallcap 250 Index Fund - Direct - Growth'}]
    code = int(url.rsplit('/', 1)[-1])
    days = pd.bdate_range('2024-01-01', periods=500)
    nav = 100 * np.exp(np.cumsum(np.full(500, 0.0008 * code)))
    return {'data': [{'date': d.strftime('%d-%m-%Y'), 'nav': f'{v:.4f}'} for d, v in zip(days, nav)]}


def test_only_direct_growth_smallcap_schemes_are_kept():
    funds = smallcap_funds(fetch=_fake_fetch)
    assert set(funds) == {'Alpha Small Cap Fund - Direct Plan - Growth',
                          'Gamma Nifty Smallcap 250 Index Fund - Direct - Growth'}


def test_curve_stats():
    s = pd.Series([100.0, 110.0, 99.0, 121.0] * 10,
                  index=pd.bdate_range('2025-01-01', periods=40))
    st = curve(s, s.index[0], s.index[-1])
    assert abs(st['ret'] - 0.21) < 1e-9 and st['maxdd'] < 0


def test_bot_variants_run_on_synthetic_data():
    rng = np.random.default_rng(3)
    u = {f'S{i}': daily_frame(300 * np.exp(np.cumsum(rng.normal(0.0008 + 0.0002 * i, 0.015, 420))),
                              400_000.0) for i in range(14)}
    idx = daily_frame(20000 * np.exp(np.cumsum(np.full(420, 0.0006))))
    curves, P = bot_variants(u, idx, P_start := pd.Timestamp(u['S0']['datetime'].iloc[260]))
    assert {'M1 12-1 momentum, top 10, Nifty > 200-day SMA filter',
            'C1 deployed dip-reversion engine (Phase 4b)'} <= set(curves)
    assert np.isfinite(curves['M4 12-1 momentum, top 10, no filter (reference)'].iloc[-1])
