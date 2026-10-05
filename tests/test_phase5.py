import numpy as np
import pandas as pd

from nsebot.research import phase5
from conftest import daily_frame


def test_window_is_strictly_before_the_selection_data():
    assert pd.Timestamp(phase5.FETCH_END) < pd.Timestamp('2023-08-14')
    assert phase5.WIN_END < pd.Timestamp('2023-08-14')
    assert len(phase5.YEARS) == 5


def test_designs_run_on_synthetic_history():
    rng = np.random.default_rng(5)
    u = {}
    for i in range(14):
        r = rng.normal(0.0008 + 0.0002 * i, 0.016, 520)
        for k in rng.choice(np.arange(260, 510), 4, replace=False):
            r[k:k + 3] = -0.04
        u[f'S{i}'] = daily_frame(300 * np.exp(np.cumsum(r)), 400_000.0, start='2017-01-02')
    idx = daily_frame(10000 * np.exp(np.cumsum(np.full(520, 0.0005))), start='2017-01-02')
    curves, P, latches = phase5.designs(u, idx)
    assert len(curves) == 8
    assert all(np.isfinite(c.dropna().iloc[-1]) for c in curves.values())
    assert len(phase5.yearly(curves['D2 12-1 momentum top 5 + 200-day filter'])) == 5
    assert set(latches) == {'D1 deployed engine (dip reversion, Phase 4b)',
                            'E1 concentrated dip: 33% x 3 slots', 'E2 broader dip: 5% trigger'}


def test_leveraged_sleeve_pays_financing_only_while_invested(monkeypatch):
    dates = pd.bdate_range('2018-01-01', periods=300)
    flat = pd.Series(50_000.0, index=dates)                   # D2 in cash the whole time

    import nsebot.backtest as bt
    monkeypatch.setattr(bt, 'run_backtest', lambda *a, **k: (None, flat, [], type('L', (), {'dir': '/nonexistent/x'})()))
    monkeypatch.setattr(phase5, 'simulate_rotation', lambda *a, **k: (flat, None))
    monkeypatch.setattr(phase5, 'Panel', lambda u: type('P', (), {
        'df': {'close': pd.DataFrame({'A': np.ones(300)}, index=dates)}, 'dates': dates,
        'liquid': np.ones((300, 1), dtype=bool)})())
    idx = pd.DataFrame({'datetime': dates, 'close': np.linspace(100, 130, 300)})
    curves, _, _ = phase5.designs({}, idx)
    e4 = curves['E4 leveraged momentum: D2 at 1.5x, 15%/yr financing']
    assert e4.iloc[-1] == 50_000.0
