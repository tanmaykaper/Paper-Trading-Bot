import json

import numpy as np
import pandas as pd

from nsebot.research import checkpoint
from conftest import daily_frame


def test_base_rates_count_windows_ahead_and_losing():
    idx = pd.bdate_range('2018-07-02', periods=600)
    book = pd.Series(1.02 ** (np.arange(600) / 21), index=idx)          # +2% a month
    mkt = pd.Series(1.01 ** (np.arange(600) / 21), index=idx)           # +1% a month
    ahead, lose, excess = checkpoint.base_rates([book], mkt, 126, start=idx[0])
    assert ahead == 1.0 and lose == 0.0 and excess > 0


def test_flagged_sessions_reads_the_oldest_ca_suspect(tmp_path):
    dates = pd.bdate_range('2026-11-02', periods=15)
    with open(tmp_path / 'state.json', 'w') as fh:
        json.dump({'positions': [{'symbol': 'A', 'meta': {}},
                                 {'symbol': 'B', 'meta': {'ca_suspect': {'date': str(dates[2].date())}}}]}, fh)
    assert checkpoint.flagged_sessions(str(tmp_path), dates) == 12
    assert not checkpoint.latch_tripped(str(tmp_path))
    (tmp_path / 'BREAKER_TRIPPED_momentum').write_text('x')
    assert checkpoint.latch_tripped(str(tmp_path))


def test_run_end_to_end_on_fake_sources(tmp_path):
    rng = np.random.default_rng(4)
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

    state = tmp_path / 'state'
    state.mkdir()
    pd.DataFrame({'session': ['2026-10-05', '2026-11-02', '2026-11-03', str(dates[-1].date())],
                  'equity': [50_000.0, 45_000.0, 145_000.0, 150_000.0]}).to_csv(state / 'equity.csv', index=False)
    with open(state / 'state.json', 'w') as fh:
        json.dump({'initial_cash': 50_000.0, 'positions': [],
                   'flows': [{'session': '2026-11-03', 'amount': 100_000.0}]}, fh)
    report = checkpoint.run(str(tmp_path / 'out'), fetch=fetch, get_list=lambda: (syms, 'fake list'),
                            state_dir=str(state))
    for heading in ('## 1. Does the live bot', '## 2. The V4 forward test', '## 3. What 6, 12 or 24 months',
                    '## 4. The checkpoint'):
        assert heading in report
    assert 'net of ₹1,00,000 moved in' in report or 'net of ₹100,000 moved in' in report
    assert '| 6 months |' in report and (tmp_path / 'out' / 'CHECKPOINT.md').exists()
