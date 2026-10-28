import json

import numpy as np
import pandas as pd

from nsebot.research import diagnosis
from nsebot.research.experiments import Panel, locked_bars, simulate_rotation
from conftest import daily_frame


def _world(n=30, T=600, seed=3, start='2022-01-03'):
    rng = np.random.default_rng(seed)
    u = {}
    for i in range(n):
        drift = rng.normal(0.0006, 0.0012)
        u[f'S{i:02d}'] = daily_frame(rng.uniform(150, 450) * np.exp(np.cumsum(rng.normal(drift, 0.02, T))),
                                     1_000_000.0, start=start)
    return u


def _demerge(df, k, keep=0.1):
    """From bar k on, the stock trades at `keep` of its old price: the gap a
    demerger leaves when Yahoo does not adjust the history."""
    df = df.copy()
    for col in ('open', 'high', 'low', 'close'):
        df.loc[k:, col] = df.loc[k:, col] * keep
    return df


def test_rotate_without_a_buy_filter_is_the_research_simulator():
    P = Panel(_world())
    c = P.df['close']
    mom = (c.shift(21) / c.shift(252) - 1).to_numpy()
    buy_ok, sell_ok = locked_bars(P)
    ref, ref_costs = simulate_rotation(P, mom, P.liquid, n=5, buy_ok=buy_ok, sell_ok=sell_ok)
    eq, costs, held = diagnosis.rotate(P, mom, P.liquid, n=5, buy_ok=buy_ok, sell_ok=sell_ok, log=True)
    assert np.allclose(eq.to_numpy(), ref.to_numpy(), equal_nan=True) and costs == ref_costs
    assert held.shape == (len(P.dates), len(P.symbols)) and (held > 0).sum(axis=1).max() == 5


def test_a_buy_filter_blocks_new_buys_but_never_forces_a_sale():
    P = Panel(_world())
    c = P.df['close']
    mom = (c.shift(21) / c.shift(252) - 1).to_numpy()
    _, _, free = diagnosis.rotate(P, mom, P.liquid, n=5, log=True)
    T = len(P.dates)
    late = np.zeros_like(P.liquid)
    late[T - 100:] = True                                  # buying allowed only in the last 100 sessions
    _, _, gated = diagnosis.rotate(P, mom, P.liquid, n=5, buy_filter=late, log=True)
    assert (gated[:T - 99] == 0).all() and (gated[-1] > 0).any()
    blocked = np.zeros_like(P.liquid)
    blocked[:T - 100] = True                               # buying stops 100 sessions before the end
    _, _, kept = diagnosis.rotate(P, mom, P.liquid, n=5, buy_filter=blocked, log=True)
    assert (kept[T - 101] > 0).sum() == 5                  # the book was full when buying stopped
    assert ((kept[T - 99:] > 0) & (kept[T - 101] == 0)).sum() == 0     # nothing new after that


def test_a_demerger_gap_is_found_booked_as_a_loss_and_kept_once_adjusted():
    u = _world()
    T = len(u['S00'])
    k = T - 50
    winner = 'S07'
    u[winner] = daily_frame(200 * np.exp(np.cumsum(np.full(T, 0.004))), 1_000_000.0, start='2022-01-03')
    u[winner] = _demerge(u[winner], k)
    P = Panel(u)
    ev = diagnosis.ca_events(P)
    j = P.symbols.index(winner)
    assert ev[:, j].sum() == 1 and P.dates[np.flatnonzero(ev[:, j])[0]] == P.dates[k]
    assert ev.sum() == 1                                   # ordinary 2% days are never flagged

    c = P.df['close']
    mom = (c.shift(21) / c.shift(252) - 1).to_numpy()
    eq, _, held = diagnosis.rotate(P, mom, P.liquid, n=5, log=True)
    hits = diagnosis.ca_hits(P, ev, held)
    assert [h[1] for h in hits] == [winner] and hits[0][3] > 0 and hits[0][4] < -0.85

    Q = diagnosis.adjusted(P, ev)
    assert np.allclose(Q.a['close'][k:, j], P.a['close'][k:, j])          # after the gap: untouched
    assert abs(Q.a['open'][k, j] / Q.a['close'][k - 1, j] - 1) < 1e-9     # no gap any more
    cq = Q.df['close']
    eq_adj, _, _ = diagnosis.rotate(Q, (cq.shift(21) / cq.shift(252) - 1).to_numpy(), P.liquid, n=5)
    assert eq_adj.iloc[-1] > eq.iloc[-1] + 0.8 * hits[0][3]


def test_scores_are_oriented_higher_is_better():
    u = _world(n=40, T=400)
    rng = np.random.default_rng(1)
    u['SMOOTH'] = daily_frame(200 * np.exp(np.cumsum(np.full(400, 0.003))), 1_000_000.0, start='2022-01-03')
    jumps = np.zeros(400)
    jumps[::40] = 0.11                                     # same 12-1 gain, made in ten jumps
    u['JUMPY'] = daily_frame(200 * np.exp(np.cumsum(jumps + rng.normal(0, 0.002, 400))), 1_000_000.0,
                             start='2022-01-03')
    P = Panel(u)
    s = diagnosis.scores(P, P.liquid)
    t = len(P.dates) - 1
    a, b = P.symbols.index('SMOOTH'), P.symbols.index('JUMPY')
    assert s['smoothness (−frog-in-the-pan ID)'][t, a] > s['smoothness (−frog-in-the-pan ID)'][t, b]
    assert s['−largest daily gain, 63 sessions'][t, a] > s['−largest daily gain, 63 sessions'][t, b]
    assert s['12-1 / volatility'][t, a] > s['12-1 / volatility'][t, b]
    assert s['52-week-high proximity'][t, a] == 1.0
    names = [v[0] for v in diagnosis.variants(P, P.liquid, s)]
    assert names[0].startswith('B4') and len(names) == 9


def test_run_end_to_end_on_fake_sources(tmp_path):
    rng = np.random.default_rng(5)
    dates = pd.bdate_range('2016-07-01', pd.Timestamp.today())
    T = len(dates)
    k = int(np.searchsorted(dates, pd.Timestamp('2026-10-09')))
    syms = [f'S{i:03d}' for i in range(40)]

    def frame(sym):
        drift = 0.0002 + 0.00004 * (sum(map(ord, sym)) % 20)
        return daily_frame(300 * np.exp(np.cumsum(rng.normal(drift, 0.018, T))), 400_000.0, start='2016-07-01')

    universe = {s: frame(s) for s in syms}
    universe['S001'] = _demerge(universe['S001'], k)

    def fetch(symbols, start, end):
        out = {}
        for s in symbols:
            if s.startswith('^'):
                out[s] = daily_frame(10_000 * np.exp(np.cumsum(np.full(T, 0.0004))), start='2016-07-01')
            elif s in universe:
                out[s] = universe[s]
        return out

    state = tmp_path / 'state'
    state.mkdir()
    pd.DataFrame({'session': ['2026-10-05', str(dates[-1].date())], 'equity': [50_000.0, 45_000.0]}) \
        .to_csv(state / 'equity.csv', index=False)
    with open(state / 'state.json', 'w') as fh:
        json.dump({'positions': [{'symbol': 'S001', 'qty': 10, 'entry_time': '2026-10-06 00:00:00'}]}, fh)

    out = tmp_path / 'out'
    report = diagnosis.run(str(out), fetch=fetch, get_list=lambda: (syms, 'fake list'),
                           get_series=lambda: {'S002': 'BE'}, state_dir=str(state))
    for heading in ('## 1. The live window', '## 2. Corporate actions', '## 3. Which score predicts',
                    '## 4. Candidate books'):
        assert heading in report
    assert '| S001 |' in report and 'value kept across corporate-action gaps' in report
    assert '| B4 deployed: 12-1, top 10, every 5 sessions |' in report
    assert (out / 'DIAGNOSIS.md').exists() and (out / 'diagnosis_variants.csv').exists()
