import numpy as np
import pandas as pd

from nsebot.research import phase5b
from nsebot.research.experiments import Panel, locked_bars, simulate_rotation
from conftest import daily_frame


def _equity_list_csv(n_eq=320):
    rows = ['SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING, ISIN NUMBER']     # NSE pads headers
    rows += [f'S{i:04d},Company {i},EQ,01-JAN-2010,INE{i:07d}' for i in range(n_eq)]
    rows += ['TRADE1,T2T name,BE,01-JAN-2012,INE9', 'SME1,SME name,SM,01-JAN-2020,INE8']
    return '\n'.join(rows) + '\n'


def test_list_falls_back_and_keeps_only_main_board_series():
    calls = []

    def get(url):
        calls.append(url)
        return None if len(calls) == 1 else _equity_list_csv()

    syms, label, url = phase5b.listed_equities(get)
    assert url == phase5b.LIST_SOURCES[1][1] and 'EQUITY_L' in label
    assert 'TRADE1' in syms and 'SME1' not in syms and len(syms) == 321


def test_list_rejects_error_pages_and_short_files():
    pages = iter(['<html>Access Denied</html>', _equity_list_csv(50), None, None])
    assert phase5b.listed_equities(lambda url: next(pages)) == ([], None, None)


def test_ever_eligible_uses_d4s_turnover_and_price_rule():
    liquid = daily_frame(np.full(60, 300.0), 400_000.0)          # ₹12 cr a day
    thin = daily_frame(np.full(60, 300.0), 10_000.0)             # ₹30 lakh a day
    penny = daily_frame(np.full(60, 20.0), 10_000_000.0)         # liquid, but under ₹50
    assert phase5b.ever_eligible(liquid)
    assert not phase5b.ever_eligible(thin) and not phase5b.ever_eligible(penny)


def test_locked_upper_circuit_bars_cannot_be_bought():
    n = 300
    hot = daily_frame(100 * np.exp(np.cumsum(np.full(n, 0.01))), 1_000_000.0)
    k = n - 60
    for col in ('open', 'high', 'low'):
        hot.loc[k:, col] = hot.loc[k:, 'close']                   # one price all day, rising: frozen
    cold = daily_frame(np.full(n, 100.0), 1_000_000.0)
    P = Panel({'HOT': hot, 'COLD': cold})
    buy_ok, sell_ok = locked_bars(P)
    j = P.symbols.index('HOT')
    assert not buy_ok[k:, j].any() and sell_ok[k:, j].all() and buy_ok[:k, j].all()

    score = np.full((n, 2), np.nan)
    score[k:, j] = 1.0                                            # HOT is the only candidate
    eligible = np.ones((n, 2), dtype=bool)
    free, _ = simulate_rotation(P, score, eligible, n=1, every=1, capital=50_000)
    locked, _ = simulate_rotation(P, score, eligible, n=1, every=1, capital=50_000,
                                  buy_ok=buy_ok, sell_ok=sell_ok)
    assert free.iloc[-1] > 50_000 * 1.2                           # it would have ridden the move
    assert locked.iloc[-1] == 50_000                              # no seller, no fill, no gain


def test_july_years_end_with_a_partial_year():
    ys = phase5b.july_years(pd.Timestamp('2026-10-05'))
    assert ys[0] == (pd.Timestamp('2018-07-01'), pd.Timestamp('2019-06-30'))
    assert ys[-1] == (pd.Timestamp('2026-07-01'), pd.Timestamp('2026-10-05')) and len(ys) == 9
    assert [phase5b._ylabel(*y) for y in (ys[0], ys[-1])] == ['18-19', '26-27*']


def test_run_end_to_end_on_fake_sources(tmp_path):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range('2016-07-01', pd.Timestamp.today())
    T = len(dates)
    broad = [f'S{i:04d}' for i in range(24)]

    def frame(sym):
        drift = 0.0002 + 0.00004 * (sum(map(ord, sym)) % 20)
        return daily_frame(300 * np.exp(np.cumsum(rng.normal(drift, 0.018, T))), 400_000.0, start='2016-07-01')

    def fetch(symbols, start, end):
        out = {}
        for s in symbols:
            if s == '^NSEI':
                out[s] = daily_frame(10_000 * np.exp(np.cumsum(np.full(T, 0.0004))), start='2016-07-01')
            elif s.startswith('S0') or s in ('RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ITC', 'LT',
                                             'SBIN', 'AXISBANK', 'MARUTI', 'TITAN', 'WIPRO', 'ONGC'):
                out[s] = frame(s)
        return out

    def funds():
        return {f'Fund {k} Small Cap - Direct Plan - Growth':
                pd.Series(10 * np.exp(np.cumsum(np.full(T, 0.0004 + 0.0001 * k))), index=dates)
                for k in range(5)}

    report = phase5b.run(str(tmp_path), fetch=fetch, get_list=lambda: (broad, 'fake list', 'fake://'),
                         get_funds=funds)
    assert '## Primary: untouched window' in report and '| B4 momentum, broad list (judged) |' in report
    assert '## Verdict (pre-registered)' in report
    assert any(v in report for v in ('**PASS**', '**FAIL**'))
    assert (tmp_path / 'PHASE5B.md').exists()


def test_run_without_a_list_reports_no_result(tmp_path):
    report = phase5b.run(str(tmp_path), fetch=lambda *a: {}, get_list=lambda: ([], None, None),
                         get_funds=dict)
    assert 'NOT RUN' in report
