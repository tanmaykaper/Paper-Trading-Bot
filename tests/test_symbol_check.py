import pandas as pd

from nsebot.research import symbol_check


def test_missing_symbols_are_matched_by_candidate_and_company_name(tmp_path):
    from nsebot.universe import INTRADAY_UNIVERSE, SWING_UNIVERSE
    listed = [s for s in dict.fromkeys(SWING_UNIVERSE + INTRADAY_UNIVERSE) if s != 'JKBANK']
    table = pd.DataFrame({'SYMBOL': listed + ['J&KBANK'],
                          'NAME': ['X'] * len(listed) + ['The Jammu & Kashmir Bank Limited'],
                          'SERIES': 'EQ'})
    report = symbol_check.run(str(tmp_path), table=table, rows={'J&KBANK': 21, 'JKBANK': 0})
    assert '1 missing from EQUITY_L' in report
    assert '| JKBANK | 0 | J&KBANK (listed · 21) |' in report
    assert 'J&KBANK = The Jammu & Kashmir Bank Limited [EQ] · 21 bars' in report


def test_no_list_means_no_result(tmp_path):
    assert 'NOT RUN' in symbol_check.run(str(tmp_path), rows={}, get=lambda url: None)


def test_equity_table_parses_nse_headers():
    text = 'SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING\nJ&KBANK,Jammu Bank,EQ,01-JAN-2000\n'
    df = symbol_check.equity_table(lambda url: text)
    assert list(df['SYMBOL']) == ['J&KBANK'] and list(df['SERIES']) == ['EQ']
