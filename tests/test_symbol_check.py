import pandas as pd

from nsebot.research import symbol_check


def test_missing_symbols_are_matched_by_candidate_and_company_name(tmp_path, monkeypatch):
    import nsebot.universe as uni
    monkeypatch.setattr(uni, 'SWING_UNIVERSE', ['RELIANCE', 'JKBANK'])
    monkeypatch.setattr(uni, 'INTRADAY_UNIVERSE', ['RELIANCE'])
    table = pd.DataFrame({'SYMBOL': ['RELIANCE', 'J&KBANK'],
                          'NAME': ['Reliance Industries Limited', 'The Jammu & Kashmir Bank Limited'],
                          'SERIES': 'EQ'})
    report = symbol_check.run(str(tmp_path), table=table, rows={'J&KBANK': 21, 'JKBANK': 0})
    assert '2 universe symbols, 1 missing from EQUITY_L' in report
    assert '| JKBANK | 0 | J&KBANK (listed · 21) |' in report
    assert 'J&KBANK = The Jammu & Kashmir Bank Limited [EQ] · 21 bars' in report


def test_the_live_universes_are_all_listed():
    """The five dead tickers the check found are fixed: every live symbol is a
    current NSE symbol (as of the Oct 2026 list the check ran against)."""
    from nsebot.universe import SWING_UNIVERSE
    for dead, live in (('JKBANK', 'J&KBANK'), ('APOLLOMICRO', 'APOLLO'), ('BIRLASOFT', 'BSOFT'),
                       ('SUVENPHAR', 'COHANCE'), ('ORIENTGREEN', 'GREENPOWER')):
        assert dead not in SWING_UNIVERSE and live in SWING_UNIVERSE


def test_no_list_means_no_result(tmp_path):
    assert 'NOT RUN' in symbol_check.run(str(tmp_path), rows={}, get=lambda url: None)


def test_equity_table_parses_nse_headers():
    text = 'SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING\nJ&KBANK,Jammu Bank,EQ,01-JAN-2000\n'
    df = symbol_check.equity_table(lambda url: text)
    assert list(df['SYMBOL']) == ['J&KBANK'] and list(df['SERIES']) == ['EQ']
