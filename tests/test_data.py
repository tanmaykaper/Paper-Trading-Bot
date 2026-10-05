import numpy as np
import pandas as pd

from nsebot.data import YahooProvider, sanitize, to_yahoo, from_yahoo


def _yf_like(tickers, n=5, intraday=False):
    """The shape yf.download(group_by='ticker', auto_adjust=True) returns."""
    idx = (pd.date_range('2026-10-01 03:45', periods=n, freq='5min', tz='UTC') if intraday
           else pd.date_range('2026-09-01', periods=n, freq='B'))
    cols = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume']],
                                      names=['Ticker', 'Price'])
    data = np.tile([100.0, 101.0, 99.0, 100.5, 1000.0], (n, len(tickers)))
    df = pd.DataFrame(data, index=idx, columns=cols)
    df[(tickers[-1], 'Close')] = np.nan            # a ticker Yahoo failed to resolve
    return df


def test_split_and_sanitize_multi_ticker_daily():
    raw = _yf_like(['RELIANCE.NS', 'TCS.NS', 'DEAD.NS'])
    frames = YahooProvider._split(raw, ['RELIANCE.NS', 'TCS.NS', 'DEAD.NS'])
    clean = {s: sanitize(f) for s, f in frames.items()}
    assert clean['RELIANCE'] is not None and len(clean['RELIANCE']) == 5
    assert clean['DEAD'] is None                    # all-NaN close -> dropped, not a crash


def test_intraday_timestamps_become_ist():
    raw = _yf_like(['INFY.NS', 'X.NS'], intraday=True)
    f = sanitize(YahooProvider._split(raw, ['INFY.NS', 'X.NS'])['INFY'], intraday=True)
    assert str(f['datetime'].dt.tz) == 'Asia/Kolkata'
    assert f['datetime'].iloc[0].strftime('%H:%M') == '09:15'


def test_symbol_mapping():
    assert to_yahoo('M&M') == 'M&M.NS' and to_yahoo('^NSEI') == '^NSEI'
    assert from_yahoo('BAJAJ-AUTO.NS') == 'BAJAJ-AUTO'
