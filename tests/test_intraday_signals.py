import pandas as pd

from nsebot.config import IntradaySignalConfig
from nsebot.signals import IntradaySignalEngine
from conftest import intraday_sessions


def _now_after(bars, hhmm):
    last_day = pd.Timestamp(bars['datetime'].iloc[-1]).normalize()
    h, m = map(int, hhmm.split(':'))
    return last_day + pd.Timedelta(hours=h, minutes=m)


def test_long_orb_fires_on_the_breakout_bar_only():
    bars = intraday_sessions(breakout_time='10:00', side='LONG')
    eng = IntradaySignalEngine()
    sigs = eng.scan({'AAA': bars}, now=_now_after(bars, '10:05'))
    assert len(sigs) == 1
    s = sigs[0]
    assert s.side == 'LONG' and s.trigger == 'orb_vwap'
    assert s.asof.strftime('%H:%M') == '10:00'
    assert s.stop < s.ref_price
    cfg = IntradaySignalConfig()
    assert cfg.min_stop_pct - 1e-9 <= s.stop_pct <= cfg.max_stop_pct + 1e-9
    # Ten minutes later the same breakout is old news.
    assert eng.scan({'AAA': bars}, now=_now_after(bars, '10:15')) == []


def test_short_orb_mirror():
    bars = intraday_sessions(breakout_time='10:30', side='SHORT')
    sigs = IntradaySignalEngine().scan({'BBB': bars}, now=_now_after(bars, '10:35'))
    assert len(sigs) == 1 and sigs[0].side == 'SHORT'
    assert sigs[0].stop > sigs[0].ref_price


def test_no_entries_after_the_window_closes():
    bars = intraday_sessions(breakout_time='13:45', side='LONG')
    assert IntradaySignalEngine().scan({'CCC': bars}, now=_now_after(bars, '13:50')) == []


def test_not_in_play_without_opening_volume():
    bars = intraday_sessions(breakout_time='10:00', or_volume=10_000.0)   # normal opening volume
    assert IntradaySignalEngine().scan({'DDD': bars}, now=_now_after(bars, '10:05')) == []


def test_index_alignment_blocks_longs_on_a_red_index():
    bars = intraday_sessions(breakout_time='10:00', side='LONG')
    index = intraday_sessions(base=25000.0)
    day = index['datetime'].dt.date == index['datetime'].dt.date.iloc[-1]
    first = index[day].index[0]
    index.loc[day, 'close'] = index.loc[first, 'open'] - 50.0           # index below its open all day
    sigs = IntradaySignalEngine().scan({'EEE': bars}, now=_now_after(bars, '10:05'), index_bars=index)
    assert sigs == []


def test_exclude_prevents_a_second_trade_in_the_same_name():
    bars = intraday_sessions(breakout_time='10:00', side='LONG')
    assert IntradaySignalEngine().scan({'FFF': bars}, now=_now_after(bars, '10:05'),
                                       exclude={'FFF'}) == []


def test_forming_bar_is_ignored():
    bars = intraday_sessions(breakout_time='10:00', side='LONG')
    # At 10:03 the 10:00 bar is still forming; nothing may fire on it.
    assert IntradaySignalEngine().scan({'GGG': bars}, now=_now_after(bars, '10:03')) == []
