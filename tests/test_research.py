import numpy as np
import pandas as pd

from nsebot.research.event_study import intraday_study, swing_study, _swing_table, _intraday_table
from conftest import daily_frame, intraday_sessions, uptrend_then_breakout


def test_swing_study_runs_point_in_time_on_synthetic_universe():
    universe = {f'S{i}': uptrend_then_breakout(n=260, seed=i) for i in range(8)}
    # Extend each with 25 flat sessions so forward windows exist after the breakout.
    for k, df in universe.items():
        tail = daily_frame(np.full(25, df['close'].iloc[-1]), start=df['datetime'].iloc[-1])
        universe[k] = pd.concat([df, tail.iloc[1:]], ignore_index=True)
    index_df = daily_frame(20000 * np.exp(np.cumsum(np.full(285, 0.001))))
    index_df['datetime'] = universe['S0']['datetime']
    ev, summary = swing_study(universe, index_df)
    assert summary['events'] == len(ev) > 0
    assert {'fwd_5', 'xs_5', 'trade_r', 'regime'} <= set(ev.columns)
    # Entry is the NEXT open, never the signal close.
    assert (ev['entry'] != 0).all()
    assert '| ALL |' in _swing_table(ev)


def test_intraday_study_simulates_stop_or_square_off():
    bars = {'AAA': intraday_sessions(n_sessions=5, breakout_time='10:00'),
            'BBB': intraday_sessions(n_sessions=5, breakout_time='10:30', side='SHORT')}
    ev, summary = intraday_study(bars, {}, None)
    assert summary['trades'] == 2
    assert set(ev['side']) == {'LONG', 'SHORT'}
    assert ev['cost_pct'].gt(0).all()
    assert '| ALL |' in _intraday_table(ev)
