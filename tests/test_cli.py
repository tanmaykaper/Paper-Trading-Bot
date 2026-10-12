import numpy as np
import pandas as pd

import nsebot.data
from nsebot.__main__ import main
from nsebot.ledger import Ledger
from conftest import daily_frame


class FakeProvider:
    def __init__(self, *a, **k):
        pass

    def daily(self, symbols, lookback_days=400):
        rng = np.random.default_rng(1)
        out = {}
        for s in symbols:
            if s == '^NSEI':
                out[s] = daily_frame(20000 * np.exp(np.cumsum(np.full(320, 0.0006))), start='2025-07-01')
            else:
                out[s] = daily_frame(300 * np.exp(np.cumsum(rng.normal(0.001, 0.015, 320))), 400_000.0,
                                     start='2025-07-01')
        return out


def test_swing_cli_end_to_end_with_fake_data(tmp_path, monkeypatch):
    monkeypatch.setattr(nsebot.data, 'YahooProvider', FakeProvider)
    monkeypatch.chdir(tmp_path)
    assert main(['swing', '--state', str(tmp_path / 'state')]) == 0
    L = Ledger(str(tmp_path / 'state'), 'swing', 50_000)
    assert L.state['last_processed'] is not None
    assert not L.equity_history().empty
    # Same session again: idempotent, still exit 0, no second equity row.
    assert main(['swing', '--state', str(tmp_path / 'state')]) == 0
    assert len(Ledger(str(tmp_path / 'state'), 'swing', 50_000).equity_history()) == 1


def test_swing_cli_refuses_to_trade_on_broken_data(tmp_path, monkeypatch):
    class Empty(FakeProvider):
        def daily(self, symbols, lookback_days=400):
            return {}
    monkeypatch.setattr(nsebot.data, 'YahooProvider', Empty)
    monkeypatch.chdir(tmp_path)
    assert main(['swing', '--state', str(tmp_path / 'state')]) == 3
    assert not (tmp_path / 'state' / 'swing' / 'state.json').exists()


def test_a_late_intraday_run_exits_quietly(tmp_path, monkeypatch):
    """GitHub's scheduler can start the intraday job after the square-off; that
    run must do nothing and send no email."""
    import nsebot.__main__ as cli
    import nsebot.engine.intraday as intraday
    sent = []
    monkeypatch.setattr(nsebot.data, 'YahooProvider', FakeProvider)
    monkeypatch.setattr(intraday.IntradayEngine, 'run_session', lambda self, provider, **k: [])
    monkeypatch.setattr(cli, 'send_email', lambda *a, **k: sent.append(a))
    monkeypatch.chdir(tmp_path)
    assert main(['intraday', '--state', str(tmp_path / 'state')]) == 0
    assert sent == []
