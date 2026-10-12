"""The gate every scheduled run passes first (scripts/should_run.py)."""

import importlib.util
import json
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('should_run', os.path.join(ROOT, 'scripts', 'should_run.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def ist(s):
    return datetime.fromisoformat(s).replace(tzinfo=gate.IST)


def test_end_of_day_bots_wait_for_the_settled_session():
    # Tuesday 6 Oct 2026
    assert gate.expected_session(ist('2026-10-06 18:00')).isoformat() == '2026-10-06'
    assert gate.expected_session(ist('2026-10-06 15:00')).isoformat() == '2026-10-05'   # bar not final yet
    assert gate.expected_session(ist('2026-10-07 06:13')).isoformat() == '2026-10-06'   # next morning
    assert gate.expected_session(ist('2026-10-10 10:00')).isoformat() == '2026-10-09'   # Saturday -> Friday
    assert gate.expected_session(ist('2026-10-12 08:37')).isoformat() == '2026-10-09'   # Monday morning -> Friday


def test_end_of_day_slots_skip_once_the_session_is_done():
    now = ist('2026-10-06 21:13')
    assert gate.decide('swing', now, {'last_processed': '2026-10-05'}) == \
        (True, 'session 2026-10-06 not processed yet (last: 2026-10-05)')
    assert gate.decide('momentum', now, {'last_processed': '2026-10-06'}) == \
        (False, 'session 2026-10-06 already processed')
    assert gate.decide('swing', now, None)[0]                                   # never run: run
    assert gate.decide('swing', now, {'last_processed': '2026-10-06'}, 'workflow_dispatch')[0]


def test_intraday_runs_only_when_a_session_can_still_trade():
    assert gate.decide('intraday', ist('2026-10-06 09:22'), {})[0]
    assert gate.decide('intraday', ist('2026-10-06 11:22'), {})[0]
    assert not gate.decide('intraday', ist('2026-10-06 14:00'), {'positions': []})[0]
    assert gate.decide('intraday', ist('2026-10-06 14:00'), {'positions': [{'symbol': 'X'}]})[0]
    assert not gate.decide('intraday', ist('2026-10-06 16:18'), {})[0]          # today's 7-hour-late slot
    assert not gate.decide('intraday', ist('2026-10-10 10:00'), {})[0]          # Saturday
    assert gate.decide('intraday', ist('2026-10-06 20:00'), {}, 'workflow_dispatch')[0]


def test_main_writes_the_step_output(tmp_path, monkeypatch):
    (tmp_path / 'state' / 'swing').mkdir(parents=True)
    (tmp_path / 'state' / 'swing' / 'state.json').write_text(json.dumps({'last_processed': '1999-01-01'}))
    out = tmp_path / 'out'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_OUTPUT', str(out))
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'schedule')
    assert gate.main(['swing']) == 0
    assert out.read_text().startswith('run=true\nreason=session ')
