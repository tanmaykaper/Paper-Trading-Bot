"""The schedules and queues the live bots depend on (docs/SCHEDULING.md)."""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wf(name):
    return open(os.path.join(ROOT, '.github', 'workflows', name)).read()


def _crons(text):
    return re.findall(r"cron:\s*'([^']+)'", text)


def _group(text):
    return re.search(r'concurrency:\s*\n\s*group:\s*(\S+)', text).group(1)


def test_each_bot_has_its_own_queue():
    groups = [_group(_wf(f)) for f in ('main.yml', 'momentum.yml', 'intraday.yml')]
    assert len(set(groups)) == 3, groups


def test_end_of_day_bots_have_backup_runs():
    for f in ('main.yml', 'momentum.yml'):
        crons = _crons(_wf(f))
        assert len(crons) >= 3, (f, crons)
        assert any(c.split()[4] == '2-6' for c in crons), f'{f}: no next-morning backup'


def test_intraday_session_fits_inside_the_job_limit():
    text = _wf('intraday.yml')
    minute, hour = (int(x) for x in _crons(text)[0].split()[:2])
    start_ist = hour * 60 + minute + 330                     # UTC -> IST
    end_ist = 15 * 60 + 12                                   # square-off 15:10 + 2 minutes
    timeout = int(re.search(r'timeout-minutes:\s*(\d+)', text).group(1))
    assert timeout <= 360                                    # GitHub's cap for hosted runners
    assert start_ist >= 9 * 60 + 15                          # never before the open
    assert end_ist - start_ist + 5 <= timeout                # 5 minutes for setup and the final commit
