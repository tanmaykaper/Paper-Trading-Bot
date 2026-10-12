#!/usr/bin/env python3
"""Does this scheduled run have any work to do?

    python3 scripts/should_run.py swing|momentum|intraday

GitHub's scheduler runs this repo's jobs hours late and drops some
(docs/SCHEDULING.md), so each bot is scheduled many times a day. This check
is the first step of every run and uses only the standard library, so it
runs before any dependency is installed: a slot with nothing to do finishes
in seconds, and the slot that finds work runs the bot.

  swing, momentum  run unless the latest settled session (today after 15:50
                   IST on a weekday, else the previous weekday) is already in
                   state/<mode>/state.json
  intraday         run inside the entry window (09:15-13:30 IST on weekdays),
                   or until 15:05 while positions are open

A manual or external run (workflow_dispatch) always runs. Writes run=true or
run=false and a reason to $GITHUB_OUTPUT.
"""

import json
import os
import sys
from datetime import datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
SETTLE = time(15, 50)                 # Yahoo's daily bar is final (the engine uses 15:45)
OPEN, ENTRIES_END, LAST_MANAGE = time(9, 15), time(13, 30), time(15, 5)


def expected_session(now):
    """The latest weekday whose close has settled. NSE holidays are not known
    here; on one, the bot runs, finds the last session done and does nothing."""
    d = now.date()
    if now.weekday() < 5 and now.time() >= SETTLE:
        return d
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def decide(mode, now, state, event='schedule'):
    if event == 'workflow_dispatch':
        return True, 'manual or external trigger: always runs'
    state = state or {}
    if mode in ('swing', 'momentum'):
        want = expected_session(now).isoformat()
        last = state.get('last_processed')
        if last and last >= want:
            return False, f'session {want} already processed'
        return True, f'session {want} not processed yet (last: {last or "never"})'
    if mode == 'intraday':
        if now.weekday() >= 5:
            return False, 'weekend'
        t = now.time()
        if OPEN <= t < ENTRIES_END:
            return True, 'inside the entry window'
        if ENTRIES_END <= t < LAST_MANAGE and state.get('positions'):
            return True, 'open positions to manage until the square-off'
        return False, 'outside the session window'
    raise ValueError(f'unknown mode {mode!r}')


def main(argv=None):
    mode = (argv if argv is not None else sys.argv[1:])[0]
    path = os.path.join('state', mode, 'state.json')
    state = None
    if os.path.exists(path):
        with open(path) as fh:
            state = json.load(fh)
    run, reason = decide(mode, datetime.now(IST), state, os.environ.get('GITHUB_EVENT_NAME', 'schedule'))
    print(f'{mode}: {"run" if run else "skip"} — {reason}')
    out = os.environ.get('GITHUB_OUTPUT')
    if out:
        with open(out, 'a') as fh:
            fh.write(f'run={"true" if run else "false"}\nreason={reason}\n')
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary and not run:
        with open(summary, 'a') as fh:
            fh.write(f'nsebot {mode}: nothing to do ({reason}).\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
