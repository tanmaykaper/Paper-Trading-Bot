"""Which sleeves trade: state/sleeves.json, edited by hand.

    {"momentum": {"status": "active"},
     "swing":    {"status": "retired", "since": "2026-11-03", "sweep_to": "momentum", "why": "..."}}

  active    trades as designed (also the default for a sleeve not listed)
  retired   takes no new entries; open positions still run to their exits.
            Once flat, the sleeve named in `sweep_to` takes its cash on that
            sleeve's next run, and the schedule gate stops running it.

Standard library only: the schedule gate (scripts/should_run.py) and the
README chart (scripts/progress.py) read it before anything is installed.
"""

import json
import os

PATH = os.path.join('state', 'sleeves.json')


def load(path=PATH):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        plan = json.load(fh)
    return {k: v for k, v in plan.items() if isinstance(v, dict)}


def plan_for(mode, plan=None):
    return (plan if plan is not None else load()).get(mode, {'status': 'active'})


def is_active(mode, plan=None):
    return plan_for(mode, plan).get('status', 'active') == 'active'


def retired_note(mode, plan=None):
    p = plan_for(mode, plan)
    since = f" since {p['since']}" if p.get('since') else ''
    return f'sleeve retired{since} (state/sleeves.json): no new entries, exits still run'
