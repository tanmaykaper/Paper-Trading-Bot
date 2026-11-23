"""Checkpoint — the 6-month read on the live momentum book.

    python -m nsebot.research.checkpoint --out research_results

The owner (Nov 2026) will study the paper book for at most 6 months, then
put real money on it. Six months of returns cannot prove or disprove the
edge (see section 3), so the checkpoint asks what six months CAN answer:
does the live bot do what the research says it does? It reports:

  1 Fidelity: the live sleeve's time-weighted return against the research
    simulator replaying the deployed design over the same sessions, from
    the same first decision (5 Oct 2026), on today's data.
  2 The V4 forward test (round 10's follow-up): B4 top 10 against V4 top 10
    from 3 Nov 2026, when V4 was coded and left off, on data neither saw.
  3 What a stretch of 6, 12 or 24 months says: how often B4 top 10 was ahead
    of an equal-weight hold of every eligible stock over such a stretch since
    Jul 2018, and how often it lost money, all 5 phases.
  4 The pre-registered checkpoint, with each condition's status today.

PRE-REGISTERED CHECKPOINT (written before the data it judges exists). On or
after 31 May 2027, real money goes on the deployed design when all hold:
  a the live sleeve's time-weighted return since 5 Oct 2026 is within 5
    points of the replay's (round 7 found the engine within a few tenths of
    the research; Yahoo revisions and corporate actions explain the rest);
  b the 55% drawdown latch has not tripped;
  c no holding has sat flagged as a suspected corporate action (ca_suspect)
    for more than 10 sessions.
The live return against the market is reported but is NOT a condition:
section 3 shows how often a working B4 trails the market over 6 months,
and acting on that would mean buying after luck and quitting after bad luck.
V4 is switched on at the checkpoint only if it is ahead of B4 at 4 or more
of the 5 phases from 3 Nov 2026 AND its median lead is at least 3 points.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

from ..listing import nse_equities
from .diagnosis import JUMP, LIVE_START, adjusted, ca_events, live_ledger, live_twr, rotate, scores
from .experiments import Panel, locked_bars
from .fast import market, phase_runs
from .phase5 import FETCH_START, WIN_START, fetch_range
from .phase5b import ever_eligible

logger = logging.getLogger(__name__)

CHECKPOINT = pd.Timestamp('2027-05-31')
V4_FROM = pd.Timestamp('2026-11-03')
FIDELITY = 0.05
V4_PHASES, V4_LEAD = 4, 0.03
CA_SESSIONS = 10
LIVE_CAPITAL = 50_000.0
CAPITAL = 150_000.0
SPANS = ((126, '6 months'), (252, '12 months'), (504, '24 months'))


def replay(Q, score, eligible, t0, capital, buy_filter=None, shift=0):
    """The research book started fresh at the close of session t0 + shift."""
    m = score.copy()
    m[:t0 + shift] = np.nan
    buy_ok, sell_ok = locked_bars(Q)
    return rotate(Q, m, eligible, n=10, capital=capital, buy_ok=buy_ok, sell_ok=sell_ok,
                  buy_filter=buy_filter)[0]


def base_rates(curves, mkt, span, start=WIN_START, step=5):
    """(share of `span`-session windows ahead of the market, share losing
    money, median excess), pooled over the curves."""
    ahead, lose, excess = [], [], []
    for e in curves:
        e = e[e.index >= start].dropna()
        m = mkt.reindex(e.index).ffill()
        for i in range(0, len(e) - span, step):
            b, x = e.iloc[i + span] / e.iloc[i] - 1, m.iloc[i + span] / m.iloc[i] - 1
            ahead.append(b > x)
            lose.append(b < 0)
            excess.append(b - x)
    if not ahead:
        return np.nan, np.nan, np.nan
    return float(np.mean(ahead)), float(np.mean(lose)), float(np.median(excess))


def flagged_sessions(state_dir, dates):
    """The longest any current holding has sat flagged ca_suspect, in sessions."""
    p = os.path.join(state_dir, 'state.json')
    if not os.path.exists(p):
        return 0
    with open(p) as fh:
        st = json.load(fh)
    worst = 0
    for pos in st.get('positions') or []:
        flag = (pos.get('meta') or {}).get('ca_suspect')
        if flag:
            try:
                d = pd.Timestamp(str(flag.get('date') if isinstance(flag, dict) else flag)[:10])
            except (TypeError, ValueError):
                d = dates[-1]
            worst = max(worst, int((dates > d).sum()))
    return worst


def latch_tripped(state_dir):
    return any(n.startswith('BREAKER_TRIPPED') for n in os.listdir(state_dir)) if os.path.isdir(state_dir) else False


def _p(v):
    return '—' if v is None or not np.isfinite(v) else f'{v * 100:+.1f}%'


def run(out_dir, fetch=None, get_list=None, state_dir=os.path.join('state', 'momentum')):
    from ..universe import INDEX_SYMBOL

    fetch = fetch or fetch_range
    os.makedirs(out_dir, exist_ok=True)
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    symbols, note = (get_list or (lambda: nse_equities(os.path.join(out_dir, 'nse_equity_list.json'))))()
    raw = fetch(symbols, FETCH_START, end)
    broad = {s_: d for s_, d in raw.items() if ever_eligible(d)}
    index_df = fetch([INDEX_SYMBOL], FETCH_START, end)[INDEX_SYMBOL]
    idx_dates = pd.DatetimeIndex(pd.to_datetime(index_df['datetime']))
    on_index = {s_: d[pd.to_datetime(d['datetime']).isin(idx_dates)].reset_index(drop=True)
                for s_, d in broad.items()}
    P = Panel({s_: d for s_, d in on_index.items() if len(d) >= 60})
    Q = adjusted(P, ca_events(P))
    s = scores(Q, P.liquid)
    mom = s['12-1 momentum (B4)']
    last = P.dates[-1]
    mkt = market(Q, P.liquid)

    md = ['# Checkpoint — the 6-month read on the live momentum book', '',
          f'Universe: {note}; {len(broad)} ever eligible. Data to {last.date()}. Checkpoint date '
          f'{CHECKPOINT.date()} ({max(int(np.busday_count(last.date(), CHECKPOINT.date())), 0)} weekdays away). '
          'The rule is in the module docstring, written before the data it judges.', '']

    # ── 1 fidelity ───────────────────────────────────────────────────────────
    md += ['## 1. Does the live bot do what the research says?', '']
    eq, positions, trades, flows, initial = live_ledger(state_dir)
    t0 = int(np.searchsorted(P.dates, LIVE_START))
    gap = live = rep = None
    if eq is not None and len(eq) and t0 < len(P.dates) and P.dates[t0] == LIVE_START:
        e = eq.set_index(pd.to_datetime(eq['session']))['equity'].astype(float)
        e = e[e.index <= last]
        live = live_twr(e, flows, initial)
        r = replay(Q, mom, P.liquid, t0, LIVE_CAPITAL)
        rep = r.loc[e.index[-1]] / r.iloc[t0] - 1 if e.index[-1] in r.index else r.iloc[-1] / r.iloc[t0] - 1
        m_ret = mkt.loc[e.index[-1]] / mkt.iloc[t0] - 1 if e.index[-1] in mkt.index else np.nan
        gap = live - rep
        md += [f'From the {LIVE_START.date()} close to {e.index[-1].date()} ({len(e) - 1} sessions):', '',
               '| | return |', '|---|---|',
               f'| Live sleeve, time-weighted (net of ₹{sum(flows.values()):,.0f} moved in) | {_p(live)} |',
               f'| Research replay of B4 top 10 from the same first decision | {_p(rep)} |',
               f'| **Gap (live − replay)** | **{_p(gap)}** |',
               f'| Equal-weight hold of every eligible stock | {_p(m_ret)} |', '',
               '_The replay uses corporate-action-adjusted prices and today\'s Yahoo data; the live book '
               'traded on the data each day had. A gap within a few points is noise; a wide one is a bug or '
               'a data fault to chase._', '']
    else:
        md += ['_No live ledger, or the first decision date is not in the data: skipped._', '']

    # ── 2 V4 forward ─────────────────────────────────────────────────────────
    md += ['## 2. The V4 forward test, from 3 Nov 2026', '']
    tv = int(np.searchsorted(P.dates, V4_FROM))
    v4_ok = False
    if tv + 1 < len(P.dates):
        with np.errstate(invalid='ignore'):
            nojump = -s['−largest daily gain, 63 sessions'] < JUMP
        rows, leads = [], []
        for k in range(5):
            b = replay(Q, mom, P.liquid, tv, CAPITAL, shift=k)
            v = replay(Q, mom, P.liquid, tv, CAPITAL, buy_filter=nojump, shift=k)
            rb, rv = b.iloc[-1] / b.iloc[tv + k] - 1, v.iloc[-1] / v.iloc[tv + k] - 1
            leads.append(rv - rb)
            rows.append(f'| +{k} | {_p(rb)} | {_p(rv)} | {_p(rv - rb)} |')
        ahead = int(np.sum(np.asarray(leads) > 0))
        v4_ok = ahead >= V4_PHASES and np.median(leads) >= V4_LEAD
        md += ['| phase | B4 top 10 | V4 top 10 | V4 lead |', '|---|---|---|---|'] + rows
        md += ['', f'V4 ahead at {ahead} of 5 phases, median lead {_p(float(np.median(leads)))}. '
                   f'Switch-on bar at the checkpoint: 4 of 5 and +3 points.', '']
    else:
        md += ['_No data after 3 Nov 2026 yet._', '']

    # ── 3 what a stretch says ────────────────────────────────────────────────
    curves = phase_runs(Q, mom, P.liquid)
    md += ['## 3. What 6, 12 or 24 months of results can tell you', '',
           '| stretch | B4 ahead of the market | B4 lost money | median lead over the market |',
           '|---|---|---|---|']
    for span, label in SPANS:
        a, lo, x = base_rates(curves, mkt, span)
        md.append(f'| {label} | {a:.0%} | {lo:.0%} | {_p(x)} |' if np.isfinite(a) else f'| {label} | — | — | — |')
    md += ['', '_Every start date since Jul 2018, all 5 phases, ₹1.5L book, CNC costs. The market is an '
               'equal-weight hold of every eligible stock. If B4 is as good as its backtest, a 6-month stretch '
               'behind the market is this common anyway; if it has no edge at all, about half are._', '']

    # ── 4 checkpoint ─────────────────────────────────────────────────────────
    flagged = flagged_sessions(state_dir, P.dates)
    tripped = latch_tripped(state_dir)
    conds = [('a live within 5 points of the replay', gap is not None and abs(gap) <= FIDELITY,
              f'gap {_p(gap)}' if gap is not None else 'no ledger'),
             ('b 55% latch not tripped', not tripped, 'tripped' if tripped else 'not tripped'),
             (f'c no ca_suspect flag older than {CA_SESSIONS} sessions', flagged <= CA_SESSIONS,
              f'oldest {flagged} sessions')]
    md += ['## 4. The checkpoint, as it stands today', '', '| condition | today | holds? |', '|---|---|---|']
    md += [f'| {c} | {t} | {"yes" if ok else "**no**"} |' for c, ok, t in conds]
    verdict = 'on track: every condition holds' if all(ok for _, ok, _ in conds) else \
        'not on track: chase the failing condition before the checkpoint'
    md += ['', f'**{verdict}.** V4: {"meets" if v4_ok else "does not meet"} its switch-on bar today. '
               f'Final read on or after {CHECKPOINT.date()}.', '']

    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'CHECKPOINT.md'), 'w') as fh:
        fh.write(report + '\n')
    step = os.environ.get('GITHUB_STEP_SUMMARY')
    if step:
        with open(step, 'a') as fh:
            fh.write(report + '\n')
    print(report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='research_results')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        stream=sys.stdout)
    run(args.out)


if __name__ == '__main__':
    main()
