"""Circuit breakers: halt on realised evidence, not on forecasts.

V2's regime gate halted on a FORECAST about the market and froze a working
pipeline for three weeks. These halt on what this strategy has actually done
to this account — a far better reason to stop:

  CONSECUTIVE LOSSES  N losers in a row -> no new entries for the cooldown
                      (intraday: rest of the session; swing: K sessions),
                      then reduced size until the next winner OR a session
                      limit, whichever is first (0 disables the breaker —
                      swing runs without it, see config.swing_breakers).
  DAILY LOSS LIMIT    today's realised + open P&L below -X% of sleeve equity
                      -> flatten (intraday) and halt until tomorrow (0 disables
                      it — the momentum sleeve runs without one).
  MAX DRAWDOWN        sleeve equity X% below its peak -> halt ALL new
                      entries until a human deletes the latch file. This is
                      the one that is meant to be annoying. Live runs keep
                      the file with the sleeve's state (committed by the
                      workflow) so it survives a fresh checkout; deleting it
                      resumes trading and restarts the drawdown from that
                      day's equity, so it doesn't instantly re-trip.
  TRADES PER DAY      intraday overtrading cap.
  KILL SWITCH         a file named STOP_TRADING in the working directory.

Exits are never blocked. A breaker that traps you in open risk is not a
safety feature.

State is a plain dict so the runner can persist it to JSON and commit it —
the exact persistence step V2 forgot for its own state (AUTOPSY §2).
"""

import os
from dataclasses import dataclass
from datetime import date

DRAWDOWN_LATCH = 'BREAKER_TRIPPED'


@dataclass
class BreakerVerdict:
    entries_allowed: bool
    size_mult: float
    reasons: list
    flatten: bool = False        # intraday daily-loss: close everything now


class CircuitBreakers:

    def __init__(self, cfg, mode, state=None, workdir='.', latch_dir=None):
        self.cfg = cfg
        self.mode = mode
        self.workdir = workdir                       # the kill switch lives here
        self.latch_dir = latch_dir or workdir        # ...the drawdown latch here
        self.s = {'consecutive_losses': 0, 'cooldown_left': 0, 'reduced': False, 'reduced_left': 0,
                  'session': None, 'trades_today': 0, 'realised_today': 0.0,
                  'peak_equity': None, 'halted_for_session': False, 'latched': False}
        if state:
            self.s.update(state)

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start_session(self, session_date):
        session_date = str(session_date)
        if self.s['session'] == session_date:
            return
        if self.s['session'] is not None and self.s['cooldown_left'] > 0:
            self.s['cooldown_left'] -= 1
        if self.s['session'] is not None and self.s.get('reduced'):
            # Reduced size expires on the clock too. Clearing only on a winner
            # deadlocks the book the moment reduced size drops below the
            # economic floor: no trade, so no win, so reduced forever.
            self.s['reduced_left'] = int(self.s.get('reduced_left', 0)) - 1
            if self.s['reduced_left'] <= 0:
                self.s['reduced'] = False
                self.s['reduced_left'] = 0
        self.s.update({'session': session_date, 'trades_today': 0, 'realised_today': 0.0,
                       'halted_for_session': False})

    def on_entry(self):
        self.s['trades_today'] += 1

    def on_exit(self, pnl):
        self.s['realised_today'] += float(pnl)
        if pnl > 0:
            self.s['consecutive_losses'] = 0
            self.s['reduced'] = False
            return
        self.s['consecutive_losses'] += 1
        if self.cfg.max_consecutive_losses and \
                self.s['consecutive_losses'] >= self.cfg.max_consecutive_losses:
            if self.mode == 'intraday':
                self.s['halted_for_session'] = True
            else:
                self.s['cooldown_left'] = self.cfg.loss_cooldown_sessions
            self.s['reduced'] = True
            self.s['reduced_left'] = int(getattr(self.cfg, 'reduced_max_sessions', 5) or 1) + \
                (self.cfg.loss_cooldown_sessions if self.mode != 'intraday' else 0)
            self.s['consecutive_losses'] = 0

    def mark_equity(self, equity):
        peak = self.s['peak_equity']
        self.s['peak_equity'] = float(equity) if peak is None else max(float(peak), float(equity))

    # ── the check ───────────────────────────────────────────────────────────
    def check(self, equity, open_pnl=0.0):
        reasons, flatten = [], False
        self.mark_equity(equity)

        if os.path.exists(os.path.join(self.workdir, self.cfg.kill_switch_file)):
            reasons.append(f'kill switch file {self.cfg.kill_switch_file} present')

        latch = os.path.join(self.latch_dir, f'{DRAWDOWN_LATCH}_{self.mode}')
        if self.s.get('latched') and not os.path.exists(latch):
            # A human deleted the latch: resume, measuring drawdown from here.
            self.s['latched'] = False
            self.s['peak_equity'] = float(equity)
        peak = self.s['peak_equity'] or equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd >= self.cfg.max_drawdown_pct and not os.path.exists(latch):
            self.s['latched'] = True
            os.makedirs(self.latch_dir, exist_ok=True)
            with open(latch, 'w') as fh:
                fh.write(f'{self.mode} drawdown {dd:.1%} from peak {peak:,.0f} on {self.s["session"] or date.today()}\n'
                         f'Delete this file to resume new entries.\n')
        if os.path.exists(latch):
            reasons.append(f'max drawdown breaker latched ({latch}) — delete it to resume')

        day_pnl = self.s['realised_today'] + float(open_pnl)
        if self.cfg.daily_loss_limit_pct and equity > 0 and day_pnl <= -self.cfg.daily_loss_limit_pct * equity:
            reasons.append(f'daily loss limit: {day_pnl:,.0f} <= -{self.cfg.daily_loss_limit_pct:.0%} of equity')
            self.s['halted_for_session'] = True
            flatten = self.mode == 'intraday'

        if self.s['halted_for_session']:
            reasons.append('halted for the rest of this session')
        if self.s['cooldown_left'] > 0:
            reasons.append(f'loss-streak cooldown: {self.s["cooldown_left"]} session(s) left')
        if self.cfg.max_trades_per_day and self.s['trades_today'] >= self.cfg.max_trades_per_day:
            reasons.append(f'max {self.cfg.max_trades_per_day} trades/day reached')

        size_mult = self.cfg.resume_size_mult if self.s['reduced'] else 1.0
        return BreakerVerdict(not reasons, size_mult, list(dict.fromkeys(reasons)), flatten)

    def state(self):
        return dict(self.s)
