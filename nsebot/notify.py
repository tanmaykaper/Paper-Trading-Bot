"""Reports and alerts. Email if EMAIL_SENDER / EMAIL_PASSWORD / EMAIL_RECIPIENT
are set (the same repo secrets the old bot used); always the log and, on
GitHub Actions, the job summary page."""

import logging
import os
import smtplib
from email.message import EmailMessage

import pandas as pd

logger = logging.getLogger(__name__)


def send_email(subject, body):
    sender, pwd, to = (os.environ.get(k) for k in ('EMAIL_SENDER', 'EMAIL_PASSWORD', 'EMAIL_RECIPIENT'))
    if not (sender and pwd and to):
        return False
    msg = EmailMessage()
    msg['Subject'], msg['From'], msg['To'] = subject, sender, to
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL(os.environ.get('EMAIL_SMTP', 'smtp.gmail.com'), 465, timeout=20) as s:
            s.login(sender, pwd)
            s.send_message(msg)
        return True
    except Exception as e:                    # an alert failure must not kill the trading run
        logger.warning(f'email not sent: {e}')
        return False


def job_summary(markdown):
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if path:
        with open(path, 'a') as fh:
            fh.write(markdown + '\n')


def _rows(items, header):
    if not items:
        return ['_none_']
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for it in items:
        lines.append('| ' + ' | '.join(f'{x:,.2f}' if isinstance(x, float) else str(x) for x in it) + ' |')
    return lines


def swing_report(rep, initial_cash):
    eq = rep.get('equity', initial_cash)
    lines = [f"## nsebot swing — {rep['asof']}", '',
             f"**Equity ₹{eq:,.0f}** ({(eq / initial_cash - 1) * 100:+.2f}% on ₹{initial_cash:,.0f}) · "
             f"cash ₹{rep.get('cash', 0):,.0f} · regime {rep.get('regime', {}).get('score', '—')} "
             f"(size ×{rep.get('regime', {}).get('size_mult', '—')}, "
             f"max {rep.get('regime', {}).get('max_new_entries', '—')} new) · "
             f"signals today {rep.get('signals', 0)}", '']
    if rep.get('status') != 'ok':
        lines += [f"_{rep['status']}_", '']
    if rep.get('breakers'):
        lines += ['**Breakers:** ' + '; '.join(rep['breakers']), '']
    e = rep.get('edge')
    if e:
        lines += [f"Edge monitor: win {e['win_rate']:.0%} · payoff {e['payoff']:.2f} · Kelly "
                  f"{e['kelly_full']:+.3f} · {e['n_realised']} own trades "
                  f"({e['prior_weight']:.0%} prior)", '']
    for w in rep.get('warnings') or []:
        lines += [f'⚠️ **{w}**', '']
    lines += ['### Filled'] + _rows(rep.get('filled'), ['symbol', 'qty', 'price'])
    lines += ['', '### Closed'] + _rows(rep.get('closed'), ['symbol', 'reason', 'net ₹', 'net R'])
    lines += ['', '### New plans (fill at next open)'] + _rows(rep.get('placed'),
                                                               ['symbol', 'qty', 'ref', 'stop', 'binding'])
    if rep.get('declined'):
        # Every signal that did not become a plan, and why: "signals today 2" with
        # no plans must never be a mystery (V2 looked exactly like that).
        lines += ['', '### Signals not taken'] + _rows(rep['declined'], ['symbol', 'reason'])
    lines += ['', '### Open'] + _rows(rep.get('open_positions'), ['symbol', 'qty', 'entry', 'stop'])
    if rep.get('cancelled'):
        lines += ['', '### Cancelled'] + _rows(rep['cancelled'], ['symbol', 'reason'])
    return '\n'.join(lines)


def fy_tax_estimate(trades, asof):
    """Realised net P&L in the Indian financial year containing `asof` (April
    to March) and a rough tax on it, for comparing an actively traded book with
    a fund held for years. Listed-equity rates since 23 Jul 2024: short-term
    (held 12 months or less) 20%, long-term 12.5% above ₹1.25 lakh a year,
    both plus 4% cess. Short-term losses offset long-term gains, not the other
    way round. An estimate, not tax advice: net P&L here is after all charges,
    including STT, which is not deductible."""
    asof = pd.Timestamp(asof)
    fy_start = pd.Timestamp(year=asof.year if asof.month >= 4 else asof.year - 1, month=4, day=1)
    if trades is None or len(trades) == 0:
        return {'fy': f'FY{(fy_start.year + 1) % 100:02d}', 'short': 0.0, 'long': 0.0, 'tax': 0.0}
    t = trades.copy()
    t['exit_time'] = pd.to_datetime(t['exit_time'])
    t = t[(t['exit_time'] >= fy_start) & (t['exit_time'] <= asof + pd.Timedelta(days=1))]
    held = (t['exit_time'] - pd.to_datetime(t['entry_time'])).dt.days
    short = float(t.loc[held <= 365, 'net_pnl'].sum())
    long_ = float(t.loc[held > 365, 'net_pnl'].sum())
    st_gain, lt_gain = short, long_
    if st_gain < 0:
        lt_gain, st_gain = lt_gain + st_gain, 0.0
    tax = 1.04 * (0.20 * st_gain + 0.125 * max(lt_gain - 125_000.0, 0.0))
    return {'fy': f'FY{(fy_start.year + 1) % 100:02d}', 'short': short, 'long': long_, 'tax': round(tax, 2)}


def momentum_report(rep, initial_cash, trades=None):
    eq = rep.get('equity', initial_cash)
    when = ('**rebalanced today**' if rep.get('rebalanced')
            else f"next rebalance in {rep.get('next_rebalance_in', '—')} session(s)")
    lines = [f"## nsebot momentum — {rep['asof']}", '',
             f"**Equity ₹{eq:,.0f}** ({(eq / initial_cash - 1) * 100:+.2f}% on ₹{initial_cash:,.0f}) · "
             f"cash ₹{rep.get('cash', 0):,.0f} · {when}"
             + (f" · {rep['eligible']} eligible stocks" if rep.get('eligible') is not None else ''), '']
    if rep.get('status') != 'ok':
        lines += [f"_{rep['status']}_", '']
    if rep.get('universe_note'):
        lines += [f"_Universe: {rep['universe_note']}_", '']
    if rep.get('breakers'):
        lines += ['**Breakers (sales still run, buys stop):** ' + '; '.join(rep['breakers']), '']
    for w in rep.get('warnings') or []:
        lines += [f'⚠️ **{w}**', '']
    if trades is not None:
        tx = fy_tax_estimate(trades, rep['asof'])
        lines += [f"{tx['fy']} realised: short-term ₹{tx['short']:+,.0f}, long-term ₹{tx['long']:+,.0f} · "
                  f"rough tax ₹{tx['tax']:,.0f} (not modelled in the backtests)", '']
    lines += ['### Bought at the open'] + _rows(rep.get('filled'), ['symbol', 'qty', 'price'])
    sold = [(s, why, pnl, f'{r * 100:+.1f}%' if r is not None and r == r else '—')
            for s, why, pnl, r in rep.get('closed') or []]
    lines += ['', '### Sold at the open'] + _rows(sold, ['symbol', 'reason', 'net ₹', 'net return'])
    lines += ['', '### Orders for the next open — sales'] + _rows(rep.get('selling'), ['symbol', 'qty', 'reason'])
    lines += ['', '### Orders for the next open — buys'] + _rows(rep.get('placed'),
                                                                ['symbol', 'qty', 'ref', 'rank', '12-1 momentum %'])
    hold = [(s, q, e, c, f'{(c / e - 1) * 100:+.1f}%' if c and e else '—')
            for s, q, e, c in rep.get('holdings') or []]
    lines += ['', '### Holdings'] + _rows(hold, ['symbol', 'qty', 'entry', 'last close', 'P&L'])
    if rep.get('kept') or rep.get('cancelled'):
        lines += ['', '### Not filled'] + _rows((rep.get('kept') or []) + (rep.get('cancelled') or []),
                                                ['symbol', 'why'])
    return '\n'.join(lines)


def intraday_report(reports, ledger):
    closed = ledger.closed_trades()
    session = ledger.state.get('session')
    today = closed[closed['exit_time'].astype(str).str.startswith(str(session))] if len(closed) else closed
    net = float(today['net_pnl'].sum()) if len(today) else 0.0
    lines = [f'## nsebot intraday — {session}', '',
             f"**Cash ₹{ledger.cash:,.0f}** · trades today {len(today)} · net today ₹{net:+,.0f}", '']
    if len(today):
        lines += _rows(today[['symbol', 'side', 'qty', 'entry_price', 'exit_price', 'exit_reason',
                              'net_pnl']].itertuples(index=False, name=None),
                       ['symbol', 'side', 'qty', 'entry', 'exit', 'reason', 'net ₹'])
    halts = sorted({r for rep in reports for r in rep.get('breakers', []) or []})
    if halts:
        lines += ['', '**Breakers fired:** ' + '; '.join(halts)]
    return '\n'.join(lines)
