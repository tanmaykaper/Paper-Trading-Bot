"""Reports and alerts. Email if EMAIL_SENDER / EMAIL_PASSWORD / EMAIL_RECIPIENT
are set (the same repo secrets the old bot used); always the log and, on
GitHub Actions, the job summary page."""

import logging
import os
import smtplib
from email.message import EmailMessage

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
    lines += ['### Filled'] + _rows(rep.get('filled'), ['symbol', 'qty', 'price'])
    lines += ['', '### Closed'] + _rows(rep.get('closed'), ['symbol', 'reason', 'net ₹', 'net R'])
    lines += ['', '### New plans (fill at next open)'] + _rows(rep.get('placed'),
                                                               ['symbol', 'qty', 'ref', 'stop', 'binding'])
    lines += ['', '### Open'] + _rows(rep.get('open_positions'), ['symbol', 'qty', 'entry', 'stop'])
    if rep.get('cancelled'):
        lines += ['', '### Cancelled'] + _rows(rep['cancelled'], ['symbol', 'reason'])
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
