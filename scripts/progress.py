#!/usr/bin/env python3
"""Draw how the money is doing, for the top of the README.

    python3 scripts/progress.py

Rewrites the block between the progress markers in README.md, and the
charts in docs/progress/ (one for GitHub's light theme, one for dark). Each
bot's commit step (scripts/commit_state.sh) runs it after the bot's state is
saved.

The three bots' books are added together, session by session. The headline
is what the money would be worth in your pocket if the paper trades had been
real and you had sold everything at that day's close:

  value     each bot's cash plus its open positions at the close. Zerodha's
            itemised charges and 5 bps of slippage are already paid on every
            fill (nsebot/broker/paper.py).
  − selling what closing the open positions would cost: STT, exchange, SEBI
            and GST on the sale, the DP charge per stock, and slippage.
  − tax     on the profit: 20.8% on delivery (short-term capital gains, 20%
            plus 4% cess) and 31.2% on intraday (speculative business income
            is taxed at your slab; the top slab is assumed). A loss pays
            nothing. Tax-year boundaries, carried-forward losses and
            dividends are not modelled.

Standard library only, so it runs before anything is installed. The output
depends only on state/, so running it again changes nothing.
"""

import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nsebot.costs import DEFAULT_CHARGES  # noqa: E402  (stdlib-only module)

BOTS = (('momentum', 'Momentum'), ('swing', 'Swing'), ('intraday', 'Intraday'))
DELIVERY = ('momentum', 'swing')                  # CNC; intraday is MIS
DEFAULT_CAPITAL = 50_000.0                        # nsebot/__main__.py, until a bot's first run
SLIPPAGE = 5.0 / 1e4                              # PaperBroker's default
STCG_TAX = 0.20 * 1.04
SPECULATIVE_TAX = 0.30 * 1.04
MIN_SPAN = 0.04                                   # y-axis covers at least ±2% of the money put in

START = '<!-- progress:start'
END = '<!-- progress:end -->'
CHARTS = 'docs/progress'


# ── data ────────────────────────────────────────────────────────────────────
@dataclass
class Bot:
    mode: str
    name: str
    capital: float
    rows: list = field(default_factory=list)      # (date, equity, open_positions, open_value), one per session
    trades: int = 0
    holding: int = 0

    def on(self, day):
        """(equity, open positions, open value) at the close of `day`; the money
        put in, uninvested, before the bot's first session."""
        last = (self.capital, 0, 0.0)
        for d, eq, n, ov in self.rows:
            if d > day:
                break
            last = (eq, n, ov)
        return last


def _num(x, cast=float, default=0):
    try:
        return cast(float(x))
    except (TypeError, ValueError):
        return default


def load_bot(state_dir, mode, name):
    d = os.path.join(state_dir, mode)
    state = {}
    if os.path.exists(os.path.join(d, 'state.json')):
        with open(os.path.join(d, 'state.json')) as fh:
            state = json.load(fh)
    bot = Bot(mode, name, _num(state.get('initial_cash'), default=DEFAULT_CAPITAL),
              holding=len(state.get('positions') or []))
    if os.path.exists(os.path.join(d, 'equity.csv')):
        by_day = {}
        with open(os.path.join(d, 'equity.csv'), newline='') as fh:
            for r in csv.DictReader(fh):
                try:
                    day = date.fromisoformat(r['session'])
                except (KeyError, TypeError, ValueError):
                    continue
                by_day[day] = (day, _num(r.get('equity'), default=bot.capital),
                               _num(r.get('open_positions'), int), _num(r.get('open_value')))
        bot.rows = [by_day[k] for k in sorted(by_day)]
    if os.path.exists(os.path.join(d, 'trades.csv')):
        with open(os.path.join(d, 'trades.csv'), newline='') as fh:
            bot.trades = sum(1 for _ in csv.DictReader(fh))
    return bot


def load(state_dir):
    return [load_bot(state_dir, mode, name) for mode, name in BOTS]


def selling_cost(mode, open_value, open_positions):
    """Rupees to close every open position at the close: the sale's charges
    and slippage, plus the DP charge once per delivery stock."""
    if open_value <= 0:
        return 0.0
    if mode in DELIVERY:
        legs = DEFAULT_CHARGES.cnc(0.0, open_value, dp_charged=False)
        legs += open_positions * DEFAULT_CHARGES.dp_charge_inr
    else:
        legs = DEFAULT_CHARGES.mis(0.0, open_value)
    return legs + open_value * SLIPPAGE


@dataclass
class Snapshot:
    day: date
    value: float            # all bots, cash plus open positions
    selling: float          # cost of selling the open positions
    tax: float
    by_bot: dict            # mode -> equity

    @property
    def pocket(self):
        return self.value - self.selling - self.tax


def snapshot(bots, day):
    value = selling = 0.0
    gain = {'delivery': 0.0, 'intraday': 0.0}
    by_bot = {}
    for b in bots:
        eq, n, ov = b.on(day)
        cost = selling_cost(b.mode, ov, n)
        value += eq
        selling += cost
        by_bot[b.mode] = eq
        gain['delivery' if b.mode in DELIVERY else 'intraday'] += eq - cost - b.capital
    tax = STCG_TAX * max(gain['delivery'], 0.0) + SPECULATIVE_TAX * max(gain['intraday'], 0.0)
    return Snapshot(day, value, selling, tax, by_bot)


def history(bots):
    days = sorted({r[0] for b in bots for r in b.rows})
    return [snapshot(bots, d) for d in days]


# ── formatting ──────────────────────────────────────────────────────────────
def inr(x, sign=False):
    """₹ with Indian digit grouping: 150000 -> ₹1,50,000."""
    n = int(round(abs(x)))
    s = str(n)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r'(\d)(?=(\d\d)+$)', r'\1,', head)
        s = f'{head},{tail}'
    lead = '−' if x < 0 and n else ('+' if sign and n else '')
    return f'{lead}₹{s}'


def pct(x):
    v = round(x * 100, 2)
    return f'{"+" if v > 0 else "−" if v < 0 else ""}{abs(v):.2f}%'


def when(d, year=True):
    return f'{d.day} {d:%b}' + (f' {d.year}' if year else '')


def _esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


# ── chart ───────────────────────────────────────────────────────────────────
THEMES = {   # dataviz reference palette; the line colour passes its checks on GitHub's backgrounds
    'light': {'line': '#2a78d6', 'ink': '#0b0b0b', 'ink2': '#52514e', 'grid': '#e1e0d9',
              'base': '#a8a79f', 'page': '#ffffff'},
    'dark': {'line': '#3987e5', 'ink': '#ffffff', 'ink2': '#c3c2b7', 'grid': '#2c2c2a',
             'base': '#5a5955', 'page': '#0d1117'},
}
W, H = 800, 320
LEFT, RIGHT, TOP, BOTTOM = 86, 112, 46, 38
FONT = "system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"


def nice_ticks(lo, hi, target=5):
    raw = (hi - lo) / (target - 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    v, ticks = math.floor(lo / step) * step, []
    while v < hi + step * 0.999:
        ticks.append(v)
        v += step
    return ticks


def chart_svg(points, capital, theme):
    """A line of `points` [(date, rupees)], with the money put in as a reference."""
    c = THEMES[theme]
    vals = [v for _, v in points] + [capital]
    lo, hi = min(vals), max(vals)
    if hi - lo < capital * MIN_SPAN:
        mid = (hi + lo) / 2
        lo, hi = mid - capital * MIN_SPAN / 2, mid + capital * MIN_SPAN / 2
    ticks = nice_ticks(lo, hi)
    y0, y1 = ticks[0], ticks[-1]
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM
    n = len(points)

    def x(i):
        return LEFT + (pw * i / (n - 1) if n > 1 else pw / 2)

    def y(v):
        return TOP + ph * (1 - (v - y0) / (y1 - y0))

    first, last = points[0], points[-1]
    gain = last[1] / capital - 1
    title = (f'Money in your pocket after charges and tax: {inr(last[1])} on {when(last[0])}, '
             f'{pct(gain)} on the {inr(capital)} put in on {when(first[0])}.')
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'role="img" aria-labelledby="t" font-family="{_esc(FONT)}">',
           f'<title id="t">{_esc(title)}</title>']
    key = ((c['line'], 2, 'In your pocket, after charges and tax'), (c['base'], 1, f'Put in: {inr(capital)}'))
    kx = LEFT                                                           # key above the plot
    for colour, width, label in key:
        out.append(f'<line x1="{kx}" x2="{kx + 18}" y1="15" y2="15" stroke="{colour}" stroke-width="{width}" '
                   f'stroke-linecap="round"/>')
        out.append(f'<text x="{kx + 26}" y="19" font-size="12" fill="{c["ink2"]}">{_esc(label)}</text>')
        kx += 26 + 7 * len(label) + 28
    for t in ticks:                                                     # gridlines and y labels
        out.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y(t):.1f}" y2="{y(t):.1f}" '
                   f'stroke="{c["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{LEFT - 10}" y="{y(t) + 4:.1f}" text-anchor="end" font-size="12" '
                   f'fill="{c["ink2"]}" style="font-variant-numeric:tabular-nums">{inr(t)}</text>')
    yc = y(capital)                                                     # the money put in
    out.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{yc:.1f}" y2="{yc:.1f}" '
               f'stroke="{c["base"]}" stroke-width="1"/>')
    k = min(n, 6)                                                       # x labels: up to 6 sessions
    idx = sorted({round(j * (n - 1) / (k - 1)) for j in range(k)}) if k > 1 else [0]
    for j, i in enumerate(idx):
        d = points[i][0]
        anchor = 'start' if (i == 0 and n > 1) else 'end' if i == n - 1 and n > 1 else 'middle'
        label = when(d, year=(j == 0 or d.year != points[idx[j - 1]][0].year))
        out.append(f'<text x="{x(i):.1f}" y="{H - 12}" text-anchor="{anchor}" font-size="12" '
                   f'fill="{c["ink2"]}">{label}</text>')
    pts = ' '.join(f'{x(i):.1f},{y(v):.1f}' for i, (_, v) in enumerate(points))
    if n > 1:
        area = f'M{x(0):.1f},{TOP + ph:.1f} L' + pts.replace(' ', ' L') + f' L{x(n - 1):.1f},{TOP + ph:.1f} Z'
        out.append(f'<path d="{area}" fill="{c["line"]}" fill-opacity="0.08"/>')
        out.append(f'<polyline points="{pts}" fill="none" stroke="{c["line"]}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
    ex, ey = x(n - 1), y(last[1])                                       # end dot and label
    out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="5" fill="{c["line"]}" stroke="{c["page"]}" '
               f'stroke-width="2"/>')
    ly = min(max(ey, TOP + 14), TOP + ph - 18)
    out.append(f'<text x="{ex + 12:.1f}" y="{ly:.1f}" font-size="14" font-weight="600" '
               f'fill="{c["ink"]}">{inr(last[1])}</text>')
    out.append(f'<text x="{ex + 12:.1f}" y="{ly + 17:.1f}" font-size="12" fill="{c["ink2"]}">'
               f'{pct(gain)}</text>')
    out.append('</svg>')
    return '\n'.join(out) + '\n'


# ── README block ────────────────────────────────────────────────────────────
def block(bots, snaps):
    capital = sum(b.capital for b in bots)
    head = [f'{START} — drawn by scripts/progress.py after every bot run; edits here are overwritten -->',
            '## How the money is doing', '']
    if not snaps:
        return '\n'.join(head + [f'_{inr(capital)} put in; the chart starts after the first session._', END])
    now, prev = snaps[-1], (snaps[-2] if len(snaps) > 1 else None)
    alt = (f'Line chart of the money in your pocket after charges and tax, from {inr(capital)} on '
           f'{when(snaps[0].day)} to {inr(now.pocket)} on {when(now.day)}.')
    lines = head + [
        f'**{inr(now.pocket)} in your pocket**, from {inr(capital)} put in on {when(snaps[0].day)} '
        f'({pct(now.pocket / capital - 1)}), at the close on {when(now.day)}.', '',
        '<picture>',
        f'  <source media="(prefers-color-scheme: dark)" srcset="{CHARTS}/chart-dark.svg">',
        f'  <img alt="{_esc(alt)}" src="{CHARTS}/chart-light.svg">',
        '</picture>', '',
        'That is what the three bots\' paper trades would have left you with as real trades, had you sold '
        'everything at that close: after Zerodha\'s charges and slippage on every trade, the cost of selling '
        'the open positions, and income tax on the profit.', '',
        f'| Bot | Since | Value | Return | Change on {when(now.day, year=False)} | Closed trades | Open now |',
        '|---|---|---|---|---|---|---|']
    for b in bots:
        v = now.by_bot[b.mode]
        change = inr(v - prev.by_bot[b.mode], sign=True) if prev else '—'
        if b.rows:
            lines.append(f'| {b.name} | {when(b.rows[0][0])} | {inr(v)} | {pct(v / b.capital - 1)} | {change} '
                         f'| {b.trades} | {b.holding} |')
        else:
            lines.append(f'| {b.name} | not started | {inr(v)} | — | — | {b.trades} | {b.holding} |')
    change = inr(now.value - prev.value, sign=True) if prev else '—'
    lines += [
        f'| **All three** | | **{inr(now.value)}** | **{pct(now.value / capital - 1)}** | {change} '
        f'| {sum(b.trades for b in bots)} | {sum(b.holding for b in bots)} |',
        f'| Selling the open positions | | {inr(-now.selling)} | | | | |',
        f'| Tax on the profit | | {inr(-now.tax)} | | | | |',
        f'| **In your pocket** | | **{inr(now.pocket)}** | **{pct(now.pocket / capital - 1)}** | | | |', '',
        '<sub>Value is cash plus open positions at the close, after the charges already paid. Tax is 20.8% on '
        'delivery profits (short-term capital gains) and 31.2% on intraday profits (speculative income at the '
        'top slab); a loss pays none. Dividends and tax-year boundaries are not modelled. Updated by each bot '
        'after its run.</sub>',
        END]
    return '\n'.join(lines)


def replace_block(readme, new):
    """`readme` with the progress block replaced, or None if it has no markers
    (deleting the block from the README switches it off)."""
    a, b = readme.find(START), readme.find(END)
    if a < 0 or b < a:
        return None
    return readme[:a] + new + readme[b + len(END):]


def main(root=ROOT):
    bots = load(os.path.join(root, 'state'))
    snaps = history(bots)
    capital = sum(b.capital for b in bots)
    if snaps:
        os.makedirs(os.path.join(root, CHARTS), exist_ok=True)
        points = [(s.day, s.pocket) for s in snaps]
        for theme in THEMES:
            with open(os.path.join(root, CHARTS, f'chart-{theme}.svg'), 'w') as fh:
                fh.write(chart_svg(points, capital, theme))
    path = os.path.join(root, 'README.md')
    with open(path) as fh:
        readme = fh.read()
    new = replace_block(readme, block(bots, snaps))
    if new is None:
        print('README.md has no progress markers; chart files only')
    elif new != readme:
        with open(path, 'w') as fh:
            fh.write(new)
    if snaps:
        s = snaps[-1]
        print(f'progress: {inr(s.pocket)} in your pocket on {s.day} ({pct(s.pocket / capital - 1)})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
