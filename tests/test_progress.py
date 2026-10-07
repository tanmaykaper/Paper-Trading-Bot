"""The README's progress block: scripts/progress.py and scripts/commit_state.sh."""

import csv
import json
import os
import shutil
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import date

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import progress as P  # noqa: E402

from nsebot.broker.paper import PaperBroker  # noqa: E402
from nsebot.costs import DEFAULT_CHARGES  # noqa: E402

README = '# title\n\n<!-- progress:start -->\n<!-- progress:end -->\n\nThe rest.\n'


def write_bot(root, mode, rows, capital=50_000.0, positions=0, trades=0):
    d = os.path.join(root, 'state', mode)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'state.json'), 'w') as fh:
        json.dump({'mode': mode, 'initial_cash': capital, 'positions': [{}] * positions}, fh)
    with open(os.path.join(d, 'equity.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['session', 'equity', 'cash', 'open_positions', 'open_value', 'realised_today',
                    'peak_equity', 'drawdown_pct'])
        for day, eq, n, ov in rows:
            w.writerow([day, eq, eq - ov, n, ov, 0, eq, 0])
    if trades:
        with open(os.path.join(d, 'trades.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['trade_id', 'net_pnl'])
            for i in range(trades):
                w.writerow([f't{i}', 10])


# ── numbers ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('x, want', [(0, '₹0'), (999, '₹999'), (1000, '₹1,000'), (150000, '₹1,50,000'),
                                     (12345678, '₹1,23,45,678'), (-567.4, '−₹567'), (0.4, '₹0')])
def test_inr_uses_indian_grouping(x, want):
    assert P.inr(x) == want


def test_signed_amounts_and_percentages():
    assert P.inr(567, sign=True) == '+₹567' and P.inr(-567, sign=True) == '−₹567' and P.inr(0, sign=True) == '₹0'
    assert P.pct(0.0113) == '+1.13%' and P.pct(-0.05) == '−5.00%' and P.pct(0.00001) == '0.00%'


def test_slippage_matches_the_paper_broker():
    assert P.SLIPPAGE == pytest.approx(PaperBroker().slip)


# ── the money ───────────────────────────────────────────────────────────────
def test_a_bot_is_its_uninvested_capital_before_its_first_session_and_carries_forward(tmp_path):
    write_bot(tmp_path, 'swing', [('2026-10-06', 51_000, 1, 10_000), ('2026-10-08', 49_000, 0, 0)])
    b = P.load_bot(tmp_path / 'state', 'swing', 'Swing')
    assert b.on(date(2026, 10, 5)) == (50_000, 0, 0.0)
    assert b.on(date(2026, 10, 7)) == (51_000, 1, 10_000)
    assert b.on(date(2026, 10, 9)) == (49_000, 0, 0.0)


def test_selling_cost_is_the_sale_legs_dp_per_stock_and_slippage():
    want = DEFAULT_CHARGES.cnc(0, 46_000, dp_charged=False) + 10 * DEFAULT_CHARGES.dp_charge_inr + 46_000 * 0.0005
    assert P.selling_cost('momentum', 46_000, 10) == pytest.approx(want)
    # by hand: STT 46.00 + exchange 1.37 + SEBI 0.05 + GST 0.25 + DP 10 × 15.34 + slippage 23.00
    assert want == pytest.approx(224.07, abs=0.01)
    assert P.selling_cost('intraday', 20_000, 1) == pytest.approx(DEFAULT_CHARGES.mis(0, 20_000) + 10)
    assert P.selling_cost('swing', 0, 0) == 0.0


def test_tax_is_on_profit_only_and_intraday_losses_do_not_offset_delivery_gains():
    def bot(mode, eq):
        b = P.Bot(mode, mode, 50_000.0)
        b.rows = [(date(2026, 10, 6), eq, 0, 0.0)]
        return b
    day = date(2026, 10, 6)
    s = P.snapshot([bot('momentum', 52_000), bot('swing', 49_000), bot('intraday', 45_000)], day)
    assert s.tax == pytest.approx(0.208 * 1_000)                             # +2,000 − 1,000 delivery; intraday ignored
    assert s.pocket == pytest.approx(146_000 - 208)
    s = P.snapshot([bot('momentum', 48_000), bot('swing', 50_000), bot('intraday', 51_000)], day)
    assert s.tax == pytest.approx(0.312 * 1_000)
    s = P.snapshot([bot('momentum', 48_000), bot('swing', 50_000), bot('intraday', 50_000)], day)
    assert s.tax == 0.0 and s.pocket == s.value == 148_000


def test_history_covers_every_session_any_bot_processed(tmp_path):
    write_bot(tmp_path, 'momentum', [('2026-10-05', 50_000, 0, 0), ('2026-10-06', 50_567.14, 10, 46_338.52)])
    write_bot(tmp_path, 'swing', [('2026-10-06', 50_000, 0, 0), ('2026-10-07', 50_100, 0, 0)])
    snaps = P.history(P.load(tmp_path / 'state'))
    assert [s.day.day for s in snaps] == [5, 6, 7]
    assert snaps[0].value == 150_000                                         # swing and intraday not started: cash
    assert snaps[-1].by_bot == {'momentum': 50_567.14, 'swing': 50_100, 'intraday': 50_000}
    assert snaps[-1].selling == pytest.approx(P.selling_cost('momentum', 46_338.52, 10))


# ── output ──────────────────────────────────────────────────────────────────
def test_chart_is_valid_svg_with_one_point_per_session_and_an_honest_axis():
    pts = [(date(2026, 10, d), 150_000 + 30 * d) for d in range(5, 10)]
    svg = P.chart_svg(pts, 150_000, 'light')
    root = ET.fromstring(svg)
    ns = '{http://www.w3.org/2000/svg}'
    line = root.find(f'{ns}polyline')
    assert len(line.get('points').split()) == 5
    ys = [float(t.text.replace('₹', '').replace(',', '')) for t in root.iter(f'{ns}text')
          if t.text.startswith('₹') and t.get('text-anchor') == 'end']
    assert max(ys) - min(ys) >= 150_000 * P.MIN_SPAN                        # a 0.2% wiggle is not blown up
    assert 'In your pocket, after charges and tax' in svg and 'Put in: ₹1,50,000' in svg
    ET.fromstring(P.chart_svg(pts[:1], 150_000, 'dark'))                     # a single session draws too


def test_block_replacement_is_idempotent_and_needs_the_markers():
    new = P.block([], [])
    once = P.replace_block(README, new)
    assert once.startswith('# title\n\n<!-- progress:start') and once.endswith('\n\nThe rest.\n')
    assert P.replace_block(once, new) == once
    assert P.replace_block('# no markers\n', new) is None


def test_main_writes_both_charts_and_the_table_and_a_rerun_changes_nothing(tmp_path):
    (tmp_path / 'README.md').write_text(README)
    write_bot(tmp_path, 'momentum', [('2026-10-05', 50_000, 0, 0), ('2026-10-06', 50_567.14, 10, 46_338.52)],
              positions=10)
    write_bot(tmp_path, 'swing', [('2026-10-05', 50_000, 0, 0), ('2026-10-06', 50_000, 0, 0)], trades=3)
    P.main(str(tmp_path))
    text = (tmp_path / 'README.md').read_text()
    for want in ('₹1,50,271 in your pocket', '| Momentum | 5 Oct 2026 | ₹50,567 | +1.13% | +₹567 | 0 | 10 |',
                 '| Swing | 5 Oct 2026 | ₹50,000 | 0.00% | ₹0 | 3 | 0 |', '| Intraday | not started |',
                 '| Selling the open positions | | −₹225 |', '| Tax on the profit | | −₹71 |',
                 'srcset="docs/progress/chart-dark.svg"', 'src="docs/progress/chart-light.svg"'):
        assert want in text, want
    assert text.endswith('<!-- progress:end -->\n\nThe rest.\n')
    charts = {t: (tmp_path / 'docs' / 'progress' / f'chart-{t}.svg').read_text() for t in ('light', 'dark')}
    assert '#2a78d6' in charts['light'] and '#3987e5' in charts['dark']
    P.main(str(tmp_path))
    assert (tmp_path / 'README.md').read_text() == text
    assert {t: (tmp_path / 'docs' / 'progress' / f'chart-{t}.svg').read_text() for t in charts} == charts


def test_before_any_session_the_block_says_so_and_no_chart_is_drawn(tmp_path):
    (tmp_path / 'README.md').write_text(README)
    P.main(str(tmp_path))
    assert '₹1,50,000 put in; the chart starts after the first session.' in (tmp_path / 'README.md').read_text()
    assert not (tmp_path / 'docs').exists()


def test_a_readme_without_markers_is_left_alone(tmp_path):
    (tmp_path / 'README.md').write_text('# mine\n')
    write_bot(tmp_path, 'swing', [('2026-10-05', 50_000, 0, 0)])
    P.main(str(tmp_path))
    assert (tmp_path / 'README.md').read_text() == '# mine\n'


def test_the_real_readme_has_the_block():
    text = open(os.path.join(ROOT, 'README.md')).read()
    assert P.replace_block(text, 'x') is not None


# ── commit_state.sh: two bots pushing at once ───────────────────────────────
def _git(cwd, *args):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
def test_bots_that_race_each_other_all_land_and_the_readme_shows_all_of_them(tmp_path):
    origin, a, b = tmp_path / 'origin.git', tmp_path / 'a', tmp_path / 'b'
    _git(tmp_path, 'init', '-q', '--bare', '-b', 'main', str(origin))
    seed = tmp_path / 'seed'
    for rel in ('scripts/progress.py', 'scripts/commit_state.sh', 'nsebot/__init__.py', 'nsebot/costs.py'):
        os.makedirs(seed / os.path.dirname(rel), exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), seed / rel)
    (seed / 'README.md').write_text(README)
    _git(seed, 'init', '-q', '-b', 'main')
    _git(seed, 'add', '-A')
    _git(seed, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'seed')
    _git(seed, 'push', '-q', str(origin), 'main')
    for clone in (a, b):
        _git(tmp_path, 'clone', '-q', str(origin), str(clone))
    env = {**os.environ, 'GITHUB_REF_NAME': 'main', 'RETRY_PAUSE': '0'}
    env.pop('GIT_DIR', None)

    def bot(clone, mode, eq, msg):
        write_bot(clone, mode, [('2026-10-06', eq, 0, 0)])
        return subprocess.run(['bash', 'scripts/commit_state.sh', msg], cwd=clone, env=env,
                              capture_output=True, text=True)

    assert bot(a, 'swing', 50_100, 'swing').returncode == 0
    # b is now behind. Its first push also loses a race: a pre-push hook lands intraday from `a` first.
    hook = b / '.git' / 'hooks' / 'pre-push'
    hook.write_text('#!/usr/bin/env bash\n'
                    f'[ -e "{tmp_path}/raced" ] && exit 0\n'
                    f'touch "{tmp_path}/raced"\n'
                    'unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE\n'
                    f'cd "{a}" && mkdir -p state/intraday\n'
                    'echo \'{"mode": "intraday", "initial_cash": 50000}\' > state/intraday/state.json\n'
                    'printf "session,equity,cash,open_positions,open_value\\n2026-10-06,50200,50200,0,0\\n"'
                    ' > state/intraday/equity.csv\n'
                    'bash scripts/commit_state.sh intraday >/dev/null\n'
                    'exit 1\n')
    hook.chmod(hook.stat().st_mode | stat.S_IEXEC)
    r = bot(b, 'momentum', 50_300, 'momentum')
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / 'raced').exists()

    check = tmp_path / 'check'
    _git(tmp_path, 'clone', '-q', str(origin), str(check))
    assert _git(check, 'log', '--format=%s').split() == ['momentum', 'intraday', 'swing', 'seed']
    for sha in _git(check, 'log', '--format=%H', '-3').split():               # each bot: one commit, state + chart
        files = _git(check, 'show', '--name-only', '--format=', sha).split()
        assert 'README.md' in files and any(f.startswith('state/') for f in files), files
    text = (check / 'README.md').read_text()
    for row in ('| Momentum | 6 Oct 2026 | ₹50,300 |', '| Swing | 6 Oct 2026 | ₹50,100 |',
                '| Intraday | 6 Oct 2026 | ₹50,200 |'):
        assert row in text, row
