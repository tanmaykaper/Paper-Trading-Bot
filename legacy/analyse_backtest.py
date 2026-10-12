# analyse_backtest.py ── post-run diagnostics on backtest_results_run_d.csv
#
# The printed report answers "how much"; this answers "why", and the rotation
# question in particular cannot be settled from the summary alone. Slot Rotation
# was 46% of exits in the last run — whether that is the allocator recycling
# capital well or churning it away is decided entirely by the P&L of those
# rotated trades, which only the per-trade file contains.
#
#   python3 analyse_backtest.py

import sys
import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else 'backtest_results_run_d.csv'
df = pd.read_csv(PATH)
df = df[df['status'] == 'CLOSED'] if 'status' in df else df
if len(df) == 0:
    raise SystemExit(f"no closed trades in {PATH}")

pnl = df['net_pnl'].astype(float)
print(f"\n{len(df)} closed trades | net ₹{pnl.sum():+,.0f} | "
      f"win {(pnl > 0).mean() * 100:.1f}% | expectancy ₹{pnl.mean():+,.0f}\n")


def table(col, label, extra=None):
    if col not in df:
        return
    g = df.groupby(col)['net_pnl'].agg(
        n='size', total='sum', mean='mean', win=lambda s: (s > 0).mean() * 100).round(1)
    if extra and extra in df:
        g['avg_hold'] = df.groupby(col)[extra].mean().round(1)
    print(f"── by {label} " + "─" * (58 - len(label)))
    print(g.sort_values('total', ascending=False).to_string(), "\n")


# THE question: is rotation recycling capital or destroying it? If rotated
# trades carry a worse mean than the rest, the allocator is cutting positions
# that would have paid — and the brakes added in this version are the fix.
if 'exit_reason' in df:
    df['exit_group'] = df['exit_reason'].astype(str).str.split('(').str[0].str.strip()
    table('exit_group', 'exit reason', 'hold_days')
    rot = df[df['exit_group'] == 'Slot Rotation']['net_pnl'].astype(float)
    rest = df[df['exit_group'] != 'Slot Rotation']['net_pnl'].astype(float)
    if len(rot) and len(rest):
        print(f"  ROTATION VERDICT: rotated ₹{rot.mean():+,.0f}/trade (n={len(rot)}) "
              f"vs ₹{rest.mean():+,.0f} for everything else (n={len(rest)})")
        print(f"  {'Rotation is DESTROYING value — the brakes are justified.'if rot.mean() < rest.mean() else 'Rotation is ADDING value — consider loosening the brakes.'}\n")

table('entry_type', 'pattern', 'hold_days')
table('market_state_at_entry', 'market state at entry')

for col, label in (('quality_score', 'entry quality'), ('momentum_percentile', 'momentum rank')):
    if col in df and df[col].notna().sum() > 6:
        d = df.dropna(subset=[col]).copy()
        d['band'] = pd.qcut(d[col].astype(float), 3,
                            labels=['low', 'mid', 'high'], duplicates='drop')
        g = d.groupby('band', observed=True)['net_pnl'].agg(
            n='size', total='sum', mean='mean', win=lambda s: (s > 0).mean() * 100).round(1)
        print(f"── by {label} band " + "─" * (53 - len(label)))
        print(g.to_string())
        lo, hi = g['mean'].iloc[0], g['mean'].iloc[-1]
        print(f"  {'monotone and positive — usable as a gate' if hi > lo else 'NOT monotone — this score is not yet predictive'}\n")

if 'commission' in df:
    notional = df['entry_price'].astype(float) * df['position_size'].astype(float)
    comm = df['commission'].astype(float)
    gross = df['gross_pnl'].astype(float).sum() if 'gross_pnl' in df else None
    print(f"── costs " + "─" * 56)
    print(f"  ₹{comm.sum():,.0f} total | ₹{comm.mean():.0f}/trade | "
          f"{(comm / notional).mean() * 1e4:.0f} bps avg"
          + (f" | {comm.sum() / gross * 100:.0f}% of gross" if gross and gross > 0 else ""))
