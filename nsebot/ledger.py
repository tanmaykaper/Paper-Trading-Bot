"""Durable state for one trading sleeve (swing or intraday).

Everything the next run needs lives under state/<mode>/ and is committed by
the workflow — the persistence step V2 skipped for its own pending book,
which is why every plan it ever placed vanished overnight (AUTOPSY §2):

  state.json   cash, open positions, pending entries, breaker state,
               last processed session — the whole state machine
  trades.csv   every closed trade, append-only, with net P&L and net R
  equity.csv   one mark-to-market row per processed session

Writes are atomic (temp file + rename), so a runner killed mid-write leaves
the previous state intact rather than a truncated JSON the next run cannot
parse.
"""

import csv
import json
import os
import tempfile
from dataclasses import asdict

import pandas as pd

from .risk.exits import Position

TRADE_FIELDS = ['trade_id', 'mode', 'symbol', 'side', 'qty', 'entry_time', 'entry_price',
                'initial_stop', 'exit_time', 'exit_price', 'exit_reason', 'bars_held',
                'gross_pnl', 'costs', 'net_pnl', 'net_r', 'trigger', 'score']
EQUITY_FIELDS = ['session', 'equity', 'cash', 'open_positions', 'open_value', 'realised_today',
                 'peak_equity', 'drawdown_pct']


def _atomic_write(path, text):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or '.', prefix='.tmp_')
    try:
        with os.fdopen(fd, 'w') as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def position_to_dict(p):
    d = asdict(p)
    d['entry_time'] = str(p.entry_time)
    return d


def position_from_dict(d):
    d = dict(d)
    d['entry_time'] = pd.Timestamp(d['entry_time'])
    return Position(**d)


class Ledger:

    def __init__(self, root, mode, initial_cash):
        self.mode = mode
        self.dir = os.path.join(root, mode)
        self.state_path = os.path.join(self.dir, 'state.json')
        self.trades_path = os.path.join(self.dir, 'trades.csv')
        self.equity_path = os.path.join(self.dir, 'equity.csv')
        self.state = self._load(initial_cash)

    # ── state ───────────────────────────────────────────────────────────────
    def _load(self, initial_cash):
        if os.path.exists(self.state_path):
            with open(self.state_path) as fh:
                raw = json.load(fh)
            raw['positions'] = [position_from_dict(p) for p in raw.get('positions', [])]
            return raw
        return {'mode': self.mode, 'initial_cash': float(initial_cash), 'cash': float(initial_cash),
                'positions': [], 'pending': [], 'breakers': {}, 'last_processed': None,
                'traded_today': [], 'trade_seq': 0}

    def save(self):
        payload = dict(self.state)
        payload['positions'] = [position_to_dict(p) for p in self.state['positions']]
        _atomic_write(self.state_path, json.dumps(payload, indent=1, default=str) + '\n')

    @property
    def positions(self):
        return self.state['positions']

    @property
    def cash(self):
        return float(self.state['cash'])

    def next_trade_id(self, symbol, when):
        self.state['trade_seq'] = int(self.state.get('trade_seq', 0)) + 1
        return f"{self.mode[0].upper()}{pd.Timestamp(when):%Y%m%d}-{symbol}-{self.state['trade_seq']}"

    # ── journals ────────────────────────────────────────────────────────────
    def append_trade(self, row):
        self._append(self.trades_path, TRADE_FIELDS, row)

    def append_equity(self, row):
        self._append(self.equity_path, EQUITY_FIELDS, row)

    @staticmethod
    def _append(path, fields, row):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            if new:
                w.writeheader()
            w.writerow({k: row.get(k) for k in fields})

    def closed_trades(self):
        if not os.path.exists(self.trades_path):
            return pd.DataFrame(columns=TRADE_FIELDS)
        return pd.read_csv(self.trades_path)

    def realised_r(self):
        """Net-of-cost R multiples of closed trades — what the Kelly sizer learns from."""
        t = self.closed_trades()
        return [] if t.empty else pd.to_numeric(t['net_r'], errors='coerce').dropna().tolist()

    def equity_history(self):
        if not os.path.exists(self.equity_path):
            return pd.DataFrame(columns=EQUITY_FIELDS)
        return pd.read_csv(self.equity_path)
