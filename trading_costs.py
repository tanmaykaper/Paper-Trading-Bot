# trading_costs.py  ── v2  (ITEMISED NSE DELIVERY COSTS + SAME-DAY DP LEDGER)
# ═════════════════════════════════════════════════════════════════════════════
# v1 modelled costs as one blended percentage plus a flat charge per sell. The
# blend was roughly right and it fixed a genuine 100-200x understatement, but it
# hid two things that matter now that position sizes and tranche counts are
# themselves decision variables:
#
# 1. THE FLAT DP CHARGE IS PER SCRIP PER DAY, NOT PER SELL.
#    A depository levies its charge once per scrip per day regardless of how
#    many sell instructions hit it. v1 charged it per CLOSED ROW, so a
#    3-tranche position exiting together was billed ~₹60 where a broker bills
#    ~₹20. That is a 200% overstatement on exactly the exits the scaled-exit
#    system is designed to produce, and it biased every tranching decision —
#    including the tranche_ok gate in signal_generator — against a structure
#    that is cheaper in reality than the model claimed.
#    DPChargeLedger fixes it by remembering which (scrip, date) pairs have
#    already been billed.
#
# 2. THE COMPONENTS BEHAVE DIFFERENTLY AND SHOULD BE VISIBLE SEPARATELY.
#    STT is symmetric across both legs, stamp duty is buy-side only, GST
#    applies only to the taxable services and not to STT or stamp duty, and the
#    DP charge does not scale at all. Collapsing them into one percentage means
#    that when a broker changes one line item you cannot tell which constant to
#    move, and it makes the cost report in the daily brief uninformative about
#    WHY a trade was expensive.
#
# Figures are for a typical Indian discount broker on CASH DELIVERY with zero
# brokerage, as of the last update. They shift, and they vary by broker — treat
# them as "roughly right", and edit the block below if yours differ. This is
# still the single place that needs changing.
# ═════════════════════════════════════════════════════════════════════════════

from datetime import date

# ── Statutory and exchange components ───────────────────────────────────────
BROKERAGE_PCT        = 0.0        # most discount brokers: ₹0 on delivery
STT_PCT              = 0.001      # 0.1% on BOTH legs for delivery
EXCHANGE_TXN_PCT     = 0.0000297  # NSE cash
SEBI_TURNOVER_PCT    = 0.000001   # ₹10 per crore
STAMP_DUTY_PCT       = 0.00015    # 0.015%, BUY side only
GST_PCT              = 0.18       # on brokerage + exchange txn + SEBI, not on STT/stamp
DP_CHARGE_PER_SCRIP  = 20.0       # flat, per scrip per day, SELL side only

# Retained under their v1 names so anything importing them keeps working. The
# blended figure is now derived from the itemised components rather than
# asserted alongside them, so the two cannot drift apart.
PCT_COST_PER_LEG = (STT_PCT + EXCHANGE_TXN_PCT + SEBI_TURNOVER_PCT
                    + (EXCHANGE_TXN_PCT + SEBI_TURNOVER_PCT + BROKERAGE_PCT) * GST_PCT
                    + STAMP_DUTY_PCT / 2.0)
FLAT_CHARGE_PER_SELL = DP_CHARGE_PER_SCRIP


def leg_cost(value, side):
    """Itemised cost of one leg. side is 'buy' or 'sell'. Returns a dict."""
    value = max(float(value), 0.0)
    brokerage = value * BROKERAGE_PCT
    stt = value * STT_PCT
    exch = value * EXCHANGE_TXN_PCT
    sebi = value * SEBI_TURNOVER_PCT
    stamp = value * STAMP_DUTY_PCT if side == 'buy' else 0.0
    gst = (brokerage + exch + sebi) * GST_PCT
    return {'brokerage': brokerage, 'stt': stt, 'exchange': exch, 'sebi': sebi,
            'stamp_duty': stamp, 'gst': gst,
            'total': brokerage + stt + exch + sebi + stamp + gst}


def cost_breakdown(entry_price, exit_price, position_size, dp_charge=DP_CHARGE_PER_SCRIP):
    """
    Full itemised round-trip cost. dp_charge can be passed as 0.0 when a
    DPChargeLedger has already billed this scrip today — which is the whole
    point of the ledger.
    """
    buy = leg_cost(float(entry_price) * int(position_size), 'buy')
    sell = leg_cost(float(exit_price) * int(position_size), 'sell')
    total = buy['total'] + sell['total'] + float(dp_charge)
    return {
        'buy': {k: round(v, 2) for k, v in buy.items()},
        'sell': {k: round(v, 2) for k, v in sell.items()},
        'dp_charge': round(float(dp_charge), 2),
        'total': round(total, 2),
    }


def round_trip_commission(entry_price, exit_price, position_size,
                          dp_charge=DP_CHARGE_PER_SCRIP):
    """
    Unchanged signature and meaning — total rupee cost of one full entry+exit.
    Every existing call site keeps working, and callers that know the DP charge
    has already been billed for this scrip today pass dp_charge=0.0.
    """
    return cost_breakdown(entry_price, exit_price, position_size, dp_charge)['total']


class DPChargeLedger:
    """
    Bills the depository charge once per scrip per day.

    Used by the live manager and the backtester so both price a multi-tranche
    same-day exit the way a broker actually does. Without it, a position that
    closes its quick, core and runner tranches on one day pays the charge three
    times in the model and once in reality — a distortion that falls entirely
    on the tranched structure and nowhere else, which is precisely the kind of
    asymmetric modelling error that makes a design look worse than it is.

    Deliberately explicit rather than global: a backtest and a live run must not
    share a ledger, and a backtest replaying two policies needs one each.
    """

    def __init__(self):
        self._billed = set()

    def charge_for(self, symbol, on_date=None):
        """Returns the DP charge owed for selling this scrip on this date —
        the full amount the first time, zero on every subsequent sell that day."""
        key = (str(symbol), str(on_date or date.today()))
        if key in self._billed:
            return 0.0
        self._billed.add(key)
        return DP_CHARGE_PER_SCRIP

    def round_trip(self, symbol, entry_price, exit_price, position_size, on_date=None):
        return round_trip_commission(entry_price, exit_price, position_size,
                                     dp_charge=self.charge_for(symbol, on_date))

    def reset(self):
        self._billed.clear()


def cost_report(trades_df):
    """
    Portfolio-level cost accounting for the daily brief and the backtest report:
    where the money actually went, in rupees and in basis points of notional.

    Reported because the single largest realised leak in this project's history
    was invisible until someone summed this column: ₹573 of costs against ₹767
    of gross profit across 24 rows.
    """
    if trades_df is None or len(trades_df) == 0 or 'commission' not in trades_df:
        return {'n': 0}
    import pandas as pd
    d = trades_df.dropna(subset=['commission'])
    if len(d) == 0:
        return {'n': 0}
    notional = (d['entry_price'].astype(float) * d['position_size'].astype(float))
    comm = d['commission'].astype(float)
    gross = d['gross_pnl'].astype(float).sum() if 'gross_pnl' in d else None
    bps = (comm / notional.replace(0, pd.NA)).astype(float) * 1e4
    return {
        'n': int(len(d)),
        'total_cost': round(float(comm.sum()), 2),
        'avg_cost': round(float(comm.mean()), 2),
        'avg_bps': round(float(bps.mean()), 1),
        'worst_bps': round(float(bps.max()), 1),
        'worst_symbol': str(d.loc[bps.idxmax(), 'symbol']) if 'symbol' in d else None,
        'pct_of_gross': (round(float(comm.sum()) / gross * 100, 1)
                         if gross and gross > 0 else None),
    }
