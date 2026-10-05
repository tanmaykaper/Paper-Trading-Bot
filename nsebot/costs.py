"""Zerodha charges for NSE equity, itemised, for CNC (delivery) and MIS (intraday).

Costs were 70% of V1's net loss (₹911 of ₹1,307), mostly because three
tranches paid the flat DP charge three times. MIS pays no DP charge at all,
which is a large part of why intraday is worth having alongside swing.

Rates are Zerodha's published equity schedule. Brokers and regulators revise
these; every rate is a field on ZerodhaCharges, so a change is one number.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ZerodhaCharges:
    # CNC (delivery)
    cnc_brokerage_pct: float = 0.0
    cnc_stt_pct: float = 0.001            # 0.1% on buy AND sell
    cnc_stamp_pct: float = 0.00015        # 0.015%, buy only
    dp_charge_inr: float = 15.34          # per scrip per sell day, incl. GST
    # MIS (intraday)
    mis_brokerage_pct: float = 0.0003     # 0.03% or ₹20 per executed order, whichever lower
    mis_brokerage_cap_inr: float = 20.0
    mis_stt_pct: float = 0.00025          # 0.025%, sell only
    mis_stamp_pct: float = 0.00003        # 0.003%, buy only
    # Both
    exchange_pct: float = 0.0000297       # NSE transaction charge
    sebi_pct: float = 0.000001            # ₹10 per crore
    gst_pct: float = 0.18                 # on brokerage + exchange + SEBI

    def _common(self, value, brokerage):
        exch = value * self.exchange_pct
        sebi = value * self.sebi_pct
        return exch, sebi, (brokerage + exch + sebi) * self.gst_pct

    def cnc(self, buy_value, sell_value, dp_charged=True):
        """Round-trip delivery cost in rupees."""
        total = 0.0
        for value, side in ((buy_value, 'buy'), (sell_value, 'sell')):
            brokerage = value * self.cnc_brokerage_pct
            exch, sebi, gst = self._common(value, brokerage)
            stt = value * self.cnc_stt_pct
            stamp = value * self.cnc_stamp_pct if side == 'buy' else 0.0
            total += brokerage + exch + sebi + gst + stt + stamp
        return total + (self.dp_charge_inr if dp_charged else 0.0)

    def mis(self, buy_value, sell_value):
        """Round-trip intraday cost in rupees (long or short — same legs)."""
        total = 0.0
        for value, side in ((buy_value, 'buy'), (sell_value, 'sell')):
            brokerage = min(value * self.mis_brokerage_pct, self.mis_brokerage_cap_inr)
            exch, sebi, gst = self._common(value, brokerage)
            stt = value * self.mis_stt_pct if side == 'sell' else 0.0
            stamp = value * self.mis_stamp_pct if side == 'buy' else 0.0
            total += brokerage + exch + sebi + gst + stt + stamp
        return total

    def round_trip(self, product, buy_value, sell_value, dp_charged=True):
        if product == 'MIS':
            return self.mis(buy_value, sell_value)
        return self.cnc(buy_value, sell_value, dp_charged)


DEFAULT_CHARGES = ZerodhaCharges()
