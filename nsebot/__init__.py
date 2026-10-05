"""nsebot — a lean NSE trading engine built on free data.

Two modes share one risk and execution core:
  swing     CNC delivery, daily bars, decided after the close
  intraday  MIS, 5-minute bars, squared off before 15:15 IST

Market data is free (Yahoo Finance via yfinance). Execution is paper by
default; the Zerodha Kite adapter switches on only when Kite Connect
credentials are configured.
"""

__version__ = '3.0.0'
