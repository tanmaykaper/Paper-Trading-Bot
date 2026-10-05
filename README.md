# nsebot — NSE paper trading on free data

A lean NSE trading engine with three modes that share one risk core:

| Mode | Product | Data | When it runs | Strategy |
|---|---|---|---|---|
| **swing** | CNC (delivery) | Yahoo daily bars (free) | 17:15 IST on weekdays, GitHub Actions | Dip reversion inside long-term uptrends |
| **intraday** | MIS (5x margin) | Yahoo 5-minute bars (free) | 09:18–15:12 IST, GitHub Actions | Opening-range breakout with VWAP and volume confirmation, long and short |
| **momentum** | CNC (delivery), paper only | NSE's equity list + Yahoo daily bars (free) | 17:40 IST on weekdays, GitHub Actions | The 10 strongest of every NSE-listed stock by 12-1 momentum, re-ranked every 5 sessions |

Execution is **paper by default**. A Zerodha Kite Connect adapter switches on when API credentials are configured (see [Kite](#zerodha-kite)).

This is V3, a rebuild after V2 stopped trading on 14 Sep 2026. The full story is in [`docs/AUTOPSY_V2.md`](docs/AUTOPSY_V2.md) (why V2 froze) and [`docs/RESEARCH.md`](docs/RESEARCH.md) (what the data supports). The V1/V2 code lives in [`legacy/`](legacy/).

---

## Honest status

All numbers below come from the bot's own code run on real NSE data, net of Zerodha costs.

- **Momentum (the main candidate for the smallcap-fund target).** The only design that passed a pre-registered test on a symbol list without hindsight (every NSE-listed stock), net of Zerodha costs:
  - Untouched Jul 2018 – Jul 2023: **+22.6% CAGR**, against +21.7% for the median active smallcap fund and +28.7% for the best.
  - Jul 2018 – Oct 2026: **+30.2% CAGR**, against +18.5% for the median fund and +24.5% for the best.
  - Worst drawdown **−52.8%**, deeper than any fund's. It lost about 15% in 2018–19 and in 2021–22, years when the median fund made money.
  - Still flattering: stocks delisted before today are missing from the data. Short-term capital gains tax is not modelled (the report estimates it).
  - Expect returns somewhere between fund-like and better, with deep falls. Beating the best fund is plausible, not proven. Details: [`docs/RESEARCH.md`](docs/RESEARCH.md), rounds 5–6.
- **Swing.** S4b dip reversion passed a pre-registered out-of-sample test on 2023–26 data (**+0.69% per trade, 64% win rate, 168 trades**; 95% interval about −0.06% to +1.44%).
  - It **failed on untouched 2018–2020 data**: −4.9% and −11.8% in the two years it traded, then it hit the 25% drawdown latch in the COVID crash.
  - It keeps running on paper because that costs nothing. Don't expect it to carry the target.
- **Intraday.** No variant has passed yet: all are net-negative after costs on Yahoo's 59-day window. The Kelly sizer therefore trades the 0.5% risk floor. Treat this mode as live data collection until its own trades earn more.
- **The V2 breakout signal lost money** in both halves of the history (out-of-sample −0.87% per trade) and has been removed from trading.
- **Full-engine backtest** (₹50k, Jun 2024 to Oct 2026, the real engine end to end):
  - The first production config lost **−12.7%**.
  - Attribution traced the loss to three causes:
    1. Kelly sizing on net results shrank positions to about ₹3k, where the flat DP charge and STT eat about 1% per trade.
    2. A loss-streak breaker whose reduced-size mode deadlocked.
    3. A regime size dial that pushed positions below economic size.
  - The current config (notional sizing with an ₹8k floor, no streak breaker or size dial on swing) returned **+6.7%** with a −18.7% max drawdown. In-sample it lost −7.3%; out-of-sample it made +15.2%.
  - That config was chosen after seeing the data, so only forward paper trading tests it cleanly. Details are in [`docs/RESEARCH.md`](docs/RESEARCH.md), round 3.

Run `python -m nsebot backtest` (or the `nsebot research` workflow) for the full-engine portfolio backtest on current data.

---

## How a day works

```
Yahoo (free) ──► DailyMarket / 5m bars ──► signal engine ──► allocator + half-Kelly sizer
                                                                      │
   state/  ◄── ledger (state.json, trades.csv, equity.csv) ◄── broker (paper | kite)
     ▲                                                                │
     └──── committed by the workflow ◄── exits (stop / target / trail / 15:10 square-off)
                                         circuit breakers (loss streak, daily loss, drawdown, kill file)
```

**Swing, end of day:**
1. Fill yesterday's plans at today's open. A plan cancels if the open is already through its stop.
2. Walk each position through every bar it hasn't seen yet. Exits: gap-aware stop, then the first close above the 5-day EMA, then a 7-session time stop.
3. Check the circuit breakers.
4. Plan tomorrow's entries.
5. Persist state.

A session that has already been processed is skipped, so re-runs are safe. Missed days catch up bar by bar.

**Momentum, end of day (`nsebot/engine/momentum.py`):**
1. Fill the last rebalance's orders at today's open: sales first, then buys in rank order. A stock frozen at its circuit limit all day, or one that didn't trade, gets no fill. A sale that can't fill waits for the next rebalance.
2. Mark the book to today's closes and check the breakers (55% drawdown latch, kill switch).
3. Every 5 sessions, rank every NSE-listed stock that is eligible today (₹5 cr median traded value, price ≥ ₹50, a year of history) by 12-1 momentum. Sell holdings that fell out of the top 30 or stopped being eligible. Fill empty slots from the top at a tenth of equity each.
4. Persist state.

The live engine is checked against the research simulator: run session by session on the same history, it produces the same equity every day and the same total costs (`tests/test_momentum.py`).

**Intraday, polled after every 5-minute bar:**
1. Fill pending orders at the next bar's open.
2. Run exits: square-off at **15:10**, then the gap-aware stop, then the trail (breakeven at +1R, then a chandelier trail at 2× ATR, tightening to 1.2× ATR).
3. Check breakers. Hitting the daily loss limit closes everything.
4. Place new opening-range breakout entries (09:30–13:30, at most one trade per stock per day).
5. Persist state after every step.

---

## Risk architecture (`nsebot/risk/`)

| Control | Swing | Intraday | Momentum |
|---|---|---|---|
| Sizing | **Notional**: 20% of equity per position. A trade that can't reach ₹8k is skipped, never shrunk. Below ₹40k equity, where 20% is under ₹8k, positions are raised to ₹8k while every other cap allows it, so a shrunken book doesn't freeze. | Half-Kelly on net R, shrunk toward the out-of-sample prior | **Equal weight**: a tenth of equity per slot, as tested |
| Risk per trade | Capped at 3% of equity. Kelly is monitored and reported, but doesn't set the size. | 0.5% floor (measured edge is negative) to 2% cap | No stop: a holding leaves when it drops out of the top 30 or stops being eligible |
| Buying power | Cash × 1 (CNC) | Cash × 5 (MIS margin) | Cash × 1 (CNC); sale proceeds fund the same open's buys |
| Other caps | 5 positions, 2 per sector, 15% total open risk, ≤1% of the stock's daily traded value. The regime dial limits only the number of new entries per day. | 3 positions, 5% total open risk, 1 per sector | 10 positions; no sector cap (none was tested) |
| Consecutive-loss breaker | **None.** In dip reversion, losses cluster just before the best rebounds. | 3 losses: done for the day. The next session trades at half size until a winner, then reduced size expires anyway. | None |
| Edge monitor | Warns when the bot's own Kelly estimate is negative after 60 or more trades | Same | — |
| Paralysis alert | Warns (and emails) after 3 sessions in a row with signals and free slots but no entry | — | Warns after 2 rebalances in a row that buy nothing despite free slots; skips (and warns about) a rebalance that sees implausibly few eligible stocks, a data fault, rather than selling the book |
| Daily loss limit | 5% | 3%, closes everything | None |
| Drawdown latch | 25% from peak: entries stop until a human deletes `state/<mode>/BREAKER_TRIPPED_<mode>` (committed by the workflow, so it survives fresh checkouts). Deleting it restarts the drawdown from that day's equity. | Same | **55%** from peak (its backtested worst was 52.8%); stops buys, never sales. Same file rule |
| Kill switch | Create a file named `STOP_TRADING` | Same | Same |

Exits are never blocked by a breaker. Prices are snapped to the instrument's tick size in the conservative direction: the band table as a fallback, Kite's instrument master when available.

---

## Run it

```bash
pip install -r requirements.txt

python -m nsebot swing        # end-of-day cycle (after 17:00 IST)
python -m nsebot intraday     # blocks until the 15:10 square-off
python -m nsebot momentum     # momentum sleeve, end of day (paper only)
python -m nsebot status       # cash, open positions, pending orders, P&L per sleeve
python -m nsebot backtest     # full-engine portfolio backtest on real data

python -m pytest tests diagnostics -q
```

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `SWING_CAPITAL`, `INTRADAY_CAPITAL`, `MOMENTUM_CAPITAL` | 50000 | Paper starting capital for each sleeve |
| `NSEBOT_BROKER` | `paper` | `paper` or `kite` (swing and intraday; momentum is paper only) |
| `EMAIL_SENDER`, `EMAIL_PASSWORD`, `EMAIL_RECIPIENT` | — | Daily report and alert emails (Gmail app password) |

**GitHub Actions:**
- `main.yml` runs the swing cycle, `intraday.yml` the intraday session and `momentum.yml` the momentum sleeve. All three commit `state/` and share a concurrency group, so only one writes state at a time.
- On rebalance days `momentum.yml` downloads NSE's equity list (saved to `state/momentum/` as a fallback for up to 30 days) and about 2,600 symbols from Yahoo, which takes a few minutes. Other days it fetches only the holdings.
- Scheduled workflows only run from the **default branch**, so this code must be merged into `main` to go live.
- `research.yml` runs the tests and the backtest or experiments on `claude/**` branches.

---

## Zerodha Kite

```bash
pip install -r requirements-kite.txt
export KITE_API_KEY=... KITE_API_SECRET=...
python -m nsebot.kite_login                  # once per trading day: open the URL, log in
python -m nsebot.kite_login <request_token>  # writes .kite_token.json (gitignored)
NSEBOT_BROKER=kite python -m nsebot swing
```

What the adapter (`nsebot/broker/kite.py`) handles:
- **Rate limits.** Per-endpoint token buckets plus caps on orders per minute and per day.
- **Daily token expiry (~06:00 IST).** An expired token raises `TokenExpired`: the bot alerts and halts new entries. Exchange-side GTT / SL-M stops keep open positions protected even after the token dies.
- **Safe retries.** After an ambiguous failure, the adapter searches the order book for the order's tag before retrying, so a timeout can't turn into a duplicate position.
- **Pre-trade margin check.** A shortfall raises an error instead of the exchange silently rejecting the order.
- **Exchange-correct prices.** Tick and lot sizes come from the instrument master.

Requirements and caveats:
- You need a Kite Connect subscription. The free personal account doesn't include the API.
- Under SEBI's retail-algo rules, live order placement requires a **static IP registered with Zerodha**. GitHub Actions runners don't have one, so use a VPS. Check Zerodha's current requirements first.
- The adapter is unit-tested against a fake client only. Do the first live runs at minimum size.

---

## Layout

```
nsebot/
  signals/      reversion.py (swing, live) · momentum.py (12-1 ranking) · intraday.py (ORB) · swing.py (breakout, research only)
  risk/         sizing.py · exits.py · breakers.py · allocator.py
  engine/       swing.py · momentum.py · intraday.py · book.py · market_view.py
  broker/       paper.py · kite.py · base.py
  research/     event_study.py · experiments.py · attribution.py · hurdle.py · phase5*.py · symbol_check.py
  backtest.py   data.py  listing.py  market.py  ledger.py  costs.py  ratelimit.py  regime.py  notify.py
state/          live paper state, committed by the workflows
docs/           AUTOPSY_V2.md · RESEARCH.md
diagnostics/    offline reproduction of the V2 locks
legacy/         V1/V2 code and their trade logs
```

---

## Disclaimer

This trades paper money. It is not investment advice. The edges are statistically weak and the momentum sleeve's drawdowns are deep, as documented above. Anyone adapting it for real capital does so at their own risk, and should start small with a Kite shadow run.
