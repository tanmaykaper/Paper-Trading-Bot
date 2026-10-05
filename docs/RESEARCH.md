# Signal research — real NSE data, out-of-sample tested

All numbers come from the bot's own signal code run on free Yahoo data on
GitHub Actions (`.github/workflows/research.yml`). Every entry is at the next
bar's open, every exit is gap-aware, and Zerodha costs are charged on both
legs. Reproduce with `python -m nsebot.research.experiments` or
`python -m nsebot.research.event_study`.

## Round 1: Phase 2 signals (run 37261644396)

| Signal | Sample | Result |
|---|---|---|
| Swing breakout / thrust | 5,562 triggers, 255 symbols | Candidates per session: **8.6** (V2: 0). Excess vs the average stock at 5/10/20 days: **−0.29% / −0.32% / −0.25%**. Stop-or-15-day trade: 42% win, **−0.95%/trade**, Kelly **−0.10** |
| Intraday ORB + VWAP | 31 trades, 57 sessions | 23% win, **−0.51R**, longs 0 of 12 |

Verdict: the bot trades again, but neither signal makes money.

## Round 2: Phase 2b pre-registered experiments (run 37265316785)

- **History:** 14 Aug 2023 to 5 Oct 2026 (774 sessions, 254 symbols).
- **In-sample (IS):** up to 3 Jul 2025. **Out-of-sample (OOS):** everything after.
- Variants and acceptance rules were written down before the results were seen.
- **Only the OOS columns count as evidence.**

### Swing event strategies (₹15k per trade, CNC costs)

| Variant | IS n | IS net/trade | IS t | OOS n | OOS win | OOS net/trade | OOS t | OOS Kelly | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| S1 breakout/thrust, 15-day time exit | 1,485 | −0.73% | −3.3 | 1,168 | 41% | −0.87% | −4.1 | −0.081 | no |
| S1t same, chandelier trail | 1,549 | −0.81% | −4.0 | 1,235 | 43% | −0.88% | −4.3 | −0.119 | no |
| S1t same, top 2/day | 412 | −1.23% | −3.1 | 384 | 41% | −0.47% | −1.0 | −0.046 | no |
| S2 breakout in weak tape, time exit | 160 | −1.33% | −2.3 | 215 | 43% | +0.11% | +0.2 | +0.065 | no |
| S2t breakout in weak tape, trail | 167 | −1.73% | −3.5 | 227 | 46% | +0.07% | +0.1 | +0.038 | no |
| S4a dip ≥ 5% in 3 days above EMA200, exit > EMA5 | 1,932 | +0.57% | +5.0 | 764 | 66% | +0.15% | +1.0 | +0.147 | no (t just under 1.0) |
| **S4b dip ≥ 8% in 3 days above EMA200, exit > EMA5** | **597** | **+1.53%** | **+6.5** | **168** | **64%** | **+0.69%** | **+1.8** | **+0.236** | **YES** |

### Swing rotation (₹50k book, integer shares, CNC costs)

| Variant | IS CAGR | IS Sharpe | OOS CAGR | OOS Sharpe | OOS maxDD | Accepted |
|---|---|---|---|---|---|---|
| Equal-weight liquid universe (benchmark, no costs) | +32.9% | 1.61 | −1.1% | 0.00 | −15.9% | — |
| Nifty 50 (benchmark) | +15.8% | 1.16 | −9.4% | −0.72 | −15.2% | — |
| R1 6-month momentum, top 5 | +0.1% | 0.16 | +12.8% | 0.54 | −22.0% | no |
| R2 6-month momentum, top 10 | +0.4% | 0.14 | +16.2% | 0.71 | −15.4% | no |
| R3 12-1 momentum, top 5 | −12.7% | −0.48 | +48.8% | 1.72 | −18.3% | no |
| R4 12-1 momentum, top 10 | −13.2% | −0.70 | +24.2% | 1.11 | −20.9% | no |
| R5 6-month momentum, top 5, above EMA200 | −1.7% | 0.10 | +14.9% | 0.60 | −22.1% | no |

Momentum rotation crushed the benchmark out-of-sample but lost to it badly in-sample (drawdowns of 37–45%, consistent with the late-2024 momentum crash in Indian mid caps). That is regime dependence, not a stable edge, so it fails the rule. It stays on the watch list.

### Intraday (₹50k notional, MIS costs)

**Low confidence:** Yahoo serves only about 59 days of 5-minute data.

| Variant | IS n | IS net/trade | OOS n | OOS win | OOS net/trade | Accepted |
|---|---|---|---|---|---|---|
| I1 ORB baseline | 16 | −0.46% | 29 | 48% | −0.09% | no |
| I2 ORB, range-opposite stop, to 14:00 | 16 | −0.51% | 31 | 52% | −0.10% | no |
| I3 VWAP pullback in trend | 11 | −0.38% | 15 | 53% | −0.06% | no |

No intraday edge has been shown. The OOS gross R is slightly positive, but every variant loses after MIS costs, and the samples are far too small to conclude anything either way.

## Round 3: full-engine backtest and attribution (runs 37289944455, 37291050170)

The deployed Phase 4 engine, run end to end on a ₹50k paper book from Jun 2024 to Oct 2026 (574 sessions), returned **−12.7%**. The max drawdown was −14.5%, over 187 trades with a 48% net win rate.

An attribution ladder runs the real `SwingEngine`, adding one deployed difference at a time. Run it with `python -m nsebot.research.attribution`.

| Variant | Trades | Avg notional | Net/trade (all) | Net/trade (OOS) | Book return / max DD |
|---|---|---|---|---|---|
| P0 parity: all signals, ₹15k fixed, no slippage / regime / breakers | 468 | ₹14,243 | +0.42% | **+0.74%** | — |
| P1 + 5 bps slippage and tick snapping | 468 | ₹14,250 | +0.32% | +0.64% | — |
| P2 + half-Kelly sizing and breakers | 262 | **₹3,051** | −1.05% | −0.40% | −17.2% / −19.1% |
| P3 + capacity (slots, sector, regime entry budget) | 202 | ₹3,478 | −0.84% | +0.15% | −12.9% / −15.9% |
| P4 = deployed (+ regime size dial) | 187 | ₹3,167 | −1.00% | −0.22% | −12.7% / −14.5% |
| P5 deployed, shallowest dip first | 187 | ₹3,048 | −0.90% | −0.32% | −10.5% / −14.5% |
| **C1: notional 20% of equity, ₹8k floor, 3% risk cap, no streak breaker, no size dial** | **183** | ₹9,115 | **+0.24%** | **+1.02% (t 2.1)** | **+6.7% / −18.7%** |
| C2 = C1 + loss-streak breaker | 28 | ₹9,274 | −0.29% | none (locked) | −1.6% / −5.4% |
| C3 = C1 + regime size dial | 47 | ₹8,741 | −1.31% | 4 trades | −11.0% / −14.1% |

What this shows:

1. **The engine is faithful.** At parity it reproduces the event study's out-of-sample result (+0.74% vs +0.69%).
2. **Position size collapse.**
   - Kelly sizing on 3×ATR stops, learning from net R, shrank CNC positions to about ₹3k.
   - At that size the flat DP charge plus STT cost about 1% per trade.
   - The resulting losses lowered the Kelly estimate, which shrank positions further. Gross P&L was about flat; net was −1%.
3. **The breaker deadlock.**
   - "Half size until the next win" sat below the ₹8k economic floor.
   - No trade could happen, so no win could reset it, and C2 never traded again.
   - Even without a floor, the streak breaker skips the rebound trades a dip-reversion strategy earns its money on.
4. **The regime size dial** pushed positions below economic size in weak markets. That is V2's regime gate by another route.

Phase 4b deploys C1. `tests/test_attribution.py` proves the deployed engine places exactly C1's trades. C1 was selected after seeing this data, so its +6.7% is in-sample by construction. The deeper drawdown than Nifty (−18.7% vs −14.5%) and the IS loss (−7.3%) are part of the result, not footnotes.

## What the evidence supports

1. **Swing: trade S4b dip reversion.** The story is consistent across the board:
   - breakouts lose because short-horizon moves tend to reverse;
   - buying the reversal wins, and wins more the deeper the dip (S4a +0.15% → S4b +0.69% OOS);
   - it held up out-of-sample during a falling market (Nifty −9.4% annualised over the OOS window).
2. **Honest uncertainty on S4b:**
   - OOS t = 1.8, so the 95% interval for net per trade is about **−0.06% to +1.44%**. Probably positive, but not proven.
   - The edge shrank from IS to OOS (+1.53% → +0.69%), which is normal and should be expected to continue.
   - Frequency is about 0.5 signals per session across 255 names.
3. **Intraday: run at the sizing floor** as live data collection. The bot's own trades move the Kelly estimate; aggressive sizing has to be earned first.
4. **Kelly priors use the OOS numbers** (the weaker half), never the IS numbers.
5. **Swing sizes by notional with an economic floor, not by Kelly** (round 3). Kelly on the bot's own results is reported every run and raises a warning when negative on 60+ trades. It never becomes an automatic state the book can't trade its way out of.
6. **Honest expectation for swing:** small, regime-dependent, and probably positive. Forward paper trading is the real test.
