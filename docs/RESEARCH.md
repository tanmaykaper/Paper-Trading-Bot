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

## Round 4: the smallcap-fund hurdle (run 37293283239)

The owner's target is to beat the best-performing smallcap funds. This round measures that bar on the full-engine backtest's exact window, **11 Jun 2024 to 5 Oct 2026**.

- **Funds:** every Direct-Growth smallcap fund with full history in the window, from mfapi.in's free NAV data. That is 25 active funds plus passive index funds.
- **Bot designs:** same universe, ₹50k book, Zerodha CNC costs, next-open fills.

| | Total | CAGR | Max DD |
|---|---|---|---|
| **Best active smallcap fund** (Motilal Oswal Small Cap, Direct) | **+45.6%** | **+17.7%** | −23.7% |
| Top quartile of 25 active funds | — | +11.8% | — |
| Median active fund (Quantum Small Cap) | +19.9% | +8.2% | −18.2% |
| Worst active fund | — | +0.7% | — |
| Nifty Smallcap 50 index funds | +15.7% | +6.5% | −25.0% |
| Nifty Smallcap 250 index funds | +7.0–7.7% | +3.0–3.3% | −26.2% |
| Bot: deployed C1, Phase 4b backtest (same window) | +6.9% | +3.0% | −18.7% |
| Bot: deployed C1 started on a longer history, measured on the same window | −4.2% | −1.8% | −24.0% |
| Bot: M2 12-1 momentum, top 5, Nifty > 200-day SMA | +15.7% | +6.5% | −26.0% |
| Bot: M1 same, top 10 | +10.6% | +4.4% | −29.3% |
| Bot: M3 6-month momentum, top 10, same filter | +1.5% | +0.6% | −28.6% |
| Bot: M4 12-1 momentum, top 10, no filter | +5.6% | +2.4% | −45.1% |
| Bot: B1 70% M1 + 30% C1 | +6.1% | +2.6% | −25.3% |

Findings:

1. **No bot design beats even the median active smallcap fund on this window.** The best fund beats every design by more than 11 points of CAGR. The best design (M2) only matches the passive Smallcap 50 index.
2. **The deployed engine's window return depends on its start date.** It reads +6.9% or −4.2% depending only on how much history the backtest started from, which changes position sizes and which slots are occupied when the window opens. Over this window it cannot be told apart from roughly flat.
3. The best fund is an ex-post maximum out of 25, and the market-filter variants were designed after seeing momentum's late-2024 crash. Both make the comparison *kinder* to the bot than it deserves, and the bot still falls short.
4. Every design in rounds 2–4 was selected on Aug 2023 to Oct 2026 data. **Daily history from before mid-2023 has never been used to choose anything.** It is the only clean place left to test whether any of these designs can beat smallcap funds.

## Round 5: untouched history, Jul 2018 to Jul 2023 (runs 37295540936, 37296721706)

Pre-registered in commit 569f52d before the run: frozen designs, return boosters and acceptance rules (`nsebot/research/phase5.py`). Rounds 1–4 never used data from before Aug 2023, so nothing below was chosen on this window. It covers the 2018–19 smallcap bear market, the COVID crash, the 2020–21 boom and the 2022 chop.

- **Universe:** 243 of 260 symbols resolved; 207 were trading by Jul 2018.
- **Book:** ₹50k, Zerodha CNC costs, next-open fills.
- **Benchmarks:** the 11 active Direct-Growth smallcap funds with full-window NAVs on mfapi.in.
- **Rules:**
  - *QUALIFIES* means CAGR above the median fund **and** beating it in at least 3 of the 5 July–June years.
  - *MEETS TARGET* means CAGR above the best fund.

| | Total | CAGR | Max DD | Years > median fund | Qualifies | Meets target |
|---|---|---|---|---|---|---|
| **Best active fund** (Quant Small Cap) | +260.8% | **+28.7%** | −46.7% | | | |
| Top quartile of 11 funds | — | +23.2% | — | | | |
| **Median active fund** (SBI Small Cap) | +171.2% | **+21.7%** | −33.8% | | | |
| Worst active fund | — | +11.9% | — | | | |
| Nifty 50 | +85.4% | +12.9% | −38.4% | | | |
| D1 deployed engine (S4b dip reversion) | −16.2% | −3.4% | −16.9% | 0/5 | no | no |
| D2 12-1 momentum, top 5, Nifty > 200-day SMA | +243.5% | +27.5% | −39.0% | 2/5 | no | no |
| D3 same, top 10 | +194.7% | +23.7% | −36.2% | 2/5 | no | no |
| **D4 12-1 momentum, top 10, no filter** | **+404.1%** | **+37.5%** | −37.5% | 4/5 | **yes** | **yes** |
| E1 dip, 33% × 3 slots | −17.3% | −3.7% | −19.5% | 0/5 | no | no |
| E2 dip on a 5% trigger | 0.0% | 0.0% | 0.0% | 1/5 | no | no |
| E3 60% D2 + 40% D1 | +139.6% | +18.8% | −33.4% | 1/5 | no | no |
| E4 D2 at 1.5×, 15%/yr financing | +329.8% | +33.3% | −56.6% | 2/5 | no | yes (CAGR only) |

Year by year, median fund vs D4: 2018–19 +2.5% vs +13.1%; 2019–20 −4.5% vs +3.9%; 2020–21 +90.3% vs +180.3%; 2021–22 +6.1% vs −3.6%; 2022–23 +36.3% vs +60.5%.

**Post-hoc diagnostics** (`phase5_diag`; these explain the result and change no design or rule):

| Design | Equity at window start | Floor binds below | First below floor | First 25% drawdown | Entries in window | Last entry |
|---|---|---|---|---|---|---|
| D1 | ₹44,535 | ₹40,000 | 2020-03-09 | 2020-03-09 | 70 | 2020-03-04 |
| E1 | ₹47,508 | ₹24,242 | never | 2021-12-20 | 179 | 2021-12-16 |
| E2 | ₹38,317 | ₹40,000 | 2018-02-05 | never | 0 | 2018-02-02 |

| | CAGR | Max DD | 18–19 | 19–20 | 20–21 | 21–22 | 22–23 |
|---|---|---|---|---|---|---|---|
| Equal-weight hold of the list's liquid names (no costs) | +24.0% | −38.0% | +7.3% | −4.9% | +93.0% | +1.6% | +46.9% |
| D4 minus that | +13.5 pts | | +5.8 | +8.8 | +87.3 | −5.2 | +13.5 |
| D2 minus that | +3.5 pts | | −18.3 | +2.6 | +100.5 | +2.0 | −15.9 |

Findings:

1. **Dip reversion fails on untouched data.**
   - D1 lost 4.9% in 2018–19 and 11.8% in 2019–20 over 70 trades. It then tripped the 25% drawdown latch in the COVID crash and never traded again.
   - E1 also lost money.
   - The S4b edge from rounds 2–3 (OOS t = 1.8) does not survive a different regime.
2. **The engine has a silent freeze.**
   - At 20% notional, the ₹8k "skip, never shrink" floor binds once equity falls below ₹40k, a 20% loss on ₹50k. From then on every signal is skipped, permanently.
   - E2 froze this way in Feb 2018 without ever reaching the 25% latch. It made no trades and raised no alert.
   - Live, this looks exactly like V2: it runs every day and opens nothing. The latch message also stamps the wall-clock date instead of the session date, so the raw Phase 5 report shows 2026-10-05.
3. **D4 is the only design that passes both pre-registered rules, but the evidence is weak.** This is the case the bias warning, written before the run, described.
   - A cost-free, equal-weight hold of today's list already beats the median fund (+24.0% vs +21.7%), so the list itself is doing much of the work.
   - D4's margin over that hold (+13.5 points, positive in 4 of 5 years) is the part that may be skill.
   - Even that margin is probably inflated: momentum concentrates in names that are in today's list *because* they went up.
4. **D4 failed in the period least affected by survivorship.**
   - Round 4 (Jun 2024 to Oct 2026, close to the date the list was drawn up) gave the same design +2.4% CAGR with a −45.1% drawdown, against +8.2% for the median fund.
   - Survivorship bias shrinks near the list date, and that is where D4 lost.
5. **The 200-day market filter hurt here.** D3, top 10 with the filter, made +23.7%; D4, top 10 without it, made +37.5%. The filter was added after seeing momentum's late-2024 crash, which makes it a fit to round-4 data, and it did not carry over.
6. **Taxes are not modelled.** Weekly rotation realises gains as short-term capital gains (15% during this window, 20% since Jul 2024), while a fund investor defers tax until redemption. That drag counts against every rotation design.

## What the evidence supports

1. **Swing: S4b dip reversion is not proven, and round 5 points negative.**
   - Rounds 2–3 found it the only variant to pass out-of-sample. Its edge grew with dip depth (S4a +0.15% → S4b +0.69% OOS) and it held up while Nifty fell.
   - On untouched 2018–2020 data it lost money (D1 −4.9%, then −11.8%) and froze.
2. **Honest uncertainty on S4b:**
   - The OOS 95% interval was already about **−0.06% to +1.44%** per trade (t = 1.8), and round 5 sits at the bad end of it.
   - Keep it running only as a paper experiment, and only once the floor freeze (round 5, finding 2) is fixed.
3. **Intraday: run at the sizing floor** as live data collection. The bot's own trades move the Kelly estimate; aggressive sizing has to be earned first.
4. **Kelly priors use the OOS numbers** (the weaker half), never the IS numbers.
5. **Swing sizes by notional with an economic floor, not by Kelly** (round 3). Kelly on the bot's own results is reported every run and raises a warning when negative on 60+ trades. It never becomes an automatic state the book can't trade its way out of.
6. **Honest expectation for swing:** dip reversion is roughly flat to negative across regimes. Forward paper trading is the remaining test.
7. **Against the owner's smallcap-fund target:**
   - Round 4: nothing qualified.
   - Round 5: 12-1 momentum, top 10, unfiltered (D4) passed both pre-registered rules on untouched 2018–23 data (+37.5% CAGR vs +28.7% for the best fund).
   - It comes with two strong caveats: the symbol list alone beats the median fund, and the same design lost to the median fund over 2024–26.
   - The next test that can settle it is a universe without hindsight. A point-in-time or much broader symbol list would show whether D4's margin survives.
