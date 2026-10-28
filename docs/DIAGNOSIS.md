# Diagnosis — Oct 2026: why the book is down, and what to fix first

Written on 28 Oct 2026, after 17 sessions of live paper trading (5–28 Oct). Sources:

- the bots' own ledgers (`state/*/trades.csv`, `equity.csv`, `state.json`);
- their GitHub Actions logs;
- a code audit;
- a real-data study run on GitHub Actions for this note (`nsebot/research/diagnosis.py`, run 40734077784).

Everything after the live losses is post-hoc. Section 4 says which conclusions are firm and which are only effect sizes.

---

## 1. The short answer

**₹1,39,551 from ₹1,50,000 (−6.97%).** The ₹10,449 loss has four sources, and only one of them is the market:

| Source | ₹ | Share of the loss | What it is |
|---|---|---|---|
| Momentum: BHAGYANGR booked at −89% | −4,466 | 43% | **Accounting.** A demerger the ledger can't represent (§2.1) |
| Intraday: trading | −1,661 | 16% | A strategy with no edge (§2.2) |
| Intraday: Zerodha charges | −2,437 | 23% | Costs larger than the trading loss itself |
| Momentum: everything else | −2,802 | 27% | Real price moves in 11 small-cap holdings |
| Swing | +1,143 | — | 3 trades, 2 winners; 89% of its money sat idle (§2.3) |
| Cost of selling the open positions | −227 | 2% | |

What that means:

1. **About two-fifths of the reported loss never happened economically.** Bhagyanagar India spun its copper business off into Tieramet Ltd, one Tieramet share per share, record date 8 Oct 2026. The paper ledger kept only the shell company and booked the copper business's value as a loss.
2. **Another two-fifths is intraday, which the research never found an edge for.** It has done exactly what the research predicted: lose a little before costs, then pay ₹44 a trade in charges.
3. **The momentum sleeve's real result is about −5.6% in 17 sessions.** For a book whose tested worst drawdown is about −50%, that is ordinary noise; §3 measures how ordinary.

---

## 2. Findings, ranked by money × certainty

### 2.1 Corporate actions: a demerger booked as an 89% loss — bug, certain

**What happened.**

| Date | Event |
|---|---|
| 25 Sep | Bhagyanagar's board fixes 8 Oct as the record date for the Tieramet demerger (1:1) |
| 5 Oct | The sleeve's first rebalance ranks BHAGYANGR #4 on 12-1 momentum (+318%). Part of that run was the market pricing the demerger |
| 6 Oct | Bought 12 shares @ ₹418.15 |
| 9 Oct | Yahoo shows the shell at ₹43.45 (−89.6%). The run's log lists it as `-89.6%` and raises no warning |
| 12 Oct | The rebalance finds it "no longer eligible": ₹43 is under the ₹50 price floor |
| 13 Oct | Sold at ₹47.82, net −₹4,465.85. The 12 Tieramet shares a real holder owns don't exist in the ledger |

**Why the guards missed it.** `MomentumEngine._rescale_for_splits` compares each holding's *anchor* bar with what Yahoo shows for that date now.

- **A demerger is not a split.** Yahoo does not rescale the history, and the price ratio isn't a small-number fraction, so the guard can never match it.
- **The anchor moves to the latest bar every run** (`_reanchor`). If Yahoo applies a split adjustment a day or more late, the anchor is already the post-split bar, so the guard compares two adjusted prices and sees nothing. A *late-adjusted split* is missed the same way.
- **Nothing alerts on a one-day collapse.** NSE's 2–20% circuit bands make a 30%+ gap down impossible as an ordinary market move in almost every stock, so it is the clearest signal there is.

**Predicted outcome of the fix.**

- Immediate: about ₹4.4k of reported value back on the momentum sleeve once the Tieramet entitlement is credited (+9 points on the sleeve, +3 on the whole book), whatever Tieramet lists at.
- Going forward: §3.2 counts how often B4 walked into one of these gaps on 2018–26 data, what the ledger booked, and B4's CAGR with value kept across them.

**Fix.**

1. A holding whose session opens and closes 30%+ under the previous close is a probable corporate action. Freeze it: no forced sale, no "ineligible" exit, no drawdown-latch input. Alert, and wait for a human to confirm.
2. Add a manual `adjust` command to credit a demerged entity (symbol, ratio, cost split).
3. Keep a fixed entry-date anchor alongside the rolling one, so a late split adjustment is still caught.
4. Optional, later: before buying, skip names with a demerger, split or bonus record date inside the next rebalance window, using NSE's corporate-actions feed.

### 2.2 Intraday has no edge, and costs exceed its trading loss — strategy, high confidence

**Results, 55 trades over 13 sessions:**

| | Value |
|---|---|
| Win rate after costs | 38% |
| Average winner | +0.97R |
| Average loser | −1.09R |
| Gross R per trade | **−0.11R** (t = −0.75; 95% interval −0.42R to +0.19R) |
| Cost per trade | **0.19R**: 10.6 bps of notional against stops 0.3–1.0% wide |
| Net R per trade | **−0.30R** (95% interval −0.60R to 0.00R) |
| Stopped out within 2 bars (10 minutes) | 12 of 55 |
| Daily loss limit hit | 21 Oct, 3% of equity, in the first 25 minutes |

**What it means.**

- **Even the top of the 95% interval only breaks even after costs.** This matches research round 2, where every ORB variant was net-negative after MIS costs, and round 1, where the baseline made −0.51R.
- **The 0.5%-risk floor doesn't save it.** About ₹250 at risk per trade and about ₹44k notional per trade, at 4.2 trades a session.

**Predicted outcome of switching it off.** It saves an expected **₹300–₹320 per session (about ₹1,500 a week)** on the paper book. That is +0.2% of the whole ₹1.5L every session the bot would otherwise have traded. At the current pace the sleeve's 25% latch would trip in about 5 more weeks anyway (₹7.4k of room at ₹315 a session).

**Fix.**

1. Stop placing intraday trades now.
2. If the data is wanted, run the signal in **shadow mode**: log what it would have done, with no capital and no costs.
3. Bring it back only if 100+ shadow trades show gross R above the 0.19R cost.

Tuning on these 55 trades isn't worth doing. Entries at 09:50–10:30 made money, for example, but 10 trades is a coincidence until shown otherwise.

### 2.3 Swing sits 89% in cash — structural

- **Capital is mostly idle.** Over 16 sessions the swing book averaged 11% invested. S4b fires about once every two sessions across 255 names, each trade takes 20% of the sleeve, and most last 1–7 days.
- **The edge doesn't scale.** Even at the S4b out-of-sample edge (+0.69% a trade), a sleeve deployed 11% of the time earns a few percent a year at best, less than a liquid fund on the idle cash.
- **The edge itself failed on untouched data.** It lost money in 2018–20 (research round 5).
- **Live results don't change that picture.** +₹1,143 on 3 trades is real, but three trades say nothing about the edge.

**Predicted outcome.** Parking the idle ₹40–45k in a liquid ETF (LIQUIDBEES/LIQUIDCASE, about 6.5% a year) adds about ₹230 a month with no new risk. Moving swing's capital into the momentum sleeve trades that for momentum's return and its ~50% drawdowns (§3.4).

### 2.4 Momentum: the right family, but it buys the most stretched names in the market — design

The first ten picks had a median 12-1 return of **+290%**:

| Stock | 12-1 return |
|---|---|
| CUPID | +532% |
| STLTECH | +480% |
| SIGMAADV | +367% |
| BLISSGVS | +330% |
| BHAGYANGR | +318% |
| VENUSREM | +263% |
| INDSWFTLAB | +254% |
| HFCL | +199% |
| AEROFLEX | +191% |
| ATHERENERG | +190% |

Most are small or micro caps that moved in jumps and circuit runs. Four of nine current holdings are down 10–18% in three weeks. That is the profile the literature warns about:

- **Momentum crashes** (Daniel & Moskowitz, 2016): the highest-momentum names give back the most when the trend turns.
- **Volatility.** Raw momentum's risk is concentrated in its most volatile winners. Scaling by volatility keeps the premium with far smaller crashes (Barroso & Santa-Clara, 2015).
- **Frog in the pan** (Da, Gurun & Warachka, 2014): winners built from many small up days keep going; winners built from a few jumps don't.
- **The MAX effect** (Bali, Cakici & Whitelaw, 2011): stocks with extreme single-day gains underperform.
- **NSE's own momentum indices** (Nifty200 Momentum 30, Midcap150 Momentum 50) rank on 6- and 12-month returns *divided by volatility*, not raw returns.

Also in the design:

- **Universe.** It includes series BE (trade-for-trade surveillance) and BZ (companies not complying with listing rules).
- **Tax.** Weekly rotation realises short-term gains at 20%, which no backtest charges.

§3.3–3.4 test whether the risk-adjusted, smooth-path and no-jump alternatives would have predicted returns better, and what they would have done to the book.

### 2.5 Operations

| Finding | Effect | Fix |
|---|---|---|
| `EMAIL_SENDER/PASSWORD/RECIPIENT` are empty in every run log | Nobody was told when a holding fell 89% overnight, or when intraday hit its daily loss limit | Add the three repository secrets (docs/SCHEDULING.md, about 5 minutes) |
| 12 Oct: an intraday run dispatched at 08:26 IST hit GitHub's 6-hour job limit at 14:27, before the 15:10 square-off. The next run's state commit then failed on a rebase conflict in `state/intraday/state.json` | JSWSTEEL (₹1.26 lakh notional) was carried overnight in the ledger and closed the next morning at −₹426. Real MIS would have been squared off by Zerodha at 15:20 | `commit_state.sh`: on a conflict in the bot's own `state/<mode>/`, keep the newer local copy. Refuse to start an intraday session that can't reach 15:12 inside 6 hours |
| `_reanchor` moves the split anchor every run | A split Yahoo adjusts late is never detected (§2.1) | Keep an entry-date anchor too |

---

## 3. Real-data study (run 40734077784)

_Filled in from `research_results/DIAGNOSIS.md` of the run._

---

## 4. What to fix, in order, and what each fix should do

_Filled in after §3._

---

## 5. Expectations

_Filled in after §3._
