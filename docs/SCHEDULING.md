# Scheduling: how the bots run on their own

## The problem

GitHub Actions' scheduler is best-effort. In GitHub's own words, scheduled runs "can be delayed during periods of high loads … High load times include the start of every hour. If the load is sufficiently high enough, some queued jobs may be dropped."

For this repo that means:

| When | What GitHub did |
|---|---|
| V2 era, Sep 2026 | The 22:30 IST schedule ran 2–5 hours late every day. |
| 5 Oct, evening | Swing (due 17:15) ran at 01:13. Momentum (due 17:40) started at 02:01, waited 15 minutes for a machine, and was cancelled without running ("All jobs were cancelled"). |
| 6 Oct, morning | Intraday (due 09:22) ran at 16:18, after the market closed. |
| 6 Oct, evening | Neither the 17:15 swing slot nor the 17:40 momentum slot had started by 18:46. |

Nothing inside the repo can make GitHub's scheduler punctual. What the repo can do is make sure a late or dropped run never costs a session, without anyone looking after it.

## What the repo does on its own (no setup, no Claude)

1. **Many schedule slots.** Each slot lands off the round minutes (:13, :37, :22), away from the busy start of the hour.

   | Bot | Scheduled (IST) |
   |---|---|
   | Swing (`main.yml`) | hourly at :13 from 17:13 to 00:13, then 02:13, 04:13, 06:13, 08:13 |
   | Momentum (`momentum.yml`) | hourly at :37 from 17:37 to 00:37, then 02:37, 04:37, 06:37, 08:37 |
   | Intraday (`intraday.yml`) | 09:22, with a retry at 11:22 |

2. **A ten-second check first.** Each run starts with `scripts/should_run.py`, which needs nothing installed. If the day's session is already processed (swing, momentum), or the market isn't open (intraday), the run stops right there. So the extra slots cost almost nothing, and the first slot that gets through does the work.
3. **Runs are safe to repeat, and they catch up.**
   - A session already processed is never redone.
   - A late run processes the right session. Before 15:45 IST the next day, that's the previous session.
   - Swing walks open positions through every bar it missed.
   - Momentum fills queued orders at the open they were meant for.
4. **One queue per bot**, so one bot's waiting run can't cancel another's.

Expect a long list of short green runs under **Actions**, most of them finishing in seconds with "nothing to do". That's the design working, not a problem.

**The upshot:** as long as any one slot runs before the next morning's open, nothing is lost. Even GitHub's worst delay seen here, about 8 hours, gets the 17:13 slot through by about 01:00 IST.

**What it can't fix:**
- End-of-day reports can arrive late at night.
- Intraday needs to run during market hours, so a late GitHub start still means a short or missed session.

The trigger below fixes both.

---

## Make it punctual (optional, about 10 minutes, free)

A run started through GitHub's "Run workflow" API begins within seconds, unlike a scheduled one. A free scheduler such as [cron-job.org](https://cron-job.org) can call that API at fixed times. Runs started this way always run in full. The GitHub schedules stay as the safety net, and their check sees the day is done and stops.

### 1. Create a fine-grained GitHub token

1. On GitHub, open your profile menu, then **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
2. Fill in the form:
   - **Token name:** `nsebot scheduler`
   - **Expiration:** pick a date, for example 1 year out, and put a renewal reminder in your calendar.
   - **Repository access:** **Only select repositories**, then choose `tanmaykaper/Paper-Trading-Bot`.
   - **Permissions → Repository permissions → Actions:** **Read and write**. "Metadata: Read-only" is added automatically; leave everything else at **No access**.
3. Click **Generate token** and copy it. GitHub shows it only once.

This token can only start, cancel and view workflow runs in this one repository. It can't read secrets, change code or reach other repositories. You can revoke it at any time on the same page.

### 2. Check that it works (optional)

```bash
curl -i -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/tanmaykaper/Paper-Trading-Bot/actions/workflows/momentum.yml/dispatches \
  -d '{"ref":"main"}'
```

You should get `HTTP/2 204` back, and a new **nsebot momentum** run appears under **Actions**. It's safe at any time: a session that's already processed is skipped.

### 3. Create three jobs on cron-job.org

Sign up (free). For each row below, click **Create cronjob** and fill in the same settings, changing only the title, workflow and time.

| Title | Workflow file in the URL | Time (Asia/Kolkata), Mon–Fri |
|---|---|---|
| `nsebot swing` | `main.yml` | 17:20 |
| `nsebot momentum` | `momentum.yml` | 17:45 |
| `nsebot intraday` | `intraday.yml` | 09:22 |

The settings for each job:

- **URL:** `https://api.github.com/repos/tanmaykaper/Paper-Trading-Bot/actions/workflows/<workflow file>/dispatches`
- **Schedule:** **Custom**, at the time in the table, **Monday to Friday** only.
- **Time zone:** **Asia/Kolkata**. If the job form doesn't offer one, set it in your account settings.
- **Advanced tab:**
  - **Request method:** `POST`
  - **Request body:** `{"ref":"main"}`
  - **Headers:**

    | Key | Value |
    |---|---|
    | `Accept` | `application/vnd.github+json` |
    | `Authorization` | `Bearer YOUR_TOKEN` |
    | `X-GitHub-Api-Version` | `2022-11-28` |
    | `Content-Type` | `application/json` |

- **Notifications:** turn on **notify me when execution fails**. A failure usually means the token expired.

Save each job, then click **Test run**. It should report `204`, and a run should appear under **Actions**.

### Notes

- **Why 09:22 for intraday.** GitHub stops any job after 6 hours. Starting at 09:22 leaves time for setup and the final save before the 15:10 square-off. The opening range (09:15–09:30) is read from Yahoo's history, so nothing is lost by starting then.
- **Why 17:20 and 17:45.** Yahoo's daily bar is final by about 15:45 IST, so any time after that works. The two jobs are staggered so they don't hit Yahoo at the same moment.
- **NSE holidays.** The bots find no new session and do nothing.
- **Renewing the token.** Generate a new token and paste it into each job's `Authorization` header in place of the old one. If you forget, the GitHub schedules keep the bots running, just late.

---

## Get the reports by email (optional, about 5 minutes)

Every run writes its report to the run's summary page under **Actions**. To get it by email instead, add three repository secrets. The bots then email you when they trade, when a breaker trips, when a warning fires, and when data fails.

1. **Create a Gmail app password.** Google Account → **Security** → turn on **2-Step Verification** if it's off → **App passwords** → create one named `nsebot` and copy the 16-character code.
2. **Add the secrets.** On GitHub, open the repo → **Settings → Secrets and variables → Actions → New repository secret**, and add these three:

   | Name | Value |
   |---|---|
   | `EMAIL_SENDER` | the Gmail address |
   | `EMAIL_PASSWORD` | the 16-character app password |
   | `EMAIL_RECIPIENT` | where the reports should go (it can be the same address) |

The next run that has something to report sends an email. Nothing else needs changing.

---

## How you'll know if something is wrong

- **GitHub emails the repository owner whenever a run fails** (the default notification setting). That covers data outages, which exit with an error and leave state untouched, and cancelled runs.
- **cron-job.org emails you** if its call to GitHub fails, usually because the token expired.
- **A run marked "cancelled" is usually harmless.** When GitHub releases several delayed slots at once, a bot's queue keeps one waiting run and cancels the rest, which had nothing to do anyway.
- **If every slot for a day is lost,** the next run catches up: swing walks its positions bar by bar, and momentum fills queued orders at the right open. A swing signal day that was never processed is the one thing that can't be recovered.
- **The bots' own warnings** appear in each report and in the emails: NOT TRADING, data faults, stale prices, split adjustments and breaker trips.
