# Scheduling: making the bots run on time

## Why this is needed

GitHub Actions' scheduler is best-effort, and for this repo it runs late:

- On 5 Oct 2026 the swing run scheduled for 17:15 IST started at 01:13 IST, and the momentum run scheduled for 17:40 IST started at 02:01 IST. Both were about 8 hours late.
- The V2 bot's 22:30 IST schedule ran 2–5 hours late on every day in its logs.
- Some runs never start. The 02:01 IST momentum run waited 15 minutes for a machine, was cancelled by GitHub and never ran a line of code. Its email says "All jobs were cancelled".

**Swing and momentum.** These are end-of-day jobs, and every run is safe to repeat. A run that comes late still processes the right session. A session that has already been processed is skipped, and missed days are caught up. Two settings in the workflows cover the rest:

- **Backup runs.** `main.yml` (swing) fires at 17:15, 19:45, 22:45 and 06:45 IST the next morning. `momentum.yml` fires at 17:40, 20:10, 23:10 and 07:10 IST. The first run that gets through does the day's work; the others find nothing to do. Momentum checks this before its big download, so a backup costs seconds.
- **One queue per bot.** GitHub keeps only one waiting run per queue, so when all three bots shared one queue, a waiting momentum run could cancel a waiting swing run. Now each bot has its own queue. They write different `state/<mode>/` folders, and the commit step rebases and retries its push.

**Intraday.** It must run during market hours, 09:22–15:12 IST. A late start means a short or missed session. It doesn't cause bad trades: a run that starts after 15:10 exits at once and sends no email. GitHub's schedule can't fix this, but a free external scheduler that presses "Run workflow" for you can. Runs started that way begin within seconds.

---

## Set up an on-time trigger for intraday (about 10 minutes, free)

You need a GitHub token that can start workflows in this one repository, and a free [cron-job.org](https://cron-job.org) account that sends the start request each weekday at 09:22 IST.

### 1. Create a fine-grained GitHub token

1. On GitHub, open your profile menu, then **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
2. Fill in the form:
   - **Token name:** `nsebot intraday trigger`
   - **Expiration:** pick a date, for example 1 year out, and put a reminder in your calendar to renew it.
   - **Repository access:** **Only select repositories**, then choose `tanmaykaper/Paper-Trading-Bot`.
   - **Permissions → Repository permissions → Actions:** **Read and write**. "Metadata: Read-only" is added automatically. Leave everything else at **No access**.
3. Click **Generate token** and copy it. GitHub shows it only once.

This token can only start, cancel and view workflow runs in this one repository. It can't read your secrets, change code or touch other repositories. You can revoke it at any time on the same page.

### 2. Check that the token works (optional, from any terminal)

```bash
curl -i -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/tanmaykaper/Paper-Trading-Bot/actions/workflows/intraday.yml/dispatches \
  -d '{"ref":"main"}'
```

You should get `HTTP/2 204` back, and a new **NSE Intraday Bot** run appears under the repo's **Actions** tab. Outside market hours that run exits within a minute without trading or emailing, so the test is safe at any time.

### 3. Create the scheduled job on cron-job.org

1. Sign up at cron-job.org (free) and click **Create cronjob**.
2. On the **Common** tab:
   - **Title:** `nsebot intraday`
   - **URL:** `https://api.github.com/repos/tanmaykaper/Paper-Trading-Bot/actions/workflows/intraday.yml/dispatches`
   - **Execution schedule:** **Custom**, at **09:22**, on **Monday to Friday** only.
   - **Time zone:** **Asia/Kolkata**. If the job form doesn't offer a time zone, set it in your account settings.
3. On the **Advanced** tab:
   - **Request method:** `POST`
   - **Headers:** add these four.

     | Key | Value |
     |---|---|
     | `Accept` | `application/vnd.github+json` |
     | `Authorization` | `Bearer YOUR_TOKEN` |
     | `X-GitHub-Api-Version` | `2022-11-28` |
     | `Content-Type` | `application/json` |

   - **Request body:** `{"ref":"main"}`
4. Under notifications, turn on **notify me when execution fails**. A failure usually means the token expired.
5. Save, then use **Test run**. It should report status `204`, and a new run should appear in Actions.

### 4. Confirm it the next trading day

After 09:22 IST, open **Actions → NSE Intraday Bot**. There should be a run started by **workflow_dispatch** at about 09:22 IST that keeps running until about 15:12.

### Notes

- **Why 09:22 and not earlier.** GitHub stops any job after 6 hours. Starting at 09:22 leaves time for setup and the final save before the 15:10 square-off. The opening range (09:15–09:30) is read from Yahoo's history, so nothing is lost.
- **The fallback schedule stays.** `intraday.yml`'s own 09:22 schedule remains as a backup. If both fire, the second waits in the bot's queue until the first finishes, then exits immediately and quietly.
- **NSE holidays.** The bot sees no session data, logs it and stops. No trades are placed.
- **Swing and momentum (optional).** The same method can start them on time too. Add two more cron-job.org jobs with the same headers and body, but with `main.yml` (at 17:15) or `momentum.yml` (at 17:40) in place of `intraday.yml` in the URL. The backup runs make this optional.
- **Renewing the token.** Generate a new token, then paste it into the cron-job.org job's `Authorization` header in place of the old one.
