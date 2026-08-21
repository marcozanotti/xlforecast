# 09 — Running the Add-in

How to get the add-in loaded in Excel so you can work through
[`08-ADDIN-CHECKLIST.md`](08-ADDIN-CHECKLIST.md).

Three things have to be running: the **API**, the **pane's dev server**, and **Excel with the
manifest sideloaded**. The order matters only in that the API should be up before you press
anything in the pane.

---

## 0. One-time setup

```bash
# Python side
uv sync --all-extras

# Add-in side
cd addin && npm ci
```

### Install the development certificate — do this first

Office **refuses to load a task pane over HTTP**, and it refuses a self-signed certificate
too. The failure mode is unhelpful: the pane shows blank or "add-in could not be started",
with nothing in any log pointing at the certificate.

```bash
cd addin
npm run certs          # installs a locally trusted CA; will ask for your password
npm run certs:verify   # should print that the certificate is trusted
```

If you skip this, checklist item **A1 fails for reasons that have nothing to do with the
add-in**.

---

## 1. Start the API

Two options. Start with the simple one.

### Option A — no Redis (recommended for the checklist)

```bash
export XLF_OBJECT_ROOT="$PWD/.xlf-data"
export XLF_TOKEN_SECRET="anything-for-local-use"
uv run uvicorn xlforecast.api.main:app --port 8000 --reload
```

Jobs run in a **background thread of the API process**, using the same subprocess executor the
real worker uses — so cancellation, checkpointing and resume behave identically. What you do
not get is durability across an API restart.

Confirm which mode you are in:

```bash
curl -s localhost:8000/v1/health
# {"status":"ok","version":"0.1.0","mode":"inline (development)"}
```

That `mode` field exists so a deployment cannot quietly be in development mode without anyone
noticing. For the checklist, `inline (development)` is the expected and correct answer.

> **One caveat**: `--reload` watches for file changes and restarts the API. Because inline mode
> keeps job state in memory, a reload mid-run loses the job. Drop `--reload` if you are testing
> **D2/D3** (pane and workbook reattachment).

### Option B — full stack with Redis

Needed only if you want to exercise the real queue, or to test that a job survives an API
restart.

```bash
export XLF_TOKEN_SECRET="anything-for-local-use"
docker compose up --build
```

This starts Redis, the API on port 8000, and an `arq` worker. `/v1/health` will report
`"mode":"queued"`. The worker requires `REDIS_URL` and refuses to start without it — it shares
job state with the API through Redis, and an earlier version that fell back to its own
in-memory store meant submitted jobs sat at "queued" forever with nothing in either log to say
why.

---

## 2. Start the pane's dev server

```bash
cd addin
npm run dev
```

Serves `https://localhost:3000` and **proxies `/v1` to the API on port 8000**. That proxy is
load-bearing rather than a convenience: if the pane were served from `https://localhost:3000`
and called `http://localhost:8000` directly, the browser would block it as mixed content, and
the session cookie would never be sent because the request is cross-origin. Proxying makes
everything same-origin, which is also how it is deployed.

If your API is somewhere else:

```bash
XLF_API=http://localhost:9000 npm run dev
```

Sanity check before touching Excel — open `https://localhost:3000/src/taskpane.html` in a
browser. You should see the pane with no certificate warning. A warning here means step 0 did
not take.

---

## 3. Sideload into Excel

The manifest points at `https://localhost:3000`, so the dev server must already be running.

### Windows

```bash
cd addin
npm run sideload        # opens Excel with the add-in loaded
npm run sideload:stop   # unregisters it afterwards
```

If that fails, do it by hand: share a folder, add it under
**File → Options → Trust Center → Trust Center Settings → Trusted Add-in Catalogs**, tick
*Show in Menu*, restart Excel, then **Insert → My Add-ins → Shared Folder**.

### Mac

```bash
mkdir -p ~/Library/Containers/com.microsoft.Excel/Data/Documents/wef
cp addin/manifest.xml ~/Library/Containers/com.microsoft.Excel/Data/Documents/wef/
```

Restart Excel, then **Insert → My Add-ins → Developer Add-ins**.

To see console output, enable the developer menu:
`defaults write com.microsoft.Excel OfficeWebAddinDeveloperExtras -bool true`, then right-click
the pane and choose *Inspect Element*.

### Excel on the web

Open a workbook at office.com, then **Insert → Add-ins → Upload My Add-in** and choose
`addin/manifest.xml`.

The browser must trust the localhost certificate, which is what step 0 arranged. In this host
you get normal browser dev tools, which makes it the easiest place to debug — **but not a
substitute for the other two**, since the whole reason for the checklist is that the three
webviews differ.

---

## 4. Prepare a test workbook

Generate a panel that will actually produce a leaderboard:

```bash
uv run python -c "
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
dates = pd.date_range('2019-01-31', periods=72, freq='ME')
rows = []
for i in range(20):
    level, amp = rng.uniform(80, 400), rng.uniform(0.15, 0.35)
    t = np.arange(72)
    y = level * (1 + amp*np.sin(2*np.pi*(t + rng.integers(0,12))/12)) * (1 + 0.08*rng.standard_normal(72))
    rows += [{'sku': f'SKU{i:02d}', 'month': d, 'units': round(max(v,0),2)} for d, v in zip(dates, y)]
pd.DataFrame(rows).to_csv('test_panel.csv', index=False)
print('wrote test_panel.csv')
"
```

Open it in Excel, select the used range **including the header row**, and press **Read
selection**. Set frequency to `ME` and horizon to `6`.

For specific checklist items you will want variants:

| item | what to prepare |
|---|---|
| **B4** (row cap) | A sheet with more than 500,000 data rows |
| **C9** (output overflow) | 2,000 series with horizon 52 — the Run button should be disabled with an explanation |
| **E1** (baseline wins) | A panel of pure random walks: `y = np.cumsum(rng.normal(0,1,72)) + 100` |
| **C6** (protected sheet) | Run once, then protect `XLF_Forecast` and run again |

---

## 5. Where to look when something fails

| symptom | most likely cause |
|---|---|
| Pane blank, or "add-in could not be started" | Certificate not installed, or the dev server is not running |
| "Failed to fetch" in the pane | API is not up, or the Vite proxy is pointing somewhere else |
| Job stays at "queued" forever | Option B without a worker running, or `REDIS_URL` set for the API but not the worker |
| Job "failed" immediately | Check the API console; the engine's error carries a remedy |
| Dates profile as 1970 | A date column mapped to something that is not a date — the pane converts Excel serials, but only from the column you mapped as the date |
| Everything works on web, nothing on desktop | The certificate is trusted by your browser but not by the OS store; re-run `npm run certs` |

The pane surfaces the server's own error message and its suggested fix rather than a stack
trace, so what you see in the red box should be actionable. If it ever is not, that is a
finding worth recording against **FS §4's error-presentation rule** — a vague error is a bug,
not a nuisance.

---

## 6. What is not wired yet

So you do not spend time hunting these:

- **Auth.** There is none. The API treats every caller as `anonymous`; the real scheme is an
  `HttpOnly; Secure; SameSite=Lax` session cookie. This does not affect any checklist item.
- **"Explain results" / "Explain this cell".** Phase 6. The buttons exist and do nothing
  useful.
- **Charts (FR-705).** Not built.
- **Natural language (S2's Ask box).** Phase 6. The manual form is the source of truth and is
  what the checklist exercises.
