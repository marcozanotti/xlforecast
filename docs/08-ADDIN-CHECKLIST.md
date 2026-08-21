# 08 — Add-in Host Verification Checklist (Gate G5)

**This is a human checklist. It cannot be automated, and pretending otherwise is how add-ins
ship broken on one host.** Office.js behaves differently across Excel for Windows (WebView2),
Excel for Mac (WKWebView) and Excel on the web, and the differences are in exactly the places
this add-in touches: range I/O at volume, custom properties, streaming responses, and sheet
visibility.

Run every row on all three hosts before signing off G5. Record the date, the build, and the
Excel version — "it worked" without a version is not a result.

| | Windows | Mac | Web |
|---|---|---|---|
| Excel version tested | | | |
| Date | | | |

---

## A. Sideloading

| # | Check | Expected |
|---|---|---|
| A1 | Add-in loads from the manifest | Pane opens, no certificate warning |
| A2 | Ribbon button appears under Home | Labelled "xlforecast" |
| A3 | Reopening the pane preserves nothing sensitive | No credential visible in the workbook's custom properties |

## B. Reading the grid (FR-107, TS §7.1)

| # | Check | Expected |
|---|---|---|
| B1 | Select a 300-series × 156-row range, read | Completes; progress text updates rather than freezing |
| B2 | Select whole columns (e.g. `A:D`) | Treated as the *used* range, not 1,048,576 rows — a panel of 300 must not be refused |
| B3 | Read a range with 500,000 data rows | Completes; note the wall-clock per host |
| B4 | Read a range with 500,001 data rows | **Refused** with the message pointing at file input, before any upload |
| B5 | Header row populates the mapping dropdowns | All columns listed |
| B6 | Cancel mid-read | Pane returns to S1 cleanly |

**Record B3's timing per host.** The estimate in TS §7.1 is "tens of seconds" for the native
case; if a host is materially worse, that is a finding, not a nuisance.

## C. Writing the sheets (FR-701, FR-702, FR-703, FR-703a)

| # | Check | Expected |
|---|---|---|
| C1 | Run a job to completion | **Five** sheets written: Forecast, Forecast_Long, Leaderboard, Diagnostics, Manifest |
| C2 | `XLF_Manifest` is hidden | Very hidden; visible only via VBA/Name Manager |
| C3 | Interval columns match the requested levels | `levels=[50,80,95]` yields six band columns, not four |
| C4 | **Re-run the same workbook** | All five sheets overwritten, **including the manifest** — the manifest must describe the *new* run |
| C5 | Rename `XLF_Leaderboard`, re-run | Sheet recreated; no duplicate |
| C6 | Protect `XLF_Forecast`, re-run | **Whole write refused** with a named message — not four sheets written and one skipped |
| C7 | Put `=XLF_Forecast!C2` in another sheet, re-run | Warning naming the referring cell, then proceeds on confirmation |
| C8 | Open the workbook in co-authoring, run | Refused with an explanation |
| C9 | Request an output that overflows (2,000 series, h=52, 9 models) | **Refused before writing**, offering winner-only or file export. Nothing partially written |

C4 and C6 are the two most likely to regress and the two least likely to be noticed: a stale
manifest looks like a fresh one, and a partial write looks like a successful one.

## D. Job lifecycle (FR-801, FR-802, TS §7.3)

| # | Check | Expected |
|---|---|---|
| D1 | Progress updates arrive per model per fold | "Fold 2 of 3 · AutoETS", not a bare percentage |
| D2 | Close the pane mid-run, reopen it | Reattaches to the running job and resumes showing progress |
| D3 | Close the **workbook** mid-run, reopen it | Same — the job id lives in a custom property |
| D4 | Cancel mid-run | Job stops; the pane says so; completed folds remain downloadable |
| D5 | Streaming works | If SSE is unavailable on a host, the pane falls back to polling **without erroring** |
| D6 | Kill the server mid-run | Pane shows a named error with a remedy, not a stack trace |

D5 is the one to watch. `fetch` streaming support differs across the three webviews, and the
fallback is the whole reason `EventSource` was rejected.

## E. Honesty of the result (FR-406, FR-408, FR-303)

| # | Check | Expected |
|---|---|---|
| E1 | Run a panel where nothing beats the baseline | Notice shown **prominently**; recommendation is `SeasonalNaive` |
| E2 | Coverage line shown in S5 | Quotes the **out-of-calibration** figure |
| E3 | Per-series selection | Winner's score labelled as selection-biased, with the leave-one-fold-out figure beside it |
| E4 | Excluded series | Named individually with reasons, never a bare count |

E1 is a product check as much as a technical one. If "nothing beat seasonal naive" reads as a
failure of the tool rather than a finding about the data, the wording is wrong — and the
wording is the product.

---

## Sign-off

G5 passes only when every row above is checked on all three hosts. A row that fails on one
host is a G5 failure, not a caveat: the add-in is a single distribution channel and a user on
Mac does not care that it works on Windows.

| | Name | Date |
|---|---|---|
| Verified | | |
