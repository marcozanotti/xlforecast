/**
 * Task pane entry point: the S1-S5 state machine (FS section 4).
 *
 * The pane holds no credentials and no forecasting logic. It reads a range, encodes it, posts
 * it, follows a job, and writes five sheets. Every number it displays came from the engine.
 *
 * Everything worth testing lives in the modules this file wires together. What remains here
 * is Office.js and DOM plumbing, which is verified by hand against three hosts
 * (docs/08-ADDIN-CHECKLIST.md) rather than against a mock that would agree with whatever was
 * written.
 */

import { ApiClient, isTerminal, XlfError, type JobProgress } from "./api";
import { encodePanel } from "./encode";
import { estimateRuntime } from "./estimate";
import { estimateOutput, refuseIfTooLarge } from "./layout";
import { readSelection, selectionShape } from "./ranges";
import { buildAllSheets, coverageNote, verdict, type RunResult } from "./results";
import { classifyIssues, preflight, writeAll } from "./sheets";
import { WorkbookState, type PropertyBag } from "./state";

type State = "S1" | "S2" | "S3" | "S4" | "S5";

// Same-origin by default: the dev server and the deployment both proxy `/v1` to the API.
// An absolute origin here would be mixed content in development and cross-origin in
// production, and the session cookie would be dropped in both cases. Overridable at build
// time only -- the pane must not let a page it did not build point it at another host.
const api = new ApiClient(import.meta.env.VITE_API_BASE ?? "");

const DEFAULT_MODELS = [
  "SeasonalNaive", "HistoricAverage", "WindowAverage", "AutoARIMA", "AutoETS",
  "DynamicOptimizedTheta", "CrostonClassic", "LocalLinear", "LocalLGBM", "LocalXGB",
  "GlobalLinear", "GlobalLGBM", "GlobalXGB",
];

interface Session {
  header: string[];
  rows: unknown[][];
  dataId: string | null;
  nSeries: number;
  request: Record<string, unknown> | null;
  confirmationToken: string | null;
  jobId: string | null;
}

const session: Session = {
  header: [], rows: [], dataId: null, nSeries: 0,
  request: null, confirmationToken: null, jobId: null,
};

/* ------------------------------------------------------------------ DOM helpers */

function $(id: string): HTMLElement | null {
  return document.getElementById(id);
}

function show(state: State): void {
  document.querySelectorAll<HTMLElement>("[data-state]").forEach((section) => {
    section.hidden = section.dataset["state"] !== state;
  });
  clearError();
}

function setText(id: string, text: string): void {
  const node = $(id);
  if (node) node.textContent = text;
}

function clearError(): void {
  const box = $("error");
  if (box) box.hidden = true;
}

function fail(error: unknown): void {
  const box = $("error");
  if (!box) return;
  // FS section 4: name the fault and state the fix. Never a stack trace.
  box.textContent =
    error instanceof XlfError ? error.display : "Something went wrong. Please try again.";
  box.hidden = false;
}

function renderTable(id: string, header: readonly string[], rows: readonly unknown[][]): void {
  const table = $(id);
  if (!table) return;
  const head = `<tr>${header.map((h) => `<th>${h}</th>`).join("")}</tr>`;
  const body = rows
    .map((row) => `<tr>${row.map((c) => `<td>${c ?? ""}</td>`).join("")}</tr>`)
    .join("");
  table.innerHTML = head + body;
}

/* ------------------------------------------------------------------ Office.js glue */

/** Custom properties, behind the Protocol `state.ts` is written against. */
function propertyBag(): PropertyBag {
  return {
    async get(key) {
      return Excel.run(async (context) => {
        const property = context.workbook.properties.custom.getItemOrNullObject(key);
        property.load(["value", "isNullObject"]);
        await context.sync();
        return property.isNullObject ? null : String(property.value);
      });
    },
    async set(key, value) {
      await Excel.run(async (context) => {
        context.workbook.properties.custom.add(key, value);
        await context.sync();
      });
    },
    async remove(key) {
      await Excel.run(async (context) => {
        const property = context.workbook.properties.custom.getItemOrNullObject(key);
        property.load("isNullObject");
        await context.sync();
        if (!property.isNullObject) property.delete();
        await context.sync();
      });
    },
  };
}

const workbook = new WorkbookState(propertyBag());

/* ------------------------------------------------------------------ S1: data */

async function readAndUpload(): Promise<void> {
  const shape = await Excel.run(async (context) => selectionShape(context));
  if (shape.rowCount === 0) {
    throw new XlfError("The selection is empty.", "Select the range containing your data.");
  }

  const read = await Excel.run(async (context) =>
    readSelection(context, shape, (done, total) => {
      // Silence on a long read is indistinguishable from a freeze (TS section 7.1).
      setText("selection-summary", `Reading ${done.toLocaleString()} of ${total.toLocaleString()} rows...`);
    }),
  );
  if (read.refusal) throw new XlfError(read.refusal.message, read.refusal.fix);

  session.header = read.header;
  session.rows = read.rows;
  populateMapping(read.header);
  setText("selection-summary", `${read.rows.length.toLocaleString()} rows read from ${shape.address}.`);
}

function populateMapping(header: readonly string[]): void {
  const guesses: Record<string, string[]> = {
    "map-unique-id": ["unique_id", "id", "sku", "item", "series"],
    "map-ds": ["ds", "date", "week", "month", "period"],
    "map-y": ["y", "value", "units", "qty", "quantity", "sales"],
  };
  for (const [id, hints] of Object.entries(guesses)) {
    const select = $(id) as HTMLSelectElement | null;
    if (!select) continue;
    select.replaceChildren(
      ...header.map((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        return option;
      }),
    );
    // A guess the user can correct beats an empty dropdown they must fill three times.
    const guess = header.find((h) => hints.includes(h.toLowerCase().trim()));
    if (guess) select.value = guess;
  }
}

function mappingFromForm(): { uniqueIdCol: string; dsCol: string; yCol: string } {
  return {
    uniqueIdCol: ($("map-unique-id") as HTMLSelectElement | null)?.value ?? "",
    dsCol: ($("map-ds") as HTMLSelectElement | null)?.value ?? "",
    yCol: ($("map-y") as HTMLSelectElement | null)?.value ?? "",
  };
}

async function uploadPanel(): Promise<void> {
  const mapping = mappingFromForm();
  const encoded = encodePanel(session.header, session.rows, mapping);
  const config = configFromForm();

  const response = await api.uploadPanel(encoded.bytes.buffer as ArrayBuffer, {
    ...mapping,
    freq: String(config["freq"]),
    horizon: Number(config["h"]),
  });
  session.dataId = response.data_id;

  const profile = response.profile as {
    n_series: number;
    validation: { n_series_in: number; n_series_out: number; excluded_detail: Record<string, string> };
  };
  session.nSeries = profile.validation.n_series_out;

  // S1's live validation summary. Excluded series are named, never a bare count (FS section 6).
  const excluded = Object.entries(profile.validation.excluded_detail);
  const summary =
    excluded.length === 0
      ? `${profile.validation.n_series_out} series valid.`
      : `${profile.validation.n_series_out} of ${profile.validation.n_series_in} series valid. ` +
        `Excluded: ${excluded.map(([id]) => id).join(", ")}.`;
  setText("validation-summary", summary);
  if (encoded.skipped > 0) {
    setText(
      "validation-summary",
      `${summary} ${encoded.skipped} row(s) skipped for a missing id or date.`,
    );
  }
}

/* ------------------------------------------------------------------ S2/S3: configure and confirm */

function configFromForm(): Record<string, unknown> {
  const value = (id: string): string => ($(id) as HTMLInputElement | null)?.value ?? "";
  const season = value("cfg-season");
  return {
    h: Number(value("cfg-h") || 13),
    freq: value("cfg-freq") || "W",
    season_length: season === "" ? null : Number(season),
    n_windows: Number(value("cfg-windows") || 3),
    selection: ($("cfg-selection") as HTMLSelectElement | null)?.value ?? "pooled",
    models: DEFAULT_MODELS,
  };
}

function renderConfirmation(): void {
  const config = configFromForm();
  session.request = config;

  const list = $("confirm-summary");
  if (list) {
    list.innerHTML = Object.entries(config)
      .map(([key, val]) => `<dt>${key}</dt><dd>${Array.isArray(val) ? val.length + " models" : val ?? "inferred"}</dd>`)
      .join("");
  }

  // FR-402: per-series selection on few windows overfits, and the warning belongs where the
  // choice is made rather than in the results.
  const warning = $("selection-warning");
  if (warning) {
    const risky = config["selection"] === "per_series" && Number(config["n_windows"]) < 5;
    warning.hidden = !risky;
    warning.textContent = risky
      ? `Per-series selection with ${config["n_windows"]} windows overfits: each winner is an ` +
        `argmin over models on ${config["n_windows"]} folds. Its score will be labelled as biased.`
      : "";
  }

  const estimate = estimateRuntime({
    nSeries: session.nSeries,
    models: config["models"] as string[],
    nWindows: Number(config["n_windows"]),
    seasonLength: Number(config["season_length"] ?? 52),
    workers: 8,
  });
  setText("runtime-estimate", estimate.text);

  // FR-708: refuse before compute is spent, not after.
  const output = estimateOutput(
    session.nSeries,
    Number(config["h"]),
    (config["models"] as string[]).length,
    [80, 95],
  );
  const refusal = refuseIfTooLarge(output);
  const box = $("output-refusal");
  const run = $("btn-run") as HTMLButtonElement | null;
  if (box) {
    box.hidden = refusal === null;
    box.textContent = refusal ? `${refusal.message} ${refusal.fix}` : "";
  }
  if (run) run.disabled = refusal !== null;
}

/* ------------------------------------------------------------------ S4: running */

async function runJob(): Promise<void> {
  if (!session.dataId || !session.request) {
    throw new XlfError("No data has been read yet.", "Go back and read the selection first.");
  }
  // AC-503: confirm, then submit. The token is bound to this exact configuration, so a
  // setting changed after confirming invalidates it.
  const { confirmation_token } = await api.confirm(session.dataId, session.request);
  session.confirmationToken = confirmation_token;

  const mapping = mappingFromForm();
  const { job_id } = await api.submitJob({
    data_id: session.dataId,
    request: session.request,
    mapping: {
      unique_id_col: mapping.uniqueIdCol,
      ds_col: mapping.dsCol,
      y_col: mapping.yCol,
      exog: [],
    },
    confirmation_token,
  });
  session.jobId = job_id;
  await workbook.attach(job_id, session.dataId);
  show("S4");
  await followJob(job_id);
}

async function followJob(jobId: string): Promise<void> {
  const bar = $("progress") as HTMLProgressElement | null;
  if (bar) bar.dataset["jobId"] = jobId;

  try {
    for await (const event of api.streamProgress(jobId)) {
      renderProgress(event);
      if (isTerminal(event.status)) {
        if (event.status !== "completed") {
          throw new XlfError(
            `The job ${event.status.replace("_", " ")}.`,
            "Any folds that finished are still available.",
          );
        }
        break;
      }
    }
  } catch (error) {
    // TS section 7.3: streaming support differs across the three webviews, so a stream that
    // will not open falls back to polling rather than failing the job.
    if (error instanceof XlfError && error.status) throw error;
    await pollUntilDone(jobId);
  }

  const results = (await api.results(jobId)) as unknown as RunResult;
  await writeResults(results);
  await workbook.detach();
  renderResults(results);
  show("S5");
}

async function pollUntilDone(jobId: string): Promise<void> {
  for (;;) {
    const event = await api.jobStatus(jobId);
    renderProgress(event);
    if (isTerminal(event.status)) {
      if (event.status !== "completed") {
        throw new XlfError(`The job ${event.status.replace("_", " ")}.`, "Try running it again.");
      }
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

function renderProgress(event: JobProgress): void {
  const progress = event.progress;
  const bar = $("progress") as HTMLProgressElement | null;
  if (progress && bar) {
    const total = Math.max(1, progress.folds_total * progress.models_total);
    const done = progress.folds_done * progress.models_total + progress.models_done_in_fold;
    bar.value = Math.min(100, (done / total) * 100);
  }
  // Per model per fold, because that is what a user can interpret. "47%" is not.
  setText(
    "progress-detail",
    progress
      ? `Fold ${Math.min(progress.folds_done + 1, progress.folds_total)} of ${progress.folds_total}` +
        (progress.current_model ? ` - ${progress.current_model}` : "")
      : `Status: ${event.status}`,
  );
}

/* ------------------------------------------------------------------ S5: results */

async function writeResults(results: RunResult): Promise<void> {
  const payloads = buildAllSheets(results);
  await Excel.run(async (context) => {
    const issues = classifyIssues(await preflight(context));
    if (issues.blocking.length > 0) {
      const first = issues.blocking[0]!;
      // FR-703a: refuse the whole write rather than producing four correct sheets and one
      // stale one, which nothing on the surface would distinguish.
      throw new XlfError(
        first.kind === "protected"
          ? `Cannot write results: the sheet '${first.sheet}' is protected.`
          : "Cannot write results while the workbook is being co-authored.",
        "Unprotect the xlforecast sheets, or close co-authoring, and run again.",
      );
    }
    await writeAll(context, payloads);
  });
}

function renderResults(results: RunResult): void {
  setText("verdict", verdict(results));
  setText("coverage-note", coverageNote(results) ?? "");
  const panel = results.leaderboard.rows.filter((row) => row.scope === "panel");
  renderTable(
    "leaderboard",
    ["model", "MASE", "CRPS", "vs baseline", "vs incumbent"],
    panel.map((row) => [
      row.model,
      row.mase?.toFixed(3) ?? "n/a",
      row.scaled_crps?.toFixed(4) ?? "n/a",
      row.vs_baseline_pct === null ? "n/a" : `${row.vs_baseline_pct.toFixed(1)}%`,
      row.vs_incumbent_pct === null ? "n/a" : `${row.vs_incumbent_pct.toFixed(1)}%`,
    ]),
  );
}

/* ------------------------------------------------------------------ wiring */

function on(id: string, handler: () => Promise<void> | void): void {
  $(id)?.addEventListener("click", () => {
    clearError();
    void Promise.resolve(handler()).catch(fail);
  });
}

Office.onReady(async (info) => {
  if (info.host !== Office.HostType.Excel) return;

  on("btn-read", async () => {
    await readAndUpload();
    await uploadPanel();
    show("S2");
  });
  on("btn-configure", () => {
    renderConfirmation();
    show("S3");
  });
  on("btn-run", runJob);
  on("btn-cancel", async () => {
    if (session.jobId) await api.cancelJob(session.jobId);
  });
  on("btn-rerun", () => show("S2"));
  on("btn-export-manifest", async () => {
    if (!session.jobId) return;
    // The manifest is already on its hidden sheet; this surfaces it for a board pack.
    const results = (await api.results(session.jobId)) as unknown as RunResult;
    setText("verdict", `Manifest: ${JSON.stringify(results.manifest).slice(0, 200)}...`);
  });

  // A reopened pane rejoins a running job rather than losing it (TS section 7.3). Checked
  // before anything else, because a user who reopens mid-run should not be shown S1.
  try {
    const attached = await workbook.attached();
    if (attached) {
      session.jobId = attached.jobId;
      session.dataId = attached.dataId;
      show("S4");
      await followJob(attached.jobId);
      return;
    }
  } catch (error) {
    fail(error);
  }
  show("S1");
});
