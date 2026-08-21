/**
 * Task pane entry point: the S1–S5 state machine (FS §4).
 *
 * The pane holds no credentials and no forecasting logic. It reads a range, posts it, polls a
 * job, and writes five sheets. Every number it displays came from the engine.
 */

import { ApiClient, isTerminal, XlfError } from "./api";
import { estimateOutput, forecastColumns, refuseIfTooLarge, SHEETS } from "./layout";
import { readSelection, selectionShape } from "./ranges";
import { classifyIssues, preflight, writeAll, type SheetPayload } from "./sheets";
import { WorkbookState, type PropertyBag } from "./state";

type State = "S1" | "S2" | "S3" | "S4" | "S5";

// The API origin is build-time configuration, not runtime state: the pane must not let a
// page it did not build point it at another host.
const api = new ApiClient(import.meta.env.VITE_API_BASE ?? "https://localhost:8000");

function show(state: State): void {
  document.querySelectorAll<HTMLElement>("[data-state]").forEach((section) => {
    section.hidden = section.dataset["state"] !== state;
  });
}

function fail(error: unknown): void {
  const box = document.getElementById("error");
  if (!box) return;
  // FS §4: name the fault and state the fix. Never a stack trace.
  box.textContent =
    error instanceof XlfError ? error.display : "Something went wrong. Please try again.";
  box.hidden = false;
}

/** Office.js custom properties, behind the Protocol `state.ts` is written against. */
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

Office.onReady(async (info) => {
  if (info.host !== Office.HostType.Excel) return;
  const workbook = new WorkbookState(propertyBag());

  // A reopened pane rejoins a running job rather than losing it (TS §7.3).
  const attached = await workbook.attached();
  if (attached) {
    show("S4");
    void followJob(attached.jobId, workbook);
  } else {
    show("S1");
  }

  document.getElementById("btn-read")?.addEventListener("click", () => {
    void Excel.run(async (context) => {
      const shape = await selectionShape(context);
      const result = await readSelection(context, shape, (done, total) => {
        const summary = document.getElementById("selection-summary");
        if (summary) summary.textContent = `Read ${done.toLocaleString()} of ${total.toLocaleString()} rows…`;
      });
      if (result.refusal) {
        fail(new XlfError(result.refusal.message, result.refusal.fix));
        return;
      }
      populateMapping(result.header);
      show("S2");
    }).catch(fail);
  });

  document.getElementById("btn-cancel")?.addEventListener("click", () => {
    const current = document.getElementById("progress")?.dataset["jobId"];
    if (current) void api.cancelJob(current).catch(fail);
  });
});

function populateMapping(header: readonly string[]): void {
  for (const id of ["map-unique-id", "map-ds", "map-y"]) {
    const select = document.getElementById(id) as HTMLSelectElement | null;
    if (!select) continue;
    select.replaceChildren(
      ...header.map((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        return option;
      }),
    );
  }
}

/** Poll or stream a job to completion, then write the sheets. */
async function followJob(jobId: string, workbook: WorkbookState): Promise<void> {
  const bar = document.getElementById("progress") as HTMLProgressElement | null;
  if (bar) bar.dataset["jobId"] = jobId;
  try {
    for await (const event of api.streamProgress(jobId)) {
      renderProgress(event);
      if (isTerminal(event.status)) break;
    }
    const results = await api.results(jobId);
    await writeResults(results);
    await workbook.detach();
    show("S5");
  } catch (error) {
    fail(error);
  }
}

function renderProgress(event: { status: string; progress: unknown }): void {
  const detail = document.getElementById("progress-detail");
  const progress = event.progress as
    | { folds_done: number; folds_total: number; current_model: string | null }
    | null;
  if (detail && progress) {
    // Per model per fold, because that is what a user can interpret. "47%" is not.
    detail.textContent = `Fold ${progress.folds_done + 1} of ${progress.folds_total}` +
      (progress.current_model ? ` · ${progress.current_model}` : "");
  }
}

async function writeResults(results: Record<string, unknown>): Promise<void> {
  const payloads = buildPayloads(results);
  await Excel.run(async (context) => {
    const issues = classifyIssues(await preflight(context));
    if (issues.blocking.length > 0) {
      // FR-703a: refuse the whole write rather than producing four correct sheets and one
      // stale one, which nothing on the surface would distinguish.
      throw new XlfError(
        `Cannot write results: ${issues.blocking[0]!.kind} sheet.`,
        "Unprotect the xlforecast sheets and try again.",
      );
    }
    await writeAll(context, payloads);
  });
}

function buildPayloads(results: Record<string, unknown>): SheetPayload[] {
  const levels = ((results["forecast"] as { levels?: number[] } | undefined)?.levels) ?? [80, 95];
  const estimate = estimateOutput(0, 0, 0, levels);
  const refusal = refuseIfTooLarge(estimate);
  if (refusal) throw new XlfError(refusal.message, refusal.fix);
  return SHEETS.map((name) => ({
    name,
    header: name === "XLF_Forecast" ? forecastColumns(levels) : [],
    rows: [],
    ...(name === "XLF_Manifest" ? { hidden: true } : {}),
  }));
}
