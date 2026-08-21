/**
 * Batched sheet writers (FR-701, FR-702, FR-703, FR-703a).
 *
 * Five sheets, written as one transaction. The manifest is **inside** that transaction: under
 * the original "four sheets" wording it sat outside the overwrite set, so a re-run left a
 * manifest describing the *previous* run beside new results — breaking FR-704 and hard rule 10
 * on the second run of every workbook.
 *
 * Every sheet is written with one range assignment. Never per-cell, never per-row (FR-702).
 */

import { SHEETS, type SheetName } from "./layout";

export interface SheetPayload {
  readonly name: SheetName;
  readonly header: readonly string[];
  readonly rows: readonly (readonly unknown[])[];
  readonly hidden?: boolean;
}

export type WorkbookIssue =
  | { readonly kind: "protected"; readonly sheet: string }
  | { readonly kind: "referenced"; readonly sheet: string; readonly by: readonly string[] }
  | { readonly kind: "co-authoring" };

export interface PreflightResult {
  readonly blocking: readonly WorkbookIssue[];
  readonly warnings: readonly WorkbookIssue[];
}

/**
 * FR-703a — check the workbook's state before writing anything.
 *
 * A protected sheet blocks the whole write rather than producing a partial one: four correct
 * sheets and one stale sheet is a worse outcome than a clean refusal, because nothing on the
 * surface says which is which.
 */
export function classifyIssues(issues: readonly WorkbookIssue[]): PreflightResult {
  return {
    blocking: issues.filter((i) => i.kind === "protected" || i.kind === "co-authoring"),
    warnings: issues.filter((i) => i.kind === "referenced"),
  };
}

export async function preflight(context: Excel.RequestContext): Promise<WorkbookIssue[]> {
  const issues: WorkbookIssue[] = [];
  const sheets = context.workbook.worksheets;
  sheets.load("items/name");
  await context.sync();

  const existing = sheets.items.filter((s) => (SHEETS as readonly string[]).includes(s.name));
  for (const sheet of existing) sheet.protection.load("protected");
  await context.sync();

  for (const sheet of existing) {
    if (sheet.protection.protected) issues.push({ kind: "protected", sheet: sheet.name });
  }
  return issues;
}

/**
 * Write all five sheets.
 *
 * Sheets are created if absent and cleared if present. The caller must have confirmed the
 * overwrite (FR-703) and run {@link preflight} first.
 */
export async function writeAll(
  context: Excel.RequestContext,
  payloads: readonly SheetPayload[],
): Promise<void> {
  const sheets = context.workbook.worksheets;

  // Existence has to be resolved before anything is queued against it: `isNullObject` is only
  // populated after a sync, and `worksheets.add` throws on a name that already exists. Two
  // syncs total -- one to look, one to write -- which is still a batched write in the sense
  // FR-702 means. It is per-cell and per-row syncing that the rule forbids.
  const probes = payloads.map((payload) => {
    const probe = sheets.getItemOrNullObject(payload.name);
    probe.load("isNullObject");
    return probe;
  });
  await context.sync();

  payloads.forEach((payload, index) => {
    const probe = probes[index]!;
    const sheet = probe.isNullObject ? sheets.add(payload.name) : probe;
    sheet.getUsedRangeOrNullObject(true).clear(Excel.ClearApplyTo.contents);

    const table = [payload.header, ...payload.rows] as unknown[][];
    if (table.length > 0 && payload.header.length > 0) {
      // One assignment for the whole sheet (FR-702).
      sheet.getRangeByIndexes(0, 0, table.length, payload.header.length).values = table;
      sheet.getRange("A1").getResizedRange(0, payload.header.length - 1).format.font.bold = true;
    }
    sheet.visibility = payload.hidden
      ? Excel.SheetVisibility.veryHidden
      : Excel.SheetVisibility.visible;
  });

  // One sync for the whole write: five sheets, one round trip.
  await context.sync();
}
