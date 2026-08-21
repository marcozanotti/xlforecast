/**
 * Chunked range reading (TS §7.1, FR-107, hard rule 9).
 *
 * Hard rule 9 forbids a sync whose cost scales with cells, not the bounded loop of chunked
 * syncs this performs — reading 500,000 rows in one `context.sync()` is what the rule is
 * about. Progress is reported throughout, because on a large range this takes tens of seconds
 * and silence reads as a freeze.
 */

import { planChunks, refuseIfTooManyInputRows, type OverflowRefusal } from "./layout";

export type ProgressReporter = (done: number, total: number) => void;

export interface RangeShape {
  readonly rowCount: number;
  readonly columnCount: number;
  readonly address: string;
}

export interface ReadResult {
  readonly header: string[];
  readonly rows: unknown[][];
  readonly refusal: OverflowRefusal | null;
}

/** Describe the selection without reading it, so the cap can be enforced first. */
export async function selectionShape(context: Excel.RequestContext): Promise<RangeShape> {
  const range = context.workbook.getSelectedRange();
  // `getUsedRange` on the selection: a user who selects whole columns would otherwise appear
  // to have selected 1,048,576 rows and be refused for a panel of 300.
  const used = range.getUsedRangeOrNullObject(true);
  used.load(["rowCount", "columnCount", "address", "isNullObject"]);
  await context.sync();
  if (used.isNullObject) return { rowCount: 0, columnCount: 0, address: "" };
  return {
    rowCount: used.rowCount,
    columnCount: used.columnCount,
    address: used.address,
  };
}

/**
 * Read the selection in bounded chunks.
 *
 * The first row is taken as the header, which is what the S1 column-mapping dropdowns are
 * populated from.
 */
export async function readSelection(
  context: Excel.RequestContext,
  shape: RangeShape,
  onProgress?: ProgressReporter,
): Promise<ReadResult> {
  const dataRows = Math.max(0, shape.rowCount - 1);
  const refusal = refuseIfTooManyInputRows(dataRows);
  if (refusal) return { header: [], rows: [], refusal };
  if (dataRows === 0) return { header: [], rows: [], refusal: null };

  const sheet = context.workbook.worksheets.getActiveWorksheet();
  const anchor = sheet.getRange(shape.address);

  const headerRange = anchor.getRow(0);
  headerRange.load("values");
  await context.sync();
  const header = (headerRange.values[0] ?? []).map((cell) => String(cell ?? ""));

  const plan = planChunks(dataRows, shape.columnCount);
  const rows: unknown[][] = [];
  let done = 0;
  for (const chunk of plan.chunks) {
    // +1 skips the header row.
    const block = anchor.getOffsetRange(chunk.startRow + 1, 0).getResizedRange(
      chunk.rowCount - 1,
      0,
    );
    block.load("values");
    await context.sync();
    rows.push(...block.values);
    done += chunk.rowCount;
    onProgress?.(done, dataRows);
  }
  return { header, rows, refusal: null };
}
