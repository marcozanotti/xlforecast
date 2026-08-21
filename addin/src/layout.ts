/**
 * Sheet layouts and the output-size precheck (FR-701, FR-708).
 *
 * Pure functions, deliberately: this is where two spec-review findings live, and both are
 * arithmetic rather than Office.js behaviour, so they can be tested without Excel.
 */

/** Excel's hard row limit, including the header. */
export const EXCEL_MAX_ROWS = 1_048_576;

/** FR-107 — the grid ingestion cap. File input has no such limit (ADR-008). */
export const MAX_INGEST_ROWS = 500_000;

/** §5 constraint: Office.js range throughput degrades past roughly this per `context.sync()`. */
export const MAX_CELLS_PER_SYNC = 50_000;

export const SHEETS = [
  "XLF_Forecast",
  "XLF_Forecast_Long",
  "XLF_Leaderboard",
  "XLF_Diagnostics",
  "XLF_Manifest",
] as const;

export type SheetName = (typeof SHEETS)[number];

/**
 * Header for `XLF_Forecast` (FS §5).
 *
 * Interval columns are generated per requested level rather than hard-coded to 80/95: the
 * original layout could not express `levels=[50, 80, 95]` at all. Order follows sorted
 * levels so the sheet is deterministic.
 */
export function forecastColumns(levels: readonly number[]): string[] {
  const sorted = [...new Set(levels)].sort((a, b) => a - b);
  const bands = sorted.flatMap((level) => [`y_hat_lo_${level}`, `y_hat_hi_${level}`]);
  return ["unique_id", "ds", "y_hat", ...bands, "model"];
}

export interface OutputEstimate {
  readonly longRows: number;
  readonly forecastRows: number;
  readonly fits: boolean;
  readonly worstSheet: SheetName;
}

/**
 * Rows `XLF_Forecast_Long` will occupy (FR-708).
 *
 * `series × h × models × (1 + 2·levels)`, because that sheet holds *every* model's forecast,
 * not just the winner's. FR-107 caps the input; nothing capped the output, and the constraint
 * table's claim that the 500k input cap "leaves headroom for output sheets" is arithmetically
 * false — 2,000 series at h=52 with 9 models and 2 levels is ~4.7M rows.
 */
export function estimateOutput(
  nSeries: number,
  horizon: number,
  nModels: number,
  levels: readonly number[],
): OutputEstimate {
  const perPoint = 1 + 2 * new Set(levels).size;
  const longRows = nSeries * horizon * nModels * perPoint;
  const forecastRows = nSeries * horizon;
  return {
    longRows,
    forecastRows,
    // +1 for the header row.
    fits: longRows + 1 <= EXCEL_MAX_ROWS,
    worstSheet: "XLF_Forecast_Long",
  };
}

export type Degradation = "winner-only" | "external-file";

export interface OverflowRefusal {
  readonly message: string;
  readonly fix: string;
  readonly options: readonly Degradation[];
}

/**
 * Refuse before writing anything (FR-708, NFR-12).
 *
 * Returning the degradations rather than silently choosing one: truncating a sheet would
 * leave a user with a plausible-looking file missing rows they never learn about, which is
 * the same class of failure as dropping a series without saying so (FS §6).
 */
export function refuseIfTooLarge(estimate: OutputEstimate): OverflowRefusal | null {
  if (estimate.fits) return null;
  return {
    message:
      `The full results need ${estimate.longRows.toLocaleString()} rows on ` +
      `XLF_Forecast_Long, over Excel's ${EXCEL_MAX_ROWS.toLocaleString()} row limit.`,
    fix: "Write only the selected model's forecast, or export the full results to a file.",
    options: ["winner-only", "external-file"],
  };
}

/** FR-107 — refuse a grid selection above the cap, pointing at file input. */
export function refuseIfTooManyInputRows(rows: number): OverflowRefusal | null {
  if (rows <= MAX_INGEST_ROWS) return null;
  return {
    message:
      `The selected range has ${rows.toLocaleString()} rows, over the ` +
      `${MAX_INGEST_ROWS.toLocaleString()} row limit for reading from the grid.`,
    fix: "Save the data as CSV or Parquet and use file input instead.",
    options: ["external-file"],
  };
}

export interface ChunkPlan {
  readonly chunks: readonly { readonly startRow: number; readonly rowCount: number }[];
  readonly totalCells: number;
}

/**
 * Split a range into bounded reads (TS §7.1).
 *
 * Hard rule 9 forbids a sync whose cost scales with cells; a bounded loop of chunked syncs is
 * required, not forbidden. Rows per chunk are derived from the column count so that wide
 * panels do not silently exceed the cell budget.
 */
export function planChunks(
  totalRows: number,
  totalColumns: number,
  maxCellsPerSync: number = MAX_CELLS_PER_SYNC,
): ChunkPlan {
  if (totalRows <= 0 || totalColumns <= 0) return { chunks: [], totalCells: 0 };
  const rowsPerChunk = Math.max(1, Math.floor(maxCellsPerSync / totalColumns));
  const chunks: { startRow: number; rowCount: number }[] = [];
  for (let start = 0; start < totalRows; start += rowsPerChunk) {
    chunks.push({ startRow: start, rowCount: Math.min(rowsPerChunk, totalRows - start) });
  }
  return { chunks, totalCells: totalRows * totalColumns };
}
