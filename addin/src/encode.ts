/**
 * Cell values → Arrow IPC (TS §6).
 *
 * The pane holds the grid as arrays of JavaScript values. Arrow is the one columnar format it
 * can produce in the browser without a WASM dependency, and the API accepts it alongside
 * Parquet for exactly this reason.
 *
 * Excel dates arrive as serial numbers, not strings, which is the detail most likely to
 * corrupt a panel silently: a serial read as a number gives a valid-looking column of five-
 * digit integers that profiles as "yearly data starting in 1970".
 */

import { tableToIPC, tableFromArrays } from "apache-arrow";

/** Excel's day-zero. The 1900 leap-year bug means serial 60 is a date that never existed. */
const EXCEL_EPOCH_UTC = Date.UTC(1899, 11, 30);
const MS_PER_DAY = 86_400_000;

export function isExcelSerialDate(value: unknown): value is number {
  // Serials below ~20,000 (1954) or above ~80,000 (2119) in a date column are far more likely
  // to be a mis-mapped numeric column than a real date, so the caller is told rather than
  // guessed at.
  return typeof value === "number" && Number.isFinite(value) && value > 0 && value < 2_958_466;
}

export function excelSerialToISO(serial: number): string {
  const ms = EXCEL_EPOCH_UTC + Math.round(serial * MS_PER_DAY);
  return new Date(ms).toISOString().slice(0, 10);
}

/** Normalise one cell from a date column to `YYYY-MM-DD`. */
export function normaliseDate(value: unknown): string | null {
  if (value === null || value === undefined || value === "") return null;
  if (isExcelSerialDate(value)) return excelSerialToISO(value);
  const text = String(value).trim();
  // Already a date string: hand it through and let the server's frequency layer judge it,
  // rather than reimplementing date parsing in the pane.
  return text === "" ? null : text;
}

export function normaliseNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  // Excel hands back strings for text-formatted cells; strip thousands separators only.
  const cleaned = String(value).replace(/[\s,]/g, "");
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
}

export interface ColumnMapping {
  readonly uniqueIdCol: string;
  readonly dsCol: string;
  readonly yCol: string;
}

export interface EncodedPanel {
  readonly bytes: Uint8Array;
  readonly rowCount: number;
  readonly skipped: number;
}

/**
 * Build an Arrow IPC stream from the header and rows read off the grid.
 *
 * Rows missing an id or a date are skipped and counted rather than sent as nulls: the server
 * would exclude them anyway, and a count the pane can show is more use than a silent
 * discrepancy between what the user selected and what was profiled.
 */
export function encodePanel(
  header: readonly string[],
  rows: readonly (readonly unknown[])[],
  mapping: ColumnMapping,
): EncodedPanel {
  const idIndex = header.indexOf(mapping.uniqueIdCol);
  const dsIndex = header.indexOf(mapping.dsCol);
  const yIndex = header.indexOf(mapping.yCol);
  if (idIndex < 0 || dsIndex < 0 || yIndex < 0) {
    const missing = [
      idIndex < 0 ? mapping.uniqueIdCol : null,
      dsIndex < 0 ? mapping.dsCol : null,
      yIndex < 0 ? mapping.yCol : null,
    ].filter(Boolean);
    throw new Error(`Column ${missing[0]} is not in the selected range.`);
  }

  const ids: string[] = [];
  const dates: string[] = [];
  const values: (number | null)[] = [];
  let skipped = 0;

  for (const row of rows) {
    const id = row[idIndex];
    const date = normaliseDate(row[dsIndex]);
    if (id === null || id === undefined || id === "" || date === null) {
      skipped += 1;
      continue;
    }
    ids.push(String(id));
    dates.push(date);
    values.push(normaliseNumber(row[yIndex]));
  }

  const table = tableFromArrays({
    unique_id: ids,
    ds: dates,
    // Nulls survive: the server's FR-105 rules decide what excess missingness means, and the
    // pane must not pre-empt that by zero-filling.
    y: Float64Array.from(values.map((v) => (v === null ? NaN : v))),
  });
  return { bytes: tableToIPC(table, "stream"), rowCount: ids.length, skipped };
}
