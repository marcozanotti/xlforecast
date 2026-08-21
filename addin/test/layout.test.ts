import { describe, expect, it } from "vitest";
import {
  EXCEL_MAX_ROWS,
  MAX_INGEST_ROWS,
  estimateOutput,
  forecastColumns,
  planChunks,
  refuseIfTooLarge,
  refuseIfTooManyInputRows,
} from "../src/layout";

describe("forecastColumns (FR-701)", () => {
  it("generates interval columns per requested level", () => {
    expect(forecastColumns([80, 95])).toEqual([
      "unique_id", "ds", "y_hat",
      "y_hat_lo_80", "y_hat_hi_80", "y_hat_lo_95", "y_hat_hi_95",
      "model",
    ]);
  });

  it("supports level sets the original fixed layout could not express", () => {
    expect(forecastColumns([50, 80, 95])).toContain("y_hat_lo_50");
  });

  it("is deterministic regardless of the order levels arrive in", () => {
    expect(forecastColumns([95, 80])).toEqual(forecastColumns([80, 95]));
  });

  it("deduplicates repeated levels", () => {
    expect(forecastColumns([80, 80, 95])).toEqual(forecastColumns([80, 95]));
  });

  it("handles a single level", () => {
    expect(forecastColumns([90])).toEqual([
      "unique_id", "ds", "y_hat", "y_hat_lo_90", "y_hat_hi_90", "model",
    ]);
  });
});

describe("estimateOutput (FR-708)", () => {
  it("counts every model's forecast, not just the winner's", () => {
    // 10 series x 6 horizon x 3 models x (1 point + 2 levels x 2 bounds)
    expect(estimateOutput(10, 6, 3, [80, 95]).longRows).toBe(10 * 6 * 3 * 5);
  });

  it("shows the input cap does not imply output headroom", () => {
    // The constraint table claimed the 500k input cap "leaves headroom for output sheets".
    // A 2,000-series panel is well inside that cap and still overflows by 4x.
    const estimate = estimateOutput(2000, 52, 9, [80, 95]);
    expect(estimate.longRows).toBeGreaterThan(4_000_000);
    expect(estimate.fits).toBe(false);
  });

  it("accepts a panel that genuinely fits", () => {
    expect(estimateOutput(300, 13, 9, [80, 95]).fits).toBe(true);
  });

  it("accounts for the header row at the boundary", () => {
    expect(estimateOutput(EXCEL_MAX_ROWS - 1, 1, 1, []).fits).toBe(true);
    expect(estimateOutput(EXCEL_MAX_ROWS, 1, 1, []).fits).toBe(false);
  });
});

describe("refuseIfTooLarge (FR-708 / NFR-12)", () => {
  it("returns null when the output fits", () => {
    expect(refuseIfTooLarge(estimateOutput(50, 12, 5, [80]))).toBeNull();
  });

  it("refuses before writing and offers named degradations", () => {
    const refusal = refuseIfTooLarge(estimateOutput(2000, 52, 9, [80, 95]));
    expect(refusal).not.toBeNull();
    // Never truncates: a plausible-looking file missing rows the user never learns about is
    // the same failure as dropping a series silently.
    expect(refusal!.options).toEqual(["winner-only", "external-file"]);
    expect(refusal!.fix).toBeTruthy();
  });

  it("states the actual numbers rather than 'too large'", () => {
    const refusal = refuseIfTooLarge(estimateOutput(2000, 52, 9, [80, 95]))!;
    expect(refusal.message).toMatch(/1,048,576/);
    expect(refusal.message).toMatch(/4,/);
  });
});

describe("refuseIfTooManyInputRows (FR-107)", () => {
  it("allows a range at the cap", () => {
    expect(refuseIfTooManyInputRows(MAX_INGEST_ROWS)).toBeNull();
  });

  it("refuses one row over, pointing at file input", () => {
    const refusal = refuseIfTooManyInputRows(MAX_INGEST_ROWS + 1)!;
    expect(refusal.fix).toMatch(/CSV or Parquet/);
  });
});

describe("planChunks (TS §7.1, hard rule 9)", () => {
  it("bounds each chunk by cells, not rows", () => {
    // Wide panels must not silently exceed the budget: the same row count yields smaller
    // chunks as columns grow. Both ranges are long enough that the cell budget, rather than
    // the range length, is what limits the chunk.
    expect(planChunks(60_000, 10, 50_000).chunks[0]!.rowCount).toBe(5_000);
    expect(planChunks(60_000, 2, 50_000).chunks[0]!.rowCount).toBe(25_000);
  });

  it("never returns a chunk longer than the range", () => {
    expect(planChunks(10_000, 2, 50_000).chunks).toEqual([{ startRow: 0, rowCount: 10_000 }]);
  });

  it("covers every row exactly once", () => {
    const plan = planChunks(12_345, 7, 50_000);
    const covered = plan.chunks.reduce((sum, c) => sum + c.rowCount, 0);
    expect(covered).toBe(12_345);
    expect(plan.chunks[0]!.startRow).toBe(0);
  });

  it("makes the round-trip count explicit for a full-cap read", () => {
    // 500k rows x 5 columns = 2.5M cells = 50 syncs. Under native Office.js those are local;
    // this is the number ADR-002 turned on.
    expect(planChunks(500_000, 5, 50_000).chunks).toHaveLength(50);
  });

  it("handles a range narrower than one chunk", () => {
    expect(planChunks(10, 3, 50_000).chunks).toEqual([{ startRow: 0, rowCount: 10 }]);
  });

  it("returns nothing for an empty range", () => {
    expect(planChunks(0, 5).chunks).toEqual([]);
    expect(planChunks(100, 0).chunks).toEqual([]);
  });

  it("never produces a zero-row chunk even when a row exceeds the budget", () => {
    const plan = planChunks(5, 100_000, 50_000);
    expect(plan.chunks.every((c) => c.rowCount >= 1)).toBe(true);
  });
});
