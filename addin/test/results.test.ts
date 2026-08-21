import { describe, expect, it } from "vitest";
import {
  buildAllSheets,
  buildDiagnosticsSheet,
  buildForecastSheet,
  buildLeaderboardSheet,
  buildManifestSheet,
  coverageNote,
  selectedModels,
  verdict,
  type RunResult,
} from "../src/results";

function result(overrides: Partial<RunResult> = {}): RunResult {
  return {
    job_id: "job-1",
    leaderboard: {
      any_beat_baseline: true,
      rows: [
        { scope: "panel", unique_id: null, model: "AutoETS", rank: 1, mase: 0.84,
          scaled_crps: 0.011, vs_baseline_pct: -11.5, vs_incumbent_pct: -54.5,
          selected: true, selection_biased: false, selected_lofo_score: null },
        { scope: "panel", unique_id: null, model: "SeasonalNaive", rank: 2, mase: 0.95,
          scaled_crps: 0.013, vs_baseline_pct: 0, vs_incumbent_pct: -48.6,
          selected: false, selection_biased: false, selected_lofo_score: null },
        { scope: "series", unique_id: "S1", model: "AutoETS", rank: 1, mase: 0.8,
          scaled_crps: 0.01, vs_baseline_pct: null, vs_incumbent_pct: null,
          selected: true, selection_biased: true, selected_lofo_score: 0.9 },
      ],
    },
    forecast: {
      levels: [80, 95],
      rows: [
        { unique_id: "S1", ds: "2024-01-31", model: "AutoETS", quantity: "point", level: null, value: 100 },
        { unique_id: "S1", ds: "2024-01-31", model: "AutoETS", quantity: "lo", level: 80, value: 90 },
        { unique_id: "S1", ds: "2024-01-31", model: "AutoETS", quantity: "hi", level: 80, value: 110 },
        { unique_id: "S1", ds: "2024-01-31", model: "AutoETS", quantity: "lo", level: 95, value: 85 },
        { unique_id: "S1", ds: "2024-01-31", model: "AutoETS", quantity: "hi", level: 95, value: 115 },
        { unique_id: "S1", ds: "2024-01-31", model: "SeasonalNaive", quantity: "point", level: null, value: 200 },
      ],
    },
    fold_scores: [{ fold_index: 0, cutoff: "2023-12-31", model: "AutoETS", unique_id: "S1", n_train_rows: 100 }],
    calibration: [
      { model: "AutoETS", level: 80, scope: "all", nominal: 0.8, empirical: 0.81,
        lower_tail: null, upper_tail: null },
    ],
    profile: { validation: { excluded_detail: { S9: "Series 'S9' is entirely zero. Remove it." } } },
    manifest: { engine_version: "0.1.0", data_fingerprint: "abc", autoarima_mode: "fourier", ets_mode: "mstl" },
    ...overrides,
  } as RunResult;
}

describe("XLF_Forecast (FR-701)", () => {
  it("writes only the selected model per series", () => {
    const sheet = buildForecastSheet(result());
    expect(sheet.rows).toHaveLength(1);
    expect(sheet.rows[0]![sheet.header.indexOf("model")]).toBe("AutoETS");
    // SeasonalNaive's forecast belongs on the long sheet, not here.
    expect(sheet.rows[0]![sheet.header.indexOf("y_hat")]).toBe(100);
  });

  it("generates interval columns per requested level", () => {
    const sheet = buildForecastSheet(result());
    expect(sheet.header).toContain("y_hat_lo_95");
    expect(sheet.rows[0]![sheet.header.indexOf("y_hat_lo_95")]).toBe(85);
  });

  it("supports a three-level request the fixed layout could not express", () => {
    const three = result({
      forecast: {
        levels: [50, 80, 95],
        rows: [
          { unique_id: "S1", ds: "2024-01-31", model: "AutoETS", quantity: "point", level: null, value: 100 },
          { unique_id: "S1", ds: "2024-01-31", model: "AutoETS", quantity: "lo", level: 50, value: 95 },
        ],
      },
    });
    expect(buildForecastSheet(three).header).toContain("y_hat_lo_50");
  });
});

describe("XLF_Leaderboard", () => {
  it("carries the selection-bias disclosure (FR-408)", () => {
    const sheet = buildLeaderboardSheet(result());
    const biasColumn = sheet.header.indexOf("selection_biased");
    const seriesRow = sheet.rows.find((r) => r[1] === "S1")!;
    expect(seriesRow[biasColumn]).toBe(true);
    expect(seriesRow[sheet.header.indexOf("selected_lofo_score")]).toBe(0.9);
  });

  it("prints absent metrics explicitly rather than blank (FR-214)", () => {
    const sheet = buildLeaderboardSheet(result());
    const seriesRow = sheet.rows.find((r) => r[1] === "S1")!;
    expect(seriesRow[sheet.header.indexOf("vs_baseline_pct")]).toBe("n/a");
  });
});

describe("XLF_Diagnostics", () => {
  it("names every excluded series with its reason (FS section 6)", () => {
    const flat = buildDiagnosticsSheet(result()).rows.flat();
    expect(flat).toContain("S9");
    expect(flat.some((c) => String(c).includes("entirely zero"))).toBe(true);
  });

  it("records the model-mode substitutions that changed what was fitted", () => {
    const flat = buildDiagnosticsSheet(result()).rows.flat();
    // FR-201a and FR-201c both change the fitted model, so both are auditable here.
    expect(flat).toContain("fourier");
    expect(flat).toContain("mstl");
  });
});

describe("XLF_Manifest (hard rule 10)", () => {
  it("is hidden and holds the whole manifest", () => {
    const sheet = buildManifestSheet(result());
    expect(sheet.hidden).toBe(true);
    expect(JSON.parse(String(sheet.rows[0]![0]))).toMatchObject({ engine_version: "0.1.0" });
  });

  it("is one of the five sheets, so a re-run refreshes it (FR-703)", () => {
    expect(buildAllSheets(result()).map((s) => s.name)).toEqual([
      "XLF_Forecast", "XLF_Forecast_Long", "XLF_Leaderboard", "XLF_Diagnostics", "XLF_Manifest",
    ]);
  });
});

describe("verdict (FR-406)", () => {
  it("says plainly when nothing beat the baseline", () => {
    const flat = result({
      leaderboard: { any_beat_baseline: false, rows: result().leaderboard.rows },
    });
    expect(verdict(flat)).toMatch(/No model beat/);
    expect(verdict(flat)).toMatch(/SeasonalNaive/);
  });

  it("names the winner and its margin otherwise", () => {
    expect(verdict(result())).toBe("AutoETS won (11.5% better than the baseline).");
  });
});

describe("coverageNote (FR-303)", () => {
  it("quotes the out-of-calibration figure", () => {
    expect(coverageNote(result())).toMatch(/out-of-calibration/);
    expect(coverageNote(result())).toMatch(/81%/);
  });

  it("returns nothing when no coverage was measurable", () => {
    expect(coverageNote(result({ calibration: [] }))).toBeNull();
  });
});

describe("selectedModels", () => {
  it("maps each series to its winner", () => {
    expect(selectedModels(result()).get("S1")).toBe("AutoETS");
  });
});
