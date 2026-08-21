import { describe, expect, it } from "vitest";
import { estimateRuntime, humanise } from "../src/estimate";

const DEFAULT_13 = [
  "SeasonalNaive", "HistoricAverage", "WindowAverage", "AutoARIMA", "AutoETS",
  "DynamicOptimizedTheta", "CrostonClassic", "LocalLinear", "LocalLGBM", "LocalXGB",
  "GlobalLinear", "GlobalLGBM", "GlobalXGB",
];

describe("estimateRuntime", () => {
  it("reproduces the measured NFR-01 projection", () => {
    // docs/05: 200 series x 13 models x 3 folds is ~2.9 minutes on 8 workers with the
    // overhead allowance. The estimate the user sees must agree with the measurement the
    // gate was signed off against.
    const estimate = estimateRuntime({
      nSeries: 200, models: DEFAULT_13, nWindows: 3, seasonLength: 52, workers: 8,
    });
    expect(estimate.wallSeconds).toBeGreaterThan(120);
    expect(estimate.wallSeconds).toBeLessThan(240);
  });

  it("counts a global model once per fold, not once per series", () => {
    // The whole reason model count is a bad proxy.
    const global = estimateRuntime({
      nSeries: 500, models: ["GlobalLGBM"], nWindows: 3, seasonLength: 12, workers: 1,
    });
    const local = estimateRuntime({
      nSeries: 500, models: ["LocalLGBM"], nWindows: 3, seasonLength: 12, workers: 1,
    });
    expect(local.cpuSeconds).toBeGreaterThan(global.cpuSeconds * 100);
  });

  it("charges AutoARIMA less in Fourier mode (FR-201a)", () => {
    const seasonal = estimateRuntime({
      nSeries: 100, models: ["AutoARIMA"], nWindows: 3, seasonLength: 12, workers: 1,
    });
    const fourier = estimateRuntime({
      nSeries: 100, models: ["AutoARIMA"], nWindows: 3, seasonLength: 52, workers: 1,
    });
    expect(fourier.cpuSeconds).toBeLessThan(seasonal.cpuSeconds);
    expect(seasonal.cpuSeconds / fourier.cpuSeconds).toBeCloseTo(3.1, 0);
  });

  it("scales with folds, counting the final refit", () => {
    const one = estimateRuntime({ nSeries: 50, models: ["AutoETS"], nWindows: 1, seasonLength: 12, workers: 1 });
    const three = estimateRuntime({ nSeries: 50, models: ["AutoETS"], nWindows: 3, seasonLength: 12, workers: 1 });
    // 4 fits against 2, not 3 against 1.
    expect(three.cpuSeconds / one.cpuSeconds).toBeCloseTo(2, 1);
  });

  it("divides by the worker count", () => {
    const serial = estimateRuntime({ nSeries: 100, models: ["AutoETS"], nWindows: 3, seasonLength: 12, workers: 1 });
    const parallel = estimateRuntime({ nSeries: 100, models: ["AutoETS"], nWindows: 3, seasonLength: 12, workers: 4 });
    expect(parallel.wallSeconds).toBeCloseTo(serial.wallSeconds / 4, 5);
  });

  it("names the dominant model on the default set, so the user knows what to drop", () => {
    // docs/05 measured LocalLGBM at 54% of total run cost on the 13-model default. That is
    // exactly the case where naming it is worth the words.
    const estimate = estimateRuntime({
      nSeries: 200, models: DEFAULT_13, nWindows: 3, seasonLength: 52, workers: 8,
    });
    expect(estimate.dominantModel).toBe("LocalLGBM");
    expect(estimate.text).toContain("LocalLGBM");
  });

  it("stays silent about the dominant model on a short run", () => {
    // Technically SeasonalNaive dominates three sub-millisecond models, but saying so is
    // noise when the whole answer is "under a minute".
    const estimate = estimateRuntime({
      nSeries: 10, models: ["SeasonalNaive", "HistoricAverage", "WindowAverage"],
      nWindows: 3, seasonLength: 12, workers: 1,
    });
    expect(estimate.text).toContain("under a minute");
    expect(estimate.text).not.toContain("Most of it is");
  });

  it("stays silent with only two models, where the leader is trivially over half", () => {
    // "Drop one of your two models" is not advice.
    const estimate = estimateRuntime({
      nSeries: 2000, models: ["AutoETS", "AutoCES"], nWindows: 3, seasonLength: 12, workers: 1,
    });
    expect(estimate.wallSeconds).toBeGreaterThan(30);
    expect(estimate.text).not.toContain("Most of it is");
  });

  it("names AutoARIMA when it is the one that dominates", () => {
    // The counterpart to the case above: at m=12 there is no Fourier discount, and seasonal
    // AutoARIMA is 76% of this trio. Which model leads depends on the configuration, so the
    // hint is computed rather than hardcoded.
    const estimate = estimateRuntime({
      nSeries: 3000, models: ["AutoETS", "AutoCES", "AutoARIMA"],
      nWindows: 3, seasonLength: 12, workers: 1,
    });
    expect(estimate.dominantModel).toBe("AutoARIMA");
    expect(estimate.text).toContain("AutoARIMA");
  });

  it("ignores an unknown model rather than throwing", () => {
    // A model the pane does not have a cost for must not break the confirmation card.
    const estimate = estimateRuntime({
      nSeries: 10, models: ["NotAModel"], nWindows: 3, seasonLength: 12, workers: 1,
    });
    expect(estimate.cpuSeconds).toBe(0);
  });

  it("never divides by zero workers", () => {
    const estimate = estimateRuntime({
      nSeries: 10, models: ["AutoETS"], nWindows: 1, seasonLength: 12, workers: 0,
    });
    expect(Number.isFinite(estimate.wallSeconds)).toBe(true);
  });
});

describe("humanise", () => {
  it.each([
    [5, "under a minute"],
    [44, "under a minute"],
    [90, "about 2 minutes"],
    [60, "about 1 minute"],
    [1800, "about 30 minutes"],
    [7200, "over an hour"],
  ])("%is reads as %s", (seconds, expected) => {
    expect(humanise(seconds)).toBe(expected);
  });
});
