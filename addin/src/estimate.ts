/**
 * Runtime estimate for the confirmation card (FS section 4, S3).
 *
 * "It must exist, because users will not wait for something with no ETA."
 *
 * Built from **measured** per-model cost (docs/05), not from a model count. A model count is
 * a poor proxy in both directions: a global model is one fit per fold across the whole panel
 * while a local model is one fit per series per fold, and SeasonalNaive and AutoARIMA differ
 * by four orders of magnitude at identical fit counts.
 */

/** CPU seconds per series per fit, measured on real data. See docs/05 sections 2 and 13. */
const LOCAL_COST_PER_SERIES_FIT: Record<string, number> = {
  SeasonalNaive: 0.0003,
  HistoricAverage: 0.0002,
  WindowAverage: 0.0002,
  CrostonClassic: 0.0002,
  ADIDA: 0.0002,
  IMAPA: 0.0002,
  ZeroModel: 0.0001,
  DynamicOptimizedTheta: 0.0028,
  AutoETS: 0.1397,
  AutoCES: 0.15,
  // AutoARIMA is the dominant term and depends on the FR-201a mode, so it is resolved below
  // rather than sitting in this table.
  LocalLinear: 0.0345,
  LocalLGBM: 0.6165,
  LocalXGB: 0.0474,
};

/** CPU seconds per fit for the whole panel. Measured flat in panel size (docs/05 section 2). */
const GLOBAL_COST_PER_FIT: Record<string, number> = {
  GlobalLinear: 0.041,
  GlobalLGBM: 0.82,
  GlobalXGB: 0.248,
};

/** FR-201a: Fourier above m=24 is 3.1x cheaper than a seasonal order search. */
const AUTOARIMA_SEASONAL = 0.923;
const AUTOARIMA_FOURIER = 0.297;

/**
 * Overhead that is nobody's train or predict but is real cost (FR-217a).
 *
 * A flat multiplier is crude, and it is honest about being crude: ingestion, profiling,
 * conformal calibration, ensembling and persistence all scale differently. Estimating only
 * train and predict would understate every run, which is worse than a rough allowance.
 */
const OVERHEAD_FACTOR = 1.5;

export interface EstimateInput {
  readonly nSeries: number;
  readonly models: readonly string[];
  readonly nWindows: number;
  readonly seasonLength: number;
  readonly workers: number;
}

export interface RuntimeEstimate {
  readonly cpuSeconds: number;
  readonly wallSeconds: number;
  readonly dominantModel: string | null;
  readonly text: string;
}

function costFor(model: string, seasonLength: number, nSeries: number, fits: number): number {
  if (model === "AutoARIMA") {
    const perFit = seasonLength > 24 ? AUTOARIMA_FOURIER : AUTOARIMA_SEASONAL;
    return perFit * nSeries * fits;
  }
  const global = GLOBAL_COST_PER_FIT[model];
  // A global model is fitted once per fold over the whole panel, not once per series.
  if (global !== undefined) return global * fits;
  const local = LOCAL_COST_PER_SERIES_FIT[model];
  return local === undefined ? 0 : local * nSeries * fits;
}

/** Round to something a person can act on. "About 3 minutes" beats "182 seconds". */
export function humanise(seconds: number): string {
  if (seconds < 45) return "under a minute";
  const minutes = seconds / 60;
  if (minutes < 10) return `about ${Math.round(minutes)} minute${Math.round(minutes) === 1 ? "" : "s"}`;
  if (minutes < 90) return `about ${Math.round(minutes / 5) * 5} minutes`;
  return `over an hour`;
}

export function estimateRuntime(input: EstimateInput): RuntimeEstimate {
  // Folds plus the final refit on full history.
  const fits = input.nWindows + 1;
  const perModel = input.models.map((model) => ({
    model,
    cost: costFor(model, input.seasonLength, input.nSeries, fits),
  }));
  const cpuSeconds = perModel.reduce((sum, m) => sum + m.cost, 0) * OVERHEAD_FACTOR;
  const workers = Math.max(1, input.workers);
  const wallSeconds = cpuSeconds / workers;
  const dominant = perModel.sort((a, b) => b.cost - a.cost)[0];

  const share = dominant && cpuSeconds > 0 ? (dominant.cost * OVERHEAD_FACTOR) / cpuSeconds : 0;
  // Naming the dominant model turns a number the user must accept into one they can act on:
  // if the estimate is unacceptable, it says which model to drop.
  //
  // Two guards keep it from being noise. Short runs do not need it -- nobody cares which of
  // three sub-second models led when the whole answer is "under a minute". And with fewer
  // than three models the leader is trivially over half the total, so "most of it is X" says
  // nothing a user could act on: dropping one of two models is not advice.
  const detail =
    dominant && share > 0.4 && wallSeconds >= 30 && input.models.length >= 3
      ? ` Most of it is ${dominant.model}.`
      : "";
  return {
    cpuSeconds,
    wallSeconds,
    dominantModel: dominant?.model ?? null,
    text: `Estimated runtime: ${humanise(wallSeconds)}.${detail}`,
  };
}
