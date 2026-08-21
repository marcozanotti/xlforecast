# 06 — Methodology

*Draft for publication. Written at Phase 3 as the build plan requires: if the leaderboard
cannot be explained in writing, the design is wrong.*

This page describes exactly what `xlforecast` does to your data and what its numbers mean. It
also states, plainly, several things that make the tool look less impressive than it could.
Those are the parts worth reading.

---

## 1. What a run actually does

You supply a panel: many time series in long format, keyed by series id, date and value. The
engine then:

1. **Profiles** the panel — frequency, seasonality, intermittency class, missingness.
2. **Validates** each series and excludes the ones it cannot honestly forecast, telling you
   which and why. It never drops a series silently.
3. **Splits** the history into cross-validation folds at **panel-wide calendar cutoffs**.
4. **Fits every model on identical folds** and scores them on identical evaluation points.
5. **Calibrates prediction intervals** by cross-conformal prediction.
6. **Builds an ensemble inside the cross-validation loop**, so it competes rather than being
   assembled afterwards from the winners.
7. **Selects** a model, refits on full history, and produces the forecast.
8. **Emits a manifest** sufficient to reproduce the run exactly.

Every number you see comes from step 4 or later. No number is produced by a language model.

---

## 2. Fair comparison is the whole product

A model competition is only worth reading if every competitor faced the same test. Two things
make that true here, and both cost us something.

### Panel-wide cutoffs, not per-series

Real panels are *ragged*: products launch and get discontinued, so series end on different
dates. The standard forecasting libraries derive cross-validation cutoffs from **each series'
own last observation**. On a ragged panel that gives every series a different fold date — and a
*global* model trained at one series' cutoff has then seen other series' later observations.

That is look-ahead leakage, and it flatters exactly the models it flatters most: the global
ones, which pool across the panel. Worse, it is invisible to the obvious check. If you assert
that two libraries received identical cutoffs, they did — per series, by construction.

So the engine computes its own cutoffs, one calendar date per fold for the whole panel, and
slices the training and test windows itself. It asserts equality of the **evaluation index** —
the exact `(series, date)` pairs being scored — rather than of the cutoffs. That test runs on a
deliberately ragged panel and must never be skipped.

The cost: we cannot use the libraries' own cross-validation, so we reimplemented the loop.

### Different models see different amounts of data

Even at identical cutoffs, a machine-learning model that builds lag features discards the first
`max_lag` observations of every series, while a statistical model trains on the full history. At
a 52-week lag over three series, that is 156 rows of difference at the same cutoff.

We do not hide this. Effective training rows per model family per fold are recorded in the
diagnostics, so a local-versus-global comparison can be audited rather than assumed clean.

---

## 3. Uncertainty: what the intervals mean

### Cross-conformal, not split-conformal

Prediction intervals are calibrated from cross-validation residuals. The obvious way to do this
is also wrong: if you calibrate a band on a set of residuals and then measure its coverage
against those same residuals, you will get the nominal rate **by construction**. An empirical
quantile covers its own sample. The number cannot fail, so it tells you nothing.

Instead, the band used to score fold *k* is calibrated only from the other folds. The band
delivered with your forecast uses all of them.

We check this the way you should check any calibration claim — with a control. Alongside the
honest figure we compute the circular one and require it to differ:

| level | out-of-calibration (reported) | in-calibration (control) |
|---|---|---|
| 80% | 0.809 | 0.844 |
| 95% | 0.953 | 1.000 |

The control sits systematically *above* nominal and can never fall below it. That is precisely
why it is worthless as evidence and useful as a control: it would report a comfortable number
for an interval that was far too narrow.

### Intervals on count data are one-sided, and we say so

For series with many zeros — intermittent demand, most SKU panels — the interval's lower half
does almost no work. Measured on an intermittent panel at the 80% level, **0.00% of violations
fall below the interval and 15.6% above**, against a roughly balanced 10.2% / 5.5% on smooth
data.

This is not a bug we have failed to fix. `y = 0` is a point mass, and no continuous interval
handles it gracefully. We built the alternative — asymmetric bands from signed residuals — and
measured it: it balances the tails on smooth series and **under-covers intermittent ones at
0.693 against a 0.80 target**, because its lower bound lands above zero and excludes every zero.
The symmetric band is the better choice on the data our users have.

So intervals are symmetric, clipped to the series' support (a negative lower bound on unit
demand is not a wider forecast, it is a nonsensical one), and both tails are reported separately
so you can see the asymmetry rather than infer it.

### The coverage number is honest but coarse

Per-series calibration needs `n_windows × h` residuals to clear a minimum of 20. Below that,
series fall back to pooled panel residuals — recorded per series in the diagnostics, because it
changes what the interval means.

---

## 4. What CRPS adds, and what it does not

The leaderboard reports scaled CRPS alongside the point metrics. You should know that **it will
almost always rank the models in the same order as MASE.**

That follows directly from the design: every model's interval is `point forecast ± a quantile of
that model's own residuals`, so a model with a better point forecast gets a better interval
almost automatically. CRPS earns its place by pricing interval *width* — a model that is
accurate but overconfident is penalised — not by reordering the leaderboard.

CRPS is also **grid-dependent**. It is computed from the quantiles your chosen interval levels
imply, so a run at 80%/95% and a run at 50%/80%/95% produce values that are not comparable. The
grid is recorded in the manifest for that reason.

---

## 5. Selection, and the winner's curse

Choosing the best model per series by looking at cross-validation scores, and then reporting
that model's cross-validation score as its accuracy, is optimistic. The argmin has already used
those folds. With few folds and many models, some of the winner's advantage is luck.

The default is therefore **pooled** selection — one winner for the panel. Where you choose
per-series selection, the leaderboard:

- warns when there are fewer than five cross-validation windows;
- flags the selected model's score as selection-biased;
- reports a **leave-one-fold-out** companion: select on the other folds, score on this one.

The gap between the two is the size of the winner's curse on your data.

---

## 6. The baseline is not decoration

Every leaderboard carries `SeasonalNaive`, which cannot be disabled, and `WindowAverage` at the
seasonal frequency — a trailing average over one full cycle, which is what most people
forecasting in a spreadsheet are doing today.

**If nothing beats them, the tool says so and recommends the baseline.** That is a result, not a
failure. A forecasting tool that always finds a winner is not measuring anything.

---

## 7. Validation against published results

The engine is benchmarked against the **Monash Time Series Forecasting Archive**'s published M3
baselines (Godahewa et al., 2021), using the archive's own protocol — a single forecast origin,
the last `h` observations held out — and its own MASE definition, which we verify matches ours
to twelve decimal places.

Mean MASE, ours against published:

| dataset | AutoETS | AutoARIMA | Theta |
|---|---|---|---|
| M3 Yearly (645 series) | 2.695 / 2.860 | 2.882 / 3.417 | 2.576 / 2.774 |
| M3 Quarterly (756) | 1.143 / 1.170 | 1.191 / 1.240 | 1.126 / 1.117 |
| M3 Monthly (1,428) | 0.863 / 0.865 | 0.876 / 0.873 | 0.850 / 0.864 |

M3 was chosen over M5 deliberately: 3,003 series with short histories resembles what our users
own far better than 42,840 hierarchical Walmart series does.

Note the one large gap — AutoARIMA on yearly, 15.7% better than published. We flag rather than
celebrate improvements that size, because they are also what a data leak looks like. In this
case monthly reproduces the archive to within 0.3% using the same code path, the archive's own
yearly ARIMA is anomalous (worse than its own SES), and the split was verified directly.

---

## 8. What we do not claim

- **We do not claim the winning model will win next quarter.** Cross-validation measures past
  performance on your data. That is the best available evidence, not a guarantee.
- **We do not explain causes.** Explanations describe what the fitted model did — its seasonal
  component, its feature attributions — never why the world behaved that way. If a holiday was
  not a regressor, we will not tell you a holiday caused anything.
- **We do not let a language model choose your model or produce your numbers.** It reads
  derived diagnostics and writes prose. Cross-validation selects; every numeral is checked
  against the artifact it came from.
- **We do not claim calibration we have not measured.** Coverage is reported out-of-calibration,
  with the tail split shown, and the pooled-fallback cases marked.

---

## 9. Reproducing a run

Every run emits a manifest containing the resolved configuration, package versions, the data
fingerprint, the exact fold cutoffs, the excluded series and why, the CRPS quantile grid, the
ensemble parameters, and the thread configuration. Re-running from it reproduces the leaderboard
byte for byte.

Thread configuration is in there because it has to be: float reductions reorder under thread
count, so byte-identity is defined relative to a recorded configuration rather than absolutely.
Worker count, we verified, does not affect the numbers — series are fitted independently — but
it is recorded rather than assumed.
