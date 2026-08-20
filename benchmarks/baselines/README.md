# Committed benchmark baselines

Gate G3 requires results on M5/VN1 subsets to fall within a **committed tolerance band of a
committed baseline file** — named metrics, explicit tolerances, checked in *before* Phase 3 opens.

The original wording, "consistent with your published work", is not a gate: it has no metric, no
tolerance and no artifact, so it cannot fail. G3 is the one gate explicitly empowered to kill the
project, which is precisely the gate that must be falsifiable.

Expected contents by Phase 3: one JSON file per benchmark with the metric set from FR-208, the
tolerance for each, and the provenance of the reference numbers.
