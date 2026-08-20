# LLM regression suites

Phase 6. Two suites:

- **Explanation (AC-605):** 100 cases, measured **pre-guardrail** — first-pass rejection < 5%, the
  sentence-strip path fires zero times, and post-guardrail unmatched numerals are zero. Measuring
  only the last of those tests the sanitiser, not the system: §9.4 strips offending sentences, so
  the post-guardrail figure is zero by construction and would pass a model that hallucinated in
  every case.
- **Parser:** ~60 phrasings (EN + IT) with expected `ForecastRequest` fields, including the legacy
  pandas frequency aliases (`M`, `Q`, `H`) that a parser will emit because its training data is
  full of them (FR-112).

Recorded responses only, so CI is deterministic and free.
