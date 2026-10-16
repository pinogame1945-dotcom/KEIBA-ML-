# Phase 2 (2026): original feature-generator exercise and CORE4 name audit

Date: 2026-10-16. Isolated branch: audit/2026-feature-input-phase1-20261016.

## Evidence
- Free standard CPU successful: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38872696302
- Original frozen CORE4 code and existing yearly snapshot builder unchanged.
- BACKFILL source pinned at 03950cfe132abeca15ef17dbc7c6885ceb1591b7
- Source warm-up: 2022-01-01 through 2026-10-12 (strictly prior for a prospective 2026-10-17 race).
- Snapshot generation outputs: 2025-12-28 reference and 2026-01-04 through 2026-10-12.
- No training/evaluation using 2026 outcomes. Results from earlier completed days used only for downstream historical state. Same-day outcomes are committed only after prediction feature production for that date, as in unchanged createMlDatasetProcessor.processDay().
- Both derived snapshots ephemeral in the runner; no Kaggle upload, persistent paid artifact, cache or dataset stored.

## Verified counts and immutable schema
- 2025 comparison: 356 horse rows, 24 races, one date (2025-12-28).
- 2026 output: 37,582 horse rows, 2,732 races, 87 race dates (2026-01-04..2026-10-12).
- Both year-wide CORE4 name unions have exactly 649 names with SHA-256
  eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58.
- 2026 populated feature family cells:
  * actor_ 98.22%
  * auto_ 85.91%
  * opponent_ 88.82%
  * timepace_ 89.09%
- Six auto_history_*_gap_to_{best,worst} cells were absent across all rows in the examined 2026 source. These are columns in the canonical 649 feature set but their values are consistently null; investigate before claiming full value parity.

## Race-specific blocker: important for live card validation
- 2026: 207/2,732 race panels (7.58%) have a per-race union of feature names smaller or different than the canonical 649, although year union is exact.
- 2025 Dec28 reference: 4/24 race panels also have incomplete per-race union, so this is not uniquely a 2026 source defect.
- 2026: 3,508/37,582 individual horse rows omit at least one key; historical original Pandas DataFrame.from_records naturally fills absent horse keys with NaN, hence individual sparseness is NOT by itself an error.
- The historical frozen one-race original inference path validates exactly 649 columns in its selected race DataFrame. Do not call the 207 incomplete race panels directly score-ready.
- Do NOT silently alter feature generator formulas, frozen model, inputs, purchase logic or evaluation methodology. Before operational use, distinguish deliberately nullable absent keys from incorrectly missing source data; any compatibility handling needs explicit validation and authorization if it changes agreed input data shape.

## Current test state
- Phase 2 name-union audit passed: run 38872696302.
- Phase 1 live/target panel test suite amended to compare per-race unions as the frozen model does, not 649 fields on every horse; 11/11 tests passed: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38872876140
- The 2026 history generator DOES NOT yield live prebet as-of evidence, a fully valued match to a historical snapshot, day bias/state/regime 690-column parity, a trained-model inference output, or L2.1/L3 stake predictions.

## Next required verification
Compare 2026 horse-level feature values to the original historical reference / full history
including derived state, day bias and regime; determine the precise missing per-race keys
for the 207 panels and the non-null coverage implications. Confirm as-of provenance and
pre-start target card / prices separately; preserve the 2026 evaluation seal.
