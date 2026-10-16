# 2026 Prediction Input Phase 1: 2026-10-16

Read-only inspection in isolated audit branch. No CORE4 / 7 Kings / Outsider / L1.75 / frozen L2.1 / L3 V3 changes; no changes to BACKFILL storage. Free standard CPU only, no artifacts.

## Evidence
- Source BACKFILL commit: 03950cfe132abeca15ef17dbc7c6885ceb1591b7
- Dates: 2026-10-10, 2026-10-11, 2026-10-12
- Cutoff: 2026-10-17 (exclusive)
- Run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38870025385
- Unit tests: 9/9 pass; real-source integrity errors: 0
- Source: 72 races, 980 entries, 980 results, 979 finished.
- Race venue, discipline, surface, distance, weather and track: 72/72.
- Race direction and lap availability: 69/72.
- Horse number, sex, age, carried weight, body weight, jockey ID, trainer ID: 980/980.
- Finish time, finishing position and final 3F present: 979/980.
- Above findings are only for the three audited race days, NOT a full 2026 audit.

## The source/feature boundary
These files are completed postrace BACKFILL history. They are not pre-race AI-ready features.
The original frozen CORE4 schema is 649 names with sorted-name SHA256:
eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58.
Additional original state, day bias and regime transforms enrich the frame to 690 features.
No 2026 derived panel or complete 690-feature parity has been validated.
Never claim historical FINAL odds are real-time prebet odds.

## Reuse on prediction day
The read-only checker research/audit_2026_prediction_inputs_v1.py has two distinct inputs:
1. Historical BACKFILL day packs, passed with repeated --history-day FILE and
   --prediction-cutoff YYYY-MM-DD. Reject target-day or future historical outcomes,
   inspect horse IDs/numbers and entry/result joins, and print source coverage.
2. An unscored target feature panel, with --panel FILE --target-race-id ID
   --target-date YYYY-MM-DD. Reject postrace target features, saved predictions and
   identity drift. Optional --expected-core4-keys FILE confirms each horse has the
   genuine canonical 649-feature list (validated by source digest).
This is a diagnostic entry guard. It deliberately does NOT declare full AI
prediction readiness without checking the downstream 690-feature frame, provenance,
historical-asof timestamps and actual quote source.

## Next phase (not yet implemented)
Build 2026 derived features using unchanged original builders and 2025-and-earlier
state plus strictly-prior 2026 results. Establish canonical feature parity and
horse/day state completeness. 2026 remains sealed for performance evaluation.
