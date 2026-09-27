# KODOKU-001 — WIN Feature Arena

First permanent L1 Feature Arena experiment.

- Model family: WIN Binary LightGBM
- Train: 2023-01-01 .. 2024-12-31
- Holdout: 2025-01-01 .. 2025-12-31
- Snapshot generation: e75cd10b7837f676
- BACKFILL SHA: af9d0dcf3295e8276ca39fcbe104d7e9ac4c1cdc
- Prediction phase: FINAL
- Feature selection: none
- Odds/popularity/payout as L1 predictors: forbidden
- 2026: locked

Candidates:

1. BASE
2. BASE + OPPONENT
3. BASE + NETWORK
4. BASE + LAP
5. BASE + STYLE
6. BASE + DISTANCE
7. BASE + BACKFILL
8. BASE + AUTO
9. BASE + PEDIGREE
10. BASE + ACTOR
11. BASE + TIME_PACE
12. ALL

Every execution is preserved under `attempts/run-<GitHub Run ID>/`.
Attempts are immutable. Failed attempts are retained alongside successful attempts.
No composite winner is assigned automatically.
