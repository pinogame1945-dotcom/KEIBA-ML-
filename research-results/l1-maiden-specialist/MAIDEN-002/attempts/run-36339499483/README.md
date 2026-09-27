# MAIDEN-002 — run 36339499483

- Run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/36339499483
- Code SHA: db085a8d694a67bf883f6da528259c18d9f392e0
- Snapshot generation: 1474ef105ac588c1
- Snapshot persistence: runner-local ephemeral only
- BACKFILL SHA: af9d0dcf3295e8276ca39fcbe104d7e9ac4c1cdc
- Train: 2023-01-01 .. 2024-12-31
- Holdout: 2025-01-01 .. 2025-12-31 / MAIDEN only
- Main Feature Arena ledger: separate

## Scorecard

| Candidate | Train | Top1 | Top3 | Top6 | Mean winner rank | Features | Peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| no_auto_full | ALL | 30.97% | 63.13% | 84.44% | 3.5522745411013568 | 266 | 5528.926 |
| no_auto_full_pedigree | ALL | 30.65% | 63.45% | 85.00% | 3.507581803671189 | 504 | 10899.137 |
| jockey_trainer_condition | ALL | 30.41% | 63.69% | 84.68% | 3.51316839584996 | 423 | 9124.051 |
| no_auto_jockey | ALL | 30.33% | 63.13% | 83.64% | 3.5650438946528333 | 164 | 3424.457 |
| maiden_same_full | MAIDEN | 30.17% | 61.61% | 85.40% | 3.5347166799680765 | 266 | 5666.457 |
| core4_no_pedigree | ALL | 29.61% | 64.64% | 85.40% | 3.482043096568236 | 649 | 13427.133 |
| maiden_same_full_pedigree_v1 | MAIDEN | 29.21% | 61.29% | 83.96% | 3.5411013567438148 | 504 | 10899.004 |
