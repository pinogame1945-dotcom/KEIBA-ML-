# ABSORB-001 — run 36348397121

- Run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/36348397121
- Code SHA: 2b088be349984d9036511341a5c03b7abb5a3f17
- Snapshot generation: 5e82b8e07b08853c
- Snapshot persistence: runner-local ephemeral only
- BACKFILL SHA: af9d0dcf3295e8276ca39fcbe104d7e9ac4c1cdc
- Train: 2023-01-01 .. 2024-12-31
- Holdout: 2025-01-01 .. 2025-12-31  / ALL races
- Main Feature Arena ledger: separate

## Scorecard

| Candidate | Train | Top1 | Top3 | Top6 | Mean winner rank | Features | Peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| no_auto_full | ALL | 28.79% | 59.46% | 81.57% | 3.8287037037037037 | 266 | 5665.75 |
| no_auto_full_plus_raceclass | ALL | 28.47% | 59.38% | 81.89% | 3.8113425925925926 | 300 | 6130.316 |
| no_auto_full_pedigree_v1 | ALL | 28.39% | 59.87% | 82.49% | 3.7881944444444446 | 504 | 10899.16 |
| no_auto_full_pedigree_legacy | ALL | 28.30% | 59.38% | 82.18% | 3.795428240740741 | 470 | 10297.215 |
| jockey_trainer_condition | ALL | 28.21% | 59.87% | 81.66% | 3.795138888888889 | 423 | 9123.996 |
| no_auto_jockey | ALL | 28.10% | 59.61% | 82.03% | 3.8200231481481484 | 164 | 3424.289 |
| core4_no_pedigree | ALL | 28.07% | 60.16% | 82.18% | 3.8017939814814814 | 649 | 13679.59 |
| no_auto_jockey_plus_raceclass | ALL | 27.86% | 59.66% | 81.51% | 3.8136574074074074 | 198 | 4494.77 |
| jockey_trainer_condition_plus_raceclass | ALL | 27.72% | 60.13% | 82.47% | 3.7745949074074074 | 457 | 9742.703 |
| core4_plus_raceclass_pedigree | ALL | 27.69% | 59.87% | 81.94% | 3.8122106481481484 | 683 | 13881.273 |
