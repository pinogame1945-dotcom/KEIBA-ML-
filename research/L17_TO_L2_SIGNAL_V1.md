# L17_TO_L2_SIGNAL_V1

## Purpose

Final L1.7 horse-level signal contract for L2.

Seven-King consensus rank remains authoritative and is never reordered by Outsider signals.
Outsider signals are exported as independent confidence/context features for L2.

## Per-horse fields

- `horse_id`
- `king_rank`
- Seven-King consensus diagnostics:
  - `king_borda_score`
  - `king_normalized_borda`
  - `king_mean_rank`
  - `king_rank_std`
  - `king_best_rank`
  - `king_worst_rank`
  - `king_top1_votes`
  - `king_probability_mean`
  - `king_probability_std`
- Outsider council raw signals:
  - `outsider_score` (0..39; rank1=3, rank2=2, rank3=1 across 13 Outsiders)
  - `outsider_support_count` (0..13)
  - `outsider_top1_votes`
  - `outsider_top2_votes`
  - `outsider_top3_votes`
- Podium calibration for King ranks 1..10:
  - `p3_calibrated`
  - `p3_rank_baseline`
  - `p3_delta = p3_calibrated - p3_rank_baseline`
  - `calibration_status`

## Leakage rules

Historical calibration is strictly walk-forward:
- 2022 <- train 2021
- 2023 <- train 2021-2022
- 2024 <- train 2021-2023
- 2025 <- train 2021-2024

2021 has no prior training year, so calibrated probability fields are null.
Ranks 11+ remain outside calibration scope and their calibrated probability fields are null.

No finish position, target, odds, popularity, payout, ROI, or other outcome/market field is exported in the L1.7 -> L2 input.

## Ranking rule

`ranking_policy = SEVEN_KING_UNCHANGED`

There is intentionally no L1.7 reranked field.
L2 may learn how to use King rank, Outsider signals, and calibrated podium confidence for ticket roles and expected value, but L1.7 does not override the Seven-King ordering.

## Deployment model

A JSON coefficient file is produced from 2021-2025 only for future operational inference.
2026 remains sealed and is not used for research evaluation or model fitting.

## Storage / execution

- GitHub standard `ubuntu-latest` CPU only.
- No GPU, paid runner, Artifact, or Cache.
- Kaggle file names are resolved from metadata first.
- 404 is non-retryable.
- Transient 429/502/503/504/network errors may retry with backoff.
- Final compact repository outputs are guarded at 50 MiB total per run.
