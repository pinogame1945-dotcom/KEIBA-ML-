# L1.5 Blind-Spot Gate V2 — run 36434954540

- Run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/36434954540
- Complete: True
- 2026 sealed / Odds NO
- Standard CPU only / Artifact NO / Cache NO / New Kaggle persistence NO

## 10% primary budget by year — TRAIN_QUANTILE

| Year | Blind | BASE selected | BASE recall | PATTERN selected | PATTERN recall | Δ recall pp |
|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 300 | 371 | 15.33% | 377 | 17.00% | +1.67 |
| 2023 | 343 | 254 | 15.45% | 259 | 17.20% | +1.75 |
| 2024 | 346 | 339 | 16.47% | 330 | 15.32% | -1.16 |
| 2025 | 377 | 345 | 17.24% | 340 | 19.89% | +2.65 |

## Aggregate

| Model | AUC | PR-AUC | Mode | Budget | Intervention | Blind recall | Precision | Enrichment |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| BASE | 0.6484 | 0.1599 | TEST_RANK | 5% | 5.01% | 9.88% | 19.51% | 1.97x |
| BASE | 0.6484 | 0.1599 | TEST_RANK | 10% | 10.01% | 17.86% | 17.63% | 1.78x |
| BASE | 0.6484 | 0.1599 | TEST_RANK | 15% | 14.99% | 27.01% | 17.81% | 1.80x |
| BASE | 0.6484 | 0.1599 | TEST_RANK | 20% | 19.99% | 34.26% | 16.93% | 1.71x |
| BASE | 0.6484 | 0.1599 | TEST_RANK | 25% | 25.00% | 41.29% | 16.32% | 1.65x |
| BASE | 0.6484 | 0.1599 | TRAIN_QUANTILE | 5% | 4.52% | 8.71% | 19.04% | 1.93x |
| BASE | 0.6484 | 0.1599 | TRAIN_QUANTILE | 10% | 9.47% | 16.18% | 16.88% | 1.71x |
| BASE | 0.6484 | 0.1599 | TRAIN_QUANTILE | 15% | 14.71% | 26.35% | 17.71% | 1.79x |
| BASE | 0.6484 | 0.1599 | TRAIN_QUANTILE | 20% | 19.86% | 34.26% | 17.05% | 1.73x |
| BASE | 0.6484 | 0.1599 | TRAIN_QUANTILE | 25% | 25.43% | 41.95% | 16.30% | 1.65x |
| PATTERN | 0.6500 | 0.1600 | TEST_RANK | 5% | 5.01% | 9.74% | 19.22% | 1.95x |
| PATTERN | 0.6500 | 0.1600 | TEST_RANK | 10% | 10.01% | 18.74% | 18.50% | 1.87x |
| PATTERN | 0.6500 | 0.1600 | TEST_RANK | 15% | 14.99% | 26.13% | 17.23% | 1.74x |
| PATTERN | 0.6500 | 0.1600 | TEST_RANK | 20% | 19.99% | 34.19% | 16.90% | 1.71x |
| PATTERN | 0.6500 | 0.1600 | TEST_RANK | 25% | 25.00% | 40.70% | 16.09% | 1.63x |
| PATTERN | 0.6500 | 0.1600 | TRAIN_QUANTILE | 5% | 4.64% | 8.86% | 18.85% | 1.91x |
| PATTERN | 0.6500 | 0.1600 | TRAIN_QUANTILE | 10% | 9.45% | 17.42% | 18.22% | 1.84x |
| PATTERN | 0.6500 | 0.1600 | TRAIN_QUANTILE | 15% | 14.83% | 26.13% | 17.41% | 1.76x |
| PATTERN | 0.6500 | 0.1600 | TRAIN_QUANTILE | 20% | 20.07% | 34.33% | 16.90% | 1.71x |
| PATTERN | 0.6500 | 0.1600 | TRAIN_QUANTILE | 25% | 25.46% | 41.58% | 16.14% | 1.63x |

10% race IDs are preserved in summary.json for later independent-study joins.
