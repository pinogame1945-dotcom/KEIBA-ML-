# L1.5 Outsider Router 31 V1 — run 36448207055

- Gate alerts: 1,384
- True blind inside Gate alerts: 277
- Five-outsider hindsight union: 244 / 277 = 88.09%
- All-11 hindsight union: 250 / 277 = 90.25%
- 2026 sealed / odds NO

## Train-only fixed baselines

| Policy | Rescued | /277 | End-to-end /1366 | Calls | Rescue/100 calls |
|---|---:|---:|---:|---:|---:|
| FIXED_K1 | 125 | 45.13% | 9.15% | 1384 | 9.03 |
| FIXED_K2 | 180 | 64.98% | 13.18% | 2768 | 6.50 |
| FIXED_K3 | 210 | 75.81% | 15.37% | 4152 | 5.06 |
| FIXED_K4 | 233 | 84.12% | 17.06% | 5536 | 4.21 |
| FIXED_K5 | 244 | 88.09% | 17.86% | 6920 | 3.53 |

## COMPACT learned Router

| Policy | Rescued | /277 | Δ vs fixed | End-to-end /1366 | Calls | Rescue/100 calls | Oracle capture |
|---|---:|---:|---:|---:|---:|---:|---:|
| INDIVIDUAL_TOP1 | 138 | 49.82% | +13 | 10.10% | 1384 | 9.97 | 56.56% |
| INDIVIDUAL_TOP2 | 188 | 67.87% | +8 | 13.76% | 2768 | 6.79 | 77.05% |
| INDIVIDUAL_TOP3 | 216 | 77.98% | +6 | 15.81% | 4152 | 5.20 | 88.52% |
| COMBO_K1 | 138 | 49.82% | +13 | 10.10% | 1384 | 9.97 | 56.56% |
| COMBO_K2 | 180 | 64.98% | +0 | 13.18% | 2768 | 6.50 | 73.77% |
| COMBO_K3 | 207 | 74.73% | -3 | 15.15% | 4152 | 4.99 | 84.84% |
| COMBO_K4 | 233 | 84.12% | +0 | 17.06% | 5536 | 4.21 | 95.49% |
| COMBO_K5 | 244 | 88.09% | +0 | 17.86% | 6920 | 3.53 | 100.00% |

## FULL learned Router

| Policy | Rescued | /277 | Δ vs fixed | End-to-end /1366 | Calls | Rescue/100 calls | Oracle capture |
|---|---:|---:|---:|---:|---:|---:|---:|
| INDIVIDUAL_TOP1 | 137 | 49.46% | +12 | 10.03% | 1384 | 9.90 | 56.15% |
| INDIVIDUAL_TOP2 | 184 | 66.43% | +4 | 13.47% | 2768 | 6.65 | 75.41% |
| INDIVIDUAL_TOP3 | 215 | 77.62% | +5 | 15.74% | 4152 | 5.18 | 88.11% |
| COMBO_K1 | 137 | 49.46% | +12 | 10.03% | 1384 | 9.90 | 56.15% |
| COMBO_K2 | 192 | 69.31% | +12 | 14.06% | 2768 | 6.94 | 78.69% |
| COMBO_K3 | 216 | 77.98% | +6 | 15.81% | 4152 | 5.20 | 88.52% |
| COMBO_K4 | 233 | 84.12% | +0 | 17.06% | 5536 | 4.21 | 95.49% |
| COMBO_K5 | 244 | 88.09% | +0 | 17.86% | 6920 | 3.53 | 100.00% |

Race-level decisions for all 1,384 Gate alerts are stored per year in yYYYY/decisions.csv.
No adaptive K is promoted yet because candidate-inflation/L2 cost is not measured.
