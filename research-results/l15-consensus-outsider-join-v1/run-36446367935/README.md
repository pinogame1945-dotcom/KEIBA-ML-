# L1.5 Consensus-World × Outsider Join V1

- Gate alerts (top10%): 1,384
- True seven-king blind spots inside alerts: 277
- Gate precision: 20.01%
- Gate recall over all 2022–2025 seven-king blind spots: 20.28%

## Individual outsider rescue inside Gate-caught blind spots

| Outsider | Rescue | /277 | Exclusive | End-to-end /1366 |
|---|---:|---:|---:|---:|
| 当日傾向型 | 118 | 42.60% | 15 | 8.64% |
| レース構造型 | 117 | 42.24% | 17 | 8.57% |
| 枠・コース型 | 106 | 38.27% | 13 | 7.76% |
| メンバー構成型 | 101 | 36.46% | 9 | 7.39% |
| 騎手型 | 62 | 22.38% | 8 | 4.54% |
| キメラ型 | 58 | 20.94% | 0 | 4.25% |
| 距離詳細型 | 47 | 16.97% | 2 | 3.44% |
| 位置取り型 | 43 | 15.52% | 0 | 3.15% |
| ラップ型 | 41 | 14.80% | 0 | 3.00% |
| レース条件型 | 40 | 14.44% | 0 | 2.93% |
| Elo型 | 36 | 13.00% | 0 | 2.64% |

## Group oracle ceilings

| Group | Rescue | /277 | End-to-end /1366 | Alerts per rescue |
|---|---:|---:|---:|---:|
| primary3_union | 216 | 77.98% | 15.81% | 6.41 |
| primary3_plus_shadow2 | 244 | 88.09% | 17.86% | 5.67 |
| nonhorse_union | 244 | 88.09% | 17.86% | 5.67 |
| horse_union | 76 | 27.44% | 5.56% | 18.21 |
| all11_union | 250 | 90.25% | 18.30% | 5.54 |

## Best hindsight combinations

| k | Outsiders | Rescue | /277 | End-to-end /1366 |
|---:|---|---:|---:|---:|
| 1 | 当日傾向型 | 118 | 42.60% | 8.64% |
| 2 | 枠・コース型 + レース構造型 | 180 | 64.98% | 13.18% |
| 3 | 当日傾向型 + 枠・コース型 + レース構造型 | 216 | 77.98% | 15.81% |
| 4 | 当日傾向型 + メンバー構成型 + 枠・コース型 + レース構造型 | 235 | 84.84% | 17.20% |
| 5 | 当日傾向型 + メンバー構成型 + 枠・コース型 + 騎手型 + レース構造型 | 244 | 88.09% | 17.86% |

> WARNING: outsider union/combinations are hindsight/oracle capacity, not deployable routing performance.
> The next deployable problem is choosing the outsider before the result is known.
