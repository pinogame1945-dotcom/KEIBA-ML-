# NEWCOMER-001 — 新馬SPECIALIST v0

- Run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/36335844882
- Snapshot: e75cd10b7837f676
- BACKFILL SHA: af9d0dcf3295e8276ca39fcbe104d7e9ac4c1cdc
- 学習: 2023-2024
- 検証: 2025 NEWCOMER 304R
- L1オッズ: NO
- 本線Feature Arenaとは別台帳

## 結果

| Candidate | Train | Top1 | Top3 | Top6 | 勝ち馬平均順位 | Peak MiB |
|---|---|---:|---:|---:|---:|---:|
| general_no_auto_full | ALL | 25.33% | 54.93% | 78.29% | 4.102 | 5669.2 |
| general_no_auto_jockey | ALL | 23.03% | 52.63% | 78.62% | 4.319 | 3349.6 |
| newcomer_same_jockey | NEWCOMER | 22.70% | 50.00% | 75.66% | 4.543 | 3427.9 |
| newcomer_actor_pedigree | NEWCOMER | 22.70% | 52.96% | 77.30% | 4.391 | 9028.6 |
| newcomer_actor | NEWCOMER | 21.71% | 54.61% | 75.66% | 4.405 | 4434.4 |
| newcomer_pedigree | NEWCOMER | 15.46% | 38.49% | 66.45% | 5.368 | 5363.8 |

## 判定メモ

- 全レース学習の general_no_auto_full が Top1 / Top3 / 勝ち馬平均順位で最高。
- 同一164特徴量比較では general_no_auto_jockey が newcomer_same_jockey を Top1/Top3/Top6/平均順位すべてで上回った。
- 新馬だけで学習するだけでは優位性は確認できなかった。
- PEDIGREE単独は弱い。一方 ACTOR + PEDIGREE は ACTOR単独より Top1/Top6 が改善し、相互補完の余地を残した。
- 次段階は「新馬戦という条件に対する父/母父の過去成績」を直接特徴化する NEWCOMER-specific pedigree stats。
