# Phase 3 — 2026-10-10 JRA full-day 24-race target

## Verified scope
Saturday 2026-10-10: 東京 4回東京3日 1–12R, 京都 4回京都3日 1–12R.
JRA programme: https://www.jra.go.jp/keiba/calendar2026/2026/10/1010.html

- Tokyo race IDs: 202605040301 through 202605040312
- Kyoto race IDs: 202608040301 through 202608040312
- Run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38885544122 (SUCCESS)
- Source: pinned KEIBA-BACKFILL 03950cfe132abeca15ef17dbc7c6885ceb1591b7
- Original builder history from 2022-01-01, cutoff at 2026-10-10.
- Yearly CORE4 649-column sorted SHA256:
  eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58.
- The exact BACKFILL race set (24), the whole target-day derived snapshot
  (24), entry-horse IDs (315 starters), and the frozen CORE4 column schema
  were checked successfully.
- 2026 snapshot rows scanned through Oct10: 36,917.
- All 24 target panels were written to ephemeral runner temp, without
  target outcome/results, historical postrace odds or cached target scores.
- 5 target races have original 595-column union and exactly the same
  54 auto relative/pair gaps verified on the previous 207 races; the
  original 649-column schema was aligned using null/NaN values ONLY:
  202605040304, 202605040305, 202605040306,
  202608040305, 202608040306.
- Kyoto 4R 202608040304 is a jump race. It passed 649-feature schema
  and entry-identity QA, but actual frozen original trained-model
  eligibility/inference parity for obstacle races has NOT been shown.

## Race counts by race number
| R | Tokyo (05) horses | Kyoto (08) horses |
|---|---:|---:|
| 1 | 16 | 14 |
| 2 | 13 | 9 |
| 3 | 16 | 8 |
| 4 | 16 | 14 |
| 5 | 12 | 14 |
| 6 | 14 | 11 |
| 7 | 13 | 16 |
| 8 | 11 | 6 |
| 9 | 9 | 16 |
| 10 | 16 | 15 |
| 11 | 11 | 13 |
| 12 | 16 | 16 |

## Honest status
- PASS: exact 24 JRA race IDs, source roster integrity and 649 feature names
  / generated unscored input with legacy NaN alignment.
- PASS from earlier Phase 2.5 run 38882823365: frozen 19+22=41 additional
  numeric feature equality on **one different race, 2026-10-12**, not yet
  batch-proven for these 24.
- NOT RUN: actual original L1 7王/Outsider and CORE4 inference for these 24;
  L1.75 scores, original L2.1 purchase rule and full authentic L3 V3 p3
  production for these 24.
- NOT VERIFIED: original 2026 cold-source 690 matrix parity with 2024/25
  training frames; as-of historical capture timestamps or original live
  prebet odds. 2026 outcomes remain sealed for evaluation.
- NO CHANGE: frozen model, 7 kings / Outsider, L1.75, L2.1 and L3 V3,
  original BACKFILL data structures and repository production branches.
- The 24 feature JSON outputs were ephemeral: NOT published to phone,
  GitHub data branch or a paid artifact storage facility.

The source-only input preparation is not end-to-end AI prediction completion.
