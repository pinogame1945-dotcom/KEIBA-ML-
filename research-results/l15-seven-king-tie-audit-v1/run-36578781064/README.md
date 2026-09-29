# L1.5 Seven-King Tie Audit V1

- Races: 17280
- Years: 2021-2025
- 2026: sealed
- No retraining; persisted horse-level scores only
- Safe tie-break: raw_margin_logit desc, horse_id asc

## King-level Top6 impact

| King | Any tie | Top6 boundary tie | Top6 set changed | Old capture | Safe capture | Delta |
|---|---:|---:|---:|---:|---:|---:|
| core4 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 14226 | 14226 | +0 |
| pedlegacy | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 14260 | 14260 | +0 |
| condition | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 14267 | 14267 | +0 |
| full | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 14230 | 14230 | +0 |
| light | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 14131 | 14131 | +0 |
| pedv1 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 14264 | 14264 | +0 |
| condrc | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 14272 | 14272 | +0 |

## Seven-king union impact

- Top6 union changed races: 0 (0.00%)
- Old blind spots: 1735
- Safe blind spots: 1735
- Blind-spot delta: +0
- Old blind -> safe hit: 0
- Old hit -> safe blind: 0
- Consensus Top1 changed: 0 (0.00%)
- Consensus anchor Top2 changed: 0 (0.00%)

This audit does not mutate fixed/l15-v1.
