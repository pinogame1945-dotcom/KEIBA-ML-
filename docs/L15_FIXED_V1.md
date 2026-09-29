# L1.5 FIX V1

## Status

**FROZEN / reusable**

This is the fixed interface between the seven L1 experts and downstream L2.

Do not edit V1 behavior in place. If the Gate, consensus ranking, Outsider routing,
candidate merge rule, or output schema changes, create `L15_FIXED_V2`.

## Frozen flow

```
Seven L1 experts
  -> Seven-King consensus order
  -> Consensus-World Gate (top 10% risk)
      -> non-alert: PASS_SEVEN_ONLY
      -> alert: FULL COMBO K2 Router
          -> select exactly 2 primary outsiders
          -> append only outsider Top6 horses absent from the Seven-King union
  -> L15_FIXED_OUTPUT_V1
  -> L2
```

The five routable outsiders are:

- 当日傾向型
- レース構造型
- 枠・コース型
- メンバー構成型
- 騎手型

## Seven-King consensus order

The reusable horse order is deterministic.

For every expert Top6, rank `r` contributes `7-r` Borda points.
Ties are broken by:

1. expert support count
2. Top1 vote count
3. best expert rank
4. mean expert rank
5. horse_id

The first two horses are emitted as `seven_anchor_horse_ids`.

Outsider horses are **COVER candidates**. They are appended after the Seven-King
consensus order and do not replace or re-rank the anchors inside L1.5.

## Gate semantics

Historical fixed scope is 2022-2025.

- 13,824 races
- 1,384 Gate alerts
- 277 historical Seven-King blind races caught by the Gate
- 2026 remains sealed

For the frozen historical ledger, absence of a race ID means:

`PASS_SEVEN_ONLY`

An alert row means:

`INTERVENE_FULL_K2`

## K2 acceptance evidence

Candidate Inflation V1:

- rescued blind races: 192
- novel horses added: 5,577
- average novel horses per alert: about 4.03
- rescued blind per 100 novel horses: about 3.44

L2 Walk-forward evidence, 2023-2025, edge > 10%:

- stake: 2,819,700 yen
- return: 3,084,300 yen
- profit: +264,600 yen
- ROI: 109.38%

This evidence is why FULL K2 is frozen instead of BASE, K3, or all-five Outsider use.

## Reusable output

Canonical historical alert ledgers are written to:

```
research-results/l15-fixed-v1/y2022.jsonl.gz
research-results/l15-fixed-v1/y2023.jsonl.gz
research-results/l15-fixed-v1/y2024.jsonl.gz
research-results/l15-fixed-v1/y2025.jsonl.gz
research-results/l15-fixed-v1/manifest.json
```

Each row is `L15_FIXED_OUTPUT_V1` and contains:

- race_id / year
- Gate alert and score
- Seven-King consensus order
- Seven-King anchors
- Seven-King Top6 union
- selected two Outsiders
- novel Outsider horse IDs
- final L1.5 candidate horse IDs

No odds, popularity, payout, ROI, or result-derived labels are allowed in the
reusable L1.5 output.

## How downstream code should consume V1

Historical L2 research should read the frozen year ledger directly instead of
rerunning Gate/Router research.

For a Gate alert:

- use `seven_anchor_horse_ids` as the Seven-King anchor signal;
- use `candidate_horse_ids` as the candidate pool;
- keep `novel_horse_ids` available so L2 can distinguish Outsider COVER horses.

For a non-alert race:

- use Seven-King output only;
- do not invoke Outsiders.

The helper `research/materialize_l15_fixed_v1.py` exists for exact replay and
contract verification. It may emit either every race or Gate-alert rows only.

## Frozen sources

The exact source commits/runs are pinned in:

`contracts/l15-fixed-v1.json`

The contract, frozen ledger hashes, and immutability rule are the source of truth.
