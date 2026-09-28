# L1.5 Outsider Rescue Router V1 — Experiment Specification

## 0. Status

- Experiment ID: `L15-OUTSIDER-RESCUE-ROUTER-001`
- Purpose: research only
- Production promotion: prohibited by this experiment alone
- Ability odds: **NO**
- Locked year: **2026**
- Compute: GitHub Actions standard CPU runner only
- Paid runner / GPU / paid storage: **prohibited without explicit approval**

Source-of-truth inputs:

- L1.5 seven-king branch baseline: `research/l15-role-router-v2-7kings-20260928`
- Baseline SHA: `4e20ca9381a8cc454dcc71eb6990d6796b373df2`
- Outsider source branch: `research/l1-outsider-arena-v1-20260928`
- Outsider result run: `36419042735`
- Outsider ledger commit: `d63a7c4`
- Router feature contract: `ROUTER_FEATURE_SNAPSHOT_V1`

## 1. Why this experiment exists

The seven kings have 1,735 Top6 blind-spot races across 2021–2025.

A blind spot means:

- the actual winner is outside Top6 for **all seven kings**.

The eleven outsider families have a hindsight union rescue ceiling of:

- 1,575 / 1,735 = 90.78%
- remaining both-miss = 160

This is **oracle headroom**, not deployable performance.

The research problem is therefore no longer:

> Which outsider is strongest overall?

It is:

> Before the result is known, can we detect a likely seven-king blind spot and route that race to the outsider family most likely to rescue it?

## 2. Non-goals

V1 does NOT:

- replace the seven kings;
- choose a production betting strategy;
- use odds, popularity, payout, or any post-race field;
- read 2026;
- optimize L2/L3 ROI directly;
- automatically promote any outsider;
- train on 2021–2025 and score those same rows in-sample;
- select an outsider using the race result.

V1 is additive rescue research.

## 3. Candidate tiers

### 3.1 Primary router pool — V1 live candidates

Only these three families participate in the first real routing contest:

1. `outsider_gatecourse` — 枠・コース型
   - total seven-king Top6 rescues: 751
   - rescue rate: 43.29%
   - exclusive rescue vs all other ten outsiders: 101

2. `outsider_raceshape` — レース構造型
   - total rescues: 740
   - rescue rate: 42.65%
   - exclusive rescue: 82

3. `outsider_daytrend` — 当日傾向型
   - total rescues: 764
   - rescue rate: 44.03%
   - exclusive rescue: 71

These are deliberately chosen because they are weak-to-middling standalone models but strong on seven-king blind spots.

### 3.2 Shadow candidates

Scored for diagnosis but not allowed to affect the V1 decision:

- `outsider_field` — メンバー構成型
- `outsider_jockey` — 騎手型

They can be promoted to V2 only if the V1 report shows stable incremental headroom.

### 3.3 Bench

Not used for V1 routing decisions:

- `outsider_chimera`
- `outsider_style`
- `outsider_lap`
- `outsider_network`
- `outsider_distance`
- `outsider_backfill`

Reason: their exclusive rescue contribution is currently too small relative to complexity. They remain available as later challengers.

## 4. Architecture

```
seven kings
    |
    v
Router Feature Snapshot V1
    |
    v
[Stage A] Blind-Spot Gate
    | P(seven-kings Top6 all miss)
    |
    +-- SAFE ----------> normal L1.5 path
    |
    +-- DANGER
           |
           v
[Stage B] Outsider Router
    |-- P(gatecourse rescues)
    |-- P(raceshape rescues)
    |-- P(daytrend rescues)
    |
    v
selected outsider(s)
    |
    v
additive rescue candidates + metadata
    |
    v
L1.5 FINAL / L2
```

Stage A and Stage B are evaluated separately and end-to-end.

## 5. Labels

### 5.1 Stage A — Blind-Spot Gate label

For each race:

`seven_blind_top6 = 1`

iff the actual winner is outside Top6 for all seven kings.

Otherwise:

`seven_blind_top6 = 0`

The race result is used **only to construct the historical label**.

### 5.2 Stage B — Multi-label outsider rescue targets

Train Stage B only on historical races where:

`seven_blind_top6 = 1`

For every eligible race create one binary target per outsider:

- `gatecourse_rescue`
- `raceshape_rescue`
- `daytrend_rescue`
- shadow: `field_rescue`
- shadow: `jockey_rescue`

A target is 1 iff that outsider contains the actual winner in its Top6.

Multiple targets may be 1 for the same race.

This is a multi-label problem implemented as independent one-vs-rest binary classifiers.

## 6. Feature policy

### 6.1 Mandatory base

Use `ROUTER_FEATURE_SNAPSHOT_V1` as the canonical race-level base.

Allowed groups include:

- race context
  - venue
  - surface / discipline
  - direction
  - distance
  - class
  - field size
  - weather
  - track condition
- data coverage
- seven-king pre-race score summaries
- seven-king Top1/Top3/Top6 agreement
- pairwise Jaccard
- rank disagreement
- probability dispersion
- entropy
- top1 vote concentration
- king-level probability margins
- feature-family coverage

### 6.2 Rescue-context extensions

Add only pre-race aggregate features that can be rebuilt before the target race:

#### Gate/course context

- gate distribution
- normalized gate fraction
- inner / middle / outer occupancy
- course layout
- field size interaction
- venue × surface × distance context

#### Race-shape context

- field mean/std of recent style signals
- front-heavy share
- early-position mean/std
- dispersion of running-style profiles

#### Same-day trend context

Strict rule:

- same venue;
- earlier race number only;
- current race and later races forbidden.

Allowed examples:

- prior-race winner gate tendency
- same-day inner/outer winner rate
- prior winners' running-style summaries
- number of prior usable races

The existing nonhorse builder's rule
`same venue, prior race number only`
is the required leakage boundary.

### 6.3 Forbidden

Hard fail if any training/prediction feature contains:

- winner identity
- finish position
- target
- payout
- odds
- popularity
- any result from the target race
- any same-day future race
- 2026 data

## 7. Walk-forward protocol

No random split.

Required folds:

| Fold | Train years | Test year |
|---|---|---|
| F1 | 2021 | 2022 |
| F2 | 2021–2022 | 2023 |
| F3 | 2021–2023 | 2024 |
| F4 | 2021–2024 | 2025 |

Every threshold, model hyperparameter choice, candidate count rule, and fallback policy must be derived from train years only.

2026 remains unread.

## 8. Models

### 8.1 Stage A

Binary LightGBM classifier:

`P(seven_blind_top6)`

Conservative defaults:

- CPU only
- `n_jobs <= 2`
- small tree size
- explicit regularization
- deterministic seed 1945
- no hyperparameter sweep in V1

A simple logistic / heuristic baseline must also be reported so LightGBM is not rewarded merely for complexity.

### 8.2 Stage B

One binary LightGBM per candidate:

- `P(gatecourse_rescue | blind)`
- `P(raceshape_rescue | blind)`
- `P(daytrend_rescue | blind)`

Shadow models:

- `P(field_rescue | blind)`
- `P(jockey_rescue | blind)`

No softmax.

Outsiders are not mutually exclusive.

## 9. Decision policies to compare

All policies are fixed using training years only.

### P0 — Seven kings only

No rescue intervention.

This is the true baseline.

### P1 — Always best fixed outsider

Always call the single primary outsider with the highest historical train rescue rate.

This detects whether the router is doing anything smarter than a fixed choice.

### P2 — Always primary-3 oracle-cost baseline

Always run all three primary outsiders.

This gives an upper practical-cost reference, but not hindsight selection.

### P3 — Gate + Top1 router

If Stage A opens:

- call only the Stage B outsider with highest predicted rescue probability.

### P4 — Gate + Top2 router

If Stage A opens:

- call the two highest predicted outsider probabilities.

Top2 exists because rescue labels overlap and a second specialist may cover a different blind-spot regime.

## 10. Gate threshold learning

Do not hard-code a final probability threshold from 2021–2025 aggregate results.

For each fold, choose the Stage A threshold from training years only.

Primary utility:

`utility = rescued_blind_spots - lambda * unnecessary_interventions`

where an unnecessary intervention is a gate-open race that was not a seven-king blind spot.

V1 must report a small fixed grid of `lambda` values rather than silently selecting one global optimum.

At minimum:

- recall-oriented
- balanced
- precision-oriented

The final report must show the tradeoff curve.

## 11. Additive rescue semantics

V1 does NOT replace king candidates.

If an outsider is called, it may contribute only novel horses that are not already represented by the seven kings.

Required metadata per race:

- gate probability
- gate decision
- selected outsider(s)
- outsider rescue probabilities
- selected outsider Top6
- novel horse ids
- novel horse count
- whether the actual winner was rescued — evaluation output only

The evaluation report must measure candidate inflation.

No production L1.5 merge is allowed until downstream L2/L3 evaluates whether the extra candidates improve EV/ROI rather than merely Top6 coverage.

## 12. Metrics

### 12.1 Stage A

Per fold and aggregate:

- blind-spot prevalence
- ROC-AUC
- PR-AUC
- gate precision
- gate recall
- false-positive rate
- false-negative count
- intervention rate
- calibration / probability bins

Accuracy is secondary and must not be the headline metric.

### 12.2 Stage B

Conditional on true blind spots:

- rescue rate for each routed outsider
- Top1-router rescue rate
- Top2-router rescue rate
- primary-3 fixed union rescue rate
- shadow incremental rescue
- exclusive rescue retained
- pairwise overlap / Jaccard
- oracle gap

### 12.3 End-to-end

The critical metric:

`end_to_end_rescue = gate_open AND selected_outsider_captures_winner_top6`

Report:

- rescued blind spots / all seven-king blind spots
- rescued races / all races
- remaining blind spots
- average outsiders called per race
- average novel horses added per race
- interventions on normal races
- rescue per 100 interventions

### 12.4 Stability

Every headline metric must be shown separately for 2022, 2023, 2024, 2025.

An aggregate gain that is driven by one year is not considered stable.

## 13. Baselines and hindsight ceilings

The report must retain these current reference facts:

- seven-king Top6 blind spots, 2021–2025: 1,735
- all eleven outsider hindsight union rescues: 1,575
- all-eleven hindsight rescue ceiling: 90.78%
- unresolved even by all eleven: 160

The 90.78% figure is explicitly labelled **ORACLE / HINDSIGHT ONLY**.

It must never be shown as model performance.

## 14. V1 success criteria

V1 is research-positive only if all are true:

1. Stage A beats blind-spot prevalence and simple consensus heuristics on untouched years.
2. P3 or P4 beats P1 on end-to-end blind-spot rescue in aggregate.
3. The gain is not isolated to one test year.
4. The router recovers a meaningful portion of the primary-3 fixed-union headroom.
5. Candidate inflation and normal-race intervention are measured, not hidden.
6. No forbidden feature / 2026 / odds leakage is detected.

Research-positive does **not** mean production adoption.

Production promotion requires L2/L3 EV and realized ROI validation.

## 15. Required output contract

Small permanent outputs may be committed to the research branch.

### Summary JSON

`L15_OUTSIDER_RESCUE_ROUTER_V1_SUMMARY`

Must contain:

- source SHAs / run IDs
- feature contract
- folds
- policy metrics P0–P4
- Stage A metrics
- Stage B metrics
- end-to-end metrics
- candidate-inflation metrics
- shadow diagnostics
- feature importance
- leakage audit result
- `odds_used=false`
- `locked_years=[2026]`

### Decisions JSONL

`L15_OUTSIDER_RESCUE_ROUTER_V1_DECISION`

One row per test race:

- year
- race_id
- blind probability
- gate policy
- gate open/closed
- outsider probabilities
- selected outsiders
- novel candidate ids
- evaluation labels

Large temporary training matrices/models are not to be persisted merely for convenience.

## 16. Compute / storage policy

- Standard GitHub-hosted CPU runner only.
- No GPU.
- No paid runner.
- No paid cloud instance.
- No new paid Artifact/Cache/Dataset/Model storage.
- Reuse already-existing snapshots/ledgers.
- Prefer one restored input per runner and reuse it locally.
- Do not repeatedly download the same Kaggle source per candidate.
- Do not create a new large Kaggle dataset for V1 unless its free status is explicitly confirmed.

If free status is uncertain, stop before storage.

## 17. Implementation order

1. Build `OUTSIDER_RESCUE_ROUTER_FEATURES_V1` from existing safe snapshots.
2. Build blind-spot + multi-label rescue labels from run `36419042735` ledgers.
3. Add leakage contract test.
4. Implement Stage A walk-forward.
5. Implement Stage B one-vs-rest walk-forward.
6. Evaluate P0–P4.
7. Save small summary + decision ledger.
8. Only after results: decide whether to add shadow candidates.
9. Only after that: integrate additive rescue metadata into L1.5 FINAL.
10. Final adoption remains behind L2/L3 EV/ROI gate.

## 18. Core rule

Do not ask the outsider to become a king.

The kings remain the normal path.

The router's job is to recognize the places where the throne room has gone blind, then call the specialist that sees a different world.
