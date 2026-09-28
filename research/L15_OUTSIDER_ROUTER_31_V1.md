# L1.5 Outsider Router 31 V1

## Goal

After Consensus-World Gate flags the top 10% risky races, choose which of the five useful non-horse outsiders to invoke **before the result is known**.

Candidates:

- 当日傾向型
- レース構造型
- 枠・コース型
- メンバー構成型
- 騎手型

The five candidates have 31 non-empty combinations.

## Training / evaluation

Router training uses **all historical seven-king Top6 blind spots from prior years**.

- train 2021 -> test Gate-selected 2022
- train 2021-2022 -> test Gate-selected 2023
- train 2021-2023 -> test Gate-selected 2024
- train 2021-2024 -> test Gate-selected 2025

The test population is the OOS Consensus-World Gate top-10% alerts from
`research-results/l15-consensus-world-race-ids-v1/run-36442605271/selected-top10-all.csv`.

Only the true blind races inside those alerts are used to score rescue success.
False Gate alerts still receive Router decisions and count toward invocation cost.

## Features

Two pre-race feature contracts are compared.

### COMPACT

- Consensus-World disagreement features
- pre-race race context
- data coverage

### FULL

- full `ROUTER_FEATURE_SNAPSHOT_V1` feature set

No odds, popularity, payout, finish, winner, or 2026 labels/features.

## Policies

Train-only fixed baselines:

- FIXED_K1 .. FIXED_K5: best global combination of size K on prior-year blind spots

Learned individual Router:

- INDIVIDUAL_TOP1
- INDIVIDUAL_TOP2
- INDIVIDUAL_TOP3

Five independent candidate rescue models are fit; choose the K highest predicted candidates.

Direct combination Router:

- COMBO_K1 .. COMBO_K5

For each of the 31 combinations, train a binary model whose target is:

> at least one outsider in this combination rescues the winner at Top6

At inference, for each K choose the size-K combination with highest predicted rescue score.

No adaptive call-count utility is promoted in V1 because candidate-inflation cost has not yet been measured.

## Metrics

For each policy:

- rescued Gate-caught blind spots / 277
- end-to-end rescue / all 1,366 blind spots
- selected outsider calls over all 1,384 Gate alerts
- rescue per 100 outsider calls
- yearly stability
- delta vs corresponding train-only FIXED_K baseline

Reference hindsight ceilings:

- Gate caught 277 / 1,366 blind spots
- five-outsider hindsight union: 244 / 277
- all-11 hindsight union: 250 / 277

These are oracle ceilings only.

## Output

Persist:

- summary JSON + README
- race-level Router decisions for all 1,384 Gate alerts
- per-year metrics
- selected outsider names
- predicted Router score
- evaluation-only actual rescuers / hit flag

No models, matrices, Artifact, Cache, or new Kaggle storage are persisted.
