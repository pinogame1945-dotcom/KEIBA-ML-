# L1.75 MARKET CONTEXT V1

## Purpose

L1.75 is the fixed bridge between pure L1.7 horse evaluation and L2 bet construction.

It is **not a new ranking model**.

- L1.7 produces the Seven-King full-field horse ranking without odds/popularity/payout.
- Existing safe Outsider signals remain independent context and never reorder Seven-King rank.
- L1.75 adds the current WIN market view **after** L1.7 is frozen for that race.
- L2 receives the combined context and decides how, or whether, to use the disagreement in ticket construction.

## Immutable ranking rule

`king_rank` is authoritative.

`ranking_policy = SEVEN_KING_UNCHANGED`

L1.75 MUST NOT create or export a market-adjusted L1 rank.

There is no `reranked_rank`, `blended_rank`, or equivalent field.

## Inputs from L1.7

Per horse:

- `horse_id`
- `king_rank`
- Seven-King diagnostics:
  - `king_borda_score`
  - `king_normalized_borda`
  - `king_mean_rank`
  - `king_rank_std`
  - `king_best_rank`
  - `king_worst_rank`
  - `king_top1_votes`
  - `king_probability_mean`
  - `king_probability_std`
- Safe Outsider context:
  - `outsider_score`
  - `outsider_support_count`
  - `outsider_top1_votes`
  - `outsider_top2_votes`
  - `outsider_top3_votes`
- Walk-forward podium context for King ranks 1-10:
  - `p3_calibrated`
  - `p3_rank_baseline`
  - `p3_delta`
  - `calibration_status`

The existing L1.7 contract remains `L17_TO_L2_SIGNAL_V1`.

## Market context added only in L1.75

L1.75 may read the WIN market after L1.7 is complete.

Per horse:

- `market_win_odds`
- `market_rank`

Historical research may use final WIN odds.
Operational use should pass the latest available pre-race WIN odds snapshot and preserve its timestamp outside the horse score itself.

Equal odds use competition ranking:
two equal odds share the same market rank.

## Derived disagreement fields

`signed_rank_gap = market_rank - king_rank`

Interpretation:

- positive: L1.7 rates the horse **higher** than the market
- negative: L1.7 rates the horse **lower** than the market
- zero: same rank

`abs_rank_gap = abs(signed_rank_gap)`

`dissent_direction`:

- `UP` when `signed_rank_gap >= 2`
- `DOWN` when `signed_rank_gap <= -2`
- `NEAR` otherwise

This threshold only describes the size/direction of disagreement.
It MUST NOT rerank horses or directly create a bet.

## Outsider agreement with the disagreement

For horses with valid `p3_delta`:

`outsider_alignment_score = sign(signed_rank_gap) * p3_delta`

`outsider_alignment`:

- `AGREE` when alignment score > 0
- `OPPOSE` when alignment score < 0
- `NEUTRAL` when exactly 0
- `NA` when podium calibration is unavailable

Interpretation example:

- Market rank 8, King rank 4 => Seven-King says UP.
- Positive p3_delta => Outsider context also supports the horse more than the rank baseline => AGREE.
- Negative p3_delta => Outsider context pushes against the Seven-King UP opinion => OPPOSE.

## What L1.75 does NOT do

L1.75 does not:

- change `king_rank`
- select tickets
- set stakes
- decide BUY / SKIP
- optimize ROI
- use race result / finish position
- use payout
- use future information
- create STRONG / NORMAL / CONTRA tags

The STRONG dissent experiments were not stable enough for a fixed production tag.

## L2 handoff

L2 may use the following as features:

- Seven-King rank and diagnostics
- Outsider raw support
- `p3_delta`
- WIN market odds/rank
- `signed_rank_gap`
- `abs_rank_gap`
- `dissent_direction`
- `outsider_alignment_score`
- `outsider_alignment`

L2 is responsible for learning whether these signals are useful for:

- horse roles
- pair/triple construction
- bet-type choice
- hit probability
- expected value / edge
- ticket selection

Bet-type odds (馬連・馬単・三連複・三連単) belong to L2, not L1.75.

## Leakage / research rules

- Core L1.7 remains market-free.
- Outsider source remains the safe/tie-safe persisted signal path.
- Historical labels may be joined only after L1.75 features are frozen for evaluation/training.
- 2026 remains sealed.
- No outcome field is exported as an L1.75 feature.

## Current evidence

Seven-King disagreement with the market is directionally useful across historical unknown-year evaluation.

Outsider agreement sometimes enriches that disagreement, but not in every year/direction.

Therefore Outsider context is retained as a feature, not used as a hard reranking command.

## Status

`L175_MARKET_CONTEXT_V1`

Status: **FIXED BRIDGE CONTRACT**

This contract is ready to be consumed by L2.
