# L2 CORE V1 — Redefinition

Status: DESIGN FROZEN  
Date: 2026-10-01  
Upstream: `L17_SEVEN_KING_FULLFIELD_OUTPUT_V1`

## 1. Why L2 is being redefined

The previous L2 research mixed too many jobs into one layer:

- select a race type
- choose a handcrafted ticket template
- infer anchor / mate roles
- search for profitable rules
- evaluate prices
- implicitly control how many tickets are bought

That produced many useful experiments, but it did not produce one stable L2 architecture.

The L1.7 full-field output changes the correct starting point. L2 no longer needs to begin from a truncated candidate pool or from fixed templates. It can begin from every runner in the race.

The old L1.7 A/B test already showed the important failure mode: simply giving full-field information to the old Direct Template architecture did not solve L2. On the 2025 holdout, `L17_FULLFIELD` increased hit coverage but also increased ticket count and reduced ROI. The conclusion is not that full-field information is useless. The conclusion is that the old template architecture is the wrong receiver for it.

## 2. New responsibility boundary

The project layers are now defined as:

`L1.7 = horse diagnosis`  
`L2 = ticket probability and price evaluation`  
`L3 = bankroll and portfolio allocation`

L2 does not decide stake size.

L2 must not use odds to decide horse ability. Market price is joined only after the probability model has produced its estimate.

## 3. Input

The canonical L2 input is the full L1.7 field for every race and every horse:

- consensus rank
- seven individual expert ranks / probabilities
- mean rank
- rank dispersion
- best / worst rank
- Top1 votes
- Top3 support
- Top6 support
- mean probability
- probability dispersion
- field size

Optional pre-race race context may be joined, but only data available before the race is allowed.

No candidate truncation is allowed before the probability model.

## 4. The one shared probability engine

L2 CORE V1 uses one shared finish-order model.

For each horse, the model learns a race-relative latent strength from L1.7 information. The full race distribution is then converted into ordered finish probabilities using a Plackett-Luce-style construction.

This gives one coherent source for every supported ticket type:

- WIN: probability that horse A finishes first
- EXACTA: probability that A finishes first and B second
- QUINELLA: EXACTA(A,B) + EXACTA(B,A)
- TRIFECTA: probability that A first, B second, C third
- TRIO: sum of all six first-second-third permutations for A/B/C

This is the central redesign.

There is no separate "horse-racing law" for quinella, exacta, trio, or trifecta in the core. There is one race world, and each ticket is merely a different question asked of that same world.

## 5. Why this is preferable to ticket templates

For an 18-horse race the valid ticket universe is large, but the probability model does not need to train one independent model row for every ticket.

The expensive part is reduced to one strength estimate per horse. Ticket probabilities are reconstructed from those strengths.

Therefore:

- all runners can remain alive
- no Top6 truncation is needed
- no A1/A2 hand assignment is required
- no 14-template menu is required
- horse-order probability remains internally consistent across ticket types

This also avoids turning L2 into five unrelated research projects.

## 6. Market stage

After ticket probability has been produced, market odds may be joined.

For each ticket:

`fair_odds = 1 / p_hit`

`edge = p_hit * market_odds - 1`

`expected_return_index = p_hit * market_odds`

Historical final odds may be used as an evaluation proxy, but they must not be described as a live pre-race execution price unless a timestamped historical odds snapshot proves that.

Odds, popularity, payouts, ROI, and result fields are forbidden inputs to the probability model.

## 7. What L2 outputs

L2 outputs a ranked ticket universe, not a staking plan.

For every race, downstream code must be able to reconstruct or inspect:

- every horse's modeled strength
- finish-order probabilities
- ticket hit probability
- ticket type
- ticket members / order
- fair odds
- market odds when available
- edge / expected-return index
- probability confidence / calibration metadata

Every race remains in the output. There is no "race disappeared because no handcrafted rule fired" behavior.

L2 may mark a ticket as unattractive. It may not erase the race from history.

## 8. What moves to L3

These are L3 responsibilities:

- how many yen to bet
- total race budget
- daily bankroll exposure
- simultaneous tickets that overlap heavily
- whether several positive-edge tickets should all be purchased
- drawdown-aware sizing
- portfolio optimization across bet types and races

Thus a ticket can be structurally attractive in L2 and still receive 0 yen in L3.

That distinction is intentional.

## 9. Legacy L2 research

The following are not deleted:

- Direct Template router
- normal-router V1-V6
- role-value labs
- Bet Kings arena
- K2 provisional rule
- loss anatomy research
- odds / comma-bug revalidations
- trifecta / trio / exacta / quinella branch experiments

They remain evidence.

They are no longer the definition of L2.

A legacy rule can return later only by surviving a clean comparison against L2 CORE V1 without leaking 2026 and without being promoted solely because of one profitable year.

## 10. Validation order

Implementation should proceed in this order:

1. Freeze the L1.7 input reader and schema checks.
2. Build the shared horse-strength / finish-order model.
3. Validate probability calibration before looking at profit.
4. Build ticket-probability reconstruction for the five supported bet types.
5. Join market prices after prediction.
6. Run walk-forward 2022-2025.
7. Compare against the legacy L2 research branches.
8. Only after L2 probability quality is stable, hand candidates to L3.

2026 remains sealed.

## 11. Cost rule

The first implementation must run on the standard free GitHub Actions CPU runner.

No paid runner, GPU, paid cloud compute, or paid persistent Artifact / Cache / Dataset storage is part of L2 CORE V1.

If implementation reaches a point where a paid resource appears necessary, execution stops before that boundary.
