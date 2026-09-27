# L1 Feature Set Contract V1

## Purpose

L1 research no longer depends on a cumulative stage order. The canonical selector is an explicit set of independent feature families plus a prediction phase and purpose-specific history windows.

Machine-readable source of truth:

- `contracts/l1-feature-set-contract-v1.json`

## Prediction phases

### EARLY

EARLY is the Friday / pre-race-day view after the entry and field are known.

Allowed availability classes:

- `ALWAYS`
- `ENTRY_PUBLISHED`
- `FIELD_FIXED`

The EARLY gate removes or rejects current-race facts that are only known later, including:

- weather
- track condition
- current body weight
- current body-weight difference
- AUTO interactions that depend on current body weight
- AUTO condition features that depend on the current going

Prior-race body weight remains legal historical information.

### FINAL

FINAL is the day-of-race view after race-day conditions and body weight are available.

In addition to EARLY classes, FINAL may use:

- `RACE_DAY`
- `BODY_WEIGHT_ANNOUNCED`

Both phases still reject every L1-forbidden target, market and post-race fact.

## Independent Feature Sets

Current sets:

- `BASE`
- `OPPONENT`
- `NETWORK`
- `LAP`
- `STYLE`
- `DISTANCE`
- `BACKFILL`
- `AUTO`
- `PEDIGREE`
- `ACTOR`
- `TIME_PACE`

The order does not imply cumulative inclusion. For example, `BASE,DISTANCE` means base plus distance only; STYLE, LAP, OPPONENT and AUTO are not pulled in just because older stages did so.

Old `--stage` values remain only as a migration adapter. They are translated to the historical cumulative combination and should not be used for new experiments.

## History windows

The old single `history_limit` is deprecated. Canonical defaults are:

- recent/form: 5 starts
- style/last3f trend: 10 starts
- suitability: 20 starts
- opponent: 10 starts
- AUTO rolling: 20 starts
- actor recent form: 30 starts
- time/pace horse history: 10 starts
- career starts: all prior history
- Elo: all prior history

The bounded windows are explicit experiment parameters. Their actual values are persisted with the model and OOF output.

## Small-sample policy

PEDIGREE and ACTOR keep raw observation counts. Small samples are not dropped merely because they are small.

The shared contract is `contracts/l1-small-sample-contract-v1.json`:

- raw counts/rates/means remain visible;
- rates and means may be shrunk toward a broader point-in-time prior;
- sparse condition buckets fall back to entity-wide stats, then global stats;
- rate/mean shrinkage strength and the minimum condition sample are research parameters;
- the exact policy is persisted with model/meta/schema/OOF.

The defaults are engineering baselines, not production-selected values.

## BACKFILL vs AUTO

`BACKFILL` is the data foundation: collected and normalized historical/current race facts whose source of truth is KEIBA-BACKFILL. KEIBA-ML does not scrape source websites.

`AUTO` is derived ML material built from approved point-in-time facts. AUTO is safe for L1 only when every source fact existed before the target race and no target result, future result, odds, popularity or payout can enter the calculation.

Whether a value is "derived" is not the safety test. Point-in-time availability is.

## Reproducibility

Every trained fold records:

- dataset and feature schema versions
- prediction phase
- exact Feature Set list
- exact history-window configuration
- exact ordered model feature list
- categorical levels
- Feature Catalog SHA-256
- Feature Set Contract SHA-256
- Small Sample Contract SHA-256
- exact small-sample policy
- training-config SHA-256
- model SHA-256
- BACKFILL source SHA when supplied
- ML source SHA when supplied

OOF rows also carry phase, Feature Sets, history windows, exact feature list and catalog/contract hashes.

## Production boundary

Feature Set availability does not mean production adoption. STYLE/AUTO/BACKFILL/PEDIGREE/ACTOR/OPPONENT/TIME_PACE remain research candidates until walk-forward selection is complete.

2026 remains locked for final confirmation and must not be used to choose Feature Sets or history windows.

L1 ability never uses odds.
