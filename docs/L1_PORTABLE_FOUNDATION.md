# L1 Portable Foundation

## Goal

GitHub is the training factory. Android must not replay years of historical races to reconstruct the model.

The first foundation produces:

1. a point-in-time L1 dataset from KEIBA-BACKFILL;
2. bounded-memory walk-forward folds;
3. one LightGBM model per fold;
4. out-of-fold (OOF) predictions for every validation starter;
5. exact feature order and categorical dictionaries per model;
6. metadata containing the exact BACKFILL source commit when supplied.

## Default validation shape

For holdout year Y and `train-years=3`:

- warm-up: Y-4
- train: Y-3 through Y-1
- holdout: Y

Example:

- warm-up: 2021
- train: 2022-2024
- holdout: 2025

The dataset for each fold is built independently and deleted after training by default. This bounds memory/disk pressure instead of constructing 2010-2025 as one giant in-memory dataset.

## OOF rule

The prediction for a race must come from a model that was trained without that race or any future race.

OOF records are the only L1 predictions allowed as future L2 training input.

## Experimental model storage

Do not commit every experimental dataset/model to Git history.

- temporary dataset: deleted after fold
- experiment model/OOF/diagnostics: runner-local only by default; persistent Actions artifact upload is disabled unless the user explicitly approves storage
- promoted model: store only after validation/promotion policy is defined

## Current Feature Set caution

Direct pedigree/jockey/trainer IDs are blocked from L1 predictors. Phase 2 adds point-in-time PEDIGREE and ACTOR performance features instead; the raw IDs exist only as internal state keys. The result-page `actual_start_time` is not used because it is not guaranteed to be known before the race.

## Future phases

After the L1 foundation is stable:

- model logic extraction (thresholds/tree paths/contributions);
- latest HORSE_STATE materialization;
- production model promotion;
- Android parity test;
- L2 betting ML trained from L1 OOF + historical odds/payouts.


## AUTO FEATURE FACTORY V1

Dataset version 3 / feature schema 8 contains AUTO FEATURE FACTORY V1 outputs and the optional BACKFILL feature receiver.

The first connected sources are limited to prior-race:

- finish position
- last 3F
- speed
- distance
- body weight
- carried weight

Current-field relative features are limited to:

- point-in-time Elo
- recent win rate
- recent top-3 rate
- recent average finish
- recent average last 3F
- recent average speed

Training does **not** consume these automatically. New research selects independent Feature Sets explicitly.

Use the `AUTO` Feature Set only when intentionally comparing AUTO-derived features in walk-forward research.

Final odds, popularity and payout remain forbidden AUTO inputs.


## BACKFILL receiver

The ML dataset can receive normalized race conditions through the independent `BACKFILL` Feature Set without forcing STYLE, LAP, OPPONENT or AUTO into the same experiment.

`AUTO` and `BACKFILL` may be selected independently or together.

Missing BACKFILL fields remain null. ML does not re-parse raw race-condition text.

Current-race margin is never a predictor. Prior-race normalized margin is now available only through past-only AUTO features after semantics/coverage verification.


## Feature Set contract

The canonical selection unit is now an independent Feature Set, not a cumulative stage.

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

The old stage names remain only as migration aliases for existing commands. New research must use `--feature-sets`.

Prediction inputs are also phase-bound:

- `EARLY`: only facts available by entry/field publication
- `FINAL`: may additionally use race-day conditions and announced body weight

Purpose-specific history windows replace the single research-wide `history_limit`. Actor recent-form and time/pace histories have their own windows. Exact phase, Feature Sets, window values and small-sample policy are persisted for reproducibility.

Phase 2 builders use a shared small-sample contract. Raw observation counts remain visible; sparse PEDIGREE/ACTOR condition buckets fall back to broader point-in-time stats and are shrunk rather than discarded. Shrinkage strengths are research parameters for Phase 3, not production-selected constants.

TIME_PACE stores normalized historical final-time, last3f, early-pace and late-pace performance plus FAST/EVEN/SLOW historical suitability. Target-race result/laps never enter its own prediction row.

See `docs/L1_FEATURE_SET_CONTRACT_V1.md`.


## Margin AUTO

Feature schema 8 adds past-only AUTO features from BACKFILL `margin_type` and `margin_lengths`.
Current-race margin remains a target/output value and is never used as a predictor for that race.


## BACKFILL READINESS V3

Full training must pass two layers before LightGBM starts.

1. BACKFILL's own `src/verify-range.mjs`
   - every calendar date in the requested range must be classified as SUCCESS / confirmed non-meeting / schedule exception;
   - required pack/parser versions must be current;
   - daily race counts and schedule ownership must agree;
   - result/lap/payout/race-meta contracts must pass.
2. KEIBA-ML readiness coverage checks
   - normalized race-condition coverage;
   - invalid/unknown rates;
   - year-to-year coverage gaps;
   - normalized margin coverage.

The exact BACKFILL SHA is part of the readiness contract.

## Model safety and diagnostics

Before training, model columns are checked against the Feature Catalog's L1-forbidden `model_keys`.
A forbidden direct ID, market field, post-race field, or target field blocks training.

Each fold metadata now records:

- prediction phase;
- exact Feature Set list;
- exact history-window configuration;
- exact ordered feature list;
- model SHA-256;
- training-config SHA-256;
- Feature Catalog SHA-256;
- Feature Set Contract SHA-256;
- Small Sample Contract SHA-256;
- exact small-sample policy;
- train/validation feature coverage;
- subgroup metrics by surface, venue, distance band, class, grade, course layout and field size.

Runner-local diagnostics can contain the LightGBM tree dump, split thresholds and mean absolute per-feature contribution.
OOF rows carry dataset version, feature schema version, leakage policy, prediction phase, Feature Sets, history windows, small-sample policy, exact feature list, Feature Catalog hash, Feature Set Contract hash and Small Sample Contract hash.
