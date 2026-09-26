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
- experiment model/OOF: short-retention Actions artifact
- promoted model: store only after validation/promotion policy is defined

## Current stage caution

The latest BACKFILL ladder showed that adding pedigree IDs degraded the 2025 holdout while `pedigree_dam_id` received abnormally large gain importance. Therefore the workflow defaults to `style`, not the full `distance` stage. Pedigree must be redesigned/audited before promotion.

## Future phases

After the L1 foundation is stable:

- model logic extraction (thresholds/tree paths/contributions);
- latest HORSE_STATE materialization;
- production model promotion;
- Android parity test;
- L2 betting ML trained from L1 OOF + historical odds/payouts.


## AUTO FEATURE FACTORY V1

Dataset version 3 / feature schema 5 contains AUTO FEATURE FACTORY V1 outputs and the optional BACKFILL feature receiver.

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

Training does **not** consume these automatically. The existing `style` stage remains the default and filters every `auto_` feature out.

Use `auto_v1` only when intentionally comparing the AUTO feature family in walk-forward research.

Final odds, popularity and payout remain forbidden AUTO inputs.


## BACKFILL receiver

The ML dataset can now receive the new normalized race conditions without changing the existing `style` baseline.

- `backfill_v1`: style + normalized BACKFILL race conditions
- `auto_backfill_v1`: style + AUTO + normalized BACKFILL race conditions

Missing BACKFILL fields remain null. ML does not re-parse raw race-condition text.

Margin normalization fields are readable by the adapter but are not yet model inputs. They remain blocked until their exact semantics and coverage are confirmed.
