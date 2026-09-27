# L1 Phase 3 Research / Selection

Phase 3 is the research-selection layer for feature schema 8.

It does not promote a production model. Its job is to compare candidate Feature Sets and preprocessing choices without inspecting 2026.

## Hard lock

2026 is forbidden in the current research code.

Both:
- `research/run_walk_forward.py`
- `research/train-staged-lightgbm.py`

reject train/validation or holdout dates in 2026 or later.

There is intentionally no CLI unlock switch in Phase 3. Final 2026 confirmation requires an explicit later code change.

## Research matrices

`research/run_phase3_selection.py` supports:

- `base_plus_one`: BASE and BASE + one family
- `windows`: one-at-a-time purpose-specific history-window changes
- `shrinkage`: rate/mean shrinkage strength and minimum specific-sample comparison
- `feature_selection`: no selection vs train-only selection
- `all`: union of the above, subject to an execution candidate cap

The runner is plan-only unless `--execute` is supplied.

## BASE + 1

The canonical first comparison is:

- BASE
- BASE + OPPONENT
- BASE + NETWORK
- BASE + LAP
- BASE + STYLE
- BASE + DISTANCE
- BASE + BACKFILL
- BASE + AUTO
- BASE + PEDIGREE
- BASE + ACTOR
- BASE + TIME_PACE

A good BASE + 1 result is not automatic production adoption. It is evidence for the next combination round.

## Window comparison

Default research grid:

- recent/form: 3, 5, 8
- style/last3f: 5, 10, 15
- suitability: 10, 20, 30
- opponent: 5, 10, 20
- AUTO rolling: 10, 20, 30
- actor recent: 15, 30, 50
- time/pace history: 5, 10, 20

The runner only varies windows relevant to the selected reference Feature Sets.

## Shrinkage comparison

Default engineering grid:

- rate prior strength: 5, 10, 20, 40
- mean prior strength: 5, 10, 20
- minimum specific observations: 3, 5, 10

These are research candidates, not hand-selected final values.

## Train-only Feature Selection

Mode `train_v1` is fitted inside each outer fold's training period only.

Order:

1. drop very-high-missingness columns;
2. drop constant / near-empty columns;
3. remove near-perfect numeric duplicates by correlation;
4. make a temporal inner split inside the outer train period;
5. fit a small LightGBM probe on the inner train only;
6. mark zero / configured-minimum gain features as deletion candidates;
7. if the inner probe would remove every remaining feature, abstain from the gain filter instead of deleting everything;
8. train the real outer-fold model with the selected columns;
9. evaluate only then on the untouched outer holdout.

The outer holdout cannot influence which columns survive.

The selection report records:
- exact input feature names;
- exact selected feature names;
- missingness drops;
- constant drops;
- correlation drops;
- inner-gain drops;
- thresholds;
- inner train/validation dates and row counts.

Selection-disabled and selection-enabled candidates are compared in the same walk-forward contract. Feature importance alone never promotes or deletes a feature.

## Metrics

Primary comparison metric:
- race-normalized NLL, lower is better

Secondary:
- Top1 / Top3 / Top6 winner capture
- mean reciprocal winner rank
- raw Brier score

Subgroup reports remain available for surface, venue, distance, class, grade, course layout and field size.

The research runner may order candidate IDs by the primary metric, but this does not promote a model.

## Compact persistent summary

The only Phase 3 result designed for persistence is:

`L1_PHASE3_RESEARCH_SUMMARY_V1`

It may contain:
- candidate configuration;
- fold and aggregate metrics;
- subgroup summaries;
- exact selected feature names;
- Feature Selection report without the full inner gain vector;
- source SHAs and reproducibility hashes.

It must not contain:
- model bytes;
- OOF row dumps;
- dataset dumps;
- full LightGBM tree dumps;
- contribution-row dumps.

The summary is capped at 2 MB.

Models, datasets, OOF and detailed diagnostics remain runner-local and ephemeral during research.

## GitHub Actions

`.github/workflows/l1-phase3-research.yml` is manual only.

Default mode is `plan`.

Execution uses the standard `ubuntu-latest` CPU runner. It has:
- no GPU;
- no paid runner;
- no artifact upload;
- no cache/dataset/model persistence.

A candidate-count safety cap prevents accidentally launching a very large matrix in one run.

## BACKFILL dependency

Actual unknown-year selection should begin only when the required BACKFILL range passes READINESS V3.

Until then, Phase 3 code can be smoke-tested and matrix plans can be generated, but incomplete historical coverage must not be treated as a model-selection result.

## Production boundary

Phase 3 may narrow candidates for later final validation. It does not:
- inspect 2026;
- write a promoted model;
- change Android production behavior;
- start L2;
- use odds in L1 ability.
