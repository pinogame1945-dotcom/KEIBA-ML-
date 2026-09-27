# L1 Phase 2 Feature Builders

Feature schema: 8.

Phase 2 completes research-grade, point-in-time builders. Completion does not mean production adoption.

## PEDIGREE

V1 scope is sire + damsire only.

Outputs are fixed-name numeric features for:
- overall;
- current surface;
- current venue;
- current course layout;
- current going;
- current distance band.

Each view keeps observation counts and raw values. Sparse condition views fall back to ancestor-wide stats and then the global point-in-time prior. Raw ancestor IDs/names are internal state keys only and never model columns.

Distance bands:
- <= 1400m;
- 1401-1800m;
- 1801-2200m;
- >= 2201m.

## ACTOR

The actor builder covers:
- jockey;
- trainer;
- horse x jockey.

Jockey/trainer features include overall, surface, venue, distance band, race class and recent form. Horse x jockey includes overall and recent form.

Raw jockey/trainer IDs are internal state keys only.

## OPPONENT

The relationship builder records only information known before the target date:
- historical field Elo level;
- opponent prior win/top3 strength;
- strong opponents faced;
- strong opponents beaten;
- strong-opponent beat rate;
- strong-field top3 rate;
- close performance in strong fields;
- current field strength after the field is fixed.

Same-day evaluations are computed from the day-start state and committed together after all prediction rows for the date are created.

## TIME_PACE

Raw times are not compared across unrelated conditions.

Historical performances are normalized against a point-in-time standard with fallback:
1. venue + surface + exact distance + going + layout;
2. surface + distance band;
3. surface;
4. global.

Horse history then exposes:
- normalized final time;
- normalized last3f;
- normalized early pace;
- normalized late pace;
- FAST / EVEN / SLOW prior pace classification;
- pace-class starts/rates/top3 rates/performance;
- standard fallback level.

The target race's own result and laps are never used in its prediction row.

## Small samples

Shared source of truth:
- `contracts/l1-small-sample-contract-v1.json`

Defaults are engineering baselines only:
- rate prior strength: 20;
- mean prior strength: 10;
- minimum specific observations: 5;
- actor recent window: controlled by the Feature Set history-window contract.

Phase 3 must compare these parameters inside walk-forward training without consulting 2026.

## Safety boundary

- STRICT_PRIOR_DATE_ONLY remains mandatory.
- Same-day results are invisible to same-day prediction rows.
- L1 ability does not use odds/popularity/payout.
- 2026 stays locked for final confirmation.
- Experimental model/OOF/diagnostics remain runner-local.
- No persistent Actions artifact upload is enabled.
