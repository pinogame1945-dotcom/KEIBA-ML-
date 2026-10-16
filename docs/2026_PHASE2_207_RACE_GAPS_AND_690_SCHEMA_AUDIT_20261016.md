# 2026 Phase 2: Diagnose 207 race-panel gaps and 41 original additional input columns

Run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38874323515 (SUCCESS).
Branch: audit/2026-feature-input-phase1-20261016.
BACKFILL pinned SHA: 03950cfe132abeca15ef17dbc7c6885ceb1591b7.
Source data from 2022-01-01 through 2026-10-12. No 2026 tuning/training/evaluation.
Only standard free CPU and ephemeral runner temp files.

## Findings: exactly what are the 207 race-panel missing names?
- 2026 output 37,582 horse rows, 2,732 races.
- Canonical CORE4 year union has all 649 original names and matching frozen SHA256.
- 207/2,732 race-level schema shortfalls (7.577%).
- ALL 207 races lack the SAME 54 source names and NO OTHER column names.
- ALL 54 come from the auto_field_* family: 6 source measures times 9 derived field-relative statistics.
- The 6 source measures in the unchanged AUTO_FIELD_RELATIVE_SPECS:
  network_elo_rating, recent_win_rate, recent_top3_rate,
  recent_avg_finish, recent_avg_last_3f, recent_avg_speed_mps.
- The 9 derived statistics: diff_mean, diff_median, zscore, percentile,
  rank, gap_to_best, gap_to_top2_mean, gap_to_top3_mean, field_count.
- In existing src/auto-feature-factory.mjs, addFieldRelativeFeatures explicitly
  skips field-relative output if every source value in the field is missing
  (if (!field.length) continue). This explains the exact 54-column absence.
- These races likely have starters without comparable previous history;
  full independent condition-by-condition inspection of 207 races not done.
- Most recent dates: 2026-10-10 had 5 affected races;
  2026-10-11 had 4; 2026-10-12 had 5.
- Frozen original Pandas DataFrame.from_records tolerates missing per-horse
  keys by NaN; however original fresh one-race wrapper has a strict 649-name
  assertion before model prepare() reindex, so affected race unions can fail.
- No NaN-materializing compatibility patch has been implemented or approved,
  because it affects model input data structure and requires parity testing.

## Frozen original 19 + 22 extras
- BiasState.snapshot and bias_cols with_fit: 13 day-bias, 2 horse score,
  4 fit terms = 19.
- Original horse_features and speed_cols: 22 regime_speed_resid_ terms.
- Total original input schema: 649 + 19 + 22 = 690 distinct names.
- Numeric original _build_same_day_bias_year(2026,...) ran on the historical
  fully completed 2026-10-12 sample, 24 races, 320 of 321 snapshot horse
  rows eligible by original bias-reader labels. 252 of 320 rows have a
  non-zero earlier same-day race count. The one-row difference is untraced.
- This sample is not proof of actual prebet availability; no source
  capture-time provenance has been independently verified.
- The 22 regime column names came from frozen horse_features using an
  empty-prior schema and resulting NaNs. Numeric 2026 speed-regime values
  against the full historical 2019-2025 prior state remain unverified.
- Full 690 feature VALUE parity with frozen model inputs remains UNVERIFIED.

## Remaining work and boundaries
1. Establish whether inserting only canonical null placeholders for absent
   auto_field columns gives matrix parity for the 207 affected race panels.
   Do not silently change frozen model, features, data structure or rules.
2. Reproduce numeric 22 original regime features with full strictly-prior
   2019-2025 history and check 690 model columns and values.
3. Independently verify historical as-of source timing and prebet quotes;
   do not mistake historical final odds for pre-start live prices.
4. Connect an unscored 2026 target through the unchanged L1/L1.75/frozen
   L2.1/L3 path only after these checks.
5. p3 TOP10 and UNKNOWN semantics remain unchanged.
