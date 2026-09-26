import assert from "node:assert/strict";
import {
  AUTO_FEATURE_FACTORY_VERSION,
  summarizeNumericSeries,
  buildRollingFeatureFamily,
  conditionMatches,
  buildConditionFeatureFamily,
  addFieldRelativeFeatures,
  buildSafePairFeatures,
} from "../src/auto-feature-factory.mjs";

assert.equal(AUTO_FEATURE_FACTORY_VERSION, 1);

const stats = summarizeNumericSeries([1, null, 3, 5], { higherIsBetter: true });
assert.equal(stats.observation_count, 3);
assert.equal(stats.missing_count, 1);
assert.equal(stats.mean, 3);
assert.equal(stats.median, 3);
assert.equal(stats.last, 5);
assert.equal(stats.previous_delta, 2);
assert.equal(stats.rolling_2_mean, 4);
assert.equal(stats.rolling_3_mean, 3);
assert.equal(stats.gap_to_best, 0);
assert.equal(stats.gap_to_worst, 4);
assert.ok(stats.linear_trend > 0);

const rolling = buildRollingFeatureFamily("speed", [10, 11, 12], { higherIsBetter: true });
assert.equal(rolling.auto_speed_mean, 11);
assert.equal(rolling.auto_speed_last, 12);

const flags = conditionMatches(
  {
    venue_code: "05",
    surface: "TURF",
    distance_m: 1800,
    direction: "LEFT",
    course_layout: "OUTER",
    track_condition: "GOOD",
    race_class_normalized: "GRADED",
    age_condition_raw: "3歳以上",
    weight_rule: "HANDICAP",
  },
  {
    venue_code: "05",
    surface: "TURF",
    distance_m: 1600,
    direction: "LEFT",
    course_layout: "OUTER",
    track_condition: "GOOD",
    race_class_normalized: "GRADED",
    age_condition_raw: "3歳以上",
    weight_rule: "HANDICAP",
  },
);
assert.equal(flags.same_venue, true);
assert.equal(flags.same_distance, false);
assert.equal(flags.within_200m, true);
assert.equal(flags.same_course_layout, true);

const condition = buildConditionFeatureFamily({
  prefix: "finish",
  history: [
    { race: { venue_code: "05", surface: "TURF", distance_m: 1600 }, result: { finish: 1 } },
    { race: { venue_code: "05", surface: "TURF", distance_m: 1800 }, result: { finish: 4 } },
    { race: { venue_code: "06", surface: "DIRT", distance_m: 1600 }, result: { finish: 2 } },
  ],
  currentRace: { venue_code: "05", surface: "TURF", distance_m: 1600 },
  valueOf: x => x.result.finish,
  successOf: x => x.result.finish <= 3,
  higherIsBetter: false,
});
assert.equal(condition.auto_finish_same_venue_starts, 2);
assert.equal(condition.auto_finish_same_venue_mean, 2.5);
assert.equal(condition.auto_finish_same_venue_success_rate, 0.5);
assert.equal(condition.auto_finish_same_surface_starts, 2);
assert.equal(condition.auto_finish_same_distance_starts, 2);

const field = addFieldRelativeFeatures([
  { horse_id: "A", features: { elo: 1600 } },
  { horse_id: "B", features: { elo: 1500 } },
  { horse_id: "C", features: { elo: 1400 } },
], [{ key: "elo", prefix: "elo", direction: "HIGHER_BETTER" }]);
assert.equal(field[0].features.auto_field_elo_rank, 1);
assert.equal(field[1].features.auto_field_elo_rank, 2);
assert.equal(field[2].features.auto_field_elo_rank, 3);
assert.equal(field[0].features.auto_field_elo_diff_mean, 100);
assert.equal(field[2].features.auto_field_elo_diff_mean, -100);
assert.equal(field[0].features.auto_field_elo_percentile, 1);
assert.equal(field[2].features.auto_field_elo_percentile, 0);

const pair = buildSafePairFeatures(
  { current_distance: 1600, recent_distance: 1800 },
  [{
    a: "current_distance",
    b: "recent_distance",
    prefix: "distance",
    ops: ["diff", "ratio", "normalized_diff"],
  }],
);
assert.equal(pair.auto_pair_distance_diff, -200);
assert.ok(pair.auto_pair_distance_ratio > 0.88 && pair.auto_pair_distance_ratio < 0.89);
assert.ok(pair.auto_pair_distance_normalized_diff < 0);

assert.throws(
  () => addFieldRelativeFeatures(
    [{ features: { final_win_odds: 2.5 } }],
    [{ key: "final_win_odds", direction: "LOWER_BETTER" }],
  ),
  /forbidden AUTO FEATURE source/,
);

console.log("AUTO_FEATURE_FACTORY_SMOKE_OK");
