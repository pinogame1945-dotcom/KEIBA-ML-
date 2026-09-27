import assert from "node:assert/strict";
import {
  applyPredictionPhase,
  assertPredictionPhaseSafe,
  legacyStageToFeatureSets,
  normalizeFeatureSets,
  normalizeHistoryWindows,
  normalizePredictionPhase,
  selectFeatureFamilies,
} from "../src/l1-feature-contract.mjs";

assert.equal(normalizePredictionPhase("early"), "EARLY");
assert.deepEqual(legacyStageToFeatureSets("distance_v1"), ["BASE", "OPPONENT", "NETWORK", "LAP", "STYLE", "DISTANCE"]);
assert.deepEqual(normalizeFeatureSets("BASE,DISTANCE"), ["BASE", "DISTANCE"]);
assert.deepEqual(normalizeFeatureSets("DISTANCE,BASE,DISTANCE"), ["DISTANCE", "BASE"]);

const selected = selectFeatureFamilies({
  distance_m: 1600,
  opponent_recent_avg_win_rate: 0.2,
  style_recent_front_rate: 0.5,
  distx_same_band_starts: 3,
  auto_history_speed_mps_mean: 16,
}, ["BASE", "DISTANCE"]);
assert.deepEqual(Object.keys(selected).sort(), ["distance_m", "distx_same_band_starts"]);

const early = applyPredictionPhase({
  weather: "晴",
  track_condition: "良",
  body_weight: 480,
  body_weight_diff: 2,
  auto_pair_body_weight_vs_recent_mean_diff: 5,
  auto_finish_same_going_starts: 3,
  carried_weight: 56,
}, "EARLY");
assert.deepEqual(early, { carried_weight: 56 });
assert.throws(() => assertPredictionPhaseSafe(["body_weight"], "EARLY"), /blocked model columns/);
assert.doesNotThrow(() => assertPredictionPhaseSafe(["body_weight"], "FINAL"));

assert.deepEqual(normalizeHistoryWindows(), {
  recent_form: 5,
  style_last3f: 10,
  suitability: 20,
  opponent: 10,
  auto_rolling: 20,
  career: "ALL",
  elo: "ALL",
});
assert.equal(normalizeHistoryWindows({ opponent: 12 }).opponent, 12);
assert.equal(normalizeHistoryWindows({}, 7).style_last3f, 7);

console.log("L1_FEATURE_SET_CONTRACT_SMOKE_OK");
