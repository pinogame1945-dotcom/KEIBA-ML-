#!/usr/bin/env node
import assert from "node:assert/strict";
import {
  blankOutcomeStats,
  chooseFallbackStats,
  effectiveOutcomeSnapshot,
  shrinkMean,
  shrinkRate,
  updateOutcomeStats,
} from "../src/small-sample-feature-utils.mjs";

assert.equal(shrinkRate(1, 3, 0.10, 20), (1 + 2) / 23);
assert.equal(shrinkMean(6, 3, 4, 10), 46 / 13);

const globalStats = blankOutcomeStats();
for (const finish of [1,2,3,4,5,6,7,8,9,10]) {
  updateOutcomeStats(globalStats, { result_status: "FINISHED", official_finish_position: finish });
}
const overall = blankOutcomeStats();
for (const finish of [1,4,7,8,9,10]) {
  updateOutcomeStats(overall, { result_status: "FINISHED", official_finish_position: finish });
}
const tiny = blankOutcomeStats();
updateOutcomeStats(tiny, { result_status: "FINISHED", official_finish_position: 1 });

const fallback = chooseFallbackStats(tiny, overall, globalStats, { minSpecificObservations: 5 });
assert.equal(fallback.fallbackSource, "ENTITY_OVERALL");
const effective = effectiveOutcomeSnapshot({
  specific: tiny,
  broader: overall,
  globalStats,
  policy: { minSpecificObservations: 5, ratePriorStrength: 20, meanPriorStrength: 10 },
});
assert.equal(effective.starts, 1);
assert.equal(effective.fallback_level, 1);
assert.equal(effective.effective_starts, 6);
assert.ok(effective.effective_win_rate < 0.5);

const enough = effectiveOutcomeSnapshot({
  specific: overall,
  broader: globalStats,
  globalStats,
  policy: { minSpecificObservations: 5 },
});
assert.equal(enough.fallback_level, 0);
assert.equal(enough.effective_starts, 6);

console.log("SMALL_SAMPLE_FEATURE_UTILS_SMOKE_OK");
