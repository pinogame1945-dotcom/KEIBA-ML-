#!/usr/bin/env node
import assert from "node:assert/strict";
import {
  PEDIGREE_FEATURE_BUILDER_VERSION,
  lineageFromHorseRecord,
  pedigreeDistanceBand,
  createPedigreeFeatureState,
} from "../src/pedigree-feature-builder.mjs";

assert.equal(PEDIGREE_FEATURE_BUILDER_VERSION, 3);
assert.equal(pedigreeDistanceBand(1200), "SPRINT_1400_OR_LESS");
assert.equal(pedigreeDistanceBand(1600), "MILE_1600_1800");
assert.equal(pedigreeDistanceBand(2000), "MIDDLE_2000_2200");
assert.equal(pedigreeDistanceBand(2400), "LONG_2400_PLUS");

const lineage = lineageFromHorseRecord({
  pedigree: [
    { generation: 1, slot: 0, ancestor_id: "SIRE" },
    { generation: 1, slot: 1, ancestor_id: "DAM" },
    { generation: 2, slot: 2, ancestor_id: "DAMSIRE" },
  ],
});
assert.deepEqual(lineage, { sire_key: "id:SIRE", damsire_key: "id:DAMSIRE" });

const fallbackLineage = lineageFromHorseRecord({
  pedigree: [
    { generation: 1, slot: 0, ancestor_id: "000", ancestor_name: "Foreign Sire" },
    { generation: 2, slot: 2, ancestor_id: "000", ancestor_name: "Foreign Damsire" },
  ],
});
assert.deepEqual(fallbackLineage, {
  sire_key: "name:foreign sire",
  damsire_key: "name:foreign damsire",
});

const state = createPedigreeFeatureState({
  smallSamplePolicy: {
    minSpecificObservations: 3,
    ratePriorStrength: 10,
    meanPriorStrength: 5,
  },
});
const turf = {
  surface: "TURF",
  distance_m: 1600,
  venue_code: "05",
  course_layout: "OUTER",
  track_condition: "GOOD",
};
const dirt = { ...turf, surface: "DIRT", venue_code: "06" };

const before = state.snapshot(lineage, turf);
assert.equal(before.ped_sire_all_starts, 0);
assert.equal(before.ped_sire_surface_starts, 0);
assert.equal(before.ped_sire_surface_condition_known, 1);
assert.equal(before.ped_sire_surface_fallback_level, 3);

state.add(lineage, turf, {
  result_status: "FINISHED",
  official_finish_position: 1,
  margin_type: "LENGTHS",
  margin_lengths: 0.5,
});
state.add(lineage, dirt, {
  result_status: "FINISHED",
  official_finish_position: 5,
  margin_type: "HEAD",
  margin_lengths: null,
});

const after = state.snapshot(lineage, turf);
assert.equal(after.ped_sire_all_starts, 2);
assert.equal(after.ped_sire_all_win_observations, 2);
assert.equal(after.ped_sire_all_margin_observations, 1);
assert.equal(after.ped_sire_surface_starts, 1);
assert.equal(after.ped_sire_surface_fallback_level, 1);
assert.equal(after.ped_sire_surface_effective_starts, 2);
assert.ok(after.ped_sire_all_effective_win_rate > 0);
assert.ok(after.ped_sire_all_effective_win_rate < 1);
assert.equal(after.ped_damsire_surface_fallback_level, 1);

state.add(lineage, turf, {
  result_status: "FINISHED",
  official_finish_position: 2,
  margin_type: "LENGTHS",
  margin_lengths: 1,
});
state.add(lineage, turf, {
  result_status: "FINISHED",
  official_finish_position: 3,
  margin_type: "LENGTHS",
  margin_lengths: 2,
});
const enough = state.snapshot(lineage, turf);
assert.equal(enough.ped_sire_surface_starts, 3);
assert.equal(enough.ped_sire_surface_fallback_level, 0);
assert.equal(enough.ped_sire_surface_effective_starts, 3);

const missingCondition = state.snapshot(lineage, {});
assert.equal(missingCondition.ped_sire_surface_condition_known, 0);
assert.equal(missingCondition.ped_sire_surface_fallback_level, 1);

const keys = Object.keys(enough);
assert.equal(keys.some(key => /sire_key|damsire_key|ancestor_id/i.test(key)), false);

const startsBeforeDnf = enough.ped_sire_all_starts;
state.add(lineage, turf, { result_status: "DNF", official_finish_position: null });
assert.equal(state.snapshot(lineage, turf).ped_sire_all_starts, startsBeforeDnf);

console.log(JSON.stringify({
  status: "PASS",
  version: PEDIGREE_FEATURE_BUILDER_VERSION,
  stat_rows: state.stat_rows(),
}, null, 2));
