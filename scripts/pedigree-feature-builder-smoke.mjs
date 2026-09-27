#!/usr/bin/env node
import assert from "node:assert/strict";
import {
  PEDIGREE_FEATURE_BUILDER_VERSION,
  lineageFromHorseRecord,
  createPedigreeFeatureState,
} from "../src/pedigree-feature-builder.mjs";

assert.equal(PEDIGREE_FEATURE_BUILDER_VERSION, 2);

const lineage = lineageFromHorseRecord({
  pedigree: [
    { generation: 1, slot: 0, ancestor_id: "SIRE" },
    { generation: 1, slot: 1, ancestor_id: "DAM" },
    { generation: 2, slot: 2, ancestor_id: "DAMSIRE" },
  ],
});
assert.deepEqual(lineage, { sire_key: "id:SIRE", damsire_key: "id:DAMSIRE" });

const fallback = lineageFromHorseRecord({
  pedigree: [
    { generation: 1, slot: 0, ancestor_id: "000", ancestor_name: "Foreign Sire" },
    { generation: 2, slot: 2, ancestor_id: "000", ancestor_name: "Foreign Damsire" },
  ],
});
assert.deepEqual(fallback, {
  sire_key: "name:foreign sire",
  damsire_key: "name:foreign damsire",
});

const state = createPedigreeFeatureState();
const race = {
  surface: "TURF",
  distance_m: 1600,
  venue_code: "05",
  course_layout: "OUTER",
  track_condition: "GOOD",
};

const before = state.snapshot(lineage, race);
assert.equal(before.ped_sire_all_starts, 0);
assert.equal(before.ped_damsire_all_starts, 0);

state.add(lineage, race, {
  official_finish_position: 2,
  margin_type: "LENGTHS",
  margin_lengths: 0.5,
});

const after = state.snapshot(lineage, race);
assert.equal(after.ped_sire_all_starts, 1);
assert.equal(after.ped_sire_all_top3_rate, 1);
assert.equal(after.ped_sire_all_avg_finish, 2);
assert.equal(after.ped_sire_all_avg_margin_lengths, 0.5);
assert.equal(after.ped_damsire_course_layout_outer_starts, 1);

console.log(JSON.stringify({
  status: "PASS",
  version: PEDIGREE_FEATURE_BUILDER_VERSION,
  stat_rows: state.stat_rows(),
}, null, 2));
