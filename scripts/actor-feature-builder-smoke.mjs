#!/usr/bin/env node
import assert from "node:assert/strict";
import {
  ACTOR_FEATURE_BUILDER_VERSION,
  actorDistanceBand,
  actorIdentityFromEntry,
  createActorFeatureState,
} from "../src/actor-feature-builder.mjs";

assert.equal(ACTOR_FEATURE_BUILDER_VERSION, 2);
assert.equal(actorDistanceBand(1600), "MILE_1600_1800");

const entry = {
  horse_id: "H1",
  jockey_id: "J1",
  jockey_name: "Jockey One",
  trainer_id: "T1",
  trainer_name: "Trainer One",
};
assert.deepEqual(actorIdentityFromEntry(entry), {
  horse_key: "horse:H1",
  jockey_key: "id:J1",
  trainer_key: "id:T1",
  horse_jockey_key: "horse:H1|id:J1",
});

const fallback = actorIdentityFromEntry({
  horse_id: "H2",
  jockey_id: "000",
  jockey_name: "Foreign Jockey",
  trainer_id: null,
  trainer_name: "Foreign Trainer",
});
assert.equal(fallback.jockey_key, "name:foreign jockey");
assert.equal(fallback.trainer_key, "name:foreign trainer");

const state = createActorFeatureState({
  smallSamplePolicy: {
    minSpecificObservations: 3,
    actorRecentWindow: 2,
    ratePriorStrength: 10,
    meanPriorStrength: 5,
  },
});
const turfOpen = {
  venue_code: "05",
  surface: "TURF",
  distance_m: 1600,
  race_class_normalized: "OPEN",
};
const dirtClass = {
  venue_code: "06",
  surface: "DIRT",
  distance_m: 1800,
  race_class_normalized: "3WIN",
};

const before = state.snapshot(entry, turfOpen);
assert.equal(before.actor_jockey_all_starts, 0);
assert.equal(before.actor_jockey_surface_starts, 0);
assert.equal(before.actor_jockey_race_class_condition_known, 1);

state.add(entry, turfOpen, { result_status: "FINISHED", official_finish_position: 1 });
state.add(entry, dirtClass, { result_status: "FINISHED", official_finish_position: 7 });

const after = state.snapshot(entry, turfOpen);
assert.equal(after.actor_jockey_all_starts, 2);
assert.equal(after.actor_jockey_recent_starts, 2);
assert.equal(after.actor_jockey_surface_starts, 1);
assert.equal(after.actor_jockey_surface_fallback_level, 1);
assert.equal(after.actor_jockey_surface_effective_starts, 2);
assert.equal(after.actor_trainer_race_class_starts, 1);
assert.equal(after.actor_horse_jockey_all_starts, 2);

state.add(entry, turfOpen, { result_status: "FINISHED", official_finish_position: 2 });
const recent = state.snapshot(entry, turfOpen);
assert.equal(recent.actor_jockey_recent_starts, 2);
assert.equal(recent.actor_jockey_surface_starts, 2);
assert.equal(recent.actor_jockey_surface_fallback_level, 1);

state.add(entry, turfOpen, { result_status: "FINISHED", official_finish_position: 3 });
const enough = state.snapshot(entry, turfOpen);
assert.equal(enough.actor_jockey_surface_starts, 3);
assert.equal(enough.actor_jockey_surface_fallback_level, 0);

const keys = Object.keys(enough);
assert.equal(keys.some(key => key.includes("jockey_id")), false);
assert.equal(keys.some(key => key.includes("trainer_id")), false);
assert.equal(keys.some(key => key.includes("J1") || key.includes("T1")), false);

const starts = enough.actor_jockey_all_starts;
state.add(entry, turfOpen, { result_status: "DNF", official_finish_position: null });
assert.equal(state.snapshot(entry, turfOpen).actor_jockey_all_starts, starts);

console.log(JSON.stringify({
  status: "PASS",
  version: ACTOR_FEATURE_BUILDER_VERSION,
  stat_rows: state.stat_rows(),
}, null, 2));
