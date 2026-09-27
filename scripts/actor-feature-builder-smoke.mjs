#!/usr/bin/env node
import assert from "node:assert/strict";
import {
  ACTOR_FEATURE_BUILDER_VERSION,
  actorIdentityFromEntry,
  createActorFeatureState,
} from "../src/actor-feature-builder.mjs";

assert.equal(ACTOR_FEATURE_BUILDER_VERSION, 1);

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

const state = createActorFeatureState();
const race = { venue_code: "05", surface: "TURF", distance_m: 1600 };

const before = state.snapshot(entry, race);
assert.equal(before.actor_jockey_all_starts, 0);
assert.equal(before.actor_trainer_all_starts, 0);
assert.equal(before.actor_horse_jockey_all_starts, 0);

state.add(entry, race, {
  result_status: "FINISHED",
  official_finish_position: 2,
});
const after = state.snapshot(entry, race);
assert.equal(after.actor_jockey_all_starts, 1);
assert.equal(after.actor_jockey_all_top3_rate, 1);
assert.equal(after.actor_trainer_surface_turf_starts, 1);
assert.equal(after.actor_horse_jockey_all_starts, 1);

const starts = after.actor_jockey_all_starts;
state.add(entry, race, {
  result_status: "DNF",
  official_finish_position: null,
});
assert.equal(state.snapshot(entry, race).actor_jockey_all_starts, starts);

// Caller must snapshot every race on a date before committing that date's results.
// This smoke keeps the state API result-only and does not expose jockey/trainer IDs as features.
for (const key of Object.keys(after)) {
  assert.equal(key.includes("jockey_id"), false);
  assert.equal(key.includes("trainer_id"), false);
}

console.log(JSON.stringify({
  status: "PASS",
  version: ACTOR_FEATURE_BUILDER_VERSION,
  stat_rows: state.stat_rows(),
}, null, 2));
