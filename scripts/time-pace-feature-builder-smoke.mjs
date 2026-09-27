#!/usr/bin/env node
import assert from "node:assert/strict";
import { createTimePaceFeatureState, TIME_PACE_FEATURE_BUILDER_VERSION } from "../src/time-pace-feature-builder.mjs";

assert.equal(TIME_PACE_FEATURE_BUILDER_VERSION, 1);

function race(id, horseA, timeA, lastA, horseB, timeB, lastB, laps) {
  return {
    race: {
      race_id: id,
      venue_code: "05",
      surface: "TURF",
      distance_m: 1600,
      track_condition: "GOOD",
      course_layout: "OUTER",
    },
    entries: [
      { horse_id: horseA, entry_status: "STARTED" },
      { horse_id: horseB, entry_status: "STARTED" },
    ],
    results: [
      { horse_id: horseA, result_status: "FINISHED", official_finish_position: 1, finish_time_ms: timeA, last_3f: lastA },
      { horse_id: horseB, result_status: "FINISHED", official_finish_position: 2, finish_time_ms: timeB, last_3f: lastB },
    ],
    laps: laps.map(v => ({ lap_seconds: v })),
  };
}

const state = createTimePaceFeatureState({
  historyLimit: 10,
  minStandardObservations: 2,
  paceClassThreshold: 0.25,
});

const r1 = race("T1", "A", 94000, 34.5, "B", 95000, 35.0, [12.2,12.1,12.0,11.8,11.7,11.6,11.5,11.4]);
const r2 = race("T2", "A", 93000, 34.0, "C", 96000, 35.5, [12.0,11.9,11.8,11.7,11.8,11.9,12.0,12.1]);

const e1 = state.evaluateRace(r1);
assert.equal(e1.records[0].normalized_time, null);
state.commitRaceEvaluation(e1);

const e2 = state.evaluateRace(r2);
assert.notEqual(e2.records[0].normalized_time, null);
assert.equal(e2.records[0].pace_class, null);
state.commitRaceEvaluation(e2);

const r3 = race("T3", "A", 92000, 33.5, "D", 97000, 36.0, [11.6,11.6,11.7,11.8,12.0,12.1,12.2,12.3]);
const e3 = state.evaluateRace(r3);
const a3 = e3.records.find(x => x.horseId === "A");
assert.notEqual(a3.normalized_time, null);
assert.notEqual(a3.normalized_last3f, null);
assert.equal(a3.pace_class, "FAST");
assert.equal(a3.standard_fallback_level, 0);

const beforeCommit = state.snapshot("A");
assert.equal(beforeCommit.timepace_recent_races, 2);
state.commitRaceEvaluation(e3);
const afterCommit = state.snapshot("A");
assert.equal(afterCommit.timepace_recent_races, 3);
assert.equal(afterCommit.timepace_fast_pace_starts, 1);
assert.equal(afterCommit.timepace_fast_pace_top3_rate, 1);
assert.notEqual(afterCommit.timepace_recent_avg_normalized_time, null);

const sameDay = createTimePaceFeatureState({ minStandardObservations: 2 });
const s1 = sameDay.evaluateRace(r1);
const s2 = sameDay.evaluateRace(r2);
assert.equal(s1.records[0].normalized_time, null);
assert.equal(s2.records[0].normalized_time, null);
sameDay.commitRaceEvaluation(s1);
sameDay.commitRaceEvaluation(s2);
const next = sameDay.evaluateRace(r3);
assert.notEqual(next.records[0].normalized_time, null);

console.log("TIME_PACE_FEATURE_BUILDER_SMOKE_OK");
