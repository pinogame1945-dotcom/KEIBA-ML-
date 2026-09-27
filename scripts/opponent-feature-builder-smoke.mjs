#!/usr/bin/env node
import assert from "node:assert/strict";
import { createOpponentFeatureState, OPPONENT_FEATURE_BUILDER_VERSION } from "../src/opponent-feature-builder.mjs";

assert.equal(OPPONENT_FEATURE_BUILDER_VERSION, 1);

function race(id, rows, marginType = "HEAD") {
  return {
    race: { race_id: id },
    entries: rows.map((x, i) => ({
      horse_id: x.horse,
      horse_number: i + 1,
      entry_status: "STARTED",
    })),
    results: rows.map(x => ({
      horse_id: x.horse,
      official_finish_position: x.finish,
      result_status: "FINISHED",
      margin_type: marginType,
      margin_lengths: marginType === "LENGTHS" ? 0.25 : null,
    })),
  };
}

const state = createOpponentFeatureState({
  historyLimit: 10,
  strongEloThreshold: 1500,
  closeMarginLengths: 0.5,
});

const bWin1 = race("R1", [
  { horse: "B", finish: 1 },
  { horse: "C", finish: 2 },
]);
const e1 = state.evaluateRace(bWin1);
state.commitRaceEvaluation(e1);
assert.ok(state.debugHorseElo("B").rating > 1500);

const bWin2 = race("R2", [
  { horse: "B", finish: 1 },
  { horse: "D", finish: 2 },
]);
const e2 = state.evaluateRace(bWin2);
state.commitRaceEvaluation(e2);

const aBeatsStrongB = race("R3", [
  { horse: "A", finish: 1 },
  { horse: "B", finish: 2 },
], "LENGTHS");
const beforeCommit = state.snapshot(aBeatsStrongB, aBeatsStrongB.entries[0]);
assert.equal(beforeCommit.opponent_relationship_races_measured, 0);

const e3 = state.evaluateRace(aBeatsStrongB);
assert.equal(state.snapshot(aBeatsStrongB, aBeatsStrongB.entries[0]).opponent_relationship_races_measured, 0);
state.commitRaceEvaluation(e3);

const target = race("R4", [
  { horse: "A", finish: 1 },
  { horse: "C", finish: 2 },
]);
const after = state.snapshot(target, target.entries[0]);
assert.equal(after.opponent_relationship_races_measured, 1);
assert.equal(after.opponent_strong_opponents_faced, 1);
assert.equal(after.opponent_strong_opponents_beaten, 1);
assert.equal(after.opponent_strong_opponent_beat_rate, 1);
assert.equal(after.opponent_strong_field_races, 1);
assert.equal(after.opponent_strong_field_top3_rate, 1);
assert.equal(after.opponent_close_finish_vs_strong_field_rate, 1);
assert.ok(after.opponent_history_avg_field_elo > 1500);

const sameDayState = createOpponentFeatureState({ strongEloThreshold: 1500 });
const same1 = sameDayState.evaluateRace(race("S1", [
  { horse: "X", finish: 1 },
  { horse: "Y", finish: 2 },
]));
const same2 = sameDayState.evaluateRace(race("S2", [
  { horse: "X", finish: 2 },
  { horse: "Z", finish: 1 },
]));
assert.equal(sameDayState.debugHorseElo("X").starts, 0);
assert.equal(same2.records.find(x => x.horseId === "X").field_avg_elo, 1500);
sameDayState.commitRaceEvaluation(same1);
sameDayState.commitRaceEvaluation(same2);
assert.equal(sameDayState.debugHorseElo("X").starts, 2);

console.log("OPPONENT_FEATURE_BUILDER_SMOKE_OK");
