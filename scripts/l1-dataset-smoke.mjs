import assert from "node:assert/strict";
import {
  buildMlDataset,
  buildRaceOutcomes,
  ML_DATASET_VERSION,
  ML_FEATURE_SCHEMA_VERSION,
  ML_LEAKAGE_POLICY,
} from "../src/ml-dataset.mjs";

function buildRichDataset(rows, options = {}) {
  return buildMlDataset(rows, {
    ...options,
    includeAutoFeatures: true,
    includeBackfillFeatures: true,
  });
}

function race({ id, date, finish, last3f, time, horse = "H1", jockey = "J1", distance = 1600 }) {
  return {
    race: {
      race_id: id,
      actual_date: date,
      venue_code: "05",
      meeting_no: 1,
      meeting_day: 1,
      race_no: Number(id.slice(-2)),
      discipline: "FLAT",
      surface: "TURF",
      distance_m: distance,
      direction: "LEFT",
      weather: "晴",
      track_condition: "良",
      actual_start_time: "15:40",
    },
    entries: [{
      race_id: id,
      horse_id: horse,
      gate: 2,
      horse_number: 3,
      sex: "牡",
      age: 4,
      carried_weight: 57,
      jockey_id: jockey,
      trainer_id: "T1",
      body_weight: 480,
      body_weight_diff: 2,
      entry_status: "STARTED",
    }],
    results: [{
      race_id: id,
      horse_id: horse,
      official_finish_position: finish,
      result_status: "FINISHED",
      finish_time_ms: time,
      margin_raw: "",
      last_3f: last3f,
      win_odds: 99.9,
      popularity: 18,
      prize_money: 1000,
    }],
    payouts: [{
      race_id: id,
      bet_type: "WIN",
      combination: "3",
      payout_yen: 9990,
      popularity: 18,
    }],
  };
}

const jan = race({ id: "202405010101", date: "2024-01-01", finish: 3, last3f: 34.5, time: 94500 });
jan.entries[0].body_weight_diff = null;
const janBuilt = buildRichDataset([jan], { startDate: "2024-01-01", endDate: "2024-01-01" });
assert.equal(janBuilt[0].features.body_weight_diff, null);
const feb = race({ id: "202405010201", date: "2024-02-01", finish: 1, last3f: 33.9, time: 93000, distance: 1800 });
const mar = race({ id: "202405010301", date: "2024-03-01", finish: 12, last3f: 38.0, time: 99000 });

const febOnly = buildRichDataset([jan, feb, mar], { startDate: "2024-02-01", endDate: "2024-02-01" });
assert.equal(febOnly.length, 1);
assert.equal(febOnly[0].features.prior_starts, 1);
assert.equal(febOnly[0].features.previous_finish_position, 3);
assert.equal(febOnly[0].features.distance_change_m, 200);
assert.equal(febOnly[0].features.jockey_continues, true);
assert.equal(febOnly[0].target.finish_position, 1);
assert.equal(febOnly[0].market_outcome.final_win_odds, 99.9);
assert.equal(febOnly[0].market_outcome.final_popularity, 18);
assert.equal(ML_DATASET_VERSION, 3);
assert.equal(ML_FEATURE_SCHEMA_VERSION, 6);
assert.equal(febOnly[0].ml_dataset_version, ML_DATASET_VERSION);
assert.equal(febOnly[0].feature_schema_version, ML_FEATURE_SCHEMA_VERSION);
assert.equal(febOnly[0].leakage_policy, ML_LEAKAGE_POLICY);
assert.equal(febOnly[0].features.auto_history_finish_position_mean, 3);
assert.equal(febOnly[0].features.auto_history_distance_m_mean, 1600);
assert.equal(febOnly[0].features.auto_pair_distance_vs_recent_mean_diff, 200);
assert.equal(febOnly[0].features.auto_history_body_weight_mean, 480);
assert.equal(febOnly[0].features.auto_pair_body_weight_vs_recent_mean_diff, 0);
assert.equal(Object.hasOwn(febOnly[0].features, "actual_start_time"), false);
assert.equal(Object.hasOwn(febOnly[0].features, "jockey_id"), false);
assert.equal(Object.hasOwn(febOnly[0].features, "trainer_id"), false);
assert.equal(Object.hasOwn(febOnly[0].features, "win_odds"), false);
assert.equal(Object.hasOwn(febOnly[0].features, "official_finish_position"), false);
assert.equal(Object.hasOwn(febOnly[0].features, "last_3f"), false);
assert.equal(Object.hasOwn(febOnly[0].features, "popularity"), false);
assert.equal(Object.hasOwn(febOnly[0].features, "payouts"), false);

const raceOutcomes = buildRaceOutcomes([jan, feb, mar], { startDate: "2024-02-01", endDate: "2024-02-01" });
assert.equal(raceOutcomes.length, 1);
assert.equal(raceOutcomes[0].race_id, feb.race.race_id);
assert.equal(raceOutcomes[0].payouts[0].bet_type, "WIN");
assert.equal(raceOutcomes[0].payouts[0].payout_yen, 9990);

const mutatedFuture = structuredClone(mar);
mutatedFuture.results[0].official_finish_position = 1;
mutatedFuture.results[0].last_3f = 20.0;
const febAfterFutureMutation = buildRichDataset([jan, feb, mutatedFuture], { startDate: "2024-02-01", endDate: "2024-02-01" });
assert.deepEqual(febAfterFutureMutation[0].features, febOnly[0].features);

const scratched = race({ id: "202405010399", date: "2024-03-31", finish: null, last3f: null, time: null, horse: "HS" });
scratched.entries[0].entry_status = "SCRATCHED";
scratched.results[0].result_status = "SCRATCHED";
const scratchedRows = buildRichDataset([scratched], { startDate: "2024-03-31", endDate: "2024-03-31" });
assert.equal(scratchedRows.length, 0);

const sameDayFirst = race({ id: "202405010401", date: "2024-04-01", finish: 1, last3f: 32.0, time: 90000, horse: "H2" });
const sameDaySecond = race({ id: "202405010402", date: "2024-04-01", finish: 2, last3f: 33.0, time: 91000, horse: "H2" });
const sameDay = buildRichDataset([sameDayFirst, sameDaySecond], { startDate: "2024-04-01", endDate: "2024-04-01" });
assert.equal(sameDay.length, 2);
assert.equal(sameDay[0].features.prior_starts, 0);
assert.equal(sameDay[1].features.prior_starts, 0);
assert.equal(sameDay[0].features.auto_history_finish_position_observation_count, 0);
assert.equal(sameDay[1].features.auto_history_finish_position_observation_count, 0);

console.log("ml dataset smoke: ok");


const opponentRace = race({ id: "202405010501", date: "2024-05-01", finish: 2, last3f: 34.0, time: 94000, horse: "HX" });
opponentRace.entries.push({
  ...opponentRace.entries[0],
  horse_id: "HO",
  horse_number: 4,
  jockey_id: "J2",
});
opponentRace.results.push({
  ...opponentRace.results[0],
  horse_id: "HO",
  official_finish_position: 1,
  win_odds: 2.0,
  popularity: 1,
});
const opponentFollowup = race({ id: "202405010601", date: "2024-06-01", finish: 1, last3f: 33.5, time: 93000, horse: "HO" });
const subjectTarget = race({ id: "202405010701", date: "2024-07-01", finish: 1, last3f: 33.0, time: 92000, horse: "HX" });
const opponentBuilt = buildRichDataset(
  [opponentRace, opponentFollowup, subjectTarget],
  { startDate: "2024-07-01", endDate: "2024-07-01" },
);
assert.equal(opponentBuilt.length, 1);
assert.equal(opponentBuilt[0].features.opponent_previous_known_count, 1);
assert.equal(opponentBuilt[0].features.opponent_previous_avg_starts, 2);
assert.equal(opponentBuilt[0].features.opponent_previous_avg_win_rate, 1);
assert.equal(opponentBuilt[0].features.opponent_previous_avg_top3_rate, 1);


const networkRace = race({ id: "202405010801", date: "2024-08-01", finish: 1, last3f: 33.0, time: 92000, horse: "NW" });
networkRace.entries.push({
  ...networkRace.entries[0],
  horse_id: "NL",
  horse_number: 4,
  jockey_id: "J3",
});
networkRace.results.push({
  ...networkRace.results[0],
  horse_id: "NL",
  official_finish_position: 2,
});
const networkTarget = race({ id: "202405010901", date: "2024-09-01", finish: 1, last3f: 33.0, time: 92000, horse: "NW" });
networkTarget.entries.push({
  ...networkTarget.entries[0],
  horse_id: "NL",
  horse_number: 4,
  jockey_id: "J3",
});
networkTarget.results.push({
  ...networkTarget.results[0],
  horse_id: "NL",
  official_finish_position: 2,
});
const networkBuilt = buildRichDataset(
  [networkRace, networkTarget],
  { startDate: "2024-09-01", endDate: "2024-09-01" },
);
assert.equal(networkBuilt.length, 2);
const networkWinner = networkBuilt.find(row => row.horse_id === "NW");
const networkLoser = networkBuilt.find(row => row.horse_id === "NL");
assert.ok(networkWinner.features.network_elo_rating > 1500);
assert.ok(networkLoser.features.network_elo_rating < 1500);
assert.equal(networkWinner.features.network_elo_starts, 1);
assert.ok(networkWinner.features.network_elo_vs_field_avg > 0);
assert.ok(networkWinner.features.network_expected_pairwise_score > 0.5);


assert.equal(networkWinner.features.auto_field_network_elo_rating_rank, 1);
assert.equal(networkLoser.features.auto_field_network_elo_rating_rank, 2);
assert.ok(networkWinner.features.auto_field_network_elo_rating_diff_mean > 0);
assert.ok(networkLoser.features.auto_field_network_elo_rating_diff_mean < 0);

for (const row of [...febOnly, ...networkBuilt]) {
  const names = Object.keys(row.features);
  assert.equal(names.some(name => /odds|popularity|payout/i.test(name)), false);
}


const backfillReady = race({
  id: "202405011001",
  date: "2024-10-01",
  finish: 1,
  last3f: 33.2,
  time: 92500,
  horse: "BF1",
});
Object.assign(backfillReady.race, {
  field_size: 12,
  course_layout: "OUTER",
  course_laps: 1,
  race_class_normalized: "OPEN",
  grade: "G3",
  age_min: 3,
  age_max: null,
  sex_condition: "ANY",
  weight_rule: "SPECIAL_WEIGHT",
  mixed: true,
  international: false,
  special_designated: true,
  designated: false,
  race_class_raw: "オープン",
  age_condition_raw: "3歳以上",
  course_meta_raw: "芝左 外1600m",
  race_condition_raw: "3歳以上 オープン 別定",
});
backfillReady.results[0].margin_seconds = 0.2;
backfillReady.results[0].margin_length_equivalent = 1.0;

const backfillBuilt = buildRichDataset(
  [backfillReady],
  { startDate: "2024-10-01", endDate: "2024-10-01" },
);
assert.equal(backfillBuilt.length, 1);
assert.equal(backfillBuilt[0].features.backfill_field_size, 12);
assert.equal(backfillBuilt[0].features.backfill_course_layout, "OUTER");
assert.equal(backfillBuilt[0].features.backfill_course_laps, 1);
assert.equal(backfillBuilt[0].features.backfill_race_class_normalized, "OPEN");
assert.equal(backfillBuilt[0].features.backfill_grade, "G3");
assert.equal(backfillBuilt[0].features.backfill_age_min, 3);
assert.equal(backfillBuilt[0].features.backfill_weight_rule, "SPECIAL_WEIGHT");
assert.equal(backfillBuilt[0].features.backfill_mixed, true);
assert.equal(Object.hasOwn(backfillBuilt[0].features, "race_condition_raw"), false);
assert.equal(Object.hasOwn(backfillBuilt[0].features, "margin_seconds"), false);

const backfillMissing = race({
  id: "202405011101",
  date: "2024-11-01",
  finish: 1,
  last3f: 33.0,
  time: 92000,
  horse: "BF2",
});
const backfillMissingBuilt = buildRichDataset(
  [backfillMissing],
  { startDate: "2024-11-01", endDate: "2024-11-01" },
);
assert.equal(backfillMissingBuilt[0].features.backfill_course_layout, null);
assert.equal(backfillMissingBuilt[0].features.backfill_grade, null);
assert.equal(backfillMissingBuilt[0].features.backfill_field_size, 1);


const isolatedStyle = buildMlDataset(
  [jan, feb],
  { startDate: "2024-02-01", endDate: "2024-02-01" },
);
assert.equal(
  Object.keys(isolatedStyle[0].features).some(name => name.startsWith("auto_")),
  false,
);
assert.equal(
  Object.keys(isolatedStyle[0].features).some(name => name.startsWith("backfill_")),
  false,
);

const layoutJan = race({ id: "202405011201", date: "2024-12-01", finish: 2, last3f: 34.0, time: 94000, horse: "LAY" });
layoutJan.race.course_layout = "OUTER";
layoutJan.race.race_class_normalized = "OPEN";
layoutJan.race.age_condition_raw = "3歳以上";
layoutJan.race.weight_rule = "SPECIAL_WEIGHT";
const layoutFeb = race({ id: "202505011301", date: "2025-01-01", finish: 1, last3f: 33.0, time: 92000, horse: "LAY" });
layoutFeb.race.course_layout = "OUTER";
layoutFeb.race.race_class_normalized = "OPEN";
layoutFeb.race.age_condition_raw = "3歳以上";
layoutFeb.race.weight_rule = "SPECIAL_WEIGHT";
const layoutBuilt = buildRichDataset(
  [layoutJan, layoutFeb],
  { startDate: "2025-01-01", endDate: "2025-01-01" },
);
assert.equal(layoutBuilt[0].features.auto_finish_same_course_layout_starts, 1);
assert.equal(layoutBuilt[0].features.auto_finish_same_course_layout_success_rate, 1);
assert.equal(layoutBuilt[0].features.auto_finish_same_class_starts, 1);
assert.equal(layoutBuilt[0].features.auto_finish_same_weight_rule_starts, 1);
