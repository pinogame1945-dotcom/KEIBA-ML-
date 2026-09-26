import assert from "node:assert/strict";
import {
  BACKFILL_FEATURE_ADAPTER_VERSION,
  readBackfillRaceFeatures,
  readBackfillRaceRawContext,
  readBackfillMarginContext,
} from "../src/backfill-feature-adapter.mjs";

assert.equal(BACKFILL_FEATURE_ADAPTER_VERSION, 1);

const entries = [
  { horse_id: "A", entry_status: "STARTED" },
  { horse_id: "B", entry_status: "STARTED" },
  { horse_id: "C", entry_status: "SCRATCHED" },
];

const ready = readBackfillRaceFeatures({
  field_size: 16,
  course_layout: "OUTER",
  course_laps: 2,
  race_class_normalized: "OPEN",
  grade: "G2",
  age_min: 3,
  age_max: null,
  sex_condition: "ANY",
  weight_rule: "SPECIAL_WEIGHT",
  mixed: true,
  international: false,
  special_designated: true,
  designated: false,
}, entries);

assert.equal(ready.features.backfill_field_size, 16);
assert.equal(ready.features.backfill_course_layout, "OUTER");
assert.equal(ready.features.backfill_course_laps, 2);
assert.equal(ready.features.backfill_race_class_normalized, "OPEN");
assert.equal(ready.features.backfill_grade, "G2");
assert.equal(ready.features.backfill_age_min, 3);
assert.equal(ready.features.backfill_age_max, null);
assert.equal(ready.features.backfill_weight_rule, "SPECIAL_WEIGHT");
assert.equal(ready.features.backfill_mixed, true);
assert.deepEqual(ready.warnings, []);

const missing = readBackfillRaceFeatures({}, entries);
assert.equal(missing.features.backfill_field_size, 2);
assert.equal(missing.features.backfill_course_layout, null);
assert.equal(missing.features.backfill_grade, null);
assert.deepEqual(missing.warnings, []);

const bad = readBackfillRaceFeatures({
  course_layout: "MYSTERY",
  course_laps: -1,
  grade: "SUPER_G1",
  mixed: "true",
  age_min: 5,
  age_max: 3,
}, entries);
assert.equal(bad.features.backfill_course_layout, null);
assert.equal(bad.features.backfill_course_laps, null);
assert.equal(bad.features.backfill_grade, null);
assert.equal(bad.features.backfill_mixed, null);
assert.ok(bad.warnings.includes("course_layout:unexpected:MYSTERY"));
assert.ok(bad.warnings.includes("course_laps:expected_positive_number"));
assert.ok(bad.warnings.includes("grade:unexpected:SUPER_G1"));
assert.ok(bad.warnings.includes("mixed:expected_boolean"));
assert.ok(bad.warnings.includes("age_condition:min_gt_max"));

const raw = readBackfillRaceRawContext({
  race_class_raw: "3勝クラス",
  age_condition_raw: "3歳以上",
  course_meta_raw: "芝右 外1600m",
  race_condition_raw: "3歳以上 オープン (国際)(指定) 別定",
});
assert.equal(raw.race_class_raw, "3勝クラス");
assert.equal(raw.age_condition_raw, "3歳以上");
assert.equal(raw.course_meta_raw, "芝右 外1600m");
assert.ok(raw.race_condition_raw.includes("オープン"));

const margin = readBackfillMarginContext({
  margin_raw: "1/2",
  normalized_margin: 0.5,
  margin_seconds: 0.1,
  margin_length_equivalent: 0.5,
  margin_kind: "LENGTH",
  margin_normalization_version: "v1",
});
assert.equal(margin.margin_raw, "1/2");
assert.equal(margin.normalized_margin, 0.5);
assert.equal(margin.margin_seconds, 0.1);
assert.equal(margin.margin_length_equivalent, 0.5);
assert.equal(margin.margin_kind, "LENGTH");
assert.equal(margin.margin_normalization_version, "v1");

console.log("BACKFILL_FEATURE_ADAPTER_SMOKE_OK");
