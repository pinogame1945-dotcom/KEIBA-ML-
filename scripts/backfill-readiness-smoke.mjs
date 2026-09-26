import assert from "node:assert/strict";
import { auditBackfillRows } from "../src/backfill-readiness.mjs";

function row(date, overrides = {}, resultOverrides = {}) {
  return {
    race: {
      race_id: date.replaceAll("-", "") + "0101",
      actual_date: date,
      field_size: 2,
      course_layout: "OUTER",
      course_laps: 1,
      race_class_raw: "オープン",
      race_class_normalized: "OPEN",
      grade: "NONE",
      age_condition_raw: "3歳以上",
      age_min: 3,
      age_max: null,
      sex_condition: "ANY",
      weight_rule: "SPECIAL_WEIGHT",
      mixed: false,
      international: false,
      special_designated: false,
      designated: false,
      course_meta_raw: "芝左 外1600m",
      race_condition_raw: "3歳以上 オープン 別定",
      ...overrides,
    },
    entries: [
      { horse_id: "A", entry_status: "STARTED" },
      { horse_id: "B", entry_status: "STARTED" },
    ],
    results: [
      { horse_id: "A", result_status: "FINISHED", official_finish_position: 1 },
      {
        horse_id: "B",
        result_status: "FINISHED",
        official_finish_position: 2,
        margin_raw: "1/2",
        normalized_margin: 0.5,
        margin_seconds: 0.1,
        margin_length_equivalent: 0.5,
        margin_kind: "LENGTH",
        margin_normalization_version: "v1",
        ...resultOverrides,
      },
    ],
  };
}

const good = auditBackfillRows([
  row("2024-01-01"),
  row("2025-01-01"),
], {
  minCoreKnownCoverage: 1,
  maxInvalidRate: 0,
  maxYearGap: 0,
});

assert.equal(good.ready_for_l1_research, true);
assert.equal(good.failed_gate_count, 0);
assert.equal(good.overall.races, 2);
assert.equal(good.overall.fields.course_layout.known_coverage, 1);
assert.equal(good.overall.fields.course_laps.known_coverage, 1);
assert.equal(good.overall.fields.grade.known_coverage, 1);
assert.equal(good.overall.margin.raw_coverage, 1);
assert.equal(good.overall.margin.margin_seconds_coverage, 1);
assert.equal(good.by_year["2024"].races, 1);
assert.equal(good.by_year["2025"].races, 1);

const missingOld = auditBackfillRows([
  row("2020-01-01", { course_layout: null }),
  row("2025-01-01"),
], {
  minCoreKnownCoverage: 0.9,
  maxInvalidRate: 0,
  maxYearGap: 0.2,
});

assert.equal(missingOld.ready_for_l1_research, false);
assert.equal(missingOld.overall.fields.course_layout.known_coverage, 0.5);
assert.ok(missingOld.failed_gates.some(g =>
  g.type === "CORE_KNOWN_COVERAGE" && g.field === "course_layout"
));
assert.ok(missingOld.failed_gates.some(g =>
  g.type === "YEAR_COVERAGE_GAP" && g.field === "course_layout"
));

const unknown = auditBackfillRows([
  row("2025-01-01", { course_layout: "UNKNOWN" }),
], {
  minCoreKnownCoverage: 0.9,
  maxInvalidRate: 0,
  maxYearGap: 1,
});
assert.equal(unknown.overall.fields.course_layout.coverage, 1);
assert.equal(unknown.overall.fields.course_layout.known_coverage, 0);
assert.equal(unknown.overall.fields.course_layout.unknown_rate, 1);
assert.equal(unknown.ready_for_l1_research, false);

const invalid = auditBackfillRows([
  row("2025-01-01", { grade: "SUPER_G1" }),
], {
  minCoreKnownCoverage: 0,
  maxInvalidRate: 0,
  maxYearGap: 1,
});
assert.equal(invalid.overall.fields.grade.invalid, 1);
assert.equal(invalid.overall.fields.grade.invalid_rate, 1);
assert.equal(invalid.ready_for_l1_research, false);

const partialMargin = auditBackfillRows([
  row("2025-01-01", {}, {
    normalized_margin: null,
    margin_seconds: null,
    margin_length_equivalent: null,
  }),
]);
assert.equal(partialMargin.overall.margin.raw_coverage, 1);
assert.equal(partialMargin.overall.margin.normalized_margin_coverage, 0);
assert.equal(partialMargin.overall.margin.margin_seconds_coverage, 0);

console.log("BACKFILL_READINESS_SMOKE_OK");


const missingLaps = auditBackfillRows([
  row("2025-02-01", { course_laps: null }),
], {
  minCoreKnownCoverage: 0.98,
  maxInvalidRate: 0,
  maxYearGap: 1,
});
assert.equal(missingLaps.ready_for_l1_research, false);
assert.ok(missingLaps.failed_gates.some(g =>
  g.type === "CORE_KNOWN_COVERAGE" && g.field === "course_laps"
));

const legacyMismatch = auditBackfillRows([
  row("2025-03-01", {
    race_class_raw: "3歳以上1000万下",
    race_class_normalized: "ONE_WIN",
  }),
], {
  minCoreKnownCoverage: 0,
  maxInvalidRate: 0,
  maxYearGap: 1,
});
assert.equal(legacyMismatch.ready_for_l1_research, false);
assert.equal(legacyMismatch.overall.fields.race_class_normalized.invalid, 1);
assert.ok(Object.keys(legacyMismatch.overall.warnings).some(key =>
  key.startsWith("race_class_semantic:expected_TWO_WIN")
));
