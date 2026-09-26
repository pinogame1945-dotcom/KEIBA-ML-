import {
  readBackfillRaceFeatures,
  readBackfillRaceRawContext,
  readBackfillMarginContext,
} from "./backfill-feature-adapter.mjs";

export const BACKFILL_READINESS_VERSION = 1;

export const RACE_FIELDS = [
  "field_size",
  "course_layout",
  "course_laps",
  "race_class_raw",
  "race_class_normalized",
  "grade",
  "age_condition_raw",
  "age_min",
  "age_max",
  "sex_condition",
  "weight_rule",
  "mixed",
  "international",
  "special_designated",
  "designated",
  "course_meta_raw",
  "race_condition_raw",
];

const NORMALIZED_ENUM_FIELDS = [
  "course_layout",
  "race_class_normalized",
  "grade",
  "sex_condition",
  "weight_rule",
];

const BOOLEAN_FIELDS = [
  "mixed",
  "international",
  "special_designated",
  "designated",
];

const CORE_FIELDS = [
  "course_layout",
  "race_class_normalized",
  "grade",
  "sex_condition",
  "weight_rule",
  "mixed",
  "international",
  "special_designated",
  "designated",
];

function present(value) {
  if (value == null) return false;
  if (typeof value === "string") return value.trim() !== "";
  return true;
}

function ratio(n, d) {
  return d ? n / d : null;
}

function yearOf(row) {
  const raw = row?.race?.actual_date ?? row?.race?.scheduled_date ?? "";
  const m = String(raw).match(/^(\d{4})-/);
  return m ? m[1] : "UNKNOWN";
}

function makeFieldStats() {
  return Object.fromEntries(RACE_FIELDS.map(name => [name, {
    present: 0,
    missing: 0,
    unknown: 0,
    invalid: 0,
  }]));
}

function makeBucket() {
  return {
    races: 0,
    fields: makeFieldStats(),
    warnings: {},
    margin: {
      finished_non_winner_rows: 0,
      raw_present: 0,
      normalized_margin_present: 0,
      margin_seconds_present: 0,
      margin_length_equivalent_present: 0,
      kind_present: 0,
      normalization_version_present: 0,
    },
  };
}

function increment(map, key) {
  map[key] = (map[key] ?? 0) + 1;
}

function rawFieldState(race, field) {
  const value = race?.[field];
  if (!present(value)) return "missing";
  if (NORMALIZED_ENUM_FIELDS.includes(field) && String(value).toUpperCase() === "UNKNOWN") {
    return "unknown";
  }
  return "present";
}

function warningField(warning) {
  const text = String(warning ?? "");
  if (text.startsWith("age_condition:")) return ["age_min", "age_max"];
  const field = text.split(":", 1)[0];
  return RACE_FIELDS.includes(field) ? [field] : [];
}

function markRace(bucket, row) {
  const race = row?.race ?? {};
  bucket.races += 1;

  for (const field of RACE_FIELDS) {
    const state = rawFieldState(race, field);
    bucket.fields[field][state] += 1;
  }

  const adapted = readBackfillRaceFeatures(race, row?.entries ?? []);
  for (const warning of adapted.warnings) {
    increment(bucket.warnings, warning);
    for (const field of warningField(warning)) {
      bucket.fields[field].invalid += 1;
    }
  }

  for (const result of row?.results ?? []) {
    const status = String(result?.result_status ?? "").toUpperCase();
    const finish = Number(result?.official_finish_position);
    if (status !== "FINISHED" || !Number.isFinite(finish) || finish <= 1) continue;

    const margin = readBackfillMarginContext(result);
    bucket.margin.finished_non_winner_rows += 1;
    if (present(margin.margin_raw)) bucket.margin.raw_present += 1;
    if (margin.normalized_margin != null) bucket.margin.normalized_margin_present += 1;
    if (margin.margin_seconds != null) bucket.margin.margin_seconds_present += 1;
    if (margin.margin_length_equivalent != null) bucket.margin.margin_length_equivalent_present += 1;
    if (present(margin.margin_kind)) bucket.margin.kind_present += 1;
    if (present(margin.margin_normalization_version)) {
      bucket.margin.normalization_version_present += 1;
    }
  }
}

export function createBackfillReadinessAccumulator() {
  const overall = makeBucket();
  const byYear = new Map();

  return {
    add(row) {
      const year = yearOf(row);
      if (!byYear.has(year)) byYear.set(year, makeBucket());
      markRace(overall, row);
      markRace(byYear.get(year), row);
    },
    finish(options = {}) {
      return summarizeReadiness(overall, byYear, options);
    },
  };
}

function fieldSummary(stats, races) {
  return {
    ...stats,
    coverage: ratio(stats.present + stats.unknown, races),
    known_coverage: ratio(stats.present, races),
    missing_rate: ratio(stats.missing, races),
    unknown_rate: ratio(stats.unknown, races),
    invalid_rate: ratio(stats.invalid, races),
  };
}

function marginSummary(margin) {
  const d = margin.finished_non_winner_rows;
  return {
    ...margin,
    raw_coverage: ratio(margin.raw_present, d),
    normalized_margin_coverage: ratio(margin.normalized_margin_present, d),
    margin_seconds_coverage: ratio(margin.margin_seconds_present, d),
    margin_length_equivalent_coverage: ratio(margin.margin_length_equivalent_present, d),
    kind_coverage: ratio(margin.kind_present, d),
    normalization_version_coverage: ratio(margin.normalization_version_present, d),
  };
}

function summarizeBucket(bucket) {
  return {
    races: bucket.races,
    fields: Object.fromEntries(
      Object.entries(bucket.fields).map(([name, stats]) => [name, fieldSummary(stats, bucket.races)]),
    ),
    warnings: Object.fromEntries(
      Object.entries(bucket.warnings).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])),
    ),
    margin: marginSummary(bucket.margin),
  };
}

function coverageGap(byYear, field) {
  const years = Object.entries(byYear)
    .filter(([year, bucket]) => /^\d{4}$/.test(year) && bucket.races > 0)
    .sort((a, b) => a[0].localeCompare(b[0]));
  if (years.length < 2) return null;

  const values = years
    .map(([year, bucket]) => ({
      year,
      coverage: bucket.fields?.[field]?.known_coverage,
    }))
    .filter(x => x.coverage != null);

  if (values.length < 2) return null;
  const min = values.reduce((a, b) => a.coverage <= b.coverage ? a : b);
  const max = values.reduce((a, b) => a.coverage >= b.coverage ? a : b);
  return {
    min_year: min.year,
    min_coverage: min.coverage,
    max_year: max.year,
    max_coverage: max.coverage,
    gap: max.coverage - min.coverage,
  };
}

export function summarizeReadiness(overallBucket, yearBuckets, {
  minCoreKnownCoverage = 0.98,
  maxInvalidRate = 0,
  maxYearGap = 0.10,
} = {}) {
  const overall = summarizeBucket(overallBucket);
  const byYear = Object.fromEntries(
    [...yearBuckets.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([year, bucket]) => [year, summarizeBucket(bucket)]),
  );

  const gates = [];
  for (const field of CORE_FIELDS) {
    const stats = overall.fields[field];
    gates.push({
      type: "CORE_KNOWN_COVERAGE",
      field,
      pass: stats.known_coverage != null && stats.known_coverage >= minCoreKnownCoverage,
      actual: stats.known_coverage,
      required: minCoreKnownCoverage,
    });
    gates.push({
      type: "INVALID_RATE",
      field,
      pass: stats.invalid_rate != null && stats.invalid_rate <= maxInvalidRate,
      actual: stats.invalid_rate,
      required_max: maxInvalidRate,
    });

    const gap = coverageGap(byYear, field);
    if (gap) {
      gates.push({
        type: "YEAR_COVERAGE_GAP",
        field,
        pass: gap.gap <= maxYearGap,
        actual: gap.gap,
        required_max: maxYearGap,
        detail: gap,
      });
    }
  }

  const failed = gates.filter(gate => !gate.pass);
  return {
    report_version: "BACKFILL_READINESS_V1",
    readiness_version: BACKFILL_READINESS_VERSION,
    thresholds: {
      min_core_known_coverage: minCoreKnownCoverage,
      max_invalid_rate: maxInvalidRate,
      max_year_gap: maxYearGap,
    },
    ready_for_l1_research: overall.races > 0 && failed.length === 0,
    failed_gate_count: failed.length,
    failed_gates: failed,
    core_fields: CORE_FIELDS,
    overall,
    by_year: byYear,
  };
}

export function auditBackfillRows(rows, options = {}) {
  const acc = createBackfillReadinessAccumulator();
  for (const row of rows ?? []) acc.add(row);
  return acc.finish(options);
}
