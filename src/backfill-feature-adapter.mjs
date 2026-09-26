export const BACKFILL_FEATURE_ADAPTER_VERSION = 2;

const ENUMS = {
  course_layout: new Set(["INNER", "OUTER", "NORMAL", "UNKNOWN"]),
  race_class_normalized: new Set([
    "NEWCOMER", "MAIDEN", "ONE_WIN", "TWO_WIN", "THREE_WIN", "OPEN",
  ]),
  grade: new Set(["G1", "G2", "G3", "JPN1", "JPN2", "JPN3", "L", "NONE", "UNKNOWN"]),
  sex_condition: new Set(["ANY", "FEMALE_ONLY", "MALE_ONLY", "OTHER", "UNKNOWN"]),
  weight_rule: new Set([
    "WEIGHT_FOR_AGE", "SET_WEIGHT", "SPECIAL_WEIGHT", "HANDICAP", "UNKNOWN",
  ]),
};

function finite(value) {
  if (value == null) return null;
  if (typeof value === "string" && value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function text(value) {
  if (value == null) return null;
  const out = String(value).trim();
  return out ? out : null;
}

function enumValue(name, value, warnings) {
  const raw = text(value);
  if (raw == null) return null;
  if (ENUMS[name]?.has(raw)) return raw;
  warnings.push(`${name}:unexpected:${raw}`);
  return null;
}

function booleanValue(name, value, warnings) {
  if (value == null) return null;
  if (value === true || value === false) return value;
  warnings.push(`${name}:expected_boolean`);
  return null;
}

function nonNegativeInteger(name, value, warnings) {
  const n = finite(value);
  if (n == null) return null;
  if (Number.isInteger(n) && n >= 0) return n;
  warnings.push(`${name}:expected_non_negative_integer`);
  return null;
}

function positiveNumber(name, value, warnings) {
  const n = finite(value);
  if (n == null) return null;
  if (n > 0) return n;
  warnings.push(`${name}:expected_positive_number`);
  return null;
}

function activeEntryCount(entries) {
  return (entries ?? []).filter(entry => {
    const status = String(entry?.entry_status ?? "").toUpperCase();
    return status !== "SCRATCHED" && status !== "EXCLUDED";
  }).length;
}

export function readBackfillRaceFeatures(race, entries = []) {
  const warnings = [];
  const sourceFieldSize = nonNegativeInteger("field_size", race?.field_size, warnings);
  const ageMin = nonNegativeInteger("age_min", race?.age_min, warnings);
  const ageMax = nonNegativeInteger("age_max", race?.age_max, warnings);

  if (ageMin != null && ageMax != null && ageMin > ageMax) {
    warnings.push("age_condition:min_gt_max");
  }

  return {
    features: {
      backfill_field_size: sourceFieldSize ?? activeEntryCount(entries),
      backfill_course_layout: enumValue("course_layout", race?.course_layout, warnings),
      backfill_course_laps: positiveNumber("course_laps", race?.course_laps, warnings),
      backfill_race_class_normalized: enumValue(
        "race_class_normalized",
        race?.race_class_normalized,
        warnings,
      ),
      backfill_grade: enumValue("grade", race?.grade, warnings),
      backfill_age_min: ageMin,
      backfill_age_max: ageMax,
      backfill_sex_condition: enumValue("sex_condition", race?.sex_condition, warnings),
      backfill_weight_rule: enumValue("weight_rule", race?.weight_rule, warnings),
      backfill_mixed: booleanValue("mixed", race?.mixed, warnings),
      backfill_international: booleanValue("international", race?.international, warnings),
      backfill_special_designated: booleanValue(
        "special_designated",
        race?.special_designated,
        warnings,
      ),
      backfill_designated: booleanValue("designated", race?.designated, warnings),
    },
    warnings,
  };
}

export function readBackfillRaceRawContext(race) {
  return {
    race_class_raw: text(race?.race_class_raw),
    age_condition_raw: text(race?.age_condition_raw),
    course_meta_raw: text(race?.course_meta_raw),
    race_condition_raw: text(race?.race_condition_raw),
  };
}

export function readBackfillMarginContext(result) {
  return {
    margin_raw: text(result?.margin_raw),
    margin_type: text(result?.margin_type),
    margin_lengths: finite(result?.margin_lengths),
  };
}
