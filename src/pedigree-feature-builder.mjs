import {
  blankOutcomeStats,
  effectiveOutcomeSnapshot,
  updateOutcomeStats,
  validateSmallSamplePolicy,
} from "./small-sample-feature-utils.mjs";

export const PEDIGREE_FEATURE_BUILDER_VERSION = 4;

function finite(value) {
  if (value == null || (typeof value === "string" && value.trim() === "")) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function text(value) {
  const s = value == null ? "" : String(value).trim();
  return s || null;
}

function ancestorKey(node) {
  const id = text(node?.ancestor_id);
  if (id && id !== "000") return `id:${id}`;
  const name = text(node?.ancestor_name)?.normalize("NFKC").toLowerCase();
  return name ? `name:${name}` : null;
}

export function lineageFromHorseRecord(record) {
  const nodes = Array.isArray(record?.pedigree) ? record.pedigree : [];
  const sire = nodes.find(node => Number(node?.generation) === 1 && Number(node?.slot) === 0);
  const damsire = nodes.find(node => Number(node?.generation) === 2 && Number(node?.slot) === 2);
  return {
    sire_key: ancestorKey(sire),
    damsire_key: ancestorKey(damsire),
  };
}

export function pedigreeDistanceBand(distance) {
  const d = finite(distance);
  if (d == null) return null;
  if (d <= 1400) return "SPRINT_1400_OR_LESS";
  if (d <= 1800) return "MILE_1600_1800";
  if (d <= 2200) return "MIDDLE_2000_2200";
  return "LONG_2400_PLUS";
}

function conditionValues(race = {}) {
  return {
    surface: text(race.surface),
    venue: text(race.venue_code),
    course_layout: text(race.course_layout),
    going: text(race.track_condition),
    distance_band: pedigreeDistanceBand(race.distance_m),
    race_class: text(race.race_class_normalized ?? race.grade),
  };
}

function statKey(role, ancestorKeyValue, conditionType = "all", conditionValue = "all") {
  return [role, ancestorKeyValue, conditionType, conditionValue].join("|");
}

function writePayload(out, base, payload) {
  for (const [key, value] of Object.entries(payload)) out[`${base}_${key}`] = value;
}

export function createPedigreeFeatureState({ smallSamplePolicy = {} } = {}) {
  const policy = validateSmallSamplePolicy(smallSamplePolicy);
  const stats = new Map();
  const globalByRole = new Map([
    ["sire", blankOutcomeStats({ withMargin: true })],
    ["damsire", blankOutcomeStats({ withMargin: true })],
  ]);

  function get(role, ancestor, type = "all", value = "all") {
    if (!ancestor) return null;
    return stats.get(statKey(role, ancestor, type, value)) ?? null;
  }

  function ensure(role, ancestor, type = "all", value = "all") {
    const key = statKey(role, ancestor, type, value);
    let row = stats.get(key);
    if (!row) {
      row = blankOutcomeStats({ withMargin: true });
      stats.set(key, row);
    }
    return row;
  }

  function featuresFor(role, ancestor, race = {}) {
    const prefix = `ped_${role}`;
    const out = { [`${prefix}_known`]: ancestor ? 1 : 0 };
    const globalStats = globalByRole.get(role);
    const overall = get(role, ancestor);

    writePayload(out, `${prefix}_all`, effectiveOutcomeSnapshot({
      specific: overall,
      broader: null,
      globalStats,
      policy: { ...policy, minSpecificObservations: 1 },
      withMargin: true,
      allowFallback: true,
    }));

    for (const [type, value] of Object.entries(conditionValues(race))) {
      out[`${prefix}_${type}_condition_known`] = value == null ? 0 : 1;
      const specific = value == null ? null : get(role, ancestor, type, value);
      writePayload(out, `${prefix}_${type}`, effectiveOutcomeSnapshot({
        specific,
        broader: overall,
        globalStats,
        policy,
        withMargin: true,
        allowFallback: true,
      }));
    }
    return out;
  }

  function addAncestor(role, ancestor, race, result) {
    updateOutcomeStats(globalByRole.get(role), result, { withMargin: true });
    if (!ancestor) return;
    updateOutcomeStats(ensure(role, ancestor), result, { withMargin: true });
    for (const [type, value] of Object.entries(conditionValues(race))) {
      if (value == null) continue;
      updateOutcomeStats(ensure(role, ancestor, type, value), result, { withMargin: true });
    }
  }

  return {
    policy,
    snapshot(lineage, race) {
      return {
        ...featuresFor("sire", lineage?.sire_key ?? null, race),
        ...featuresFor("damsire", lineage?.damsire_key ?? null, race),
      };
    },
    add(lineage, race, result) {
      addAncestor("sire", lineage?.sire_key ?? null, race, result);
      addAncestor("damsire", lineage?.damsire_key ?? null, race, result);
    },
    stat_rows() {
      return stats.size;
    },
  };
}
