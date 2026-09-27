import {
  blankOutcomeStats,
  effectiveOutcomeSnapshot,
  updateOutcomeStats,
  validateSmallSamplePolicy,
} from "./small-sample-feature-utils.mjs";

export const ACTOR_FEATURE_BUILDER_VERSION = 2;

function finite(value) {
  if (value == null || (typeof value === "string" && value.trim() === "")) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function text(value) {
  const s = value == null ? "" : String(value).trim();
  return s || null;
}

function actorKey(id, name) {
  const rawId = text(id);
  if (rawId && rawId !== "000") return `id:${rawId}`;
  const rawName = text(name)?.normalize("NFKC").toLowerCase();
  return rawName ? `name:${rawName}` : null;
}

export function actorDistanceBand(distance) {
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
    distance_band: actorDistanceBand(race.distance_m),
    race_class: text(race.race_class_normalized ?? race.grade),
  };
}

function statKey(role, key, type = "all", value = "all") {
  return [role, key, type, value].join("|");
}

function recentKey(role, key) {
  return [role, key].join("|");
}

function writePayload(out, base, payload) {
  for (const [key, value] of Object.entries(payload)) out[`${base}_${key}`] = value;
}

function statsFromRecent(results) {
  const stats = blankOutcomeStats();
  for (const result of results ?? []) updateOutcomeStats(stats, result);
  return stats;
}

export function actorIdentityFromEntry(entry = {}) {
  const horseId = text(entry.horse_id);
  const jockeyKey = actorKey(entry.jockey_id, entry.jockey_name);
  const trainerKey = actorKey(entry.trainer_id, entry.trainer_name);
  return {
    horse_key: horseId ? `horse:${horseId}` : null,
    jockey_key: jockeyKey,
    trainer_key: trainerKey,
    horse_jockey_key: horseId && jockeyKey ? `horse:${horseId}|${jockeyKey}` : null,
  };
}

export function createActorFeatureState({ smallSamplePolicy = {} } = {}) {
  const policy = validateSmallSamplePolicy(smallSamplePolicy);
  const stats = new Map();
  const recent = new Map();
  const globalByRole = new Map([
    ["jockey", blankOutcomeStats()],
    ["trainer", blankOutcomeStats()],
    ["horse_jockey", blankOutcomeStats()],
  ]);

  function get(role, key, type = "all", value = "all") {
    if (!key) return null;
    return stats.get(statKey(role, key, type, value)) ?? null;
  }

  function ensure(role, key, type = "all", value = "all") {
    const mapKey = statKey(role, key, type, value);
    let row = stats.get(mapKey);
    if (!row) {
      row = blankOutcomeStats();
      stats.set(mapKey, row);
    }
    return row;
  }

  function addRecent(role, key, result) {
    if (!key || String(result?.result_status ?? "").toUpperCase() !== "FINISHED") return;
    const finish = finite(result?.official_finish_position);
    if (finish == null || finish < 1) return;
    const mapKey = recentKey(role, key);
    const rows = recent.get(mapKey) ?? [];
    rows.push({
      result_status: "FINISHED",
      official_finish_position: finish,
    });
    if (rows.length > policy.actorRecentWindow) rows.splice(0, rows.length - policy.actorRecentWindow);
    recent.set(mapKey, rows);
  }

  function featuresFor(role, key, race = {}, { conditions = true, includeRecent = true } = {}) {
    const prefix = `actor_${role}`;
    const out = { [`${prefix}_known`]: key ? 1 : 0 };
    const globalStats = globalByRole.get(role);
    const overall = get(role, key);

    writePayload(out, `${prefix}_all`, effectiveOutcomeSnapshot({
      specific: overall,
      broader: null,
      globalStats,
      policy: { ...policy, minSpecificObservations: 1 },
      allowFallback: true,
    }));

    if (includeRecent) {
      const recentStats = statsFromRecent(key ? recent.get(recentKey(role, key)) ?? [] : []);
      writePayload(out, `${prefix}_recent`, effectiveOutcomeSnapshot({
        specific: recentStats,
        broader: overall,
        globalStats,
        policy,
        allowFallback: true,
      }));
    }

    if (conditions) {
      for (const [type, value] of Object.entries(conditionValues(race))) {
        out[`${prefix}_${type}_condition_known`] = value == null ? 0 : 1;
        const specific = value == null ? null : get(role, key, type, value);
        writePayload(out, `${prefix}_${type}`, effectiveOutcomeSnapshot({
          specific,
          broader: overall,
          globalStats,
          policy,
          allowFallback: true,
        }));
      }
    }
    return out;
  }

  function addRole(role, key, race, result, { conditions = true, includeRecent = true } = {}) {
    updateOutcomeStats(globalByRole.get(role), result);
    if (!key) return;
    updateOutcomeStats(ensure(role, key), result);
    if (includeRecent) addRecent(role, key, result);
    if (conditions) {
      for (const [type, value] of Object.entries(conditionValues(race))) {
        if (value == null) continue;
        updateOutcomeStats(ensure(role, key, type, value), result);
      }
    }
  }

  return {
    policy,
    snapshot(entry, race) {
      const identity = actorIdentityFromEntry(entry);
      return {
        ...featuresFor("jockey", identity.jockey_key, race, { conditions: true, includeRecent: true }),
        ...featuresFor("trainer", identity.trainer_key, race, { conditions: true, includeRecent: true }),
        ...featuresFor("horse_jockey", identity.horse_jockey_key, race, { conditions: false, includeRecent: true }),
      };
    },
    add(entry, race, result) {
      const identity = actorIdentityFromEntry(entry);
      addRole("jockey", identity.jockey_key, race, result, { conditions: true, includeRecent: true });
      addRole("trainer", identity.trainer_key, race, result, { conditions: true, includeRecent: true });
      addRole("horse_jockey", identity.horse_jockey_key, race, result, { conditions: false, includeRecent: true });
    },
    stat_rows() {
      return stats.size;
    },
  };
}
