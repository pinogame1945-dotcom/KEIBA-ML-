export const ACTOR_FEATURE_BUILDER_VERSION = 1;

function finite(value) {
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

function distanceBand(distance) {
  const d = finite(distance);
  if (d == null) return null;
  if (d <= 1400) return "SPRINT_1400_OR_LESS";
  if (d <= 1800) return "MILE_1600_1800";
  if (d <= 2200) return "MIDDLE_2000_2200";
  return "LONG_2400_PLUS";
}

function conditionKeys(race = {}) {
  const out = [];
  if (text(race.venue_code)) out.push(["venue", text(race.venue_code)]);
  if (text(race.surface)) out.push(["surface", text(race.surface)]);
  const band = distanceBand(race.distance_m);
  if (band) out.push(["distance_band", band]);
  return out;
}

function blankStats() {
  return { starts: 0, wins: 0, top3: 0, finish_sum: 0, finish_n: 0 };
}

function addResult(stats, result) {
  if (String(result?.result_status ?? "").toUpperCase() !== "FINISHED") return false;
  const finish = finite(result?.official_finish_position);
  if (finish == null || finish < 1) return false;
  stats.starts += 1;
  if (finish === 1) stats.wins += 1;
  if (finish <= 3) stats.top3 += 1;
  stats.finish_sum += finish;
  stats.finish_n += 1;
  return true;
}

function snapshot(stats) {
  if (!stats) return { starts: 0, win_rate: null, top3_rate: null, avg_finish: null };
  return {
    starts: stats.starts,
    win_rate: stats.starts ? stats.wins / stats.starts : null,
    top3_rate: stats.starts ? stats.top3 / stats.starts : null,
    avg_finish: stats.finish_n ? stats.finish_sum / stats.finish_n : null,
  };
}

function statKey(role, key, conditionType = "all", conditionValue = "all") {
  return [role, key, conditionType, conditionValue].join("|");
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

export function createActorFeatureState() {
  const stats = new Map();

  function featuresFor(role, key, race = {}, useConditions = true) {
    const prefix = `actor_${role}`;
    if (!key) return { [`${prefix}_known`]: 0, [`${prefix}_all_starts`]: 0 };
    const out = { [`${prefix}_known`]: 1 };
    for (const [name, value] of Object.entries(snapshot(stats.get(statKey(role, key))))) {
      out[`${prefix}_all_${name}`] = value;
    }
    if (useConditions) {
      for (const [type, value] of conditionKeys(race)) {
        const safe = String(value).toLowerCase().replace(/[^a-z0-9]+/g, "_");
        for (const [name, metric] of Object.entries(snapshot(stats.get(statKey(role, key, type, value))))) {
          out[`${prefix}_${type}_${safe}_${name}`] = metric;
        }
      }
    }
    return out;
  }

  function add(role, key, race, result, useConditions = true) {
    if (!key) return;
    const keys = [[role, key, "all", "all"]];
    if (useConditions) {
      keys.push(...conditionKeys(race).map(([type, value]) => [role, key, type, value]));
    }
    for (const [r, k, type, value] of keys) {
      const mapKey = statKey(r, k, type, value);
      let row = stats.get(mapKey);
      if (!row) {
        row = blankStats();
        stats.set(mapKey, row);
      }
      addResult(row, result);
    }
  }

  return {
    snapshot(entry, race) {
      const identity = actorIdentityFromEntry(entry);
      return {
        ...featuresFor("jockey", identity.jockey_key, race, true),
        ...featuresFor("trainer", identity.trainer_key, race, true),
        ...featuresFor("horse_jockey", identity.horse_jockey_key, race, false),
      };
    },
    add(entry, race, result) {
      const identity = actorIdentityFromEntry(entry);
      add("jockey", identity.jockey_key, race, result, true);
      add("trainer", identity.trainer_key, race, result, true);
      add("horse_jockey", identity.horse_jockey_key, race, result, false);
    },
    stat_rows() {
      return stats.size;
    },
  };
}
