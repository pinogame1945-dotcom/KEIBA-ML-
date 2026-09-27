export const PEDIGREE_FEATURE_BUILDER_VERSION = 2;

function finite(value) {
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
  const surface = text(race.surface);
  const venue = text(race.venue_code);
  const layout = text(race.course_layout);
  const going = text(race.track_condition);
  const band = distanceBand(race.distance_m);
  if (surface) out.push(["surface", surface]);
  if (venue) out.push(["venue", venue]);
  if (layout) out.push(["course_layout", layout]);
  if (going) out.push(["going", going]);
  if (band) out.push(["distance_band", band]);
  return out;
}

function blankStats() {
  return {
    starts: 0,
    wins: 0,
    top3: 0,
    finish_sum: 0,
    finish_n: 0,
    margin_lengths_sum: 0,
    margin_lengths_n: 0,
  };
}

function updateStats(stats, result) {
  if (String(result?.result_status ?? "").toUpperCase() !== "FINISHED") return;
  const finish = finite(result?.official_finish_position);
  if (finish == null || finish < 1) return;
  stats.starts += 1;
  if (finish != null) {
    if (finish === 1) stats.wins += 1;
    if (finish <= 3) stats.top3 += 1;
    stats.finish_sum += finish;
    stats.finish_n += 1;
  }
  if (String(result?.margin_type ?? "").toUpperCase() === "LENGTHS") {
    const margin = finite(result?.margin_lengths);
    if (margin != null) {
      stats.margin_lengths_sum += margin;
      stats.margin_lengths_n += 1;
    }
  }
}

function snapshot(stats) {
  if (!stats) {
    return {
      starts: 0,
      win_rate: null,
      top3_rate: null,
      avg_finish: null,
      avg_margin_lengths: null,
    };
  }
  return {
    starts: stats.starts,
    win_rate: stats.starts ? stats.wins / stats.starts : null,
    top3_rate: stats.starts ? stats.top3 / stats.starts : null,
    avg_finish: stats.finish_n ? stats.finish_sum / stats.finish_n : null,
    avg_margin_lengths: stats.margin_lengths_n
      ? stats.margin_lengths_sum / stats.margin_lengths_n
      : null,
  };
}

function statKey(role, ancestorId, conditionType = "all", conditionValue = "all") {
  return [role, ancestorId, conditionType, conditionValue].join("|");
}

export function createPedigreeFeatureState() {
  const stats = new Map();

  function featuresFor(role, ancestorId, race = {}) {
    const prefix = `ped_${role}`;
    if (!ancestorId) {
      return {
        [`${prefix}_known`]: 0,
        [`${prefix}_all_starts`]: 0,
      };
    }

    const out = { [`${prefix}_known`]: 1 };
    const overall = snapshot(stats.get(statKey(role, ancestorId)));
    for (const [k, v] of Object.entries(overall)) out[`${prefix}_all_${k}`] = v;

    for (const [conditionType, conditionValue] of conditionKeys(race)) {
      const snap = snapshot(stats.get(statKey(role, ancestorId, conditionType, conditionValue)));
      const safeValue = String(conditionValue).toLowerCase().replace(/[^a-z0-9]+/g, "_");
      const base = `${prefix}_${conditionType}_${safeValue}`;
      for (const [k, v] of Object.entries(snap)) out[`${base}_${k}`] = v;
    }
    return out;
  }

  function add(role, ancestorId, race, result) {
    if (!ancestorId) return;
    const keys = [[role, ancestorId, "all", "all"], ...conditionKeys(race).map(([t, v]) => [role, ancestorId, t, v])];
    for (const [r, id, type, value] of keys) {
      const key = statKey(r, id, type, value);
      let row = stats.get(key);
      if (!row) {
        row = blankStats();
        stats.set(key, row);
      }
      updateStats(row, result);
    }
  }

  return {
    snapshot(lineage, race) {
      return {
        ...featuresFor("sire", lineage?.sire_key, race),
        ...featuresFor("damsire", lineage?.damsire_key, race),
      };
    },
    add(lineage, race, result) {
      add("sire", lineage?.sire_key, race, result);
      add("damsire", lineage?.damsire_key, race, result);
    },
    stat_rows() {
      return stats.size;
    },
  };
}
