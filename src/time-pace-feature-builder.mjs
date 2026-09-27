export const TIME_PACE_FEATURE_BUILDER_VERSION = 1;

function finite(value) {
  if (value == null || (typeof value === "string" && value.trim() === "")) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function mean(values) {
  const clean = (values ?? []).map(finite).filter(v => v != null);
  return clean.length ? clean.reduce((a, b) => a + b, 0) / clean.length : null;
}

function distanceBand(distance) {
  const d = finite(distance);
  if (d == null) return "UNKNOWN";
  if (d <= 1400) return "SPRINT";
  if (d <= 1800) return "MILE";
  if (d <= 2200) return "MIDDLE";
  return "LONG";
}

function text(value, fallback = "UNKNOWN") {
  const s = value == null ? "" : String(value).trim().toUpperCase();
  return s || fallback;
}

function conditionKeys(race = {}) {
  const venue = text(race.venue_code);
  const surface = text(race.surface);
  const distance = finite(race.distance_m);
  const going = text(race.track_condition);
  const layout = text(race.course_layout);
  const band = distanceBand(distance);
  return [
    `EXACT|${venue}|${surface}|${distance ?? "UNKNOWN"}|${going}|${layout}`,
    `BAND|${surface}|${band}`,
    `SURFACE|${surface}`,
    "GLOBAL",
  ];
}

function blankMetric() {
  return { n: 0, sum: 0, sumsq: 0 };
}

function addMetric(metric, value) {
  const n = finite(value);
  if (n == null) return;
  metric.n += 1;
  metric.sum += n;
  metric.sumsq += n * n;
}

function metricSnapshot(metric) {
  if (!metric || metric.n <= 0) return { n: 0, mean: null, sd: null };
  const avg = metric.sum / metric.n;
  const variance = Math.max(0, metric.sumsq / metric.n - avg * avg);
  return { n: metric.n, mean: avg, sd: metric.n >= 2 ? Math.sqrt(variance) : null };
}

function blankStandard() {
  return {
    finish_time_ms: blankMetric(),
    last3f: blankMetric(),
    early_pace: blankMetric(),
    late_pace: blankMetric(),
    pace_delta: blankMetric(),
  };
}

function standardSnapshot(row) {
  return {
    finish_time_ms: metricSnapshot(row?.finish_time_ms),
    last3f: metricSnapshot(row?.last3f),
    early_pace: metricSnapshot(row?.early_pace),
    late_pace: metricSnapshot(row?.late_pace),
    pace_delta: metricSnapshot(row?.pace_delta),
  };
}

function lapSummary(laps) {
  const values = (laps ?? []).map(x => finite(x?.lap_seconds)).filter(v => v != null && v > 0);
  if (values.length < 2) return { early: null, late: null, delta: null };
  const split = Math.max(1, Math.floor(values.length / 2));
  const early = mean(values.slice(0, split));
  const late = mean(values.slice(split));
  return {
    early,
    late,
    delta: early != null && late != null ? early - late : null,
  };
}

function scoreLowerBetter(value, metric) {
  const v = finite(value);
  if (v == null || !metric || metric.n < 2 || metric.mean == null || !metric.sd) return null;
  return (metric.mean - v) / metric.sd;
}

function paceClass(earlyScore, threshold) {
  const z = finite(earlyScore);
  if (z == null) return null;
  if (z >= threshold) return "FAST";
  if (z <= -threshold) return "SLOW";
  return "EVEN";
}

function resultMap(row) {
  return new Map((row?.results ?? []).map(r => [String(r?.horse_id ?? ""), r]));
}

function eligible(entry, result) {
  const es = String(entry?.entry_status ?? "").toUpperCase();
  const rs = String(result?.result_status ?? "").toUpperCase();
  return !["SCRATCHED", "EXCLUDED"].includes(es)
    && rs === "FINISHED"
    && finite(result?.official_finish_position) != null;
}

function summarizeHorseHistory(history, limit) {
  const rows = (history ?? []).slice(-limit);
  const classRows = cls => rows.filter(row => row.pace_class === cls);
  const top3Rate = xs => xs.length ? xs.filter(x => x.finish <= 3).length / xs.length : null;
  const avg = (xs, key) => mean(xs.map(x => x[key]));
  const out = {
    timepace_recent_races: rows.length,
    timepace_normalized_time_observations: rows.filter(x => x.normalized_time != null).length,
    timepace_normalized_last3f_observations: rows.filter(x => x.normalized_last3f != null).length,
    timepace_recent_avg_normalized_time: avg(rows, "normalized_time"),
    timepace_recent_avg_normalized_last3f: avg(rows, "normalized_last3f"),
    timepace_recent_avg_performance_score: avg(rows, "performance_score"),
    timepace_recent_avg_standard_fallback_level: avg(rows, "standard_fallback_level"),
  };
  for (const cls of ["FAST", "EVEN", "SLOW"]) {
    const xs = classRows(cls);
    const key = cls.toLowerCase();
    out[`timepace_${key}_pace_starts`] = xs.length;
    out[`timepace_${key}_pace_top3_rate`] = top3Rate(xs);
    out[`timepace_${key}_pace_avg_performance_score`] = avg(xs, "performance_score");
  }
  return out;
}

export function createTimePaceFeatureState({
  historyLimit = 10,
  minStandardObservations = 20,
  paceClassThreshold = 0.5,
} = {}) {
  if (!Number.isInteger(historyLimit) || historyLimit < 1) throw new Error("historyLimit must be a positive integer");
  if (!Number.isInteger(minStandardObservations) || minStandardObservations < 1) {
    throw new Error("minStandardObservations must be a positive integer");
  }
  const standards = new Map();
  const historyByHorse = new Map();

  function chooseStandard(race) {
    const keys = conditionKeys(race);
    let fallback = 0;
    let bestAvailable = null;
    for (const key of keys) {
      const row = standards.get(key);
      const snap = standardSnapshot(row);
      const usable = Math.max(
        snap.finish_time_ms.n,
        snap.last3f.n,
        snap.early_pace.n,
        snap.late_pace.n,
      );
      if (usable >= minStandardObservations) return { key, fallbackLevel: fallback, snapshot: snap };
      if (!bestAvailable && usable >= 2) bestAvailable = { key, fallbackLevel: fallback, snapshot: snap };
      fallback += 1;
    }
    return bestAvailable ?? { key: null, fallbackLevel: 4, snapshot: standardSnapshot(null) };
  }

  function snapshot(horseId) {
    return summarizeHorseHistory(historyByHorse.get(String(horseId ?? "")) ?? [], historyLimit);
  }

  function evaluateRace(row) {
    const standard = chooseStandard(row?.race ?? {});
    const laps = lapSummary(row?.laps ?? []);
    const earlyScore = scoreLowerBetter(laps.early, standard.snapshot.early_pace);
    const cls = paceClass(earlyScore, paceClassThreshold);
    const results = resultMap(row);
    const records = [];
    const standardUpdates = [];

    for (const entry of row?.entries ?? []) {
      const horseId = String(entry?.horse_id ?? "");
      const result = results.get(horseId);
      if (!horseId || !eligible(entry, result)) continue;
      const normalizedTime = scoreLowerBetter(result?.finish_time_ms, standard.snapshot.finish_time_ms);
      const normalizedLast3f = scoreLowerBetter(result?.last_3f, standard.snapshot.last3f);
      records.push({
        horseId,
        finish: finite(result.official_finish_position),
        normalized_time: normalizedTime,
        normalized_last3f: normalizedLast3f,
        performance_score: mean([normalizedTime, normalizedLast3f]),
        pace_class: cls,
        standard_fallback_level: standard.fallbackLevel,
      });
      standardUpdates.push({
        finish_time_ms: finite(result?.finish_time_ms),
        last3f: finite(result?.last_3f),
      });
    }

    return {
      race: row?.race ?? {},
      pace: laps,
      records,
      standard_updates: standardUpdates,
    };
  }

  function commitRaceEvaluation(evaluation) {
    for (const record of evaluation?.records ?? []) {
      const history = historyByHorse.get(record.horseId) ?? [];
      history.push(record);
      if (history.length > historyLimit * 3) history.splice(0, history.length - historyLimit * 3);
      historyByHorse.set(record.horseId, history);
    }

    for (const key of conditionKeys(evaluation?.race ?? {})) {
      let row = standards.get(key);
      if (!row) {
        row = blankStandard();
        standards.set(key, row);
      }
      for (const update of evaluation?.standard_updates ?? []) {
        addMetric(row.finish_time_ms, update.finish_time_ms);
        addMetric(row.last3f, update.last3f);
      }
      addMetric(row.early_pace, evaluation?.pace?.early);
      addMetric(row.late_pace, evaluation?.pace?.late);
      addMetric(row.pace_delta, evaluation?.pace?.delta);
    }
  }

  return {
    snapshot,
    evaluateRace,
    commitRaceEvaluation,
    debugStandard(race) {
      return chooseStandard(race);
    },
  };
}
