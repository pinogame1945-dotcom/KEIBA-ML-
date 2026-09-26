export const AUTO_FEATURE_FACTORY_VERSION = 1;

const FORBIDDEN_SOURCE_PATTERNS = [
  /^target[._]/i,
  /^market[._]/i,
  /final_win_odds/i,
  /final_popularity/i,
  /payout/i,
];

function finite(value) {
  if (value == null) return null;
  if (typeof value === "string" && value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function mean(values) {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const m = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[m] : (sorted[m - 1] + sorted[m]) / 2;
}

function variance(values) {
  if (!values.length) return null;
  const m = mean(values);
  return mean(values.map(v => (v - m) ** 2));
}

function slope(points) {
  if (points.length < 2) return null;
  const xs = points.map(p => p.index);
  const ys = points.map(p => p.value);
  const xm = mean(xs);
  const ym = mean(ys);
  const denom = xs.reduce((s, x) => s + (x - xm) ** 2, 0);
  if (!denom) return 0;
  return points.reduce((s, p) => s + (p.index - xm) * (p.value - ym), 0) / denom;
}

function recentMean(values, n) {
  const xs = values.slice(-n);
  return xs.length ? mean(xs) : null;
}

function ewma(values, alpha) {
  if (!values.length) return null;
  let out = values[0];
  for (let i = 1; i < values.length; i += 1) out = alpha * values[i] + (1 - alpha) * out;
  return out;
}

function recencyWeightedMean(values) {
  if (!values.length) return null;
  let weighted = 0;
  let weights = 0;
  for (let i = 0; i < values.length; i += 1) {
    const w = i + 1;
    weighted += values[i] * w;
    weights += w;
  }
  return weighted / weights;
}

export function assertSafeFactorySource(featureName) {
  const name = String(featureName ?? "");
  if (!name) throw new Error("feature name is required");
  if (FORBIDDEN_SOURCE_PATTERNS.some(re => re.test(name))) {
    throw new Error(`forbidden AUTO FEATURE source: ${name}`);
  }
  return name;
}

export function summarizeNumericSeries(rawValues, {
  higherIsBetter = null,
  ewmaAlpha = 0.5,
} = {}) {
  if (!(ewmaAlpha > 0 && ewmaAlpha <= 1)) throw new Error("ewmaAlpha must be > 0 and <= 1");

  const points = (rawValues ?? []).map((raw, index) => ({ index, value: finite(raw) }));
  const observedPoints = points.filter(p => p.value != null);
  const values = observedPoints.map(p => p.value);

  if (!values.length) {
    return {
      observation_count: 0,
      missing_count: points.length,
      mean: null,
      median: null,
      min: null,
      max: null,
      stddev: null,
      variance: null,
      coefficient_of_variation: null,
      last: null,
      previous_delta: null,
      rolling_2_mean: null,
      rolling_3_mean: null,
      rolling_5_mean: null,
      ewma: null,
      linear_trend: null,
      recency_weighted_mean: null,
      gap_to_best: null,
      gap_to_worst: null,
    };
  }

  const avg = mean(values);
  const varValue = variance(values);
  const last = values.at(-1);
  const previous = values.length >= 2 ? values.at(-2) : null;
  const min = Math.min(...values);
  const max = Math.max(...values);

  let best = null;
  let worst = null;
  if (higherIsBetter === true) {
    best = max;
    worst = min;
  } else if (higherIsBetter === false) {
    best = min;
    worst = max;
  }

  return {
    observation_count: values.length,
    missing_count: points.length - values.length,
    mean: avg,
    median: median(values),
    min,
    max,
    stddev: varValue == null ? null : Math.sqrt(varValue),
    variance: varValue,
    coefficient_of_variation: avg ? Math.sqrt(varValue) / Math.abs(avg) : null,
    last,
    previous_delta: previous == null ? null : last - previous,
    rolling_2_mean: recentMean(values, 2),
    rolling_3_mean: recentMean(values, 3),
    rolling_5_mean: recentMean(values, 5),
    ewma: ewma(values, ewmaAlpha),
    linear_trend: slope(observedPoints),
    recency_weighted_mean: recencyWeightedMean(values),
    gap_to_best: best == null ? null : last - best,
    gap_to_worst: worst == null ? null : last - worst,
  };
}

export function buildRollingFeatureFamily(prefix, rawValues, options = {}) {
  const safe = assertSafeFactorySource(prefix);
  const summary = summarizeNumericSeries(rawValues, options);
  return Object.fromEntries(
    Object.entries(summary).map(([key, value]) => [`auto_${safe}_${key}`, value]),
  );
}

function normalized(value) {
  return value == null ? null : String(value).trim().toUpperCase() || null;
}

function same(a, b) {
  const aa = normalized(a);
  const bb = normalized(b);
  return aa != null && bb != null && aa === bb;
}

export function conditionMatches(priorRace, currentRace) {
  const priorDistance = finite(priorRace?.distance_m);
  const currentDistance = finite(currentRace?.distance_m);
  const distanceGap = priorDistance != null && currentDistance != null
    ? Math.abs(priorDistance - currentDistance)
    : null;

  return {
    same_venue: same(priorRace?.venue_code, currentRace?.venue_code),
    same_surface: same(priorRace?.surface, currentRace?.surface),
    same_distance: distanceGap === 0,
    within_100m: distanceGap != null && distanceGap <= 100,
    within_200m: distanceGap != null && distanceGap <= 200,
    within_300m: distanceGap != null && distanceGap <= 300,
    within_400m: distanceGap != null && distanceGap <= 400,
    same_direction: same(priorRace?.direction, currentRace?.direction),
    same_course_layout: same(priorRace?.course_layout, currentRace?.course_layout),
    same_going: same(priorRace?.track_condition, currentRace?.track_condition),
    same_class: same(priorRace?.race_class_normalized, currentRace?.race_class_normalized),
    same_age_condition:
      same(priorRace?.age_condition_raw, currentRace?.age_condition_raw) ||
      (
        finite(priorRace?.age_min) === finite(currentRace?.age_min) &&
        finite(priorRace?.age_max) === finite(currentRace?.age_max) &&
        (finite(priorRace?.age_min) != null || finite(priorRace?.age_max) != null)
      ),
    same_weight_rule: same(priorRace?.weight_rule, currentRace?.weight_rule),
  };
}

export function buildConditionFeatureFamily({
  prefix,
  history,
  currentRace,
  valueOf,
  successOf = null,
  higherIsBetter = null,
}) {
  assertSafeFactorySource(prefix);
  if (typeof valueOf !== "function") throw new Error("valueOf must be a function");
  if (successOf != null && typeof successOf !== "function") throw new Error("successOf must be a function");

  const buckets = {
    same_venue: [],
    same_surface: [],
    same_distance: [],
    within_100m: [],
    within_200m: [],
    within_300m: [],
    within_400m: [],
    same_direction: [],
    same_course_layout: [],
    same_going: [],
    same_class: [],
    same_age_condition: [],
    same_weight_rule: [],
  };

  for (const item of history ?? []) {
    const flags = conditionMatches(item?.race ?? {}, currentRace ?? {});
    for (const [condition, matched] of Object.entries(flags)) {
      if (matched) buckets[condition].push(item);
    }
  }

  const out = {};
  for (const [condition, items] of Object.entries(buckets)) {
    const values = items.map(valueOf);
    const summary = summarizeNumericSeries(values, { higherIsBetter });
    const base = `auto_${prefix}_${condition}`;
    out[`${base}_starts`] = items.length;
    out[`${base}_observations`] = summary.observation_count;
    out[`${base}_mean`] = summary.mean;
    out[`${base}_median`] = summary.median;
    out[`${base}_trend`] = summary.linear_trend;
    if (successOf) {
      const measured = items.map(successOf).filter(v => v === true || v === false);
      out[`${base}_success_rate`] = measured.length
        ? measured.filter(Boolean).length / measured.length
        : null;
      out[`${base}_success_observations`] = measured.length;
    }
  }
  return out;
}

function percentile(values, value, direction) {
  if (!values.length) return null;
  if (values.length === 1) return 1;
  const betterOrEqual = values.filter(v =>
    direction === "HIGHER_BETTER" ? v <= value : v >= value
  ).length;
  return (betterOrEqual - 1) / (values.length - 1);
}

function orientedRank(values, value, direction) {
  const sorted = [...values].sort((a, b) =>
    direction === "HIGHER_BETTER" ? b - a : a - b
  );
  return sorted.indexOf(value) + 1;
}

export function addFieldRelativeFeatures(rows, specs) {
  const output = (rows ?? []).map(row => ({
    ...row,
    features: { ...(row?.features ?? {}) },
  }));

  for (const spec of specs ?? []) {
    const key = assertSafeFactorySource(spec?.key);
    const prefix = String(spec?.prefix ?? key).replace(/[^a-zA-Z0-9_]/g, "_");
    const direction = spec?.direction;
    if (!["HIGHER_BETTER", "LOWER_BETTER"].includes(direction)) {
      throw new Error(`${key}: direction must be HIGHER_BETTER or LOWER_BETTER`);
    }

    const values = output.map(row => finite(row.features[key]));
    const field = values.filter(v => v != null);
    if (!field.length) continue;
    const avg = mean(field);
    const med = median(field);
    const varValue = variance(field);
    const sd = varValue == null ? null : Math.sqrt(varValue);
    const oriented = [...field].sort((a, b) =>
      direction === "HIGHER_BETTER" ? b - a : a - b
    );
    const best = oriented[0];
    const top2 = mean(oriented.slice(0, 2));
    const top3 = mean(oriented.slice(0, 3));

    for (let i = 0; i < output.length; i += 1) {
      const value = values[i];
      if (value == null) continue;
      const p = `auto_field_${prefix}`;
      output[i].features[`${p}_diff_mean`] = value - avg;
      output[i].features[`${p}_diff_median`] = value - med;
      output[i].features[`${p}_zscore`] = sd && sd > 0 ? (value - avg) / sd : 0;
      output[i].features[`${p}_percentile`] = percentile(field, value, direction);
      output[i].features[`${p}_rank`] = orientedRank(field, value, direction);
      output[i].features[`${p}_gap_to_best`] = value - best;
      output[i].features[`${p}_gap_to_top2_mean`] = value - top2;
      output[i].features[`${p}_gap_to_top3_mean`] = value - top3;
      output[i].features[`${p}_field_count`] = field.length;
    }
  }
  return output;
}

export function buildSafePairFeatures(features, pairSpecs) {
  const out = {};
  for (const spec of pairSpecs ?? []) {
    const aKey = assertSafeFactorySource(spec?.a);
    const bKey = assertSafeFactorySource(spec?.b);
    const prefix = String(spec?.prefix ?? `${aKey}_vs_${bKey}`).replace(/[^a-zA-Z0-9_]/g, "_");
    const a = finite(features?.[aKey]);
    const b = finite(features?.[bKey]);
    if (a == null || b == null) continue;

    for (const op of spec?.ops ?? ["diff"]) {
      if (op === "diff") out[`auto_pair_${prefix}_diff`] = a - b;
      else if (op === "ratio") out[`auto_pair_${prefix}_ratio`] = b === 0 ? null : a / b;
      else if (op === "normalized_diff") {
        const denom = Math.abs(a) + Math.abs(b);
        out[`auto_pair_${prefix}_normalized_diff`] = denom ? (a - b) / denom : 0;
      } else {
        throw new Error(`unsupported pair operation: ${op}`);
      }
    }
  }
  return out;
}
