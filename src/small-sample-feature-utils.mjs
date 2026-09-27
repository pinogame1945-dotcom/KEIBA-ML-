export const SMALL_SAMPLE_FEATURE_UTILS_VERSION = 1;

export const DEFAULT_SMALL_SAMPLE_POLICY = Object.freeze({
  ratePriorStrength: 20,
  meanPriorStrength: 10,
  minSpecificObservations: 5,
  actorRecentWindow: 30,
});

function finite(value) {
  if (value == null || (typeof value === "string" && value.trim() === "")) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export function validateSmallSamplePolicy(policy = {}) {
  const merged = { ...DEFAULT_SMALL_SAMPLE_POLICY, ...(policy ?? {}) };
  for (const key of ["ratePriorStrength", "meanPriorStrength"]) {
    const n = finite(merged[key]);
    if (n == null || n < 0) throw new Error(`${key} must be >= 0`);
    merged[key] = n;
  }
  for (const key of ["minSpecificObservations", "actorRecentWindow"]) {
    const n = Number(merged[key]);
    if (!Number.isInteger(n) || n < 1) throw new Error(`${key} must be a positive integer`);
    merged[key] = n;
  }
  return merged;
}

export function blankOutcomeStats({ withMargin = false } = {}) {
  return {
    starts: 0,
    wins: 0,
    top3: 0,
    finish_sum: 0,
    finish_n: 0,
    ...(withMargin ? { margin_sum: 0, margin_n: 0 } : {}),
  };
}

export function updateOutcomeStats(stats, result, { withMargin = false } = {}) {
  if (String(result?.result_status ?? "").toUpperCase() !== "FINISHED") return false;
  const finish = finite(result?.official_finish_position);
  if (finish == null || finish < 1) return false;
  stats.starts += 1;
  if (finish === 1) stats.wins += 1;
  if (finish <= 3) stats.top3 += 1;
  stats.finish_sum += finish;
  stats.finish_n += 1;
  if (withMargin && String(result?.margin_type ?? "").toUpperCase() === "LENGTHS") {
    const margin = finite(result?.margin_lengths);
    if (margin != null) {
      stats.margin_sum += margin;
      stats.margin_n += 1;
    }
  }
  return true;
}

export function rawOutcomeSnapshot(stats, { withMargin = false } = {}) {
  const s = stats ?? blankOutcomeStats({ withMargin });
  return {
    starts: Number(s.starts ?? 0),
    win_observations: Number(s.starts ?? 0),
    top3_observations: Number(s.starts ?? 0),
    finish_observations: Number(s.finish_n ?? 0),
    win_rate: s.starts ? s.wins / s.starts : null,
    top3_rate: s.starts ? s.top3 / s.starts : null,
    avg_finish: s.finish_n ? s.finish_sum / s.finish_n : null,
    ...(withMargin ? {
      margin_observations: Number(s.margin_n ?? 0),
      avg_margin_lengths: s.margin_n ? s.margin_sum / s.margin_n : null,
    } : {}),
  };
}

export function shrinkRate(successes, observations, priorRate, priorStrength) {
  const n = finite(observations);
  const s = finite(successes);
  const p = finite(priorRate);
  const k = finite(priorStrength);
  if (n == null || s == null || n < 0 || s < 0 || s > n) return null;
  if (n === 0 && (p == null || k === 0)) return null;
  if (p == null || k === 0) return n ? s / n : null;
  return (s + k * p) / (n + k);
}

export function shrinkMean(sum, observations, priorMean, priorStrength) {
  const n = finite(observations);
  const total = finite(sum);
  const p = finite(priorMean);
  const k = finite(priorStrength);
  if (n == null || total == null || n < 0) return null;
  if (n === 0 && (p == null || k === 0)) return null;
  if (p == null || k === 0) return n ? total / n : null;
  return (total + k * p) / (n + k);
}

export function shrunkOutcomeSnapshot(stats, priorStats, policy = {}, { withMargin = false } = {}) {
  const cfg = validateSmallSamplePolicy(policy);
  const s = stats ?? blankOutcomeStats({ withMargin });
  const prior = rawOutcomeSnapshot(priorStats, { withMargin });
  return {
    win_rate_shrunk: shrinkRate(s.wins ?? 0, s.starts ?? 0, prior.win_rate, cfg.ratePriorStrength),
    top3_rate_shrunk: shrinkRate(s.top3 ?? 0, s.starts ?? 0, prior.top3_rate, cfg.ratePriorStrength),
    avg_finish_shrunk: shrinkMean(s.finish_sum ?? 0, s.finish_n ?? 0, prior.avg_finish, cfg.meanPriorStrength),
    ...(withMargin ? {
      avg_margin_lengths_shrunk: shrinkMean(
        s.margin_sum ?? 0,
        s.margin_n ?? 0,
        prior.avg_margin_lengths,
        cfg.meanPriorStrength,
      ),
    } : {}),
  };
}

export function chooseFallbackStats(specific, broader, globalStats, policy = {}) {
  const cfg = validateSmallSamplePolicy(policy);
  if ((specific?.starts ?? 0) >= cfg.minSpecificObservations) {
    return { stats: specific, fallbackLevel: 0, fallbackSource: "SPECIFIC" };
  }
  if ((broader?.starts ?? 0) > 0) {
    return { stats: broader, fallbackLevel: 1, fallbackSource: "ENTITY_OVERALL" };
  }
  if ((globalStats?.starts ?? 0) > 0) {
    return { stats: globalStats, fallbackLevel: 2, fallbackSource: "GLOBAL" };
  }
  return { stats: null, fallbackLevel: 3, fallbackSource: "NONE" };
}

export function effectiveOutcomeSnapshot({
  specific,
  broader = null,
  globalStats = null,
  policy = {},
  withMargin = false,
  allowFallback = true,
} = {}) {
  const cfg = validateSmallSamplePolicy(policy);
  const raw = rawOutcomeSnapshot(specific, { withMargin });
  const chosen = allowFallback
    ? chooseFallbackStats(specific, broader, globalStats, cfg)
    : { stats: specific, fallbackLevel: 0, fallbackSource: "SPECIFIC" };
  const selected = chosen.stats;
  const selectedRaw = rawOutcomeSnapshot(selected, { withMargin });
  const prior = selected === globalStats ? null : globalStats;
  const shrunk = shrunkOutcomeSnapshot(selected, prior, cfg, { withMargin });
  return {
    ...raw,
    fallback_level: chosen.fallbackLevel,
    effective_starts: selectedRaw.starts,
    effective_win_rate: shrunk.win_rate_shrunk,
    effective_top3_rate: shrunk.top3_rate_shrunk,
    effective_avg_finish: shrunk.avg_finish_shrunk,
    ...(withMargin ? {
      effective_margin_observations: selectedRaw.margin_observations,
      effective_avg_margin_lengths: shrunk.avg_margin_lengths_shrunk,
    } : {}),
  };
}
