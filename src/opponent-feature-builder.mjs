export const OPPONENT_FEATURE_BUILDER_VERSION = 1;

const ELO_BASE = 1500;
const ELO_SCALE = 400;
const ELO_K = 24;

function finite(value) {
  if (value == null || (typeof value === "string" && value.trim() === "")) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function mean(values) {
  const clean = (values ?? []).map(finite).filter(v => v != null);
  return clean.length ? clean.reduce((a, b) => a + b, 0) / clean.length : null;
}

function rate(values) {
  const clean = (values ?? []).filter(v => v === true || v === false);
  return clean.length ? clean.filter(Boolean).length / clean.length : null;
}

function resultMap(row) {
  return new Map((row?.results ?? []).map(result => [String(result?.horse_id ?? ""), result]));
}

function eligible(entry, result) {
  if (!entry || !result) return false;
  const es = String(entry.entry_status ?? "").toUpperCase();
  const rs = String(result.result_status ?? "").toUpperCase();
  if (["SCRATCHED", "EXCLUDED"].includes(es) || ["SCRATCHED", "EXCLUDED"].includes(rs)) return false;
  return rs === "FINISHED" && finite(result.official_finish_position) != null;
}

function eloState(map, horseId) {
  return map.get(String(horseId ?? "")) ?? { rating: ELO_BASE, starts: 0 };
}

function expected(a, b) {
  return 1 / (1 + 10 ** ((b - a) / ELO_SCALE));
}

function smallGap(result, maxLengths) {
  const type = String(result?.margin_type ?? "").toUpperCase();
  if (["DEAD_HEAT", "NOSE", "HEAD", "NECK"].includes(type)) return true;
  if (type === "LENGTHS") {
    const n = finite(result?.margin_lengths);
    return n == null ? null : n <= maxLengths;
  }
  return null;
}

function horsePerformanceStats(map, horseId) {
  return map.get(String(horseId ?? "")) ?? { starts: 0, wins: 0, top3: 0 };
}

function performanceRates(stats) {
  return {
    win_rate: stats.starts ? stats.wins / stats.starts : null,
    top3_rate: stats.starts ? stats.top3 / stats.starts : null,
  };
}

function eloUpdates(runners) {
  if (runners.length < 2) return [];
  const deltas = new Map(runners.map(r => [r.horseId, 0]));
  const pairK = ELO_K / (runners.length - 1);
  for (let i = 0; i < runners.length; i += 1) {
    for (let j = i + 1; j < runners.length; j += 1) {
      const a = runners[i];
      const b = runners[j];
      const scoreA = a.finish < b.finish ? 1 : a.finish > b.finish ? 0 : 0.5;
      const delta = pairK * (scoreA - expected(a.rating, b.rating));
      deltas.set(a.horseId, deltas.get(a.horseId) + delta);
      deltas.set(b.horseId, deltas.get(b.horseId) - delta);
    }
  }
  return runners.map(r => ({ horseId: r.horseId, delta: deltas.get(r.horseId), starts: 1 }));
}

export function createOpponentFeatureState({
  historyLimit = 10,
  strongEloThreshold = 1525,
  closeMarginLengths = 0.5,
} = {}) {
  if (!Number.isInteger(historyLimit) || historyLimit < 1) throw new Error("historyLimit must be a positive integer");
  if (!Number.isFinite(strongEloThreshold)) throw new Error("strongEloThreshold must be finite");
  const historyByHorse = new Map();
  const eloByHorse = new Map();
  const horseStatsById = new Map();

  function currentField(row, horseId) {
    const ids = (row?.entries ?? [])
      .filter(entry => {
        const status = String(entry?.entry_status ?? "").toUpperCase();
        return !["SCRATCHED", "EXCLUDED"].includes(status);
      })
      .map(entry => String(entry?.horse_id ?? ""))
      .filter(id => id && id !== String(horseId));
    const states = ids.map(id => eloState(eloByHorse, id));
    const ratings = states.map(s => s.rating);
    return {
      ids,
      known_count: states.filter(s => s.starts > 0).length,
      avg_elo: mean(ratings),
      max_elo: ratings.length ? Math.max(...ratings) : null,
      spread: ratings.length ? Math.max(...ratings) - Math.min(...ratings) : null,
    };
  }

  function snapshot(row, entry) {
    const horseId = String(entry?.horse_id ?? "");
    const prior = (historyByHorse.get(horseId) ?? []).slice(-historyLimit);
    const current = currentField(row, horseId);
    const strongFaced = prior.reduce((sum, item) => sum + item.strong_opponents_faced, 0);
    const strongBeaten = prior.reduce((sum, item) => sum + item.strong_opponents_beaten, 0);
    const strongField = prior.filter(item => item.strong_field);
    const measuredField = prior.filter(item => item.field_avg_elo != null);
    return {
      opponent_relationship_races_measured: prior.length,
      opponent_history_avg_field_elo: mean(measuredField.map(item => item.field_avg_elo)),
      opponent_history_max_field_elo: measuredField.length
        ? Math.max(...measuredField.map(item => item.field_max_elo).filter(v => v != null))
        : null,
      opponent_history_avg_opponent_win_rate: mean(prior.map(item => item.avg_opponent_win_rate)),
      opponent_history_avg_opponent_top3_rate: mean(prior.map(item => item.avg_opponent_top3_rate)),
      opponent_strong_opponents_faced: strongFaced,
      opponent_strong_opponents_beaten: strongBeaten,
      opponent_strong_opponent_beat_rate: strongFaced ? strongBeaten / strongFaced : null,
      opponent_strong_field_races: strongField.length,
      opponent_strong_field_top3_rate: rate(strongField.map(item => item.top3)),
      opponent_close_finish_vs_strong_field_rate: rate(strongField.map(item => item.close_finish)),
      opponent_current_field_known_count: current.known_count,
      opponent_current_field_avg_elo: current.avg_elo,
      opponent_current_field_max_elo: current.max_elo,
      opponent_current_field_elo_spread: current.spread,
    };
  }

  function evaluateRace(row) {
    const results = resultMap(row);
    const runners = (row?.entries ?? []).map(entry => {
      const horseId = String(entry?.horse_id ?? "");
      const result = results.get(horseId);
      if (!horseId || !eligible(entry, result)) return null;
      return {
        horseId,
        entry,
        result,
        finish: finite(result.official_finish_position),
        rating: eloState(eloByHorse, horseId).rating,
      };
    }).filter(Boolean);

    const records = [];
    for (const runner of runners) {
      const opponents = runners.filter(other => other.horseId !== runner.horseId);
      const opponentRatings = opponents.map(other => other.rating);
      const strong = opponents.filter(other => other.rating >= strongEloThreshold);
      const oppRates = opponents.map(other => performanceRates(horsePerformanceStats(horseStatsById, other.horseId)));
      const fieldAvg = mean(opponentRatings);
      const strongField = fieldAvg != null && fieldAvg >= strongEloThreshold;
      const closeGap = smallGap(runner.result, closeMarginLengths);
      records.push({
        horseId: runner.horseId,
        field_avg_elo: fieldAvg,
        field_max_elo: opponentRatings.length ? Math.max(...opponentRatings) : null,
        avg_opponent_win_rate: mean(oppRates.map(x => x.win_rate)),
        avg_opponent_top3_rate: mean(oppRates.map(x => x.top3_rate)),
        strong_opponents_faced: strong.length,
        strong_opponents_beaten: strong.filter(other => runner.finish < other.finish).length,
        strong_field: strongField,
        top3: runner.finish <= 3,
        close_finish: strongField ? (runner.finish <= 3 || closeGap === true) : null,
      });
    }

    return {
      records,
      elo_updates: eloUpdates(runners),
      results: runners.map(r => ({ horseId: r.horseId, finish: r.finish })),
    };
  }

  function commitRaceEvaluation(evaluation) {
    for (const record of evaluation?.records ?? []) {
      const history = historyByHorse.get(record.horseId) ?? [];
      history.push(record);
      if (history.length > historyLimit * 3) history.splice(0, history.length - historyLimit * 3);
      historyByHorse.set(record.horseId, history);
    }
    for (const update of evaluation?.elo_updates ?? []) {
      const state = eloState(eloByHorse, update.horseId);
      eloByHorse.set(update.horseId, {
        rating: state.rating + update.delta,
        starts: state.starts + update.starts,
      });
    }
    for (const item of evaluation?.results ?? []) {
      const stats = horsePerformanceStats(horseStatsById, item.horseId);
      stats.starts += 1;
      if (item.finish === 1) stats.wins += 1;
      if (item.finish <= 3) stats.top3 += 1;
      horseStatsById.set(item.horseId, stats);
    }
  }

  return {
    snapshot,
    evaluateRace,
    commitRaceEvaluation,
    debugHorseElo(horseId) {
      return { ...eloState(eloByHorse, horseId) };
    },
  };
}
