export const ML_DATASET_VERSION = 2;
export const ML_FEATURE_SCHEMA_VERSION = 3;
export const ML_LEAKAGE_POLICY = "STRICT_PRIOR_DATE_ONLY";

function finite(value) {
  if (value == null) return null;
  if (typeof value === "string" && value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function isEligibleStarter(entry, result) {
  const entryStatus = String(entry?.entry_status ?? "").toUpperCase();
  const resultStatus = String(result?.result_status ?? "").toUpperCase();
  if (entryStatus === "SCRATCHED" || entryStatus === "EXCLUDED") return false;
  if (resultStatus === "SCRATCHED" || resultStatus === "EXCLUDED") return false;
  return true;
}

function mean(values) {
  const clean = values.map(finite).filter(value => value != null);
  return clean.length ? clean.reduce((sum, value) => sum + value, 0) / clean.length : null;
}

function rate(items, predicate) {
  if (!items.length) return null;
  return items.filter(predicate).length / items.length;
}

function isoDate(value) {
  const text = String(value ?? "").slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : null;
}

function daysBetween(fromDate, toDate) {
  const from = isoDate(fromDate);
  const to = isoDate(toDate);
  if (!from || !to) return null;
  const delta = Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`);
  return Number.isFinite(delta) ? Math.round(delta / 86400000) : null;
}

function stableRaceDate(row) {
  return isoDate(row?.race?.actual_date ?? row?.race?.scheduled_date);
}

function resultMap(row) {
  return new Map((row?.results ?? []).map(result => [String(result?.horse_id ?? ""), result]));
}

function eligibleStarterIds(row, results) {
  return (row?.entries ?? [])
    .filter(entry => {
      const horseId = String(entry?.horse_id ?? "");
      const result = results.get(horseId) ?? null;
      return horseId && result && isEligibleStarter(entry, result);
    })
    .map(entry => String(entry.horse_id));
}

function opponentFieldSummary(opponentHorseIds, horseStatsById) {
  const known = (opponentHorseIds ?? [])
    .map(horseId => horseStatsById.get(String(horseId)))
    .filter(stats => stats && stats.starts > 0);

  if (!known.length) {
    return {
      known_count: 0,
      avg_starts: null,
      avg_win_rate: null,
      avg_top3_rate: null,
      max_win_rate: null,
    };
  }

  const winRates = known.map(stats => stats.wins / stats.starts);
  const top3Rates = known.map(stats => stats.top3 / stats.starts);
  return {
    known_count: known.length,
    avg_starts: mean(known.map(stats => stats.starts)),
    avg_win_rate: mean(winRates),
    avg_top3_rate: mean(top3Rates),
    max_win_rate: Math.max(...winRates),
  };
}

function opponentSnapshot(history, horseStatsById, limit) {
  const recent = history.slice(-limit).reverse();
  const summaries = recent.map(item => opponentFieldSummary(item.opponentHorseIds, horseStatsById));
  const measured = summaries.filter(summary => summary.known_count > 0);
  const previous = summaries[0] ?? null;

  return {
    opponent_previous_known_count: previous?.known_count ?? 0,
    opponent_previous_avg_starts: previous?.avg_starts ?? null,
    opponent_previous_avg_win_rate: previous?.avg_win_rate ?? null,
    opponent_previous_avg_top3_rate: previous?.avg_top3_rate ?? null,
    opponent_previous_max_win_rate: previous?.max_win_rate ?? null,
    opponent_recent_races_measured: measured.length,
    opponent_recent_avg_known_count: mean(measured.map(summary => summary.known_count)),
    opponent_recent_avg_starts: mean(measured.map(summary => summary.avg_starts)),
    opponent_recent_avg_win_rate: mean(measured.map(summary => summary.avg_win_rate)),
    opponent_recent_avg_top3_rate: mean(measured.map(summary => summary.avg_top3_rate)),
    opponent_recent_max_win_rate: measured.length
      ? Math.max(...measured.map(summary => summary.max_win_rate).filter(value => value != null))
      : null,
  };
}

const ELO_BASE = 1500;
const ELO_SCALE = 400;
const ELO_K = 24;

function currentStarterIds(row) {
  return (row?.entries ?? [])
    .filter(entry => {
      const status = String(entry?.entry_status ?? "").toUpperCase();
      return status !== "SCRATCHED" && status !== "EXCLUDED";
    })
    .map(entry => String(entry?.horse_id ?? ""))
    .filter(Boolean);
}

function eloState(eloByHorse, horseId) {
  return eloByHorse.get(String(horseId ?? "")) ?? { rating: ELO_BASE, starts: 0 };
}

function eloExpected(ratingA, ratingB) {
  return 1 / (1 + 10 ** ((ratingB - ratingA) / ELO_SCALE));
}

function networkSnapshot(row, horseId, eloByHorse) {
  const own = eloState(eloByHorse, horseId);
  const opponentIds = currentStarterIds(row).filter(id => id !== String(horseId));
  if (!opponentIds.length) {
    return {
      network_elo_rating: own.rating,
      network_elo_starts: own.starts,
      network_field_known_count: 0,
      network_field_avg_elo: null,
      network_field_max_elo: null,
      network_field_elo_spread: null,
      network_elo_vs_field_avg: null,
      network_expected_pairwise_score: null,
    };
  }

  const opponentStates = opponentIds.map(id => eloState(eloByHorse, id));
  const opponentRatings = opponentStates.map(state => state.rating);
  const avg = mean(opponentRatings);
  return {
    network_elo_rating: own.rating,
    network_elo_starts: own.starts,
    network_field_known_count: opponentStates.filter(state => state.starts > 0).length,
    network_field_avg_elo: avg,
    network_field_max_elo: Math.max(...opponentRatings),
    network_field_elo_spread: Math.max(...opponentRatings) - Math.min(...opponentRatings),
    network_elo_vs_field_avg: avg != null ? own.rating - avg : null,
    network_expected_pairwise_score: mean(
      opponentRatings.map(rating => eloExpected(own.rating, rating)),
    ),
  };
}

function raceEloUpdates(row, eloByHorse) {
  const results = resultMap(row);
  const runners = (row?.entries ?? [])
    .map(entry => {
      const horseId = String(entry?.horse_id ?? "");
      const result = results.get(horseId) ?? null;
      const finish = finite(result?.official_finish_position);
      if (!horseId || !result || !isEligibleStarter(entry, result)) return null;
      if (String(result?.result_status ?? "").toUpperCase() !== "FINISHED" || finish == null) return null;
      const state = eloState(eloByHorse, horseId);
      return { horseId, finish, rating: state.rating };
    })
    .filter(Boolean);

  if (runners.length < 2) return [];
  const deltas = new Map(runners.map(runner => [runner.horseId, 0]));
  const pairK = ELO_K / (runners.length - 1);

  for (let i = 0; i < runners.length; i += 1) {
    for (let j = i + 1; j < runners.length; j += 1) {
      const a = runners[i];
      const b = runners[j];
      const scoreA = a.finish < b.finish ? 1 : a.finish > b.finish ? 0 : 0.5;
      const expectedA = eloExpected(a.rating, b.rating);
      const delta = pairK * (scoreA - expectedA);
      deltas.set(a.horseId, deltas.get(a.horseId) + delta);
      deltas.set(b.horseId, deltas.get(b.horseId) - delta);
    }
  }

  return runners.map(runner => ({
    horseId: runner.horseId,
    delta: deltas.get(runner.horseId),
    starts: 1,
  }));
}

function updateHorseStats(horseStatsById, horseId, result) {
  const key = String(horseId ?? "");
  if (!key) return;
  const stats = horseStatsById.get(key) ?? { starts: 0, wins: 0, top3: 0 };
  stats.starts += 1;
  const finish = finite(result?.official_finish_position);
  if (finish === 1) stats.wins += 1;
  if (finish != null && finish <= 3) stats.top3 += 1;
  horseStatsById.set(key, stats);
}

function historySnapshot(history, current, limit, horseStatsById) {
  const recent = history.slice(-limit).reverse();
  const currentDistance = finite(current.race?.distance_m);
  const currentSurface = current.race?.surface ?? null;
  const currentVenue = current.race?.venue_code ?? null;
  const previous = recent[0] ?? null;

  const finished = recent.filter(item => item.result?.result_status === "FINISHED" && item.result?.official_finish_position != null);
  const sameSurface = finished.filter(item => item.race?.surface && item.race.surface === currentSurface);
  const sameVenue = finished.filter(item => item.race?.venue_code && item.race.venue_code === currentVenue);
  const sameDistance = finished.filter(item => finite(item.race?.distance_m) === currentDistance && currentDistance != null);

  const speeds = finished.map(item => {
    const distance = finite(item.race?.distance_m);
    const timeMs = finite(item.result?.finish_time_ms);
    return distance != null && timeMs != null && timeMs > 0 ? distance / (timeMs / 1000) : null;
  });

  return {
    prior_starts: history.length,
    recent_window_starts: recent.length,
    days_since_last_start: previous ? daysBetween(previous.date, current.date) : null,
    previous_finish_position: finite(previous?.result?.official_finish_position),
    previous_last_3f: finite(previous?.result?.last_3f),
    previous_distance_m: finite(previous?.race?.distance_m),
    distance_change_m: previous && currentDistance != null && finite(previous?.race?.distance_m) != null
      ? currentDistance - finite(previous.race.distance_m)
      : null,
    jockey_continues: previous?.entry?.jockey_id && current.entry?.jockey_id
      ? String(previous.entry.jockey_id) === String(current.entry.jockey_id)
      : null,
    recent_avg_finish: mean(finished.map(item => item.result?.official_finish_position)),
    recent_win_rate: rate(finished, item => Number(item.result?.official_finish_position) === 1),
    recent_top3_rate: rate(finished, item => Number(item.result?.official_finish_position) <= 3),
    recent_avg_last_3f: mean(finished.map(item => item.result?.last_3f)),
    recent_avg_speed_mps: mean(speeds),
    same_surface_starts: sameSurface.length,
    same_surface_top3_rate: rate(sameSurface, item => Number(item.result?.official_finish_position) <= 3),
    same_distance_starts: sameDistance.length,
    same_distance_top3_rate: rate(sameDistance, item => Number(item.result?.official_finish_position) <= 3),
    same_venue_starts: sameVenue.length,
    same_venue_top3_rate: rate(sameVenue, item => Number(item.result?.official_finish_position) <= 3),
    ...opponentSnapshot(history, horseStatsById, limit),
  };
}

function currentFeatures(row, entry, historyFeatures, networkFeatures) {
  const race = row.race ?? {};
  return {
    race_date: stableRaceDate(row),
    venue_code: race.venue_code ?? null,
    meeting_no: finite(race.meeting_no),
    meeting_day: finite(race.meeting_day),
    race_no: finite(race.race_no),
    discipline: race.discipline ?? null,
    surface: race.surface ?? null,
    distance_m: finite(race.distance_m),
    direction: race.direction ?? null,
    weather: race.weather ?? null,
    track_condition: race.track_condition ?? null,
    actual_start_time: race.actual_start_time ?? null,
    gate: finite(entry?.gate),
    horse_number: finite(entry?.horse_number),
    sex: entry?.sex ?? null,
    age: finite(entry?.age),
    carried_weight: finite(entry?.carried_weight),
    jockey_id: entry?.jockey_id ?? null,
    trainer_id: entry?.trainer_id ?? null,
    body_weight: finite(entry?.body_weight),
    body_weight_diff: finite(entry?.body_weight_diff),
    ...historyFeatures,
    ...networkFeatures,
  };
}

function targetFrom(result) {
  const finish = finite(result?.official_finish_position);
  return {
    result_status: result?.result_status ?? null,
    finish_position: finish,
    is_win: finish === 1,
    is_top3: finish != null ? finish <= 3 : null,
    finish_time_ms: finite(result?.finish_time_ms),
    margin_raw: result?.margin_raw ?? null,
    last_3f: finite(result?.last_3f),
    prize_money: finite(result?.prize_money),
  };
}

function marketOutcomeFrom(result) {
  return {
    final_win_odds: finite(result?.win_odds),
    final_popularity: finite(result?.popularity),
  };
}

export function buildRaceOutcomes(raceRows, {
  startDate = null,
  endDate = null,
} = {}) {
  return (raceRows ?? [])
    .map(row => ({ row, date: stableRaceDate(row) }))
    .filter(item => item.date)
    .filter(item => (!startDate || item.date >= startDate) && (!endDate || item.date <= endDate))
    .sort((a, b) => a.date.localeCompare(b.date) || String(a.row?.race?.race_id ?? "").localeCompare(String(b.row?.race?.race_id ?? "")))
    .map(({ row, date }) => ({
      ml_dataset_version: ML_DATASET_VERSION,
      race_id: String(row?.race?.race_id ?? ""),
      race_date: date,
      payouts: (row?.payouts ?? []).map(payout => ({
        bet_type: payout?.bet_type ?? null,
        combination: payout?.combination ?? null,
        payout_yen: finite(payout?.payout_yen),
        popularity: finite(payout?.popularity),
      })),
    }))
    .filter(row => row.race_id);
}

export function buildMlDataset(raceRows, {
  startDate = null,
  endDate = null,
  historyLimit = 5,
} = {}) {
  if (!Number.isInteger(historyLimit) || historyLimit < 1 || historyLimit > 100) {
    throw new Error("historyLimit must be an integer from 1 to 100");
  }

  const normalized = (raceRows ?? [])
    .map(row => ({ row, date: stableRaceDate(row) }))
    .filter(item => item.date)
    .sort((a, b) => a.date.localeCompare(b.date) || String(a.row?.race?.race_id ?? "").localeCompare(String(b.row?.race?.race_id ?? "")));

  const historyByHorse = new Map();
  const horseStatsById = new Map();
  const eloByHorse = new Map();
  const out = [];
  let index = 0;

  while (index < normalized.length) {
    const date = normalized[index].date;
    const day = [];
    while (index < normalized.length && normalized[index].date === date) {
      day.push(normalized[index].row);
      index += 1;
    }

    // Generate every row for the day before committing any same-day result to history.
    // This guarantees a strict date-level anti-leakage boundary even when exact post times
    // are missing or historical schedules have been repaired.
    for (const row of day) {
      const raceId = String(row?.race?.race_id ?? "");
      const results = resultMap(row);
      for (const entry of row?.entries ?? []) {
        const horseId = String(entry?.horse_id ?? "");
        if (!raceId || !horseId) continue;
        const result = results.get(horseId) ?? null;
        if (!result || !isEligibleStarter(entry, result)) continue;
        const history = historyByHorse.get(horseId) ?? [];
        const current = { date, race: row.race ?? {}, entry };
        const features = currentFeatures(
          row,
          entry,
          historySnapshot(history, current, historyLimit, horseStatsById),
          networkSnapshot(row, horseId, eloByHorse),
        );
        const inRange = (!startDate || date >= startDate) && (!endDate || date <= endDate);
        if (inRange) {
          out.push({
            ml_dataset_version: ML_DATASET_VERSION,
            feature_schema_version: ML_FEATURE_SCHEMA_VERSION,
            leakage_policy: ML_LEAKAGE_POLICY,
            race_id: raceId,
            horse_id: horseId,
            features,
            target: targetFrom(result),
            market_outcome: marketOutcomeFrom(result),
          });
        }
      }
    }

    for (const row of day) {
      const results = resultMap(row);
      const starterIds = eligibleStarterIds(row, results);
      for (const entry of row?.entries ?? []) {
        const horseId = String(entry?.horse_id ?? "");
        const result = results.get(horseId) ?? null;
        if (!horseId || !result || !isEligibleStarter(entry, result)) continue;
        const history = historyByHorse.get(horseId) ?? [];
        history.push({
          date,
          race: row.race ?? {},
          entry,
          result,
          opponentHorseIds: starterIds.filter(id => id !== horseId),
        });
        historyByHorse.set(horseId, history);
      }
    }

    // Calculate every same-day Elo delta from the day-start ratings, then apply them together.
    // This preserves STRICT_PRIOR_DATE_ONLY for the network features as well.
    const eloUpdates = [];
    for (const row of day) {
      eloUpdates.push(...raceEloUpdates(row, eloByHorse));
    }
    const eloDeltaByHorse = new Map();
    const eloStartsByHorse = new Map();
    for (const update of eloUpdates) {
      eloDeltaByHorse.set(
        update.horseId,
        (eloDeltaByHorse.get(update.horseId) ?? 0) + update.delta,
      );
      eloStartsByHorse.set(
        update.horseId,
        (eloStartsByHorse.get(update.horseId) ?? 0) + update.starts,
      );
    }
    for (const [horseId, delta] of eloDeltaByHorse) {
      const state = eloState(eloByHorse, horseId);
      eloByHorse.set(horseId, {
        rating: state.rating + delta,
        starts: state.starts + (eloStartsByHorse.get(horseId) ?? 0),
      });
    }

    // Commit daily performance stats only after every feature row for the day was generated.
    // This keeps same-day races outside the opponent-strength view.
    for (const row of day) {
      const results = resultMap(row);
      for (const entry of row?.entries ?? []) {
        const horseId = String(entry?.horse_id ?? "");
        const result = results.get(horseId) ?? null;
        if (!horseId || !result || !isEligibleStarter(entry, result)) continue;
        updateHorseStats(horseStatsById, horseId, result);
      }
    }
  }

  return out;
}
