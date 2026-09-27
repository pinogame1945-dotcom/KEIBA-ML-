function finite(value) {
  if (value == null) return null;
  if (typeof value === "string" && value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function mean(values) {
  const clean = values.map(finite).filter(v => v != null);
  return clean.length ? clean.reduce((a, b) => a + b, 0) / clean.length : null;
}

function rate(items, predicate) {
  return items.length ? items.filter(predicate).length / items.length : null;
}

function raceDate(row) {
  const value = row?.race?.actual_date ?? row?.race?.scheduled_date ?? "";
  const text = String(value).slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : null;
}

function resultMap(row) {
  return new Map((row?.results ?? []).map(r => [String(r?.horse_id ?? ""), r]));
}

function eligible(entry, result) {
  if (!entry || !result) return false;
  const es = String(entry.entry_status ?? "").toUpperCase();
  const rs = String(result.result_status ?? "").toUpperCase();
  return !["SCRATCHED", "EXCLUDED"].includes(es) && !["SCRATCHED", "EXCLUDED"].includes(rs);
}

function lapSummary(laps) {
  const values = (laps ?? []).map(x => finite(x?.lap_seconds)).filter(v => v != null && v > 0);
  if (!values.length) return null;
  const first = values.slice(0, Math.min(3, values.length));
  const last = values.slice(Math.max(0, values.length - 3));
  const avg = mean(values);
  const variance = mean(values.map(v => (v - avg) ** 2));
  return {
    first3: mean(first),
    last3: mean(last),
    last1: values.at(-1),
    variance,
    delta_last3_first3: mean(last) - mean(first),
  };
}

function parseCorners(raw, fieldSize) {
  const positions = String(raw ?? "").match(/\d+/g)?.map(Number).filter(Number.isFinite) ?? [];
  if (!positions.length) return null;
  const first = positions[0];
  const last = positions.at(-1);
  const size = finite(fieldSize);
  return {
    first,
    last,
    gain: first - last,
    first_ratio: size && size > 0 ? first / size : null,
    last_ratio: size && size > 0 ? last / size : null,
    front: first <= 3,
    finished_front: last <= 3,
    improved: last < first,
  };
}

function historicalExtras(history, currentDistance, windows) {
  const styleRecent = history.slice(-windows.style_last3f).reverse();
  const suitabilityRecent = history.slice(-windows.suitability).reverse();
  const finished = suitabilityRecent.filter(x =>
    String(x.result?.result_status ?? "").toUpperCase() === "FINISHED" &&
    finite(x.result?.official_finish_position) != null
  );

  const lapItems = styleRecent.map(x => lapSummary(x.laps)).filter(Boolean);
  const prevLap = lapItems[0] ?? null;
  const styleItems = styleRecent.map(x => parseCorners(x.result?.corner_raw, x.fieldSize)).filter(Boolean);
  const prevStyle = styleItems[0] ?? null;

  const current = finite(currentDistance);
  const sameBand = finished.filter(x => {
    const d = finite(x.race?.distance_m);
    return d != null && current != null && Math.abs(d - current) <= 200;
  });
  const shorter = finished.filter(x => {
    const d = finite(x.race?.distance_m);
    return d != null && current != null && d < current;
  });
  const longer = finished.filter(x => {
    const d = finite(x.race?.distance_m);
    return d != null && current != null && d > current;
  });
  const distances = finished.map(x => finite(x.race?.distance_m)).filter(v => v != null);

  return {
    lap_previous_first3_avg: prevLap?.first3 ?? null,
    lap_previous_last3_avg: prevLap?.last3 ?? null,
    lap_previous_last1: prevLap?.last1 ?? null,
    lap_previous_variance: prevLap?.variance ?? null,
    lap_previous_delta_last3_first3: prevLap?.delta_last3_first3 ?? null,
    lap_recent_races_measured: lapItems.length,
    lap_recent_avg_first3: mean(lapItems.map(x => x.first3)),
    lap_recent_avg_last3: mean(lapItems.map(x => x.last3)),
    lap_recent_avg_last1: mean(lapItems.map(x => x.last1)),
    lap_recent_avg_variance: mean(lapItems.map(x => x.variance)),
    lap_recent_avg_delta_last3_first3: mean(lapItems.map(x => x.delta_last3_first3)),

    style_previous_first_position: prevStyle?.first ?? null,
    style_previous_last_position: prevStyle?.last ?? null,
    style_previous_position_gain: prevStyle?.gain ?? null,
    style_previous_first_ratio: prevStyle?.first_ratio ?? null,
    style_previous_last_ratio: prevStyle?.last_ratio ?? null,
    style_recent_races_measured: styleItems.length,
    style_recent_avg_first_position: mean(styleItems.map(x => x.first)),
    style_recent_avg_last_position: mean(styleItems.map(x => x.last)),
    style_recent_avg_position_gain: mean(styleItems.map(x => x.gain)),
    style_recent_avg_first_ratio: mean(styleItems.map(x => x.first_ratio)),
    style_recent_avg_last_ratio: mean(styleItems.map(x => x.last_ratio)),
    style_recent_front_rate: rate(styleItems, x => x.front),
    style_recent_finish_front_rate: rate(styleItems, x => x.finished_front),
    style_recent_improve_rate: rate(styleItems, x => x.improved),

    distx_recent_avg_distance_m: mean(distances),
    distx_current_minus_recent_avg_m:
      current != null && distances.length ? current - mean(distances) : null,
    distx_same_band_starts: sameBand.length,
    distx_same_band_top3_rate: rate(sameBand, x => Number(x.result.official_finish_position) <= 3),
    distx_shorter_starts: shorter.length,
    distx_shorter_top3_rate: rate(shorter, x => Number(x.result.official_finish_position) <= 3),
    distx_longer_starts: longer.length,
    distx_longer_top3_rate: rate(longer, x => Number(x.result.official_finish_position) <= 3),
  };
}

export function createHistoricalExtraProcessor(historyWindows) {
  const windows = historyWindows ?? {};
  const retain = Math.max(
    1,
    Number(windows.style_last3f) || 1,
    Number(windows.suitability) || 1,
  );
  const historyByHorse = new Map();

  function processDay(dayRows, dateValue = null, {
    startDate = null,
    endDate = null,
  } = {}) {
    const day = [...(dayRows ?? [])]
      .filter(row => raceDate(row))
      .sort((a, b) => String(a?.race?.race_id ?? "").localeCompare(String(b?.race?.race_id ?? "")));
    if (!day.length) return new Map();
    const date = dateValue ?? raceDate(day[0]);
    for (const row of day) {
      if (raceDate(row) !== date) throw new Error("historical extra day contains mixed dates");
    }

    const extras = new Map();
    const emit = (!startDate || date >= startDate) && (!endDate || date <= endDate);
    if (emit) {
      for (const row of day) {
        const raceId = String(row?.race?.race_id ?? "");
        const results = resultMap(row);
        for (const entry of row?.entries ?? []) {
          const horseId = String(entry?.horse_id ?? "");
          const result = results.get(horseId);
          if (!raceId || !horseId || !eligible(entry, result)) continue;
          extras.set(
            raceId + "|" + horseId,
            historicalExtras(historyByHorse.get(horseId) ?? [], row?.race?.distance_m, windows),
          );
        }
      }
    }

    for (const row of day) {
      const results = resultMap(row);
      const fieldSize = (row?.entries ?? []).filter(entry => {
        const result = results.get(String(entry?.horse_id ?? ""));
        return eligible(entry, result);
      }).length;
      for (const entry of row?.entries ?? []) {
        const horseId = String(entry?.horse_id ?? "");
        const result = results.get(horseId);
        if (!horseId || !eligible(entry, result)) continue;
        const history = historyByHorse.get(horseId) ?? [];
        history.push({
          race: row.race ?? {},
          result,
          laps: row.laps ?? [],
          fieldSize,
        });
        if (history.length > retain) history.splice(0, history.length - retain);
        historyByHorse.set(horseId, history);
      }
    }
    return extras;
  }

  return {
    retainedHistoryLimit: retain,
    processDay,
    debugStateSizes() {
      return { history_horses: historyByHorse.size };
    },
  };
}
