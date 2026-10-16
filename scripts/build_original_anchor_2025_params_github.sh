#!/usr/bin/env bash
set -euo pipefail
# Generate original prior-only ANCHOR_FULL 2025 model, proven against saved FINAL OOS.
# NO live odds, paid artifact, new model training, 2026 result or postrace bet claim.
out="${1:?usage: bash scripts/build_original_anchor_2025_params_github.sh OUT_DIR}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
if [ -n "${RUNNER_TEMP:-}" ]; then
  free_kb="$(df -Pk "$RUNNER_TEMP" | tail -1 | awk '{print $4}')"
  test "$free_kb" -gt 1800000 || { echo "INSUFFICIENT_EPHEMERAL_SPACE" >&2; exit 3; }
fi
fetch() {
  python3 scripts/github-readonly-archive.py --repo pinogame1945-dotcom/KEIBA-ML- \
    --ref freeze/l2.1-standard50-to-l3-20261015 \
    --path "$1" --dest "$out/$2" --max-bytes "$3"
}
fetch research-cache/market-fixed-anchor-residual-v2/fair-base-bf811fa2eab73db0.csv.gz fair-base.csv.gz 19000000
fetch research-results/market-fixed-anchor-residual-v2/run-38659856085/oos-fair-value-predictions.csv.gz saved-oos.csv.gz 11000000
fetch research-results/market-fixed-anchor-residual-v2/run-38659856085/residual-coefficients.csv coefficients.csv 20000
fetch research-results/market-fixed-anchor-residual-v2/run-38659856085/lambda-selection.csv lambda.csv 15000
python3 research/replay-original-anchor-coefficients-2025.py \
  --cache "$out/fair-base.csv.gz" --coefficients "$out/coefficients.csv" \
  --lambda-selection "$out/lambda.csv" --predictions "$out/saved-oos.csv.gz" \
  --race 202507010212 --out "$out/original-prior-2025-parity.json" \
  --export-parameters "$out/original-prior-anchor-asof-params.json"
python3 - "$out/original-prior-anchor-asof-params.json" "$out/original-prior-2025-parity.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
proof=json.load(open(sys.argv[2],encoding="utf-8"))
assert p["trained_only_on_years"]==[2022,2023,2024]
assert p["variant"]=="ANCHOR_FULL" and len(p["coefficients"])==19
assert set(p["coefficients"])==set(p["standardizer"])
assert p["not_promoted_or_authenticated_for_live_betting"] is True
assert proof["original_anchored_probs_match"] is True
assert proof["prediction_prebet_time_verified"] is False
print("ORIGINAL_ANCHOR_PRIOR_PARAMETERS_RECOVERED",sys.argv[1],flush=True)
PY
