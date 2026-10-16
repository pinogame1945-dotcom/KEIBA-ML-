# LIVE P1 - netkeiba horse_id and JRA official horse_no

Date: 2026-10-16. Code added to isolated branch. **No real GitHub-runner network collection yet.** No L1, L2, L3, MOBILE, or main changes.

## Sources

- netkeiba PRE-RACE entry page: https://race.netkeiba.com/race/shutuba.html?race_id=202605040611
- JRA formally published, numbered card for 2026-10-17 Tokyo 11R: https://www.jra.go.jp/JRADB/accessD.html?CNAME=pw01dde0105202604061120261017%2F22
- BACKFILL reads horse IDs from the actual netkeiba /horse/ID/ links on result pages. This probe extracts the same style of IDs from pre-race entry rows. Do not guess horse IDs.
- JRA odds MUST be sourced from JRA, not netkeiba. This P1 version only probes the official odds navigation; **no prices are yet fetched**.

## Official JRA pre-race entry list, Fuji Stakes, 14 horses

| JRA No | Horse | netkeiba horse_id (individually verified profile) |
|---:|---|---|
| 1 | ファンダム | [2022105836](https://db.netkeiba.com/horse/2022105836/) |
| 2 | ミトノオルフェ | [2022104053](https://db.netkeiba.com/horse/2022104053/) |
| 3 | シンフォーエバー | [2022110114](https://db.netkeiba.com/horse/2022110114/) |
| 4 | トロヴァトーレ | [2021105557](https://db.netkeiba.com/horse/2021105557/) |
| 5 | アドマイヤズーム | [2022105396](https://db.netkeiba.com/horse/2022105396/) |
| 6 | デビットバローズ | [2019104878](https://db.netkeiba.com/horse/2019104878/) |
| 7 | ウォーターリヒト | [2021100953](https://db.netkeiba.com/horse/2021100953/) |
| 8 | マジックサンズ | [2022104772](https://db.netkeiba.com/horse/2022104772/) |
| 9 | キープカルム | [2021105119](https://db.netkeiba.com/horse/2021105119/) |
| 10 | アンパドゥ | [2021110030](https://db.netkeiba.com/horse/2021110030/) |
| 11 | ファーヴェント | [2021104893](https://db.netkeiba.com/horse/2021104893/) |
| 12 | エンペラーズソード | [2022106691](https://db.netkeiba.com/horse/2022106691/) |
| 13 | シュガークン | [2021102224](https://db.netkeiba.com/horse/2021102224/) |
| 14 | エコロアルバ | [2023103184](https://db.netkeiba.com/horse/2023103184/) |

This table comes from public JRA page text checked 2026-10-16. Each horse_id above was independently checked against the publicly accessible netkeiba horse profile URL; JRA's pre-race numbered card provides the horse numbers. This is a MANUAL public-page crosswalk, NOT evidence that the GitHub runner already fetched or matched all 14 records.

## Safety gates

- Validate that the real published JRA CNAME identifies the SAME 12-digit race ID as netkeiba. Verify JRA HTML date, track, meeting number/day, race number.
- Allow per-race name matching only if all horse names match exactly and uniquely across both official and netkeiba lists. Keep netkeiba stable horse_id and JRA official horse_no/gate; never infer from order.
- A missing JRA number returns WAIT_OFFICIAL_NUMBERING. Missing horse ID, count mismatch, a conflicting horse number, duplicate, or a different race gives BLOCKED. Required expected headcount prevents quietly dropping runners.
- Fail immediately on 403 / 429 / 404. Backoff only on temporary communication or 502/503/504. Do not bypass access restrictions.
- Output JSON has prebet_odds_complete=false. Nothing is wagered, no odds from netkeiba, no race outcomes, no paid resources, no persistent raw HTML/Actions artifacts.

## Test

Install one source-parser dependency and run local fixture regression:

    python -m pip install -r scripts/requirements-live-source-probe-v1.txt
    python -m unittest discover -s tests -p 'test_live_race_source_probe_v1.py' -v

Read-only PUBLIC internet probe (to be run on a free standard runner after review):

    python scripts/live_race_source_probe_v1.py \
      --race-id 202605040611 --expected 14 \
      --jra-card-url 'https://www.jra.go.jp/JRADB/accessD.html?CNAME=pw01dde0105202604061120261017%2F22'

Manual-only workflow at .github/workflows/live-source-crosswalk-p1.yml. Actions workflow_dispatch generally requires the workflow to be present in the repository default branch first. This branch has not been merged and its workflow has not been executed.

## Remaining work

1. Live runner validation of all 14 source URLs and crosswalks (do not report fixture tests as production source success).
2. Discover real JRA entry URLs via links, not guessed CNAMEs. Handle race cancellation and changed meeting schedules.
3. Implement live JRA odds POST navigation, all WIN / QUINELLA / TRIO / TRIFECTA prices, each actual fetch time, full combination coverage and missing/invalid sale statuses. No netkeiba odds fallback.
4. Race-day changes: cancellations, body weight, jockey changes, venue turf/dirt/weather, source publication timestamps.
5. Verify rate limits / service terms and stop on restrictions. No artificial high-frequency retries.
