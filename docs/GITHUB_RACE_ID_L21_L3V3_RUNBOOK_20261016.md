# KEIBA-ML GitHub-only Race ID → 凍結L2.1 → 正式L3 V3 実行手順（2026-10-16）

## 採用済みの不変条件

- 入力はnetkeiba形式の **12桁レースID**（1件/複数）。JRAや研究データの取得・AI推論・L2.1・L3・JSON公開はすべて**GitHub側**。スマホは完成JSONを受け取り表示するだけ。スマホ側によるJRA収集やAI処理への逆送を必須としない。
- 元L2.1 STANDARD **50 line×gate**、9元買い方、元 `ANCHOR_FULL` 軸選定を変更しない。保存済み7王やOutsiderの比較順位で `AI_RAW` や軸馬を補間しない。
- L3 V3の正式配分は **REVERSE_P3_RACE_MAX300**。1券100〜300円、100円刻み、通常 `min(200円×券数, 指定レース予算, 指定レース上限)`。最低100円×券数が不可能なら全件停止。券種、馬番、組み合わせ・購入点数を変更しない。
- 同一日複数レースは、その入力済みレース分の掛け金合計を**当日払戻前に**資金確保。足りない場合はその日分全件停止。リクエストに含まれる一部レースが欠損している場合、他のレースが計算できても `PAPER_ONLY_PARTIAL_REQUESTED_BATCH` と明示する。
- 2026年の過去結果を研究検証に使わない。2023〜2025年のROIは後知恵選択であり未来利益の保証ではない。発走後FINALオッズで発走前予想を偽装しない。馬券購入APIは呼ばない。

## 実装済みの入口

- `research/run_github_race_ids_v1.py`：GitHubマニフェストから1件/複数レースIDの入力を解決、元の純粋AI順位、元のANCHOR_FULL×単勝オッズによる期待値順位、L2.1馬券、V3購入金額・資金不足をJSONへ。
- `research/l21_github_frozen_prebet.py`：以前の `research/l21-frozen-prebet-price.py` の凍結50条件を移植。変更はモバイル由来のソース識別子を `KEIBA_GITHUB_JRA` へ切り替える最小範囲。L2.1購入ルールは変更していない。
- `research/l3_v3_operational.py`：正式V3資金配分（元V2 `REVERSE_P3` と差分一致テスト済み）・レース予算・日次資金。
- `tests/test_github_race_id_integration_v1.py`：同日2レースの合成発走前カード＆428件/レースのオッズ → L2.1 → V3、CLI、欠損・FINALオッズ拒否・2026外拒否・偽p3拒否。
- `research/replay-one-race-id.py`：独立の**過去診断**。2021〜2025年の保存済み7王7体＋Outsider13体を、GitHubの正式ファイルメタデータとSHA検証で必要年だけ取得、結果列を使わず再生。**L2.1の元 `AI_RAW` / `ANCHOR_FULL` / L1.75 p3とは違う**。診断データを運用入力に昇格させない。

### GitHubが生成したソースを使った実行

```bash
python3 research/run_github_race_ids_v1.py \
  --race 202507010212 \
  --race 202507010211 \
  --manifest local/github-produced-manifest.json \
  --output out/races/batch.json
```

マニフェストは次の形。各ファイルはマニフェスト所在ディレクトリからの**相対パス**であり、入力ファイルには **GitHubで収集した本物の時点適格データ**が必要。モデルや時刻を推測して埋めない。

```json
{
  "contract": "KEIBA_GITHUB_RACE_INPUT_MANIFEST_V1",
  "races": {
    "202507010212": {
      "race_card": "202507010212/race-card.json",
      "original_l21_features": "202507010212/original-l21-features.json",
      "jra_prebet_quotes": "202507010212/jra-prebet-quotes.json",
      "original_l175_horse_p3": "202507010212/original-l175-p3.json"
    }
  },
  "budget": {
    "race_budget_yen": {"202507010212": 1500},
    "max_race_exposure_yen": 3000,
    "available_cash_by_date_yen": {"2025-01-06": 3000},
    "already_reserved_same_day_yen": {"2025-01-06": 0},
    "max_day_exposure_yen": 10000
  }
}
```

`race_card` は `KEIBA_GITHUB_JRA_RACE_CARD_V1`、`jra_prebet_quotes` は `KEIBA_JRA_COMBINATION_ODDS_ASOF_V1` かつ `source_system=KEIBA_GITHUB_JRA`、`original_l21_features` は `KEIBA_L21_ORIGINAL_ANCHOR_FEATURES_PREBET_V1`、`original_l175_horse_p3` は `KEIBA_L175_HORSE_P3_ASOF_V1`（本来の原版p3・完全な馬ID照合・学習版・出力時刻）を要求。

**重要：GitHub側でこれら4種の実レース元データを自動的に作る上流 collector/model inference は現時点では未接続。** マニフェストは機械可読の配線契約であり、空から予想を生成するAIではない。データが欠ければ `BLOCKED`。合成テスト成功を実レース発走前予想の成功と混同しない。

### 実在する過去の予測アーカイブを診断する

```bash
python3 research/replay-one-race-id.py \
  --race 202507010212 \
  --output-dir out/archive-202507010212
```

これは保存済み7王・13 Outsiderの研究順位。 `HISTORICAL_OOF_REPLAY_ONLY` として保存し、`run_github_race_ids_v1.py` に足りない能力AI／L1.75／発走前価格を代理生成**しない**。内部のGitHubメタデータを先に確認、現存名を一度ダウンロード・SHA照合、runner内で再利用。404は打ち切り、429/502/503/504のみリトライ。

## 出力

`batch.json` は `KEIBA_GITHUB_RACE_OUTPUT_BATCH_V1`。

- `races[].rankings.ai_raw`：オッズを一切混ぜない元 `distill_ai_rank`（オッズ未補正）
- `races[].rankings.expected_value`：元 `P__ANCHOR_FULL × 同時刻単勝オッズ` の降順。別の未確認EVモデルと同一だとは主張しない
- `races[].l2_1.tickets`：凍結50条件の同一馬券物理集合、時刻付き価格、該当line×gate
- `races[].l3_v3.tickets`：券ID、金額、券種、組み合わせ、元の該当line×gate、`min_p3`、欠損フラグ
- `daily_bankroll`：リクエスト内の有効レースの同日確保資金。**開催日全JRAレースを網羅した保証はない**
- `BLOCKED` / `BLOCKED_L3_V3` / `PAPER_ONLY_COMPLETE` / `PAPER_ONLY_PARTIAL_REQUESTED_BATCH` の状態と理由

入力JSONの自己申告時刻や真正性は独立立証できないため、現段階は紙上結果。**実レース未検証なのに READY 購入可扱いにはしない。**

## この次に必要な上流実装

1. GitHub側の既存BACKFILL/年度snapshotを使って、指定レースの **正式出走馬ID、時点適格特徴量** を解決する。過去結果・配当・FINALオッズはL2運用入力として使わない。
2. オリジナルの純粋AI `distill_ai_rank, distill_score_z, distill_score_gap_to_best`、Correction Risk、元 `P__ANCHOR_FULL` と L1.75 `p3` を全頭に対してモデル正本から出す。未配備なら未配備と報告し、保存7王の比較順位で埋めない。
3. GitHub側JRA当日収集器の検討は、実開催日まで取得可能性を保留する。過去に記録された本当の発走前オッズがなければ2025年の完成買い目は出せない。
4. GitHub側JSONの保管・配信先を設計し、アプリはそのJSONを読むだけ。課金不明の保存を勝手に使わない。

2026年未知レース実行時の追加時刻検証と、独立した未知年の紙上購入検証は別ゲート。研究成績から将来利益は保証できない。

