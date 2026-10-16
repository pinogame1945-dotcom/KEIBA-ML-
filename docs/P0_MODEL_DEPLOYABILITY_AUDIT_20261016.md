# P0追加監査：当日予想向け元モデルの実行可能性（2026-10-16）

## 判定
- **研究資産の発見・構成照合: 完了（静的監査）**。新規未来レースの全モデル推論: **未完成 / 未実証**。
- 対象: PR#7（`f9eedf6c2d82719571619b39b14816155b6f2cd0`）の `models/one-race-registry.json` に登録された **31件**。全件 `kind=unconfigured`。
- 内訳: 7王個体7件、Outsider候補18件（馬ベース5・非馬6・遷移7）、派生/集約ステージ6件。これらを実戦で動く31体の独立学習済みモデルと数えない。
- **22件は元の研究JSONとモデル名が完全一致**（7王4、Outsider18）。7王の3件は候補版違いがあり、名前から対応付けてはいけない。各レコードの証拠/未取得項目は `contracts/KEIBA_LIVE_P0_DEPLOYABILITY_MATRIX_V2.json` に保存。
- この静的監査ではKaggleの現行認証・モデル本体取得・そのハッシュ照合・CIは実行していない。`models/promoted`にはREADMEしかなく、元の1レース登録簿で本番モデルが有効化されていない。

## 7王：研究モデルの実体は存在する可能性、昇格は不可
- PR#7内 `research/audit-seven-king-bundles.py` は過去のKaggle学習済みLightGBMの `model.txt/model.json/schema.json/candidate.json` を検査。研究用workflowsはKaggleファイル一覧→実在名→一回取得→SHA/特徴量順序を確認する。
- `research/verify-trained-kings-vs-saved-scores.py` は**歴史の元モデルと保存スコアの再現性**を調べる。2025当該年検証ラベルでearly stoppingが行われたモデルを**2025年の真のblind予測**とは扱わない。
- `research/predict-prior-year-seven-kings.py` は対象より前の年のモデルだけを使う改善経路。ただし入力snapshotが歴史 `FINAL` で発走前利用可能時点の監査が未完。**2026で勝手に実施しない**。
- 研究名一致4件: `core4_no_pedigree`、`jockey_trainer_condition`、`no_auto_full`、`no_auto_jockey`。
- **名前整合未解決3件**: `no_auto_full_pedigree_legacy`、`no_auto_full_pedigree_v1`、`jockey_trainer_condition_plus_raceclass`。元の研究 `l1-final-walkforward-v1.json` は近い候補名を含むが、同一視不可。Kaggleの `candidate.json/schema.json/model.json` と一致検証する。
- 原7王の順位合議は凍結ルールのtie-safeを再現する。平均順位などで代用しない。

## Outsider：18候補と保存13票は別
- 馬ベース5件: `research/l1-outsider-arena-v1.json` → `BASE+STYLE/LAP/NETWORK/DISTANCE/BACKFILL`。
- 非馬6件: `research/l1-outsider-nonhorse-v1.json` → `DAY_TREND/RACE_SHAPE/GATE_COURSE/JOCKEY/FIELD_COMPOSITION/CHIMERA`。同日の将来レース結果は不許可、当日傾向は発走前に終了した同会場の前レースだけを使う。
- 遷移7件: `research/l1-outsider-transition-v1.json` → `ROTATION/CLASS_MOVE/WEIGHT_CHANGE/BODY_CHANGE/DISTANCE_TRANSITION/COURSE_TRANSITION/OPPONENT_GAP`。馬の過去履歴は `STRICT_PRIOR_DATE_ONLY`。
- 上記は**研究時のFeature Set/モード**であり、元学習済みLightGBMの**最終列名・列順・モデルハッシュ**ではない。元bundleとtraining config照合まで待つ。
- 過去の7王＋13 Outsider票の再生は `research/replay-one-race-id.py` で可能。登録簿の18候補から勝手に13体を選び直さず、13票の元モデル・選抜世代の真正性を別確認する。

## 期待値AI：新規推論のブロッカー
- 元 `ANCHOR_FULL` は19特徴量と正本係数を復元する既存処理があり、履歴1レースで確率一致を検証している。
- しかし元 `distill_ai_rank`、`distill_score_z`、`distill_score_gap_to_best`、`risk` は元能力/元FULL Correction Riskを、**締切前・全頭**で算出してこそ得られる。OOS歴史スコアやL1.7 `p3` 等の値で補完してはいけない。
- 単勝と馬連/三連複/三連単の組合せ価格は、同一観測・非FINAL・締切前の実証が必要。独立に検証されたJRA公表時刻・購入締切は現状未確認。
- 当日側の契約: `missing original model/input → BLOCKED`、未来結果/DNF事後排除→禁止。L2.1 STANDARD50とL3 V3 `REVERSE_P3_RACE_MAX300` は変更しない。

## P1までに解決する順番
1. **モデル正本と出力の所在を確定**：7王7候補のKaggle metadata実在名/ハッシュ、Outsider18候補の学習済み正本、13 Outsider合議の選抜元、元能力とCorrection Riskのモデル/学習済みparams。
2. **モデル依存行列を完成**：各 `schema.json` 等から列順・カテゴリ辞書・特徴量時点・学習/検証範囲を抽出。推測したモデル列は禁止。
3. **当日特徴量の時点追跡**：正式出馬表ID対応、結果判明前全頭、日時/天候/馬体重、同日先行レース結果、非FINALオッズを時点付きで検証。
4. **実測のない入力は止める**：モデルがないなら `BLOCKED`、日付で勝手にFINALに昇格しない。2026未知年の結果を開かない。
5. **無料CPUの範囲で静的/模擬テスト**。Kaggleはmetadata実在名を一度解決して同runner再利用。404は停止、429/502/503/504と一時通信だけbackoff。有料runner/GPU/課金保存は未許可のため使用しない。

## 監査元
- PR#7: https://github.com/pinogame1945-dotcom/KEIBA-ML-/pull/7
- `models/one-race-registry.json`（PR#7 head）
- `models/promoted/README.md`（P0基底ブランチ）
- `research/score-one-race.py`、`research/audit-seven-king-bundles.py`、`research/predict-prior-year-seven-kings.py`（PR#7 head）
- `research/l1-final-walkforward-v1.json`（P0基底）
- `research/l1-outsider-arena-v1.json` と `research/l1-outsider-nonhorse-v1.json`（`research/l16-outsider-chimera-all-races-v1-20260929`）
- `research/l1-outsider-transition-v1.json`（`research/l1-outsider-transition-v1-20260929`）

**作業の種類：設計監査、docs/contractファイル作成のみ。既存モデル・買い目・資金配分・mainは変更していない。**
