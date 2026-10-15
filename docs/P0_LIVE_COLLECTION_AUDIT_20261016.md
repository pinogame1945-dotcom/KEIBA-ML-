# P0｜当日レース収集基盤 設計・正本監査（2026-10-16）

状態: **P0 AUDIT RECORDED / NOT YET SIGNED OFF FOR LIVE**。設計監査のみ。データ収集・予想・購入は実行していない。L1/L2.1/L3・main・アプリmainは変更禁止。基底: `feature/l3-v3-official-github-producer-20261016`。同封機械台帳: `contracts/KEIBA_LIVE_DATA_P0_FEATURE_INVENTORY_V1.json`。

## A. 決定事項：責任分担

- **GitHub側**: JRA等の入力取得 → 検証 → 過去年のみの特徴量計算 → 原版L1 / ANCHOR_FULL / Correction Risk → 凍結L2.1 → L3 V3 → バージョン付き公開JSON。
- **MOBILE側**: GitHubが公開した状態と結果を読み、AI_RAW順位、元期待値AI順位、買い目・掛金、時刻とBLOCK理由を表示するだけ。既存MOBILEの直接JRA収集コードを新パイプラインの依存先にしない。旧コードの削除やアプリ改修はP0の対象外。
- BACKFILLは前日までの**確定履歴**、LIVEは当日の**時点付き現在情報**。両者の用途、信頼境界、更新頻度、保存期間を混同しない。
- ユーザー指定済みL3 V3の配分方式は `REVERSE_P3_RACE_MAX300`。基底ブランチの `chosen_stake_policy:null` は**旧文書の矛盾**。このP0はL3エンジンを改造しないが、P1統合で正本契約へ選択済み方式を反映し、1レース100～300円の配分・資金不足を検証する。L2.1券は不変。V3歴史ROIは将来保証でない。

## B. P0-1: リポジトリ監査

| 対象 | 確認できた資産 | LIVE実装としての評価 |
|---|---|---|
| `KEIBA-DATA-CORE` | `docs/JRA_ENTRY_BOUNDARY.md` に正式な番号付き `/JRADB/accessD.html`、開催単位のatomic保存、取消・順延・欠落非削除ルール | **設計再利用**、当日新GitHub collectorの実証ではない |
| `KEIBA-MOBILE` | `src/services/liveSyncService.ts`、`src/domain/live.ts` に開催・出馬表・天候・馬場・オッズ・変更・結果の型と端末取得処理 | 実行場所がユーザーのGitHub収集方針と異なる。ロジックは移植/共有候補 |
| `KEIBA-BACKFILL` | `docs/FORWARD_BACKFILL_V1.md` に前日まで/manifest・SHA・日単位確定レース収集 | 当日LIVE collectorとして使わない |
| `KEIBA-ML-` | L1 catalog、凍結L2.1 50 rules、V3資金研究、未統合の1レース推論・原ANCHOR再生 | 本物の全頭LIVE scorerとJRA発走前オッズは未実証 |

### ブランチとPRの重要な区別

- `KEIBA-ML-` PR #7: `feature/single-race-engine-20261015`、**closed / unmerged**。元ANCHOR_FULL 19特徴量の歴史再生、DNF原因監査、合成as-of配管。https://github.com/pinogame1945-dotcom/KEIBA-ML-/pull/7
- `KEIBA-ML-` PR #8: `feature/multi-race-id-pipeline-20261016`、**closed / unmerged**。2023–25歴史的ID再生は確認、当日入力はMOBILE capture依存で**合成テストのみ**。https://github.com/pinogame1945-dotcom/KEIBA-ML-/pull/8
- `KEIBA-MOBILE` PR #18: `feature/l21-prebet-odds-export-20261015`、**closed / unmerged**。4券種の同一client-observedAt export候補。GitHubへ自動供給する仕組みではない。https://github.com/pinogame1945-dotcom/KEIBA-MOBILE/pull/18
- **closedはmergedではない**。ブランチ上の検証済み機能をmain本番済みと表示しない。

## C. P0-2: AI入力全件監査と不可欠なギャップ

基底ブランチの実L1カタログは**149項目**: `AVAILABLE 108 / DERIVABLE 23 / PROHIBITED_L1 15 / L2_ONLY 3`。mainの旧カタログは137項目: `AVAILABLE 91 / DERIVABLE 16 / BACKFILL_REQUIRED 16 / PROHIBITED_L1 11 / L2_ONLY 3`。**149がこのP0正本**。古いmainの137を取り込まない。全149件の元ID・source_field・available_at・history_boundary・元catalog statusは機械台帳に記載。カタログの `AVAILABLE` はLIVE観測済みや配備済みの意味ではない。

PR #7の1レースモデル登録簿 `models/one-race-registry.json` には**31件すべて `unconfigured`**。登録簿は研究候補の一覧で、実戦対応モデル数を意味しない。7王/Outsider/L1.7保存済みOOSスコアの歴史再生は、新レースの新規推論ではない。

**元L2.1の核心**: `ANCHOR_FULL` は元モデルの19特徴量、`P__ANCHOR_FULL`、全頭の正規 `distill_ai_rank` / `distill_score_z` / `risk` / ability、発走前の単勝市場を要する。元の期待値軸は `argmax(P__ANCHOR_FULL × WIN odds)`（正本の同点規則を維持）。7王順位平均やL1.7のp3を代用品にしない。元の能力とFULL Correction Riskの**学習済み実体・学習時前処理・当日全頭推論**が現在の最重要未解決点。

**凍結L2.1**: 9元テンプレート×50 `line×gate` / `STANDARD`、券種は馬連・三連複・三連単。価格選別には締切前のWIN/QUINELLA/TRIO/TRIFECTA生スナップショットと組合せ一意性が必要。馬券追加・削除・置換・重複購入は禁止。既存PR#18の候補仕様では全組合せの完全なスナップショットを要求する。18頭なら最大5883価格（WIN18 + 馬連153 + 三連複816 + 三連単4896）。発走前の公式サーバー時刻と価格の真正性は未立証。

**未来情報禁止**: `STRICT_PRIOR_DATE_ONLY`の履歴しか現在の特徴量に使わない。当該レースの結果・払戻・事後FINALオッズ・実際の発走時刻・事後の馬場情報を流さない。取消・除外はレース前に公式に判明したものだけ適用。

**DNF重大問題**: PR#7監査では2025-01-06の24レース中5レースでDNFが元の研究OOS母集団から消えていた。結果の着順有効性で出走馬を事後フィルタしていたため。全馬照合では19/24のみ原研究券と一致。結果を用いて5頭を先に除外する操作は禁ずる。現行凍結研究成績にsurvivorship/selection-on-outcome biasの疑いがあり、LIVE予想性能の根拠には使わない。

### 必須データ→収集責任の候補表

| 分類 | 例 | 暫定取得元 | 検証ゲート |
|---|---|---|---|
| レース識別 | netkeiba形式12桁ID、JRA正式race/meeting、予定日・実施日・会場・R | JRA正式開催・正式出馬表＋ID照合 | 実在する公式キーとの**双方向・衝突なし照合**（文字列から推測生成しない） |
| 出走馬 | horse_id / 馬番 / 枠 / 斤量 / 騎手・調教師 / 取消・除外 | JRA正式 `accessD`＋実馬ID照合 | 全頭・一意・番号付き・頭数整合、開催単位atomic保存 |
| 条件 | 距離 / 芝ダート / コース / クラス / 重量条件 | 正式出馬表＋既存過去年正規化 | model schema一致、不明値を推測しない |
| 過去情報 | 出走・ラップ・相手・血統・騎手調教師の過去年特徴量 | BACKFILL日pack＋Kaggle保存モデル/履歴 | 日付境界・manifest + SHA・session再利用 |
| 当日変動 | 天候・馬場 / 馬体重 / 騎手変更 / 開催順延・中止 | JRA正式更新 | source観測時点付き、発表前null、旧ver不変 |
| 市場 | 同時点の単勝 / 馬連 / 三連複 / 三連単全候補 | JRAの実測オッズ | **非FINAL**・全必要組合せ・同時点・締切前・価格由来独立検証 |
| L3資金 | 事前資金残高・同日予約原資・一律100円単位 | ユーザー資金台帳・V3方式 | `REVERSE_P3_RACE_MAX300`、不足時 `INSUFFICIENT_BUDGET`、当日払戻先取りなし |

各ソースの取得利用条件、実際のAPI/HTML動作、アクセス制限、正式公表時刻の証跡は**P1前に個別確認が必要**。JRAの公開HTMLが存在することは、大量の定期自動取得許可の証明ではない。

## D. P0-3 / P0-4: 公表時期、収集、停止規則

- JRA公式FAQ: 通常の無番号出馬表は木曜16時すぎ、番号付きは前日10時すぎ。GIなど例外あり。時計だけでは保存許可せず、**実在する正式出馬表の構造**で判断。参照: https://jra.jp/faq/pop02/2_2.html
- 馬体重はJRAが「発表済」を付けたレースのみ反映。固定時刻の発表を前提にしない。参照: https://www.jra.go.jp/faq/pop02/2_13.html
- GitHub Actionsのcronは**最短5分**で、遅延・不実行の可能性あり。締切直前1分以内の最新価格保証はできない。作業実施前に「有効時点が取得できるか」「妥協不能な条件は何か」を独立に確認。参照: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
- **取得状態と予想状態は別**。例: `DISCOVERED → OFFICIAL_CARD_VALIDATED → DAY_UPDATES_COLLECTING → PREDICTION_INPUT_READY` と、各モデル `SCORED/BLOCKED/UNAVAILABLE`、L2.1 `BET_READY/BLOCKED`、レース `CLOSED`。未発表とソース障害と不正データを同じnullにしない。
- 形式が崩れた/HTTP404/最新metadataの実在名が消えた場合はそのデータについてSTOP。429/502/503/504・一時通信断のみ指数backoff+上限。古い正式カードを無警告で消さない。Kaggleは**manifest→metadata実在ファイル名→1取得→同runner再利用**、名前推測なし。
- 取得した各版は `race_id, race_identity, horse_id, scheduled_start_at, observed_at, source_published_at(if proven), fetched_at, source_uri, source_hash, runner_commit, model_version, input_fingerprint` を保持。受信側へステータス・時刻・原因を出す。client `observed_at` を公式サーバーが署名した時刻として扱わない。
- 保存先は**GitHub側の小型バージョン付きJSONを優先候補**。GitHub Pages・repo fileのどちらを正本配信経路にするか、更新頻度・サイズと利用枠の実測に基づきP1で選ぶ。課金や無料枠不明の保存は使わない。GitHub Actions Artifact/Cacheを既定利用しない。
- 2026年過去結果を未知年モデル選定に使用しない。紙上予測・紙上資金計算のみで、実投票は行わない。

## E. P0-5: 提案する入力/出力契約の最小単位

**入力指定**: `{schema_version, mode:"TODAY", race_ids:[12桁...], request_at}`。レースIDは発見用キー、**JRA正式IDへの解決を必須**とする。

**レース単位のGitHub公開JSON**（実装前の案）:
```json
{
  "schema_version": "KEIBA_LIVE_PUBLIC_V1",
  "race_id": "YYYYXXXXXXXX",
  "race_state": "PREDICTION_INPUT_READY",
  "status": "BLOCKED",
  "blocked_reasons": ["LIVE_ORIGINAL_ABILITY_RISK_NOT_PROMOTED"],
  "timestamps": {"generated_at": null, "source_as_of": null, "scheduled_start_at": null, "bet_cutoff_at": null},
  "provenance": {"source_hashes": [], "run_id": null, "model_versions": {}},
  "rankings": {"ai_raw": null, "expected_value": null},
  "l2_1": {"policy": "STANDARD", "contract_version": "2.1.0", "tickets": null},
  "l3_v3": {"version": "3.0.0", "stake_policy": "REVERSE_P3_RACE_MAX300", "tickets": null, "funding_status": "NOT_EVALUATED"}
}
```
`null`は欠損を隠す値ではなく、`blocked_reasons`で必ず理由を併記する。上例は**全てダミーのスキーマ例**で、実データ成功例ではない。アプリが元の順位、価格、馬券、掛金を独自算出・補完することは禁止。

## F. P0終了判定とP1のブロッカー

| P0項目 | 現在判定 | 必要な後続作業 |
|---|---|---|
| P0-1既存資産と矛盾 | **静的監査済み** | PR#7/#8/#18を本番採用しない前提で再利用設計 |
| P0-2全特徴量の台帳 | **149件を採録済み** | 正本の各配備モデル → 実際に参照するfeature_idの依存行列は**未確定** |
| P0-3取得先候補 | **仮説表作成済み** | 規約・到達可能性・公式IDとnetkeiba ID解決・時刻の実測 |
| P0-4更新・異常設計 | **暫定規則あり** | 実際のGitHub/ソース応答と性能で検証 |
| P0-5I/Oの最小スキーマ | **暫定案あり** | 実体テストと各consumer合意後に凍結 |
| 当日の実データE2E | **未実証** | 次回以降の開催で前出し時点検証 |
| L3 V3公式ステーク統合 | **方式はユーザー選定済み、実装未検証** | V3原研究関数抽出・予算台帳と原L2.1互換テスト |

**P0は「監査資料を作った」段階**であり、LIVE READYでも設計の全項目が証拠で閉じた状態でもない。特にモデルごとの推論特徴量契約、ソース利用条件、正式な発走前タイムスタンプ、配信経路がP1入口の重要ゲート。

## G. P1でまず実施するテスト候補

1. 取消/除外/順延/12R以外/仮枠番/馬番重複で正式カード保存失敗＋旧版不変。
2. 2025年既存レース（2026結果封印）で既知の全頭ソース照合・DNF頭数維持。後知恵除外で再現性が上がってもLIVE合格に数えない。
3. 16頭/18頭で必要な4券種の全物理オッズの組合せと一意性、観測時点・正式締切証拠・FINAL拒否。
4. L1元7王、Outsider、L1.75/ANCHOR_FULL、Correction Riskの未配備を正しくBLOCKする。順位の推定作成ゼロ。
5. L2.1 STANDARD 50条件の不変hash検査、物理券重複排除、L3 V3所定方式・100円単位・資金不足全件停止・当日払戻先取りなし。
6. GitHub標準無料CPU・一時領域だけ。Kaggleファイル名metadata実在検証と429/5xxだけの限定リトライ。

## H. 参照の正本
- L1 catalog: `contracts/l1-feature-catalog-v1.json`（基底149件・blob SHA `cd3c12a6971f4e56e62aeafd6a0e43b75b8bfa70`）
- CORE official entry: https://github.com/pinogame1945-dotcom/KEIBA-DATA-CORE/blob/main/docs/JRA_ENTRY_BOUNDARY.md
- BACKFILL forward: https://github.com/pinogame1945-dotcom/KEIBA-BACKFILL/blob/main/docs/FORWARD_BACKFILL_V1.md
- L2.1 frozen: `freeze/l2.1-standard50-to-l3-20261015/contracts/KEIBA_L2_1_FIXED_PURCHASE_V1.json`
- L3 branch: `feature/l3-v3-official-github-producer-20261016/docs/GITHUB_PREDICTION_PRODUCER_L3_V3_OFFICIAL_20261016.md`
- PR#7 full: https://github.com/pinogame1945-dotcom/KEIBA-ML-/pull/7
- PR#8 two-port: https://github.com/pinogame1945-dotcom/KEIBA-ML-/pull/8
- MOBILE PR#18 odds: https://github.com/pinogame1945-dotcom/KEIBA-MOBILE/pull/18

**No collected LIVE races, no deployed original forward model, no CI/run execution, no modifications to production.**
