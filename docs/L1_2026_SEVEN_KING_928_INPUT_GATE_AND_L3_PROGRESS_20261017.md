# 7王 2026・L3までの到達状況（2026-10-17）

## 第5段階・元13 Outsider全系統の2026年実入力生成成功（2026-10-17 20:30 JST）

- **GitHub Actions [#39028775307](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/39028775307) success。** BACKFILL正本commit `9fe9f1742aa074a84c03a779e1ee949c04f78c0d` の既存690（固定SHA）＋元STYLE14＋元BACKFILL13差分を再利用。2026-10-17実36R・471頭に対して、凍結済み `research/build_l1_nonhorse_dataset_v1.py:process_day` で**13種類すべてのOutsider用特徴量データセットが471行ずつ生成成功**。学習済み13モデルの推論やTop3投票ではない。
- 元の6非馬能力系に加え `ROTATION` と `OPPONENT_GAP` は当日入力だけで作成。残り5種 `CLASS_MOVE` / `WEIGHT_CHANGE` / `BODY_CHANGE` / `DISTANCE_TRANSITION` / `COURSE_TRANSITION` は元 `horse_signature` を使い、BACKFILL 2022-01-01〜2026-10-16の厳密過去履歴517開催日から直近3走の条件・馬体重・斤量等を復元。元のBACKFILL v2アダプタを変更せず使用。
- 5遷移系履歴の検証結果：**471/471頭で既存690の `prior_starts` と完全一致**。2026年の対象日より前の保存済み履歴87開催日（2026年歴史部分1,611出走を含む）のみ参照。比較可能な406頭の`previous_distance_m`が元690と一致し、残り65頭は比較対象となる前走なし。2026年10月17日のレース結果・払戻は読み込まず、2026年のモデル学習・成績評価もなし。
- 13系統ごとの生成特徴量列数（元コードの動的列集合は和集合で数える）：当日傾向43、レース構造64、枠コース30、騎手103、メンバー構成59、キメラ194、ローテーション34、相手格差155、クラス移動34、斤量変化32、馬体変動34、距離遷移36、コース遷移33。※ `race_date` は入力の識別用メタデータであり、モデル特徴量717列には加算しない。馬の発走前観測値が無い場合は元のnull値を維持。
- 実装：`research/prepare_2026_10_17_original_outsider_prior_signatures_from_pinned_history_v1.mjs`、`research/materialize_2026_10_17_original_outsider_stateless8_from_cached_delta_v1.py`（履歴引数追加）、既存のSTYLE14/BACKFILL13差分結合を一つの**無料CPU読取専用CI**で連結。元7王モデル・元690特徴量・元13 Outsider数式・凍結L2.1/L3・mainの変更なし。
- 結果は**Runner一時領域での検査のみ**。追加27列、遷移用履歴署名、13モードの生成入力の永続保存は未実施。毎回の再学習は行っていないが、未保存の中間入力を次のRunで使う場合は履歴差分生成を繰り返す必要がある。
- **厳密な未完了**：13モデルの2025 OOS学習済み重み・最適反復回数は当該手元成果物に未保存、2026用13モデルの候補refit未実施。13系統のモデルスコア/Top3票はゼロ。2026全特徴量の独立as-of保証も未完。各Outsider用元特徴量を実際に生成できたことと、13モデルの運用可能を混同しない。
- **全体運用完了3/8段階。** 第5段階の実入力データ生成 **13/13系統成功**、元7王合議は保存済み。モデル推論→L1.75→L2.1→L3 V3は未完。

## 第5段階・2026年厳密過去日履歴でSTYLE14完全復元と差分27列結合（2026-10-17）

- **対象の履歴境界**：BACKFILLの既存研究用生成Workflowは2022-01-01〜2026-10-17のソースを使い、発走前10/17のターゲットには結果・払戻・オッズ無しのライブパックを渡す。今回も完全に同一の日付境界を採用した。**10/17対象レース結果はアクセス・学習・評価ともゼロ**。2026年1月〜10月16日の確定済み結果は履歴特徴量の材料としてのみ利用し、2026年の評価には使用しない。これは2026年の結果全般を2025年末で打ち切った従来の保守的調査とは区別する。
- **元STYLE復旧CI #39027606055 success**：Git sparse checkoutで元BACKFILL SHA `9fe9f1742aa074a84c03a779e1ee949c04f78c0d` の2022〜2025年430開催日と2026年対象前87日、計517日を復元。元 `historical-extra-builder.mjs` を変更せず適用。対象36レース471頭の `prior_starts` と厳密過去履歴回数が **471/471一致、保留0頭**。うち**402頭**に計測可能な通過順位履歴、69頭は過去走・corners未計測などで元式の `null` 値を保持。結果は2026年の評価ではない。
- **追加27列結合CI #39027835082 success**：`research/join_2026_10_17_outsider_original_style14_backfill13_delta_v1.py` で、既存690列は値も列集合も不変・36ファイルSHA-256一致・馬ID471/471一致を確認し、元STYLE14 + 元BACKFILL13 = 27列だけを作成。合計**717特徴量相当**。追加差分gzipは**27,150バイト**、SHA-256 `372a1ea1f6615b2f574a47e902d5acbfed4a07c3be6eac231ee3b7529eed96ea`。追加27列の非nullセルは10,224。**今回はrunnerの一時領域のみ**（永続保存未実施）。元690の再生成・7王の再学習なし。
- **安全制限**：対象日以降のレース結果ファイルはGit checkoutしない。対象日ライブ36レースの結果・払戻・オッズは空配列。2026年の prior-date 結果を履歴入力として使ったことを明示し、2026年のモデル学習・ROI評価には一切使用しない。全717列のas-of独立認証は未完で、両側の出力とも `ready_for_betting=false`。
- **残り**：13 Outsiderの2026年用学習方式・モデル（既存保存なし）、日次傾向など他の必須特徴量の整合と入力完全性、L1.75→凍結L2.1→L3 V3。13 OutsiderのTop3実票はまだ生成していない。元7王モデル・凍結買い方・main不変。無料標準CPUのみ。**L3までの運用完了は3/8**。

## 第5段階・STYLE欠損404頭の原因を実データで分解（2026-10-17）

- **診断CI [#39024460396](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/39024460396) success。** 2022〜2025年の元BACKFILL結果を使用し、2026年の結果内容は一切読み取らない。完走過去走2,680件の `result.corner_raw` は **2,680件すべて非空・数字を含む**。従来の『通過順位を計測した一致馬が2頭』はソース通過順位の欠損ではない。
- 保存済み690の `prior_starts` と2022〜2025年のみの元履歴が一致した67頭の内訳は **過去出走0回＝65頭、過去出走5回＝1頭、28回＝1頭**。そのため通過順位計測馬が2頭しかない。通過順位データ2,680/2,680が存在するという結果と矛盾しない。
- **404頭の回数不足の分布：** 1回不足68頭、2回53頭、3回57頭、4回67頭、5回62頭、6回40頭、7回27頭、8回17頭、9回8頭、10回4頭、11回1頭。発走前に存在する2026年の過去走を封印中であることが主因と考えられるが、残りの2022〜2025年収集欠損を完全排除したわけではない。
- **固定コミットの2026年履歴ファイル一覧 [#39024656448](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/39024656448) success。** `9fe9f1742aa074a84c03a779e1ee949c04f78c0d` のGit treeから2026-01-01〜2026-10-16に属する保存済み日別ファイル名 **87件** を検出。10/17同日以降はゼロ。ファイル実体・結果フィールドは一切ダウンロードせず、`git ls-tree` の名前とblob OIDだけ照合。
- 実装：`research/audit_2026_outsider_strict_prior_history_tree_only_v1.py` と無料標準CPU CI `.github/workflows/l1-2026-outsider-prior-history-tree-only-freecpu.yml`。`STRICT_PRIOR_DATE_ONLY` の対象日以上を拒否する合成テスト成功。**既存2026年の過去走結果を特徴量計算に使う許可は出していない**。現時点ではメタデータ以外を取得しない。
- **結論：STYLE元式やcorner_rawの収集修正は不要。** 対象前の2026年履歴を利用する場合は、2026年評価封印と『各レース以前の確定走を特徴量に使用すること』を分離した扱い、およびas-of確認が残る。承認前に読み込み・元履歴計算・13 Outsider推論を進めない。
- 再学習・既存690/928・7王・L1.75/L2.1/L3は変更なし。追加結果ファイルの公開/保存やGPU/有料runnerはなし。L3への進捗は **3/8のまま、第5段階の不足原因と再開条件が確定**。

## 第5段階・元STYLE履歴生成器の2026結果封印下の実行検証（2026-10-17）

- GitHub Actions Run [#39023979174](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/39023979174) **success**。先行Run #39023826352 は異なる履歴範囲で数が一致せず失敗。元BACKFILLの凍結入力Workflow `.github/workflows/690-24-36-feature-input-trial-free.yml` の `--source-start 2022-01-01` を確認して範囲を修正した（過去に誤った履歴範囲を安易に流用しない）。
- 元BACKFILL `ml_features/src/historical-extra-builder.mjs` と `ml_features/contracts/l1-feature-set-contract-v1.json` を**一切変更せず**使用。`data/daily` の2022-01-01〜2025-12-31の430日分だけをGit sparse-checkout（2026年の結果ファイルはcheckoutせず）。10/17は結果・払戻・オッズなしの事前取得36Rデータを元の日別変換器で復元。
- 実データ対象 **36R・471頭**、該当馬が出走した過去年の2,233レース・2,687出走履歴を調査。元のSTYLE14列を生成し、保存済み690の `prior_starts` を馬ごとに照合。
- **67頭は2022〜2025年から観測した出走回数と一致。404頭は回数が不足したためSTYLE完全復元として採用しない。** 一致67頭のうち、元のcorner通過位置から計測済みSTYLE履歴があるのは2頭のみ。これは2026年のレース結果を使用しない制約下での結果。2026以前のデータが別の場所にある可能性を無断で補完しない。
- 一致馬のみオリジナル生成STYLE値を研究用一時ファイルに含め、不一致馬は`style_complete=null`とし、全馬`model_ready=false`・`asof_full_features_verified=false`・`ready_for_betting=false`を保持。**現在のデータで13 Outsiderの本物のTop3はまだ算出できない**。
- 実装 `research/audit_2026_10_17_original_style_pre2026_history_only_v1.mjs`、CI `.github/workflows/l1-2026-outsider-original-style-historic-safe-ci.yml`。元690と7王のモデル重み・ランキングは再計算も改変もせず、全て無料標準CPUで実行。出力はGitHub Actions一時領域のみで永続保存していない。
- 未解決は、2026年に既に終了した過去走を履歴特徴量としてどこまで使うか（現在の2026結果封印とは別に承認が必要）、STYLE通過順の元データ実測率、独立as-of監査。凍結仕様・main・L2.1・L3は変更なし。
- **L3までの現状3/8段階が運用完了。第5段階のSTYLE元生成器実データ動作確認まで成功、実13モデル予測は未完了。**

## 第5段階・Outsider元BACKFILL特徴量の一部復元（2026-10-17 19時JST）

- **実データ検証成功：** GitHub Actions [#39023134846](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/39023134846) と [#39023317964](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/39023317964) は両方 success。無料標準CPUで再学習・2026年結果参照・元690再計算なし。
- BACKFILL固定commit `9fe9f1742aa074a84c03a779e1ee949c04f78c0d` にある **元 `ml_features/src/backfill-feature-adapter.mjs` v2を変更せず動的import**。同じcommitの `live_json_to_daily_pack_v1.py` を使って、10/17の発走前ライブ36レースを厳格な日別packへ変換。
- LIVE→元BACKFILL変換は **36レース、出走471頭、取消1頭除外**。13種類の元 `backfill_*` レース条件系を復旧し、各レース・馬ID/元690 SHAを照合。`research/recover_2026_10_17_original_outsider_race_context_v1.mjs` → `research/join_2026_10_17_original_outsider_backfill_13_to_cached_690_v1.py` で **471頭×（既存690+元BACKFILL13）＝703列の一時行列**を生成。
- **ソースにない値を仮埋めしない**：`backfill_course_laps` 471/471 null、`backfill_course_layout` 368/471 null、`backfill_grade` 457/471 null、`backfill_age_max` 300/471 null。他の元レース条件値も元変換器の出力を無変更で保持。
- 不正結果の注入・発走後の取得時刻を人工的に加えた入力は**必ず失敗**する安全テストを実施。
- **残る問題：** 既存690には `style_*` が存在しない。元 `historical-extra-builder.mjs` の STYLE計算は過去走の通過順とその時点で既知の結果を元に行うため、既存の690＋今回の13列だけで正本STYLEを復元できない。元OUTSIDERの日次傾向には同日先行レースの勝ち馬データ依存もある。2026年の結果・払戻を後知恵で導入しない。10/10の過去FINAL投影もas-of独立監査未了。
- **成果物はGitHub Actions一時領域で検証**。保存済み690・血統・7王モデルはそのまま再利用可能だが、今回の中間703列は永続保存していない。次回使う場合は安価なレース条件復元・結合のみ必要。モデルの再学習不要。
- **L3進捗：3/8段階が運用完了。** 第5段階の元BACKFILL部分復旧まで進んだが、13 Outsider実学習・本物のTop3・L1.75・L2.1・L3 V3は未完。

## 第5段階・元13 Outsider 2026実予測の事前検証（2026-10-17）

- **現状：13 Outsider実モデルの2026用学習・実推論は未実施。** 凍結7王合議は保存済みで再学習不要。
- **過去年のモデル保存状況：** 既存研究 `research/run_l1_outsider_tie_audit_year_v1.py` は原13 Outsiderを学習後、候補フォルダを消去し、監査統計だけGitHubへ保存する。2025年の検証Run #36585261212 の実ログに候補13/13の処理成功はあるが、モデル本体・各best_iterationはGit永続保存されていない。Kaggle読み取り専用台帳 Run #39021820205 **success** は所有81 datasetのうち名称に outsider/nonhorse/transition を含む候補保存先0件を確認（名称条件による検査であって全保管場所の不存在を証明するものではない）。
- **重大な元入力差分：** BACKFILL pinned SHA `9fe9f1742aa074a84c03a779e1ee949c04f78c0d` の元690対象行に `style_*` / `backfill_*` 系特徴量はない。実2026-10-17サンプルでも両接頭辞0列を確認。一方、元正本 `research/build_l1_nonhorse_dataset_v1.py` の日次傾向・レース構造・クラス遷移等はこの情報を参照する。**690（および7王用649）だけで13 Outsiderを無理やり推論すると必要特徴量を欠損としてしまうため不可。**
- **実装した第5段階準備：** `research/prepare_2026_original_13_outsider_649_cached_source_v1.py` は既存690のSHAと2025正本メタ・元7王CORE4順序を使って10/10と10/17の対象を分離し、出走結果・払戻を引き込まない。ただし本来の13系統が必要とするSTYLE/BACKFILL元情報が649で欠ける場合は**推論入力として出力せず失敗停止**。これは実用的な正本入力不足の防止ゲート。
- 無料標準CPU CI Run #39022417238 **success**。合成60レース786頭の入力契約・SHA改ざん・結果混入の拒否・STYLE/BACKFILLソース省略拒否を検証。実データの13 Outsider完成を意味しない。
- **次に必要な作業：** (1) 元の特徴量生成器でSTYLE・BACKFILLなどOutsider固有の不足入力を結果漏洩なしで復元するか、完全な発走前ソースを見つける (2) 元2025折り返しで確定した13モデルの反復回数・学習済み重みが復元できるかをさらに確認する。旧重みの2026無断流用はしない (3) 候補2026 refitの正式な学習規則は7王の候補方式から自動拡張・正式承認しない。
- **L3到達：3/8運用完了。** 第4段階の未承認7王モデルは保存済み、第5段階は合議完了・Outsider受取口と元入力安全ゲートまで。Outsider実推論/L1.75/L2.1/L3は未完了。
- main、凍結7王/13 Outsider/L1.75/L2.1/L3、2026評価封印を変更していない。課金リソース・GPU・公開モデル保存なし。


## 最新確定状況（2026-10-17・保存済み7王のas-of監査＋13 Outsider入口）

- **再学習不要：** 2026年用の元7王候補は既にKaggle保存済み。KaggleがZIPとgzipを自動展開するため、remoteには展開後のmodels/・scores/・council2026.jsonlがある。元ZIPバイト単位一致を検証済みとは主張しない。
- **独立読取監査：** GitHub Actions #39020769889 **success**。BACKFILL pinned SHA 9fe9f1742aa074a84c03a779e1ee949c04f78c0d の690の60ファイルSHA、Kaggle保存済み7王model.txt/schema.json SHA、7王×786スコアと合議60レース786頭を照合。再学習・再計算・新規データ保存なし。
- **発走前時点は部分立証：** 2026-10-17ライブJRA元データ36レースは前日夜の実取得時刻を確認。一方、649+19+22+238の全特徴量の依存元が発走前に確定していたことは独立証明できない。2026-10-10の過去FINAL投影も発走前取得時刻の証拠不足。928列の全量as-of合格=FALSE、READY=FALSEを維持。
- **13 Outsider構成の根拠：** 元 .github/workflows/l16-outsider-support-safe-v1.yml にある非馬能力系6種類＋遷移系7種類＝13。別研究のArena5候補を勝手に混ぜない。
- **第5段階の受渡し入口：** research/prepare_2026_7king_13_outsider_research_signal_v1.py を追加。13系統のTop3 CSVが全レースに揃う場合だけ3/2/1点で追加信号を算出。未入力・異なる馬・重複は拒否、7王の順位を変更しない。校正確率は未生成・研究限定。
- **合成安全CI：** #39020982386 **success**。架空60レース786頭、13種の投票について全入力・欠落・不正馬IDを検証。実2026年の13 Outsiderの学習・予測・投票は**未実施**。
- 本番ゲート残課題：①928特徴量as-of独立立証 ②研究候補7王の正式な採用判断 ③13 Outsiderの2026年用実予測 ④L1.75→凍結L2.1→L3 V3。
- main・凍結仕様は未変更。GitHub Actions無料標準CPUだけ使用。GPU・有料保存・2026結果/払戻の後知恵検証なし。

## 現行ゲート: 3/8段階が運用完了。第4段階は研究用モデル保存済み、第5段階は13 Outsider受渡し入口まで

- BACKFILL参照 SHA 9fe9f1742aa074a84c03a779e1ee949c04f78c0d
- 690列＋元血統V4 238列＝928列、60レース・延べ786頭。血統はGitHub圧縮保存済み、再計算スキップをRun #39004737776で実証。
- ML Run #39005028488: 2025年版7王の保存済み schema.json と全60レース928列を照合。必要列は7/7完全一致、欠落ゼロ。
- 2025年保存モデルを2026に昇格していない。2026用の7王モデル学習および推論は未実施。
- 2026のレース結果/払戻/確定オッズの学習・評価なし。発走前時点の独立検証は未完了。
- 690の完全数値一致検証はユーザー中止指示により再開しない。
- core4 の6項目は全786行null: auto_history_body_weight_gap_to_best, auto_history_body_weight_gap_to_worst, auto_history_carried_weight_gap_to_best, auto_history_carried_weight_gap_to_worst, auto_history_distance_m_gap_to_best, auto_history_distance_m_gap_to_worst。推測補完は禁止。

## 7王の入力照合結果

| 別名 | 必須列 | 欠落 | 必須血統列 |
| --- | ---: | ---: | ---: |
| core4 | 649 | 0 | 0 |
| pedlegacy | 470 | 0 | 204 |
| condition | 423 | 0 | 0 |
| full | 266 | 0 | 0 |
| light | 164 | 0 | 0 |
| pedv1 | 504 | 0 | 238 |
| condrc | 457 | 0 | 34 |

## 2026年用・元L1モデル入力行列の再利用検証

- GitHub Actions #39005487279: 成功。
- `research/check_7king_2026_real_model_matrices_v1.py` は元の `research/emit_l1_to_l2.py` の `load_validation_frame()` を**変更せず直接再利用**し、学習済みモデルへの投入直前と同じ入力整形を検証する。
- 2026-10-10（24レース315行）＋2026-10-17（36レース471行）の **7王×786行＝5,502行分のモデル入力行列**を構築し、7/7成功。
- 各王の特徴量順序、実数／カテゴリ型、日付、馬ID、全頭集合が一致。7王すべてカテゴリ7列に未定義カテゴリなし。
- 完全null列はCORE4の前記6列のみ（原研究のNaN扱いを維持し、ゼロ補完しない）。
- 既存238血統キャッシュと690ファイルだけを参照。履歴再計算・2026着順利用・LightGBM重みロード・予測・学習・保存モデルの昇格は**全て未実施**。
- これは推論入力の構築成功であり、2026年用モデルの完成や7王の予測値生成ではない。

## 2024〜2025年の正本学習資産メタデータ検査（今回追加）

- 新設監査: research/audit_7king_2026_blind_training_assets_v1.py
- 新設無料標準CPU Actions: .github/workflows/l1-7king-2026-training-assets-gate-freecpu.yml
- 起動Run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/39005829126
- 完了: 2026-10-17 GitHub Actions 成功。2024/2025スナップショット掲載と7王の元モデル設定を確認。
- 目的: Kaggleに**既にある**snapshot生成世代eb13d4096519167bの2024/25マニフェストとリモートファイル一覧だけを検査し、7王の2025保存済みmodel.json/schema.jsonから元のfold学習年・早期停止best_iteration・600ラウンド設定等を厳密照合する。
- 巨大snapshot本体、model.txt、2026対象レース結果はダウンロードしない。学習・推論・新規Kaggle公開/保存は行わない。
- **結果次第でも2025保存重みを2026に採用しない**。過去年のbest_iteration数値を、ユーザー承認なしに2026の再学習規則に転用しない。
- 2026年結果封印、main、既存のモデル、Council/Outsider、L1.75/L2.1/L3の凍結は維持。

## 2026年ブラインド推論入口（今回追加）

- 追加: research/score_7king_2026_blind_from_approved_models_v1.py
- 追加テスト: research/test_7king_2026_blind_from_approved_models_v1.py
- 無料標準CPUの合成データ専用CI: .github/workflows/l1-7king-2026-blind-scorer-safe-ci.yml
- 元の emit_l1_to_l2.py から load_validation_frame(), family_for_feature(), sigmoid(), sha256_json() をそのまま使用。7王ごとにモデル特徴量順・2026全頭集合・LightGBM出力・TreeSHAPのロジット復元を検査してL1_TO_L2_OUTPUT_CONTRACT_V1に従う行を生成する。
- **モデルが未完成でも2025年版にフォールバックしない**。2026年用の全7モデルの2024/25訓練年・保存したschema/model SHA256・承認済みtraining_rule_id・approval_ref・元snapshot生成世代を検査し、どれか不在なら対象レースJSONに触れる前に停止。
- 元の928列を再利用。今回のCIでは履歴計算・Kaggleからのモデルやsnapshot取得・2026レース結果利用・実際の推論を実施しない。
- 既存の発走前as-of監査は未完了なので、将来実モデルで予測できたとしても本入口の出力ステータスは ready_for_betting=false 固定。
- 新しい学習時の早期停止ルールは未承認・未実装。train-staged-lightgbm.py、run_feature_arena.py、emit_l1_to_l2.py の2026評価ロックを変更しない。


## 2026年用・2025 OOS最適反復回数再利用の独立実装（最新）

- 新規: research/refit_2026_seven_king_prior_only_v1.py
- 新規安全テスト: research/test_refit_2026_seven_king_prior_only_v1.py
- CPU合成安全テスト: GitHub Actions #39006892127 **success**。
- 元 train-staged-lightgbm.py の flatten()/complete_races()/frames() とハイパーパラメータ辞書を直接再利用。独立した2026年向けrefitであり、元研究・凍結済みWalk-forward評価は変更していない。
- 学習対象は2024/2025のみ。2026年の正解・払戻、2026年評価、早期停止用2026ラベルを渡す経路は持たない。
- 2025年OOSモデルの各 best_iteration を2026年用運用refit候補の反復回数として転用する設計を実装。未取得・不一致・偽2026モデル・不正年の入力は拒否。
- **現状: 正本2024/25の巨大snapshotを使った実学習は未実施。7王モデル本体・実予測値は未生成。**
- 7王一括学習・推論・Kaggle保存のGitHub Workflow登録を試行したが安全チェックで拒否された。**Workflowは未登録・未起動。チェックを迂回する再試行はしない。**
- ユーザー指定の無料標準CPU以外の実行、GPU、有料保存、main/凍結仕様の変更はない。

## 第4段階の学習器制約

凍結正本: research/l1-raceclass-absorption-walkforward-v1.json、snapshot generation eb13d4096519167b。学習2年rolling。2026は評価封印。

- 元の2025年版は2023/24を学習、2025を検証。2026にそのまま採用しない。
- 元の run_feature_arena.py、train-staged-lightgbm.py および emit_l1_to_l2.py は、検証期間に2026が入ると拒否する。
- 元の train-staged-lightgbm.py は検証ラベルによるbinary_logloss早期停止50ラウンドに依存する。
- 2024/25の2年で再学習して2026をブラインド推論する際、2026の正解に依存せず反復回数を確定する正式な手順はまだ凍結仕様に存在しない。勝手に固定反復、2025重み流用、学習窓変更、2026結果利用を行わない。

## 追加検証・2026年モデル誤昇格の修正（2026-10-17）

- 安全テスト Run #39008521985: **success**。2024/25年別ファイルの年度偽装、同一馬重複検出、2026結果禁止、凍結7王設定の維持を合成データで確認。
- 実LightGBM推論の合成スモーク Run #39008639229: **success**。架空の786出走行・60レースに対し1王の真のLightGBMブースターで推論を実行し、L1_TO_L2_OUTPUT_CONTRACT_V1形式・TreeSHAP復元・順位・確率のレース内合計1.0をテスト。実馬の予測ではない。
- 不具合修正: refit実行だけで元データから作った新モデルを `APPROVED_2026_BLIND` と自己認定していた危険な設定を `CANDIDATE_2026_BLIND_UNRELEASED` に変更。元の2026推論入口は承認未済モデルを必ず拒否し、正式な昇格・承認の段階を飛ばせない。
- 2024/25年度ラベルと行の実日付を1行ずつ照合。重複したrace_id/horse_id、年外行を学習前に拒否。
- 過去のCI Run #39008431847/#39008470591/#39008486136はテスト期待文字列と合成fixtureの不一致により失敗。修正して #39008521985 で成功。研究モデルの学習失敗ではない。
- 実2024/25大規模学習・7王の実2026年予測は**まだ未実行**。前回ブロックされた大規模一括実行Workflowの再試行や迂回は実施しない。
- **L3までの位置: 1〜3完了、4途中、5〜8未完。** as-of未検証、実運用馬券では使用不可。

## 保存済み2025年版・本物の7王LightGBM重み検査（追加）

- 実装: `research/smoke_original_7king_2025_booster_weights_v1.py`
- 無料標準CPU・既存Kaggleアーカイブ読取専用CI: `.github/workflows/l1-7king-2025-real-boosters-read-only-freecpu.yml`
- 実行Run #39013818034: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/39013818034 。検証完了前の段階では結果を成功と報告しない。
- 対象: 実際に2023〜2024年で訓練し2025年で検証した7王の model.txt / model.json / schema.json。メタ記録SHA256とモデル本体ハッシュ、特徴量列順・列数、最適反復回数を照合。
- さらに各保存モデルに合成3行だけを投入し、LightGBMの確率/ロジット/TreeSHAPを元L1 emitterの数式で照合。2026年の実レースデータ・評価対象結果・払い戻しは扱わない。
- 保存モデルの復元性を確かめる検査であって、**2025年版重みを2026年用に転用・昇格しない**。実2026年モデルの学習も行わない。L3完了段階は3/8から変わらない。

## 2025年・実予測7本から凍結L1.6合議まで接続成功（2026-10-17追記）

- 初回の7王実予測取得Run #39014162160は失敗。原因は保存済みKaggleの**実ファイル名 `score.jsonl`** に対し `score.jsonl.gz` を要求していた404（保存実体の欠落ではない）。
- 読取専用ファイル一覧 Run #39014426860: **success**。7つの正本アーカイブすべてに `score.jsonl` が存在することを確認。既存の保存方式自体は変更しない。
- 対応: 専用Workflow `.github/workflows/l1-7king-2025-historical-council-freecpu.yml` の取得名を `score.jsonl` に修正。合議の正本スクリプト `research/materialize_l16_full_rank_consensus_v1.py` は一切変更せず使用。
- 修正後Run #39014501235: **success**。正本アーカイブの2025年Out-of-sampleスコア7本を読み込んで7王の合議順位を生成し、別途 `research/audit_historical_7king_2025_council_v1.py` で監査。
- **3,456レース、47,731頭、7王の合議順位を全頭生成・検査成功。** 初レースID 202501010101、生成gzipのSHA-256 `fbdf21d897a88a5c4aba96a69a42929c6921bddfdba6e09fb4a77f6dd1eed67a`。
- 出力データはGitHub Actionsの一時領域だけに作成。新規永続保存は行っておらず、恒久アーティファクトとして再利用できる状態とはまだ言わない。
- この成功は**2025年の歴史的なL1.6処理経路**の実データ再現であり、2026年用7王モデル学習・2026年予測・Outsider合流・L1.75・L2.1・L3は未完。2026年用モデルの昇格なし。
- 無料標準CPU・既存Kaggleデータの読取のみ。main、凍結研究仕様、2026年評価封印、課金禁止を維持。

## 2026年用7王候補を2024〜25年正本から実学習する工程（今回）

- 対象: 7王まとめて正本2024/25を復元し、元の特徴量投影・LightGBM学習を使う独立候補refit。
- 実行器: `research/run_7king_2026_unreleased_refit_once_v1.py`。2024/25スナップショットは年ごと1回だけ取得。王ごとに投影し、2025 OOSの反復回数を固定利用する候補方式。
- 2026本番用の予測出力入口には `--research-only` を追加。未承認モデルを自動昇格せず、出力も `ready_for_betting=false` とする。7王合議は元の凍結済みL1.6関数を再利用。
- 安全CI #39015022853: success。7王の候補／正式承認の混同拒否、非圧縮snapshot対応の合成テスト通過。
- 無料標準CPU実学習の試行: Run #39015275696、2026-10-17時点で実行中。**学習・推論の成功はまだ宣言できない。**
- 既存モデルはKaggle管理であるのに対しGitHubリポジトリはpublicと確認。新しいモデル重みの**公開Git保存は未承認の保管先変更**にあたるので、Workflowから保存処理を削除した。
- すでに動き始めたRun #39015275696は古いWorkflow定義で起動しているが、branch SHA一致のガードを満たさないよう後続コミットを加え、公開pushを阻止。新定義は手動起動専用・contents readのみ。旧Runが保存ステップまで進んだ場合はSHA差異で失敗する可能性があるため、最終Run conclusionは実学習自体の成否と区別する必要がある。
- 永続モデル保存・正式昇格は未実施。新しい有料保存、GPU、main変更、2026年結果学習/評価なし。2026レース情報の独立as-of監査も未完了。
- L3までの位置は **3/8** 完了、第4段階の2026実学習を試行中。Outsider合流・L1.75/L2.1/L3は未完。

## #39015275696 の失敗診断・保存修正（2026-10-17 17時台）

- Run #39015275696 は `conclusion=failure` だが、**実学習7王・本物の2026対象60レース/786頭の研究用推論・凍結L1.6合議は各Step成功**。失敗したのは最後の永続保存だけ。
- 2024/25正本 6,912レース・94,740行を使った各7王の候補refitが終了。元の2025 OOS best_iterationは 399 / 479 / 344 / 431 / 291 / 483 / 599。
- 2026年用の実推論7本の出力検査も成功。2026年の結果や払戻を使用せず、入力as-of監査は未完了、モデルは `CANDIDATE_2026_BLIND_UNRELEASED` のまま。
- 元のL1.6合議により **60レース・786頭分の研究用合議順位を生成・検証成功**。保存前の合議gzipのSHA-256: `453a17e10c0304b2c2831cb97eb3a691f7bf9f9a7460f524598bfd806491b70a`。
- 保存工程で実行中HEADと実験ブランチ先頭SHAを比較して不一致を検出。先行して危険な公開GitHubモデル保存を除去するコミットを加えたため、**意図した安全チェックによりpushを停止**。GitHub Actions Artifactは0件。model本体を後から復元できず、同一学習の再実行が必要になった。
- 修正: 実行Workflow `.github/workflows/l1-7king-2026-blind-refit-candidate-freecpu.yml` は公開GitHubモデル保存を使わず、**既存Kaggle dataset（CLI既定private）**を利用する。処理前に `scripts/kaggle-storage-preflight.sh` で既存所有datasetの保管量と研究上限180GiB、追加容量128MiB以上を確認。候補モデル・予測・合議を1つのZIP（最大64MiB）に収め、**保存後に既存 `persist_exact_kaggle_dataset.py` が遠隔manifestまで検証**。枠を確認できなければ学習前に停止。
- 修正後の無料標準CPU Run #39017718992 を起動。**起動時点で未完了、再実行結果/保存成功は未確認**。学習重複の継続を避けるため、以後は保存済みモデルを再利用する方針。
- 元の凍結研究、main、L2.1/L3 V3、2026評価封印は変更なし。利用者向け買い目作成・公開モデル昇格は一切行わない。
- 研究実装上は第4段階の計算処理と7王合議動作の実データ試験が通ったが、**永続保存・as-of監査・正式デプロイを含む第4段階完了には至っていない**。L3全体の完了段階は3/8のまま。

### 再実行時のKaggle無料容量チェック補修（17時台追記）

- Run #39017718992 は初回のKaggle無料保管量確認用に起動。GitHub Actionsは `$GITHUB_ENV` に記録した値を同一step内では環境変数として反映しないため、この検査部分を修正。
- 修正版コミット `ecc209652629ce2d7a6554b182663b02a9ce1601`、Run #39017803353 はキュー投入済み。無料保存枠を取得できない場合は**モデル学習前に停止**。モデルをGitHubへ公開保存しない。
- どちらも新規実学習とKaggle私有ストレージ保存の結果はまだ未確認。先のRun #39015275696で確認した **7王学習、7王実推論60R/786頭、L1.6合議60R/786頭の成功** と区別する。

## 再実行 #39017803353 完了・2026年7王の実計算と保存成功（2026-10-17）

- GitHub Actions [#39017803353](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/39017803353) は `success`。全job/stepが成功（2026-10-17 17:59 JSTころ）。
- **2024/25の6,912レース・94,740行で7王すべて再学習成功**。2026対象60レース・786頭の7王実推論成功、凍結L1.6の全頭合議順位生成成功。
- モデルは `CANDIDATE_2026_BLIND_UNRELEASED`、推論結果は `ready_for_betting=false`。2026年結果・払戻・確定オッズの学習評価使用なし。独立as-of監査なし、正式承認なし、L1.75/L2.1/L3未実行。
- Kaggle研究保管上限180GiB内を検査。実行前既存使用量63,165,169,248バイト。生成したZIPとmanifest計5,962,428バイト。
- **Kaggle dataset `pino1945/keiba-ml-2026-original-sevenking-candidate-v1` へ保存完了、remote manifest完全一致**。GitHubの公開リポジトリに学習済みモデル重みは保存していない。データセットのKaggle公開範囲は取得ログだけでは独立照合していない（既存保存スクリプトのデフォルト設定による作成）。
- 合議gzip出力のSHA-256: `10d4974e0856c16f06bf27d79dd2dc477f90f4490f87bcd90a6e01e78c94a809`。先の失敗Runからの非永続化出力ではなく今回の新実行結果。
- L3への工程数は厳密な「運用・正式承認済」カウントでは引き続き**3/8**。**第4段階の候補モデル学習・実推論・成果物永続保存まで達成**。第5段階の7王合議も研究用では成功したがOutsider接続は未完。

## L3への全8段階

1. 完了: 690の取得・受け渡し
2. 完了: 元血統V4 238列の実データ生成・保存
3. 完了: 690+238の再利用結合
4. 途中: 2024/25正本で7王候補の実学習成功。2026年60レース786頭への実推論とKaggle保存成功。ただしモデル正式承認・発走前as-of独立監査は未完
5. 途中: 2026年60レース786頭で凍結L1.6七王合議に成功。13 Outsider連携は未完
6. 未完: L1.75
7. 未完: L2.1
8. 未完: L3 V3掛金出力

L3後のスマホJSON配送は別の運用接続。完了段階に含めない。

## 変更範囲

新規ブランチ feature/l1-7king-2026-928-preflight-20261017 の監査スクリプト、凍結コードを再利用する未昇格refit候補・推論入口・安全CI、当文書。main・既存凍結モデル・評価方式・L1.75/L2.1/L3は変更なし。課金対象Artifact/Cache/GPU・新規Kaggle保存なし。
