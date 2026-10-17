# 7王 2026・L3までの到達状況（2026-10-17）

## 完了点: 3/8段階完了。第4段階は実入力行列・推論経路・再学習候補コードのテストまで完了

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

## L3への全8段階

1. 完了: 690の取得・受け渡し
2. 完了: 元血統V4 238列の実データ生成・保存
3. 完了: 690+238の再利用結合
4. 途中: 7王入力行列7/7・2026専用推論入口・2024/25学習refit候補コード実装。合成の真のLightGBM推論・全安全CI成功。ただし2026モデル未承認・長時間学習未実施・実2026予測値未生成
5. 未完: 13 Outsider・7王合議
6. 未完: L1.75
7. 未完: L2.1
8. 未完: L3 V3掛金出力

L3後のスマホJSON配送は別の運用接続。完了段階に含めない。

## 変更範囲

新規ブランチ feature/l1-7king-2026-928-preflight-20261017 の監査スクリプト、凍結コードを再利用する未昇格refit候補・推論入口・安全CI、当文書。main・既存凍結モデル・評価方式・L1.75/L2.1/L3は変更なし。課金対象Artifact/Cache/GPU・新規Kaggle保存なし。
