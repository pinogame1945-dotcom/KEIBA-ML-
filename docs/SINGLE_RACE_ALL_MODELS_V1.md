# 1レース × 全登録モデル分析エンジン V1

## Purpose

与えられた **12桁レースIDまたは netkeiba レースURL** から、BACKFILLに保存済みのレースを特定し、**当該レースの着順・払戻・確定オッズを使わず**、過去実績だけで全出走馬の特徴量を生成する。対応する実体が**検証済みで保存されているモデルに限り**順位を計算する。

コード実装とモデルの配備は別物。2026-10-15時点では `models/promoted` に学習済みモデル本体が存在せず、デフォルト登録一覧は未配備と判定する。**現在のデフォルトのままでは全モデルのAI順位は出ない。** 特定馬への偽スコアや出走馬順による疑似順位は生成しない。

## 最短経路：レースIDだけで保存済み20系統を確認

**BACKFILLの全履歴を取得する必要はない。** 2021〜2025年の
保存済みwalk-forward研究データが存在するレースなら、
ID/URLを指定して7王の各順位・13 OutsiderのTop3投票を再生できる。

```sh
python3 research/replay-one-race-id.py \
  --race 202506010101 \
  --output-dir out/race-id-only
```

`out/race-id-only/race-id-archive-analysis.json` に全頭をまとめ、
研究元ファイル2点のSHA-256、13 Outsiderの投票点、7王の合議参考順位を記録。
`--race` は netkeiba URL でもよい。
利用する GitHub 正本ファイルは年度別2点のみ（合計約1.4 MB）だが、
初回取得はGitのフィルタ付きリポジトリ参照が必要になる。
再実行時は同じディレクトリのsessionを再利用する。

これは **新しい馬券予想の推論** ではなく
**既存年度の7王・13 Outsiderの保存済み予測再生**。
Outsiderは各体のTop3外順位を推測しない。
モデルの再学習や2026未知年の開封は一切しない。

**実レース接続実績（2026-10-15）**：
GitHub Actions `38732534277`、実保存済み
`202506010101`・16頭で`REAL_RACE_ID_ONLY_20_SIGNAL_PASS`。
7王・13 Outsider投票の完全性、全馬ID、事後ラベル非使用の
race-ID-only JSON化に成功した。
これは**過去スコアの再生成功であり、新しい未来レースへのAI再推論成功ではない**。

実レースの発走前の時刻付き出馬表・オッズ・モデル本体が未整備のため、
実戦運用の稼働確認とは分離する。

## Quick start

BACKFILLとKEIBA-MLを並べてチェックアウトした場合:

```sh
node research/predict-one-race.mjs \
  --race 'https://race.netkeiba.com/race/shutuba.html?race_id=202305020811' \
  --source-root ../KEIBA-BACKFILL \
  --history-from 2021-01-01 \
  --phase EARLY \
  --features-out out/one-race-features.json \
  --results-out out/one-race-rankings.json
```

動作は以下。
1. `data/manifest.json`の**SUCCESSで実在するファイルパス**から指定レースを検索する。ID未発見または重複なら停止。
2. `history-from` から対象レース**前日まで**の全開催日についてSUCCESS/確認済み非開催/中止・延期が判定できるか監査。不明な日があれば停止。
3. 過去日のデータだけを日付順に流して既存の特徴量生成器を更新。当該日と未来日を更新しない。
4. 当該レースのアーカイブから結果・ラップ・払戻・確定オッズ・馬体重・天候・馬場状態を**除いて**出馬表相当を組み立てる。
5. 学習時と共通のBASE/OPPONENT/NETWORK/AUTO/BACKFILL/ACTOR/TIME_PACE/PEDIGREE計算と、過去走だけのLAP/STYLE/DISTANCE計算で全頭特徴量を出す。
6. 各モデルを **SCORED / BLOCKED / UNAVAILABLE** で報告し、許されたモデルのみ順位を付ける。順位の同点はhorse_id昇順で固定（結果ページ行順に依存させない）。

標準ではEARLYのみ。アーカイブのFINAL天候・馬体重は、いつ発表されたかが分からないため、現段階で当日予想の根拠に流用しない。実際の金曜時点の出走取消や延期を正確に再現できるとは限らず、**史実のレース出馬表を結果ページから再構成した模擬予想**として扱う。

## Integrated one-race command (recommended)

```sh
python3 research/analyze-one-race.py \
  --race 'https://race.netkeiba.com/race/shutuba.html?race_id=202505020811' \
  --source-root ../KEIBA-BACKFILL \
  --history-from 2021-01-01 \
  --output-dir out/one-race-example
```

Outputs `features.json` (all horses × features), `model-rankings.json`
(each declared model scored or explicitly unavailable), and `analysis.json`
(the combined machine-readable report). A model's `UNAVAILABLE` status is
not a poor prediction; it means the model artifact has not been promoted.

With an already-configured Kaggle CLI and the repository's authorized API
credential, `--seven-kings` also adds **historical saved OOF rankings** from
the verified seven king Kaggle datasets; it is **not** a newly run live model.
The EARLY feature panel and FINAL archived 7-king output have separate origin
labels and cannot be presented as if they were the same prediction phase.

```sh
python3 research/analyze-one-race.py \
  --race 202505020811 \
  --source-root ../KEIBA-BACKFILL \
  --history-from 2021-01-01 \
  --seven-kings \
  --output-dir out/one-race-example
```

Use `--require-scored` to fail explicitly if no verifiable model predicted.
No GitHub Actions artifact/cache or paid compute is required. The command only
**reads** BACKFILL and Kaggle; output files are small and local.

## Kaggleなしの既存7王＋13 Outsider（GitHub保存済み）

**20モデルの保存済み予想を、1レースIDだけで再生できる。**

正本はすでにGitHubの `research/l1-council-rebuild-v1-20261004`
ブランチに保存済みの2021〜2025年度別アーカイブ。各年の7王
`rank_matrix` は、既存の7王スコアをtie-safe再順位した後に生成された。
`outsider_support` と別保存の13体 `ballots.csv.gz` は照合して一致を要求する。

```sh
python3 research/analyze-one-race.py \
  --race 202507010212 \
  --source-root ../KEIBA-BACKFILL \
  --history-from 2021-01-01 \
  --github-saved \
  --output-dir out/race-202507010212 \
  --require-scored
```

`--github-saved` はGitHubの既存コミットから該当年の既製zip
（Council約0.8MB／Outsider ballot約0.55MB）だけを取得し、
ローカルsession内に再利用する。Git tree一覧で**実在するファイルと
サイズを確認してから**取得。Kaggle・rolecmp router・GPU・課金保存なし。

最終 `analysis.json` には全馬の `model_ranks`、
7王別順位・13 Outsider別Top3票順位・累計投票点をまとめる。
7王の公式L2.1ルールに**勝手に置換せず**、M127_S0の純粋7王比較順位
として記録する。Outsiderの「Top3外」は順位を捏造せずnullのまま。

**重要：この7王と13 Outsiderは保存済みのhistorical OOF評価を再生
している。** `prediction_phase=FINAL` の事後保存レース入力を学習検証に
使っているため、レース発走前の取得時刻を完全再現したと主張しない。
`--history-from` を短縮した実データsmokeは特徴量接続確認であり、
学習時点の全履歴と完全一致する証明ではない。

### 実レース統合検証の確定結果（2026-10-15）

**2025年1月6日 `202507010212`、16頭で1レース統合を成功確認。**
保存済み7王7体、Outsider13体、L1.7の**過去年学習／当該年検証の3着内確率（上位10頭）**を、同じrace_id / horse_idに接続した。

- 本番と同じ出走馬IDで全16頭の結果非参照特徴量を生成。
- 7王は独立7体分の保存済み順位、Outsiderは13体Top3票と得点を再生。
- L1.7は **2021〜2024の学習結果から予測された2025年のOOS保存値** を取得。L1.7 `p3` を各馬へ出力するが、**7王順位を変更しない**。
- L1.7の元CSVに含まれる `finish`, `y3` は意図的に読み飛ばす。結果列を書き換えても出力が不変な単体テストも追加。
- **L1.75市場補正とは別の研究モデル**。L1.75と偽装しない。
- archiveの前処理は大規模Git cloneを廃止。 GitHub Contents metadata に保存済み正式パスとサイズがあるものだけ取得し、SHA1 Git blob照合で真正性を確認した。404・別名当てずっぽう・課金Artifact／Cacheなし。
- `--github-saved` で3種類の保存済み情報を統合。全元モデル新規推論ではない。

<a>正式監査ログ：</a> `https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38734212514`

```sh
python3 research/analyze-one-race.py \
  --race 202507010212 \
  --source-root ../KEIBA-BACKFILL \
  --history-from 2021-01-01 \
  --github-saved \
  --require-scored \
  --output-dir out/real-one-race
```

上記のコマンドはBACKFILLに `history-from` 以降の完全な過去日が揃っている場合に利用する。CIでは既に検査済みの2025-01-05/06の2日だけに短縮した**結合スモーク**であり、過去の全履歴からの特徴量再現一致を証明するものではない。元の学習に使われた`FINAL`の配信時刻が不明なので、保存スコアを正確な購入時点の予想とみなさない。

### 実レースでの確認済み範囲

- BACKFILL実データ 2025-01-05/06 2日分を抜き出し、
  **202507010212、16頭**の特徴量生成・結果非参照・モデル欠落表示が
  無料CPU CI成功（run `38729589817`）。
- GitHub研究アーカイブから直接、
  **202506010101、16頭**の7王順位と**13 Outsider票**の一貫性検証に成功
  （run `38730555347`）。結果・確定オッズは順位に投入しない。
- この2経路を同一レースで結合し、L1.7上位10頭のOOS3着内確率まで追加した無料CPU実証が成功（run `38734212514`）。
- 2019以前など保存済みfold外のレースは、特徴量生成と保存済み20モデル
  予測再生を混同しない。モデルが存在しない場合は `UNAVAILABLE`。

## All-model registry

`models/one-race-registry.json` に各研究系統の登録枠がある。以下の形式で **実体を確認したときだけ** 種別を変更する。

### Promoted LightGBM

```json
{
  "name": "ACTUAL_PROMOTED_L1",
  "kind": "lightgbm",
  "model_path": "promoted/real-model.txt",
  "metadata_path": "promoted/real-model-meta.json"
}
```

※`model_path`等はレジストリファイルのあるディレクトリに対する相対パス。実ファイル名は例示であり、現在存在しない。

正規のモデル `metadata` が持つ特徴量リスト・順序・カテゴリ水準・学習／検証終了日・予測段階・Feature Sets・履歴窓・small-sample-policy・SHA256を厳密に比較する。予測対象の日付以降のデータを学習／検証に使ったモデルは禁止。異なる履歴窓のモデルは、**同じ特徴量出力を流用せず、プロファイルを分けて再生成する必要がある**。V1では異なるプロファイルは`BLOCKED`にする。

### Precomputed historical OOF scores

```json
{
  "name": "ACTUAL_ARCHIVED_SCORE",
  "kind": "archived_oof",
  "score_path": "scores/actual-scores.csv.gz",
  "metadata_path": "scores/actual-scores-meta.json",
  "race_field": "_race_id",
  "horse_field": "_horse_id",
  "score_field": "_score"
}
```

`metadata` は `historical_oof: true` で、予測対象より前の `split.train_end` と `split.valid_end` 、プロファイル等を記録する。単なる後知恵で再計算したスコアをOOFと偽装しないこと。リプレイは**モデル再推論ではない**とレポートに明示される。古い研究結果の学習・校正・推論時点が証明できないものは未配備扱い。

### Ensemble

```json
{
  "name": "VERIFIED_COUNCIL",
  "members": ["VERIFIED_KING_A", "VERIFIED_KING_B"],
  "weights": [1, 1],
  "method": "rank_average"
}
```

単に名前を7王/Outsiderにしただけでは正当な本物の合議ではない。**元の投票配点・同票順位処理を正本と突合できた場合だけ**正式名で登録する。

## 既存研究との境界

- L1能力モデルは市場オッズ非使用。
- L1.75/CORRECTION_RISK・Outsiderは、元の検証済みモデルもしくはスコアが特定できるまでは配備しない。
- L2.1 STANDARDの**凍結済み50組ルール、購入軸、校正確率**は不変。L2.1はレース直前のタイムスタンプ付き馬券オッズが必要で、歴史的FINALを転用してはいけない。
- 2026は未知年検証を封印。2026の着順からモデル選択・回収率検証・パラメータ調整を行わない。
- 最初の接続試験はsynthetic fixtureのみ。未検証の実データ成功や実馬AI順位の生成を主張しない。
- GitHub Actions は無料の標準CPU runnerまで。無断の有料保存や推論インフラを使わない。推論はローカルに保存し、Actions Artifact/Cacheへアップロードしない。


## Saved 7王を実際に読み込む経路（2021〜2025）

**新規モデル学習なし**で、保存済み7王 horse-level score を直接再利用する経路を追加した。

```sh
# 既存の Kaggle API token をこの端末の環境変数へ設定した状態で実行
bash research/replay-seven-kings-kaggle.sh 202505010111 out/archived-seven-king.json
```

処理は7体それぞれについて、Kaggle Dataset のメタデータ一覧取得 → 実在する `score.jsonl.gz` または `score.jsonl` を1件特定 → 当該ファイル1件をダウンロード → runner内で使い回す。レガシーの `rolecmp` router Dataset は取得しない。404なら即時停止、429/一時的5xxのみ指数バックオフ。ファイルサイズが不明または100MB超の場合も事前に停止する。

保存先は `mktemp` 配下で、正常終了・エラー時とも自動削除。レポートJSONだけを指定したローカルパスへ出力する。GitHub Actions Artifact や有料Storageへのアップロードはない。

正本の7体名は `research/l1-council-rebuild-v1-20261004` の `EXPERTS` を使用：`core4`, `pedlegacy`, `condition`, `full`, `light`, `pedv1`, `condrc`。保存済みscoreの `L1_TO_L2_OUTPUT_CONTRACT_V1` と全馬一貫性・順位の連続性を検証する。

7体の各順位はそのまま表示し、参考用に **rank-sum 7体比較順位** を算出する。これは2026-10-04研究 `order_for(mask=127,share=0)` の実装と対応するが、**凍結された7王routerやL2.1正式軸ではない**。別名を付けて識別する。

保存済みfoldのもとの学習時点・早期停止の適格性を監査するまでは「その日その場で予想した」とは主張しない。これはあくまで**保存済み歴史的予想の再生**であり、2026の新規推論ではない。

## Progress

- [x] レースIDまたはURLの解決、manifest実在パスから読み込み
- [x] 過去日限定の特徴量生成器、履歴完全性ゲート
- [x] 全頭の特徴量、結果非参照、EARLY限定
- [x] LightGBM/保存済みOOF/登録済みアンサンブルの接続口
- [x] 1コマンドで特徴量・全モデル状態・保存済み7王（任意）を統合するJSON出力
- [x] Githubからの保存済み7王＋13 Outsiderを1フラグで取得・計算（2021〜2025、各既製アーカイブの年別）
- [x] 7王・Outsider・L1.75・L2.1の配備不足を欠落ステータスで報告
- [x] 保存済み7王scoreをKaggle一覧から直接取得し、全7体順位を再生するコマンドを実装（実リモート検証は未実施）
- [x] 7王＋13 Outsiderの保存済みランク／票をGitHub実アーカイブから復元し、2025年16頭の実レースと結合
- [x] 保存済みL1.7のOOS上位10頭3着内確率を検証して統合（L1.75とは別）
- [ ] 本番の生推論用モデル実体のpromote、L1.75の元データ／モデル接続
- [ ] 特徴量窓が異なる多数モデルを1パスで高速化して計算
- [ ] アーカイブ1レースの実データ対照監査と同じ全頭スコアの再現
- [ ] EARLYに続く当日FINAL（タイムスタンプ付きJRA事前情報）
- [ ] 本番MODEL promotion、KEIBA-MOBILE表示／タイムスタンプ付き期待値
