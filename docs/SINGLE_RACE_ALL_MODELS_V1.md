# 1レース × 全登録モデル分析エンジン V1

## Purpose

与えられた **12桁レースIDまたは netkeiba レースURL** から、BACKFILLに保存済みのレースを特定し、**当該レースの着順・払戻・確定オッズを使わず**、過去実績だけで全出走馬の特徴量を生成する。対応する実体が**検証済みで保存されているモデルに限り**順位を計算する。

コード実装とモデルの配備は別物。2026-10-15時点では `models/promoted` に学習済みモデル本体が存在せず、デフォルト登録一覧は未配備と判定する。**現在のデフォルトのままでは全モデルのAI順位は出ない。** 特定馬への偽スコアや出走馬順による疑似順位は生成しない。

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
- [x] 7王・Outsider・L1.75・L2.1の配備不足を欠落ステータスで報告
- [x] 保存済み7王scoreをKaggle一覧から直接取得し、全7体順位を再生するコマンドを実装（実リモート検証は未実施）
- [ ] 実際に利用できる研究モデル実体・保存済みOOF scoreの実ダウンロードと本番レジストリ登録
- [ ] 特徴量窓が異なる多数モデルを1パスで高速化して計算
- [ ] アーカイブ1レースの実データ対照監査と同じ全頭スコアの再現
- [ ] EARLYに続く当日FINAL（タイムスタンプ付きJRA事前情報）
- [ ] 本番MODEL promotion、KEIBA-MOBILE表示／タイムスタンプ付き期待値
