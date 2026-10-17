# L1 7王 + 13 Outsider 実行資産・接続点調査（2026-10-17）

## 作業範囲
- 既存2026 L1純粋LambdaRank Run #38996450588 には一切変更しない
- 7王・Outsiderの研究ロジック、既存モデル、L1.75/L2.1/L3 V3は変更しない
- 2026年結果と過去の勝敗を混ぜない。2026年の結果評価をしない
- GitHub無料標準CPUのみ、追加Artifact/Cache/有料保存は使わない

## 正本から確認できた7王とOutsider
1. 7王エイリアス（`research/replay-archived-council.py`、`research/run_l1_council_rebuild_v1.py`）:
   `core4`, `pedlegacy`, `condition`, `full`, `light`, `pedv1`, `condrc`。
2. 元のCouncilは **7系統それぞれの `L1_TO_L2_OUTPUT_CONTRACT_V1` の全頭予測順位** を入力する。**単一LambdaRankの順位の複製では代用できない**。
3. 過去の各王の出力アーカイブはKaggleの `pino1945/keiba-l15-7k-<alias>-y<year>-v1`（2021〜2025）。再構築コード: `.github/workflows/l1-council-rebuild-v1.yml`。ここで公開されているのは既存年の各王の **予測スコアであり、2026年に再利用できる学習済み重み本体を確認したわけではない**。
4. GitHubの保存済みCouncil: branch `research/l1-council-rebuild-v1-20261004` に `research-results/l1-council-rebuild-v1/run-37183263875/races/y2021..y2025.jsonl.gz`。投票: `research-results/l16-outsider-support-safe-fast-v2/run-36724124925/y<year>/ballots.csv.gz`。`scripts/fetch-saved-council-github.sh` と `research/replay-archived-council.py` は **過去のOOF予測の再生専用**、2026年の新規推論では使わない。後者は明示的に2026年を拒否する。
5. 元の投票候補は13種類。正本 `research/replay-archived-council.py` の `OUTSIDERS` を凍結名として参照。過去の投票CSVは2026年対象の新規投票ではない。
6. 各王の元の学習候補セットは `research/l1-final-walkforward-v1.json`。特に血統・全特徴量型は `PEDIGREE` 等を使用。納品済み690列は *original CORE4 649 + 当日傾向19 + 状態22* の特定モデル入力であり、7王全部の完全な入力集合ではない。

## 出力をつなぐ時の固定条件
- 各2026年の出走馬集合は、GitHub固定 `handoff/690-feature-data-v1`（SHA: `113d6ab73bf840c95a761cf1d8df2b4a0281adf5`）の `ml_features/delivery/690/<date>/<race_id>.json` に照合する
- `race_id`, `horse_id`、全頭数、1..Nの非重複順位を **7系統それぞれ** 確認
- 投票13候補の `race_id`, `candidate`, Top3馬IDを確認
- 2025以前の投票/順位を2026年予測として読み込んだ場合は拒否
- 本コード `research/check_l1_seven_king_outsider_handoff_v1.py` は読み取り専用の **入力ゲート**。Council集計・新規推論・補欠AIによる代替は行わない
- `READY_FOR_FROZEN_REPLAY_INPUT_REVIEW` は入力が全て揃った時の合図で、本番推論・馬券購入を承認する意味ではない

## 実行例
```bash
python research/check_l1_seven_king_outsider_handoff_v1.py \
  --backfill-690 "/tmp/backfill-2026/202605040301.json" \
  --expert "core4=/tmp/score-core4-2026.jsonl" \
  --expert "pedlegacy=/tmp/score-pedlegacy-2026.jsonl" \
  --expert "condition=/tmp/score-condition-2026.jsonl" \
  --expert "full=/tmp/score-full-2026.jsonl" \
  --expert "light=/tmp/score-light-2026.jsonl" \
  --expert "pedv1=/tmp/score-pedv1-2026.jsonl" \
  --expert "condrc=/tmp/score-condrc-2026.jsonl" \
  --outsider-ballots "/tmp/outsider-ballots-2026.csv" \
  --output "/tmp/l1-readiness.json"
```
このコマンドはファイルが存在しなければ失敗/不足を報告する。仮の予測値は生成しない。

## 未解決
- **2026用の7王それぞれの真正なモデル・推論を実行可能にすること**。過去の7王OOFの保存ファイルを2026用モデルと混同しない。
- **2026用の13候補のOutsider投票を真正な処理で生成すること**。
- **690納品JSON以外に必要な特徴量の再構成と時点適格性の証明**。本作業では算出式を変えず新規作成もしない。
- 実際の発走前時点は引き続き `ASOF_UNVERIFIED_DO_NOT_BET`。7王が揃ってもそのまま購入可にはならない。

参照:
- `research/replay-archived-council.py`
- `research/run_l1_council_rebuild_v1.py` (research branch)
- `research/l1-final-walkforward-v1.json` (research branch)
- `.github/workflows/l1-council-rebuild-v1.yml` (research branch)
- `.github/workflows/l16-outsider-support-safe-fast-v2.yml` (research branch)
