# L1.5 Seven-King Blind-Spot Gate V2

## 目的

V1 で確認できた「七王 Top6 全滅には事前特徴がある」という信号を、
実運用に近い **固定介入予算** で評価する。

V1 の主問題は閾値目的関数だった。

- recall 型: 51–76% のレースへ介入
- balanced / precision 型: 0% 介入

そこで V2 は binary decision の閾値最適化を主役から降ろし、
**危険度ランキングの上位何%を救援対象にするか**へ変更する。

## 固定ルール

- 2026 は封印
- L1/L1.5 で odds / popularity / payout は禁止
- GitHub 標準 CPU runner のみ
- GPU / 有料 runner / 新規有料保存は禁止
- 新規 Kaggle dataset / GitHub Artifact / Cache は使わない
- 既存の Router Feature Snapshot V1 と outsider rescue ledger を再利用する
- race_id を結果に保持する

## 比較するモデル

### BASE

V1 と同じ Router Feature Snapshot V1 ベース。

### PATTERN

BASE に、七王の「選び方の形」を明示的に加える。

#### Top1 集中

- unique horse count
- max vote / second vote
- max vote share
- top2 combined vote share
- vote gap
- vote HHI
- normalized vote entropy
- vote pattern（例: 7 / 6-1 / 5-2 / 4-2-1 / 3-2-2）

#### Top3 / Top6 支持構造

各馬について「7王中何王の候補集合に入ったか」を数え、以下を作る。

- union size
- support max / mean / std
- support HHI
- support=1..7 の馬数
- support>=4 の馬数
- support<=2 の馬数
- unanimous horse count
- union size / field size

狙いは、
「一頭集中」「二極集中」「狭い合意」「広い分散」「全員同じ方向に寄った盲信」
をモデルが直接読めるようにすること。

## walk-forward

| test | train |
|---|---|
| 2022 | 2021 |
| 2023 | 2021–2022 |
| 2024 | 2021–2023 |
| 2025 | 2021–2024 |

## 介入予算

各 fold で以下を比較する。

- 5%
- 10%
- 15%
- 20%
- 25%

2種類を出す。

### TEST-RANK diagnostic

未知年の予測スコアだけを高い順に並べ、その年の上位 q% を選ぶ。

結果ラベルは順位選択に使わない。
これは「ランキング能力」を測る診断値。

### TRAIN-QUANTILE deployable

train 年だけの OOF 予測から上位 q% の threshold を決め、
その threshold を未知年へそのまま適用する。

こちらを実運用に近い主評価とする。

## 主指標

各 budget について:

- intervention rate
- blind spots caught
- blind recall
- precision
- enrichment = precision / blind prevalence
- lift_vs_random_recall = blind recall / intervention rate
- false positives
- false negatives

特に **10% budget** を主戦場にする。

実全滅率が約 10% なので、

> 全レースの約10%だけ見て、本物の全滅を何%詰め込めるか

を中心に見る。

## V2 の勝ち条件

PATTERN が BASE に対して、

- 10% budget の blind recall を複数年で改善
- aggregate でも改善
- 5/15/20/25% でも極端に崩れない
- AUC / PR-AUC でも少なくとも悪化しない

なら、王の選び方パターンは Gate の正規特徴へ昇格候補。

## 出力

fold ごとに:

- BASE / PATTERN の AUC, PR-AUC
- TEST-RANK 各 budget
- TRAIN-QUANTILE 各 budget
- 10% budget の race_id
- true blind race_id
- caught blind race_id
- false-positive race_id
- top feature importance

最終 ledger は小さい JSON/README のみ Git へ保存する。

## 次段

V2 の危険レース race_id と、ユーザー側の別軸研究 race_id を後で join する。

比較したいもの:

- 両方が危険判定
- Gate V2 のみ
- 別軸のみ
- 両方通常
- それぞれの真の全滅捕捉率

モデルを先に混ぜない。
まず独立研究として相関と補完性を見る。
