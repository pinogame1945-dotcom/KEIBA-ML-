# L1.5 Blind-Spot Anatomy V1

## 目的

Gate V2 で効いた「七王 Top6 の視野の形」を分解し、
**何が七王全滅の事前シグナルなのか**を特定する。

これは Gate V3 を作る前の解剖実験であり、
新しい本番 Gate を決める実験ではない。

## 固定条件

- 2026 は封印
- odds / popularity / payout は禁止
- 結果は blind-spot 教師ラベルにのみ使用
- walk-forward: 2021→2022, 2021-22→2023, 2021-23→2024, 2021-24→2025
- GitHub standard CPU runner のみ
- GPU / paid runner / Artifact / Cache / new Kaggle persistence は使わない
- race_id を保持する

## 分解する軸

### A. BREADTH — 視野の広さ

Top6 七王集合が出走馬全体のどこまで見ているか。

- union size
- union / field size
- unseen horse count
- unseen share
- total Top6 slots

### B. REDUNDANCY — 同じ場所を見すぎているか

七王の42枠（原則 7×Top6）が、どれだけ同じ馬へ重複しているか。

- duplicate slots
- redundancy ratio
- support HHI
- effective horse count
- support entropy
- max support share
- top2 support share

### C. CORE_FRINGE — コアと端役の形

各馬が何王から支持されたか。

- support=1..7 horse count/share
- support>=2 / >=3 / >=4 / >=5 horse count/share
- singleton slot share
- core slot share (support>=4)
- fringe share (support<=2)

### D. PAIRWISE — 王同士の重なり方

- Top6 pairwise Jaccard mean/min/max/std
- pairwise intersection mean/min/max/std

### E. TOP3 — 上位3頭世界の形

Top3 について BREADTH / REDUNDANCY / CORE を圧縮した特徴。

### F. TOP1 — 王の本命投票形

- unique Top1 horse count
- max / second vote
- max vote share
- top2 vote share
- vote gap
- vote HHI / entropy
- 7, 6-1, 5-2, 4-2-1, 3-2-2 等の集中形を数値化

### G. ALL_ANATOMY

A〜F 全部。

## 比較方法

全て同じ BASE Gate 特徴へ 1群だけ追加する。

- BASE
- BASE+BREADTH
- BASE+REDUNDANCY
- BASE+CORE_FRINGE
- BASE+PAIRWISE
- BASE+TOP3
- BASE+TOP1
- BASE+ALL_ANATOMY

これで「どの群が +何pt 持ってきたか」を見る。

## 固定介入予算

各未知年を危険スコア順に並べて、

- 5%
- 10%
- 15%
- 20%
- 25%

だけ選ぶ。

これは ranking diagnostic。
結果ラベルは選択に使用しない。

主戦場は 10%。

## 単変量解剖

以下は train 年の分布から quintile 境界を作り、
未知年へその境界を適用して blind rate / lift を見る。

重点:

- top6_union_to_field
- top6_unseen_share
- top6_redundancy_ratio
- top6_effective_horse_count
- top6_support_entropy
- top6_support_hhi
- top6_singleton_share
- top6_core_slot_share
- top3_union_to_field
- top1_max_vote_share
- top1_vote_entropy

これにより LightGBM の feature importance だけでなく、
「値が高い/低いと本当に全滅率が上がるのか」を確認する。

## 成功条件

Gate V3 候補へ昇格する軸は、

- 10% budget で BASE を aggregate 改善
- 4年中複数年で改善
- 5/15/20/25%でも極端に反転しない
- 単変量 OOS bin でも解釈可能な lift が見える

こと。

一年度だけの勝ちは採用理由にしない。

## 出力

fold ごとに:

- 各 group AUC / PR-AUC
- 5/10/15/20/25% blind recall / precision / lift
- 10% selected race_id
- 10% caught blind race_id
- 10% false-positive race_id
- group feature importance
- 単変量 quintile table

最終 aggregate:

- group ごとの BASE 差
- 年別差
- stable wins
- race_id overlap

後でユーザーの別軸研究と race_id join 可能にする。
