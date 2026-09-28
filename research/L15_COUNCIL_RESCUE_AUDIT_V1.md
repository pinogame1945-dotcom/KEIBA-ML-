# L1.5 合議制 救済・改悪・全滅監査 V1

## 目的

Council Router V3 の成績を「平均的に何%当たったか」ではなく、レース単位で分解する。

最終目的は L2/L3 の回収率改善であり、この監査自体を本番採用判定には使わない。

## レース分類

各 role × TopN × race を次の5群に固定する。

1. RESCUE（救済）
   - 現行 hard router は外れ
   - V3 は的中
2. REGRESSION（改悪）
   - 現行 hard router は的中
   - V3 は外れ
3. BOTH_HIT（両方的中）
   - hard router / V3 とも的中
4. MISSED_RESCUE（まだ救えた取りこぼし）
   - hard router / V3 とも外れ
   - その年・そのセルで許可された候補会議方式のどれかは的中
5. DEAD_ZONE（候補会議総崩れ）
   - その年・そのセルで許可された候補会議方式が全部外れ
   - V3 の oracle_hit=0 を用いる

DEAD_ZONE は「七王 Top6 全滅」と同義ではない。
これは Council V3 がその未知年で利用可能だった候補方式の範囲内での全滅であり、後でアウトサイダー研究と race_id で突合する。

## 主指標

各 role × TopN と全体について以下を保存する。

- レース数
- V3 介入数 / 介入率
- 救済数
- 改悪数
- 純救済 = 救済 - 改悪
- hard router 外れのうち救済できた率
- hard router 的中のうち壊した率
- まだ救えた取りこぼし数
- 候補会議総崩れ数 / 率
- 選択された会議方式の内訳
- 会議方式ごとの救済 / 改悪 / 純救済

## 特徴比較

race / data_coverage / consensus / 5王候補集合 / 5王 router 予測から、以下を比較する。

- 競馬場
- 芝/ダート/障害
- 距離
- クラス
- 頭数
- 天候 / 馬場
- 5王の候補重複率
- 5王の候補集合の広がり
- 5王が同じ馬を支持する強さ
- router 1位と2位の予測差
- router 予測のばらつき / エントロピー
- 既存 consensus / coverage 数値
- 各王 candidate_summary / expert_summary の集約値

数値特徴は、各分類群の平均が全体平均から標準偏差何個分ずれているかを計算し、絶対値上位を保存する。
カテゴリ特徴は、その分類群での出現率 ÷ 全体出現率を「持ち上がり倍率」として保存する。

## 並列設計

未知年を3本に分けて無料標準 CPU runner で並列実行する。

- 2023
- 2024
- 2025

各年 runner は ANCHOR / MAINLINE / COVER を同じ年の中で処理する。
これにより年度データを1 runner に積み上げず、前回の RAM 問題を避ける。

Kaggle への集中アクセスを避けるため開始を少しずつずらす。

## 入力

- Council Router V3 保存済み decisions
  - anchor
  - mainline
  - cover
- その年の 5王 role-features
- その年の 5王 OOS router predictions

Snapshot / 競走結果の再解析はしない。
2026 は読まない。
オッズは使わない。

## 出力

年ごとに Kaggle へ保存する。

- audit-races.jsonl.gz
  - race_id / role / top_n / 分類 / 選択方式 / hit状態 / 特徴
- summary.json
  - 各セルの救済・改悪・全滅集計
  - 会議方式別集計
  - 分類別の特徴差上位
  - カテゴリ特徴の持ち上がり倍率

保存先:
- keiba-l15-council-rescue-audit-y2023-v1
- keiba-l15-council-rescue-audit-y2024-v1
- keiba-l15-council-rescue-audit-y2025-v1

## 次段

この監査の後に作る Rescue Gate V4 は、V3 の「どれが当たりそうか」ではなく、

- hard router を変更すると救済できるか
- 変更すると改悪するか

を直接学ぶ。

基本は hard router 維持。
救済期待が改悪危険を十分上回るレースだけ会議方式へ切り替える。

アウトサイダー研究は DEAD_ZONE と race_id で合流し、
「5王会議では説明できないがアウトサイダーは救えるレース」を別系統の救援対象として扱う。
