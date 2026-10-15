# 1レース分析・L1.75原版ANCHOR_FULL：フェーズ区切り（2026-10-16）

## 今回ここまでを完了として区切る

1. **元のANCHOR_FULLのモデル式・特徴量を本当に再現**。
   - 元の`run_market_fixed_anchor_residual_v2.py`で使った19特徴量を、新しい`research/anchor-original-feature-geometry.py`として共通化。`research/score-original-anchor-asof.py`も全く同じ共通関数を呼ぶ。
   - 2025-01-06実レース`202507010212`・16頭の元の`fair-base`について、**19特徴量すべて元データと誤差0**。元の`ANCHOR_FULL`の係数、2022〜2024年だけで作った標準化（140,882馬）で評価した勝率が保存済み2025年OOS勝率と最大誤差`1.11e-16`。
   - 元データは**事後FINALオッズ**。検査専用であり、これだけで「発走前に予想した」証明にはならない。race_idと照合した`target_win`、払戻、結果列を推論特徴量として使用しない。
   - CI: [実レース16頭・本物の19特徴量と勝率の完全再現](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38758391024)

2. **発走前ソースの安全境界を強化**。
   - モバイルの公式`JraEntry.canonicalHorseId`と、L1能力・Correction Riskの各馬`horse_id/horse_number`は一頭ずつ完全一致を要求。番号だけ合う異なる馬への予想を拒否。
   - `target_win`、`finish_position`、`result_status`、`final_win_odds`等の事後結果列がcompanionsに混ざれば拒否。途中棄権(DNF)する馬も、**出走時点では予測対象から除外しない**。発走後に判明するDNFを見越した除外は絶対に使わない。
   - 元のAI能力・Correction Riskが発走前に作られたという証拠をまだ独立に確認していない。そのため**`PAPER_PRIOR_PARAMETERS_SCORED_WITH_ASSERTED_SOURCE_TIMES`は本番モデルの合格を意味しない**。
   - CI: [全券種・発走前検査・L1.75 paperからL3まで](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38758979317)

3. **最大18頭立てでの明白なオッズ取得制約を解消**。
   - 18頭立て時に必要な価格は単勝18、馬連153、三連複816、三連単（順序あり）4,896の計**5,883通り**。過去の5,000件上限は誤りで、正常な18頭立てを拒否したため、L2.1の組合せ（単勝は別配列）の上限を厳密に**5,865**へ修正。
   - 全5,883件が同一時点に揃った**合成18頭立て**を走らせ、凍結L2.1の紙上券生成まで成功した。実際のJRAが全組合せを一時点で配信することは未検証。
   - `scripts/l21-18-horse-full-odds-smoke.py`の無料標準CPU結果、`MAX_18_HORSE_JRA_WIN_AND_THREE_BETTYPE_SOURCE_COMPLETE 5883 combo 5865 physical_tickets 7`。
   - [成功ログ](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38758979317)

4. **L2.1の正本は無変更**。
   - 既存の9テンプレート×STANDARD50条件／STRICT26条件を凍結したまま使用。
   - DNFが原因で元の保存予測から漏れた5頭は、史実を説明する研究時のみに識別。事後DNF除外なら過去24R・33券に一致するが、未来の全頭予測で使ってはならない。
   - 完走馬だけで再現した当初19R・27券、および事後DNFの扱いを明記した実24R・33券の監査は`38756551224`（**両方とも事後研究であり、発走前実現の証明ではない**）。
   - L3は100円均等／予算・点数・資金上限を超えたら全件ブロック。追加の資金配分最適化はしない。

## 未達と次フェーズを限定する

**本物の前日〜当日PRE_RACEデータだけで次の２つを確保する必要がある**：

- 元のL1能力`distill_ai_rank,distill_score_z,distill_score_gap_to_best`と原版Correction Risk`risk`を、全出走馬について**結果を使わず新規推論するモデル本体・当時既知特徴量・model selection cutoff**。いまは元の2025年OOS値の再現と、これら全値が与えられたときの入力ゲートはできたが、新規のリスクモデル推論・真の当日L1.75推論は未認証。7王の順位や保存済み事後スコアでこの入力を**埋めない**。
- KEIBA-MOBILEによる実開催日・元JRAの時刻付き正式出走馬・単勝/馬連/三連複/三連単の全価格。**取得時刻・最終かどうか・実際の購入締切**は本当に検査してから、その時点のオッズだけL2.1へ渡す。時計や`source_system`が単にJSONに書かれているだけでは公開時刻の独立証明にならない。

`research/run-one-race-paper-pipeline.py` が既に`--original-anchor-parameters`、`--original-ability-risk`、`--mobile-export`、`--limits`を受け取り、揃った場合に元のP→凍結L2.1→L3の紙上接続を行う。揃わなければ`pipeline-status.json`を残して`BLOCKED`で終了し、ライブ購入を許可しない。

## 終了状態

- 対象ブランチ：`feature/single-race-engine-20261015`、レビューPR #7。
- KEIBA-MOBILEは別ブランチ`feature/l21-prebet-odds-export-20261015`、レビューPR #18。mainは未変更。
- 2026の未知年結果による評価・パラメータ選択は行わない（開催前の新規予想と歴史的2026検証を区別）。
- GitHub Actions無料標準CPUのみ。有料runner・GPU・課金Artifact/Cache/モデル保存を利用しない。
- **このフェーズをここで停止**。次は実際の原版L1能力＋Correction Riskの学習済みモデル探索・推論と、本物の前日・当日入力時点保証。断片研究や凍結L2.1書換えには進まない。
