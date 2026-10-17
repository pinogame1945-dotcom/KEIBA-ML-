# 7王 2026・L3までの到達状況（2026-10-17）

## 完了点: 3/8段階完了。第4段階の入力列照合まで完了

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

## 第4段階の学習器制約

凍結正本: research/l1-raceclass-absorption-walkforward-v1.json、snapshot generation eb13d4096519167b。学習2年rolling。2026は評価封印。

- 元の2025年版は2023/24を学習、2025を検証。2026にそのまま採用しない。
- 元の run_feature_arena.py、train-staged-lightgbm.py および emit_l1_to_l2.py は、検証期間に2026が入ると拒否する。
- 元の train-staged-lightgbm.py は検証ラベルによるbinary_logloss早期停止50ラウンドに依存する。
- 2024/25の2年で再学習して2026をブラインド推論する際、2026の正解に依存せず反復回数を確定する正式な手順はまだ凍結仕様に存在しない。勝手に固定反復、2025重み流用、学習窓変更、2026結果利用を行わない。

## L3への全8段階

1. 完了: 690の取得・受け渡し
2. 完了: 元血統V4 238列の実データ生成・保存
3. 完了: 690+238の再利用結合
4. 途中: 7王必要列照合7/7完了。2026モデル作成・推論は未完
5. 未完: 13 Outsider・7王合議
6. 未完: L1.75
7. 未完: L2.1
8. 未完: L3 V3掛金出力

L3後のスマホJSON配送は別の運用接続。完了段階に含めない。

## 変更範囲

新規ブランチ feature/l1-7king-2026-928-preflight-20261017 の読取専用検査スクリプト、GitHub無料標準CPU workflow、当文書のみ。main・既存凍結モデル・評価方式・L1.75/L2.1/L3は変更なし。課金対象Artifact/Cache/GPU・新規Kaggle保存なし。
