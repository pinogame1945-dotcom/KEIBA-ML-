# BACKFILL 690 -> ML 実データ受け取り結果（2026-10-17）

- 送信元: `pinogame1945-dotcom/KEIBA-BACKFILL`
- 正本ハンドオフ: `handoff/690-feature-data-v1` / commit `113d6ab73bf840c95a761cf1d8df2b4a0281adf5`
- 元データ: `ml_features/delivery/690/index.json`
- 10月10日: 24レース315頭
- 10月17日: 36レース471頭
- 60レース・786頭・各690特徴量。10月17日は事前観測時点の独立保証なし。
- BACKFILL索引上の全レース SHA-256 を現物と照合する。
- ML側の `research/receive_backfill_690_v1.py` を用いて2日分を日付別に取り込む。
- 受信先はGitHub Actionsの `$RUNNER_TEMP/ml-input-YYYYMMDD/` のみ。ダウンロード済みデータをGit・Artifact・Cacheへ保存しない。
- モデルの学習・推論は**行わない**。2026年の結果や成績は評価しない。
- 元研究との690特徴量値の完全一致検証は行わない。
- `readiness=STRUCTURE_ACCEPTED_RESEARCH_ONLY_NO_BETS` のまま。

実行するワークフロー:
`.github/workflows/receive-backfill-690-real-freecpu.yml`

成功した場合でも、一時ランナー上で読み込みに成功したという事実であり、
ランナー終了後の永続データ転送やML本番予想の完了を意味しない。
必要なレースIDはBACKFILLの索引から都度取得する方式。
