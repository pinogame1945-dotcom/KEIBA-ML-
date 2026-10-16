# 凍結L2.1を正本とした、既存AI → L3 V3過去レース再生

対象作業ブランチ：`feature/l3-v3-official-github-producer-20261016`

## 監査で判明したこと

凍結正本 `freeze/l2.1-standard50-to-l3-20261015`（`9655c717333636b5cc931f672e12dbc377e396bc`）の `docs/L2_1_FREEZE_TO_L3_HANDOFF_20261015.md` と `contracts/KEIBA_L2_1_FIXED_PURCHASE_V1.json` に、研究用の馬別出力・軸馬・信頼度/購入券の参照元が**すでに明示**されていた。最初から別の純粋AIモデルを探して作り直す必要はない。

### 凍結版に存在する、実在する元ファイル

| 役割 | 保存先（同一凍結ブランチ内） | 読む列・用途 |
|---|---|---|
| 純粋AI順位・補正p・Correction Risk | `research-results/market-fixed-anchor-residual-v2/run-38659856085/oos-fair-value-predictions.csv.gz` | `distill_ai_rank`, `risk`, `P__ANCHOR_FULL`, `market_rank`, `final_win_odds`, `_race_id`, `_horse_id` |
| 元能力スコア | `research-cache/market-fixed-anchor-residual-v2/fair-base-bf811fa2eab73db0.csv.gz` | 同じrace/horseキーで `distill_score_z` を照合。順位から逆算・捏造しない |
| 元ANCHOR_FULL軸馬 | `research-results/all-race-one-pick-v1/run-38668174639/all-race-one-picks.csv.gz` | `strategy=ANCHOR_FULL` の馬IDと `selected_value` を、元p×FINAL単勝価格の同点規則に照合 |
| 凍結L2.1で確定した実物理馬券・p3あり | `research-results/l3-l175-frozen-ticket-trust-audit-v1/run-38710197503/joined_ticket_trust_complete.csv.gz` | 13,264点。馬券キー、`min_p3`。元のL2.1を選び直さない |
| 同じ物理馬券・p3不明 | `research-results/l3-l175-frozen-ticket-trust-audit-v1/run-38710197503/tickets_missing_l175_signal.csv.gz` | **215点**。削除せず `UNKNOWN`。計13,479点 |
| 凍結50条件 | `contracts/KEIBA_L2_1_FIXED_PURCHASE_V1.json` | STANDARD50、9元テンプレート、原件数と記述ROI |

### 重要：前回の「L1.75 p3」説明を訂正

凍結L2.1引継書の信頼度は、**過去年だけで校正された7王＋OutsiderのL1.7 rank<=10の馬別p3**。この値が物理馬券の構成馬すべてに存在する場合のみ `min_p3` を付ける。215券は当時の保存出力にp3がなく、未知のまま残す。

したがって、V3正式配分 `REVERSE_P3_RACE_MAX300` に必要なものを「元の全馬L1.75独立モデルが存在するはず」と断定するのは誤り。**V3実証に用いた信号の正本はL1.7（保存済み7王＋Outsider過去年校正）**。GitHub上の前向きインターフェースも `KEIBA_L17_KING_OUTSIDER_P3_ASOF_V1`、`source=L17_KING_OUTSIDER_PAST_ONLY_WF_P3`、`coverage_scope=TOP10_OTHER_UNKNOWN` と明示する。スコアが提供されない出走馬を0.5等の確率で埋めず、既存馬券の数も減らさない。

## 今回の実装と成功結果

- `research/replay_frozen_l21_to_l3_v3_historical.py`：元OOSと原版能力z、元ANCHOR_FULL選択を読み、順位 `AI_RAW`, `AI_PROB`, `VALUE_EV` を元のソート・軸選択規則で復元。元の物理馬券13,479点/3年の件数・重複・p3既知/不明数を**必須整合性検査**し、`l3_v3_operational.reverse_p3_stakes` を同じ馬券に適用。**L2.1の馬券選択・元のAI順位・軸馬は変更しない**。
- `.github/workflows/frozen-l21-original-ai-to-v3-real-archive-proof.yml`：無料標準CPU GitHub Actions。1〜24個の過去race_id（2023〜25）をカンマ区切りで指定可能。正本ファイルのGitHub Metadataパス+SHA照合→同一runnerに5ファイル取得→1回だけ解析。合計13,479点の正本件数にドリフトがあれば停止。
- 2026封印。公表済みの研究`FINAL`オッズで作った出力は `HISTORICAL_FINAL_RESEARCH_ONLY` のみ。元の元評価は**2023〜25を使った50条件の後知恵選択**でもあるので、未来利益の証明ではない。レース結果・払戻列は、この再生器の得点・配分計算に読み込まない。
- **元データ実行例**：`202507010212`＝純粋AI最上位`2021106369`、期待値軸馬`2020103729`、凍結馬券**1点・V3紙上200円**。`202506010102`＝元純粋AI最上位`2022102965`、軸馬`2022103708`、凍結馬券**2点・紙上400円**。
- 成功Action: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38829589225
- 凍結L2.1原本、`main`、旧L1、研究L3 V3およびスマホアプリは変更していない。

## 実行方法（無料Actions）

GitHub Actions → **Frozen L2.1 real original AI to L3 V3 historical replay - Free CPU** → Run workflow → `race_ids` に `202507010212,202506010102` などを入力する。スクリプトは出力に `original_anchor_horse_id`, `rankings`, `original_risk_by_horse_id`, `tickets`, `l3_v3_total_yen` を記録するが、現時点では**runner内の一時JSON**であり、永続公開・スマホ配信は実装していない。Actionsログには要約のみ。課金対象となる追加Artifactを自動的にアップロードしない。

過去再生には、馬番→元horse_id対応の確定出馬表ファイルを接続していないので、馬別ランキングには馬IDのみ。購入券には元の馬番組合せが存在する。これは画面表示時のID照合が別途必要という意味。

## 「新しいレースを本当に予測する」には別の条件がある

上述の保存済みOOSスコアと元の13,479点の保存馬券は**すでに計算された2023〜25年の結果**。2026年の未知レースや未取得レースにそのまま転用できない。元の純粋AI `distill_ai_rank/score_z` とCorrection Riskを新しいレースで計算するには、元の訓練済みモデル（または厳密再現可能な学習手順・時点適格特徴量）がまだ必要。現行Gitソースツリーに保存済みLightGBM本体が見えないことと、「研究出力がない」ことは同義ではない。

当日のレースデータ収集はGitHub担当。アプリは完了JSONの受信・表示のみ。当日オッズの観測可能性は開催日まで深入りしない。今フェーズで元モデルを再訓練し直したり、新しいAIを独断で作成したりしない。
