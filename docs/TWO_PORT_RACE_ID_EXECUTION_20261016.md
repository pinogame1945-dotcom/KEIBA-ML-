# 2つの入口（2026-10-16）— レースID 1件 / 複数件

**共通入口**: \`research/execute-races-by-id.py\`。Netkeibaをブラウザで開かず、渡された12桁IDだけを識別子として扱う。結果は \`out/two-port-races/summary.json\`、各レースの個別フォルダ。

## 入口A：過去レースをIDだけで即時研究再生

\`\`\`bash
python3 research/execute-races-by-id.py --mode historical \
  --race-ids '202506050811,202507010212' \
  --out-dir out/history-by-id
\`\`\`

- 対象は元の学習/評価記録が保存された **2023〜2025**。2026結果は禁止。2022以前で元予測が揃わない場合はBLOCK。
- 元の \`ANCHOR_FULL\` OOS予測・能力キャッシュ・当時の原版軸をGitHubの凍結保存先から**年/ID分だけ一度取得してrunnerで再利用**。元の市場OOSに入っている \`_race_date\` から対象開催日を解決。レース名/馬番をウェブ検索しない。BACKFILLの対象開催日のレース・FINALオッズをmetadata/sha検証して1回取得。
- 元の凍結STANDARD50条件、9テンプレートを変更せず歴史的L2.1買い目を再生。**FINALオッズ＝事後研究**であって本当に発走前に購入できた記録ではない。
- 配管の安全境界のため、本番 \`L3.allocate\` はFINALオッズを拒否する。代わりに**L3の100円固定／レース予算・点数上限・全券or見送りと同じ制約だけを独立した研究用audit**で計算する。本番L3実行や実投票の成功と表現しない。
- 予算デフォルトは \`config/l3-one-race-example-fixed100-limits.json\` の**1500円の研究例**。投資助言・承認された購入資金ではない。
- 途中棄権DNFの予想漏れ等があれば未来の結果を遡って除外せず、そのレースだけ止める。
- 課金なし。取得キャッシュはrunner一時ディスク内のみ。

### 実検証（2026-10-16）

GitHub Actions: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38772406266

| ID | 元ANCHOR_FULL軸馬番 | 凍結L2.1買い目 | flat100研究額 |
| --- | ---: | ---: | ---: |
| 202506050811 | 3 | 6 | 600円 |
| 202507010212 | 7 | 1 | 100円 |
| 合計 | | 7 | 700円 |

これは「事後研究として原版から買い目まで自動再生した」の実証。**発走前L3予測の実証ではない**。

## 入口B：当日JRAデータ

\`\`\`bash
python3 research/execute-races-by-id.py --mode today \
  --race-ids '202606050811,202606050812' \
  --captures-dir out/keiba-mobile-captures \
  --limits config/l3-one-race-example-fixed100-limits.json \
  --out-dir out/today-by-id
\`\`\`

- \`out/keiba-mobile-captures/{ID}.json\`: **KEIBA-MOBILE \`exportL21PrebetCandidate\`の出力**。当日、発走前、JRA非FINAL確認、馬番、馬ID、全単勝/馬連/三連複/三連単物理オッズの同一取得時刻を要求。
- \`{ID}-original-features.json\`: **正規の原版ANCHOR_FULL・全頭・時刻確認済みの発走前予測**。無ければ \`original-anchor-parameters.json\` と \`{ID}-original-ability-risk.json\` の原版能力/risk組で再構築。モデル順位の代用禁止。
- 元の\`run-one-race-paper-pipeline.py\`で、そのまま凍結L2.1→凍結L3の紙上資金配分を実行。過去日や発走済みの出馬表を当日として使わせない。
- **未解決の外部入力**：GitHub runnerにスマホのデータは自動共有されない。KEIBA-MOBILEのexport自体にも当日の実JRA実績は未確認。元の能力・Correction Risk予測を全頭について本当に発走前に生成し保存する上流モデルも未整備。これらが不足すれば停止（買い目やオッズを捏造しない）。
- 2件の当日入力JSONを**合成**してL2.1→L3まで通すテストは成功：https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38772406266 。これは生JRAデータや本番の購入可能性を証明しない。

## 変更管理

- 開発ブランチ: \`feature/multi-race-id-pipeline-20261016\`、draft PR #8
- 既存の凍結 \`freeze/l2.1-standard50-to-l3-20261015\`、元の凍結STANDARD50、L3資金管理、MOBILE mainは**未変更**。
- GitHub Actionsは**無料の公開標準CPU**だけ。認証GitHub REST metadataにrunner付属のread-onlyトークンを使用（公開API 403 rate-limitを回避）。課金対象Artifact/Cache/GPUなし。
