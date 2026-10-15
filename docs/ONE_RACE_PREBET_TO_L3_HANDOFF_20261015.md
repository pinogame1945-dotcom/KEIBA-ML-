# 開催前に完成できる部分 — 1レース→L2.1→L3 紙上ハンドオフ（2026-10-15）

## 実装済み

- **GitHub ML PR #7**：`research/analyze-one-race.py` に研究アーカイブや7王推論の統合、L2.1/L3へのオプション入力 `--mobile-l21-capture`, `--original-l21-features`, `--l3-limits` を追加。既存BACKFILLレースのEARLY特徴量のための`--source-root`は今のところ必要。
- **単独L2/L3入口**：`research/run-one-race-paper-pipeline.py --race ID --out-dir DIR` は入力不足でも止まらず、`pipeline-status.json`へ不足物・段階を出す。L1.75を未promoteのまま「完成」とはしない。入力がそろうと、元の`ANCHOR_FULL`＋全頭馬別指標 → KEIBA-MOBILE発走前WIN/馬連/三連複/三連単同一スナップショット → 凍結STANDARD 50個 → 紙上買い目 → L3予算確認 をつなぐ。
- **L3 `research/l3-one-race-money.py`**：凍結L2.1物理馬券全件へ100円ずつ割当て。同一券の多重購入なし、買い目追加・削除・重み変更なし。予算、レース上限、点数上限を超えたら**全件停止**。信頼度や同年ROIを最適掛け金とみなさない。これは本番L3最適化ではなく、比較のベースライン／資金上限の確認。
- **KEIBA-MOBILE側**：別の未マージPR `https://github.com/pinogame1945-dotcom/KEIBA-MOBILE/pull/18`。正式出馬表`status=OFFICIAL`かつ未実施`raceStatus=SCHEDULED`の時だけ、`odds_history`の完全に同一`observedAt`のWIN・馬連・三連複・三連単を取り出す。必要行数はN出走馬なら `N+C(N,2)+C(N,3)+N(N-1)(N-2)`。全券の価格や1時刻が欠ける場合は拒否。結果・確定オッズを待って補完しない。
- **元の買い目再現**：2025-01-06の実24レースをFINALオッズで検証し、19レース27点が凍結原研究券と完全一致。5レースはBACKFILL`STARTED`なのに保存済み元予測から各1頭欠落していたため**BLOCKED**。欠けた結果やモデル出力を作らない。
- **2026の扱いを分離**：2026年の過去レースに対する結果バックテスト／保存済み確定オッズの再利用は引き続き封印。ただし、これから始まる**未来の2026年レース**について、端末が実際に取得を終えた発走前スナップショット（取得時刻≦現在時刻、現在時刻＜締切＜発走）を、予想・紙上検証の入口へ渡すことは許容。未知年結果は読まない。元L1.75未完成のため、2026年の「最終買い目予想」提供可能という意味ではない。

## テスト

```bash
python3 scripts/market-context-v1-smoke.py
python3 scripts/mobile-combination-bridge-smoke.py
python3 scripts/l21-frozen-prebet-price-smoke.py
python3 scripts/l3-one-race-money-smoke.py
python3 scripts/one-race-l3-pipeline-smoke.py
python3 scripts/verify_keiba_l2_1_freeze.py
```

実データの発走前入力が無い状態での現在の動作：

```bash
python3 research/run-one-race-paper-pipeline.py \
  --race 202507010212 --out-dir out/race/202507010212
# 出力: BLOCKED_MISSING_INPUTS と具体的に必要な入力をJSONに保存
```

元の**本当に時点適格な**入力がそろった後の紙上確認：

```bash
python3 research/run-one-race-paper-pipeline.py \
  --race 202507010212 \
  --mobile-export local/keiba-mobile-actual-prebet-capture.json \
  --original-features local/original-l21-prebet-horse-features.json \
  --limits config/l3-one-race-example-fixed100-limits.json \
  --out-dir out/race/202507010212
```

生成されるもの：`JRA-quote-book.json`、`L2_1-quoted-tickets.json`、`L2_1-preview.md`、`L3-fixed100-paper-portfolio.json`、`pipeline-status.json`。元のL1が別途存在する場合は`--analysis`で全頭ID・馬番を照合。完全なL1.75本番確率器は追加していない。

## まだ残る重要ゲート

1. **ライブORIGINAL ANCHOR_FULLの学習済み係数・特徴量とas-of証拠**：保存済みのFINAL研究予測を使った将来レースの本番推論は未実装。7王で軸を代理させない。
2. **KEIBA-MOBILEの実開催日の完全オッズ取得**：同一observedAtで複数種類が正確に保存されるか、三連単N×(N-1)×(N-2)の全組がJRAパーサから出るか、JRA最終オッズ判定が発走前に機能するかは当日実測が必要。現段階のサービスはコードから呼ぶエクスポート入口で、アプリUIのファイル保存・MLネットワーク連携は未実装。
3. **公式の発走・発売締切確定時刻**：締切を発走時刻の数分前と**推測しない**。外部から根拠付きで指定された時刻だけ利用。サーバー時刻の独立署名・真正性は未保証。
4. **5レース欠損馬**：保存済み元予測は15頭、BACKFILL出走STARTEDは16頭等。システムは欠落を明示し、補完せず停止。元の保存予測のカバレッジと当日取消等の照合が必要。
5. **L3利益最適化**：今回は「全件100円」のベースラインのみ。2023〜25同年選択ROIを真の未来利益の根拠にしない。購入APIは絶対に呼ばない。

**成果物は両リポジトリとも開発ブランチ／未マージPRに保存**。凍結L2.1やKEIBA-MOBILEのmain/APKを変更していない。無料標準CPUのみ。ArtifactやKaggle有料保存は行わない。
