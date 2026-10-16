# 2026-10-16 富士S: 今できる２課題の実証（JRA POST / 元L1推論入力）

レースID：202605040611（東京11R、2026-10-17）。既存のモデル、研究方法、BACKFILL、main、MOBILE、L2.1、L3 V3は未変更。

## A. JRAオッズ・エラー013

- **原因**: JRA公式出馬表の `doAction('/JRADB/accessO.html','pw15oli00/6D')` は、URLを組み立てるGETではなく、`cname=pw15oli00/6D` を `application/x-www-form-urlencoded` の**POST本文として送信**する方式。
- 正しい方式で公式レースページのリンクを実際にたどり、JRAの「オッズ開催選択」「今週のオッズ」の画面を取得できた（**エラー013は解消**）。
- **2026-10-16の発売前時点**では、取得したページに当該開催日の個別レースへ進む新しい `cname` が確認できない。トップへの自己リンクしかないため、実オッズは0件、実8券種のスクレイピングは未実証。
- 実装は公式HTMLに存在する `doAction` からCNAMEを抽出し、POSTを使用。CNAMEを計算・捏造しない。JRAの古い開催日の結果や、netkeibaの予想オッズで埋めない。
- 当初のrunはトップへの自己リンクを「ナビあり」と記録したため、**正確な状態「JRA_POST_INDEX_REACHED_NO_RACE_LINK」に修正し、再実行して成功**。
- 実測証跡：[GET/POST比較・実画面確認](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38853304940) / [正しい発売前ステータスへ修正した確認](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38853780329)。
- 販売開始後、日付・開催地→対象レース→券種オッズへ進めるかを実際に検証する必要がある。**実オッズ0件のため、L1.75/L2.1の当日実買い目は未生成**。

## B. 実際の14頭を既存1レースAI入力契約へつなぐ

- 出発点：netkeiba実ページの31特徴量、BACKFILLの過去93＋静的10項目、計134項目×14頭。元の監査 [join run](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38850791166) の同一厳格チェックを再実行。
- 2026-10-17の着順・確定オッズを混入せず `EARLY` へ。凍結 `l1-feature-set-contract-v1.json` の `BASE,OPPONENT,NETWORK,LAP,STYLE,DISTANCE,BACKFILL,AUTO,ACTOR,TIME_PACE` を適用し、`applyPredictionPhase` で当日未公表のweather, track_condition, body_weight, body_weight_diffなどを排除。
- 変換後は、元 PR#7 の `KEIBA_ONE_RACE_ASOF_V1` 形式、**14頭×670列**。これらは確定済みのモデル列順ではなく、特徴量セット・スキーマの入力候補。1頭当たりnullは17〜41件（複合統計のNULLも含む）。正本学習済みschemaとの完全一致は未検証。
- 元PR#7 commit `f9eedf6c2d82719571619b39b14816155b6f2cd0` から、**元の `research/score-one-race.py` と `models/one-race-registry.json` を無改造で取得して実行した**。
- 登録簿は31モデルすべて `unconfigured`。元推論コードは実データの入力契約を受理したが `scored_models=0, blocked_or_missing_models=31`。**予想ランキングは作らない**。この未配備状態を解消するには、原研究の学習済みモデル・metadata・列順・ハッシュ・予測時点の実証が必要であり、勝手な再学習・代理モデル構築・後知恵の利用はしない。
- 実測証跡：[既存の元推論コードへの接続＆31モデル未登録を検証](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38853698389)。
- 有料runner/GPU/Artifact/Cache不使用。元学習済みモデルのKaggle保存方式やL1/L2/L3仕様は変更していない。

## 残件

- 発売開始後：JRAオッズの開催選択→富士S→8券種の実際のPOST遷移、時刻の取れる範囲、組み合わせと馬番の真正性を実証する。
- 元7王・Outsider・L1.75の真正な推論器を導入できるか調査・照合が必要。今は実体を確認できず、入力形状だけ正しく通った。**「入力が通ること」と「オリジナルAI予想ができること」は別。**

すべて監査専用ブランチに保存。mainや凍結済みの研究仕様は変更しない。
