# 2026-10-16 19時すぎ JRA発売中オッズの実データ取得検証

対象は2026-10-17（土）東京11R富士S、netkeiba race_id `202605040611`。凍結L1/L1.75/L2.1/L3、BACKFILL、main、MOBILEは変更しない。

## 確認日時・発売時刻
- 金曜2026-10-16 18:30 JSTから**土曜日全レース**が通常発売開始。公式根拠 https://www.jra.go.jp/dento/soku/instructions/hatsubai/jra.html 。**日曜日全レース**の通常発売開始は**土曜19:30 JST**であり、10/18秋華賞を金曜18:30発売と案内するのは誤り。
- [実ページのPOST遷移検証](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38885269866): 2026-10-16 19:09 JST 時点、出馬表に従来なかった富士S固有の「オッズ」CNAMEが追加。公式に記載された `doAction` の `cname` をURL捏造なしに抽出して、HTTP **POST**（`application/x-www-form-urlencoded`）で `/JRADB/accessO.html` へアクセス。GETではエラー013だったがPOSTでは成功。
- [実HTMLテーブル検証](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38885406593): JRA出典の「単勝・複勝オッズ（馬番順）」2026-10-17 東京11R 富士S、14頭分のテーブルを取得。表示時点「19時09分現在オッズ」。馬番1ファンダム単勝36.8、馬番4トロヴァトーレ単勝4.4、馬番5アドマイヤズーム単勝5.2などを確認（**取得時の値であり随時変化**）。単勝・複勝の列が分離されている。JRAのウェブ画面には掲載時刻表示があるが、アクセスしたサーバーのHTTP Date・実リクエスト受信時刻とは区別する。
- [7画面・計8券種の公式実データ検証](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38885604621): 実出馬表→富士S固有CNAME→公式レースオッズ内で各券種の「オッズ」リンクを抽出しPOST、7画面すべて富士S/2026-10-17東京11Rであることを独立に検証。7画面すべて表示時刻「19時09分現在オッズ」。3連単は182テーブル、3連複78テーブル、単勝複勝15行、馬連・ワイドそれぞれ13テーブル、馬単14テーブル、枠連8テーブルを観測。
- 券種は**単勝、複勝、枠連、馬連、ワイド、馬単、3連複、3連単**。倍率が表示される一方、組み合わせによっては**「票数なし」**と表示されるため数値と混同しない。未確定/票数なしを推測値・最終オッズで補わない。

## 成功と未完了を区別

- **成功**：金曜夜のJRA公式発売中の富士S固有オッズページへのPOST遷移、全8券種の公式HTMLページ取得と「実際に単勝/複勝や各組合せ倍率がある」ことの証明、表示更新時刻の特定。
- **未完了**：JRA各オッズテーブルからの**全組み合わせ馬番キーの厳密パースと件数検算**、正本8券種スキーマへの変換・アプリ/凍結L2.1と元L1.75への投入、票数なしを含む使えない組み合わせの扱い。各画面の閲覧成功だけで完全な全組み合わせオッズ取得・当日AIの実運用成功とは判定しない。
- **当日の天候・馬場・馬体重**は10/17に改めて確認。**秋華賞**の発売中オッズは10/17土曜19:30以降の検証対象。
- 無料のGitHub Actions標準CPU runnerのみで検証。課金Artifact、Cache、GPU、有料runner不使用。実際の馬券購入なし。

## 証跡
- `.github/workflows/fuji-onsale-jra-probe-free.yml` と `scripts/fuji_onsale_jra_deep_probe_20261016.py`
- `.github/workflows/fuji-odds-live-table-probe-free.yml` と `scripts/fuji_odds_live_table_probe_20261016.py`
- `.github/workflows/fuji-jra-eight-type-real-probe-free.yml` と `scripts/fuji_eight_type_jra_probe_20261016.py`

検証時刻と価格を長期保存し、あとから「当時のオッズ」を上書きしないこと。今回の確認値は再取得で変動しうる。
