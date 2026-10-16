# 富士S 2026-10-17：BACKFILL由来103項目の欠損37件の原因確定

対象は富士S（race_id=202605040611）の14頭、93履歴項目＋10プロフィール・血統項目だけ。2026-10-16時点の開催前検証。

## 結論

最初の103×14=1,442セル中、null 37件を6項目ごとに追跡した。**37件の原因をすべて特定**した。ただし、元BACKFILLの37セルを修正したという意味ではない。

| 対象 | 欠損数 | 検証結果 | 状態 |
|---|---:|---|---|
| distance.shorter_top3_rate | 7 | 定義どおりの該当短距離出走0件 | 正常null |
| distance.longer_top3_rate | 3 | 同じく該当長距離出走0件 | 正常null |
| horse_jockey.top3_rate | 2 | 今回の騎手との過去出走0件 | 正常null |
| history.margin_lengths | 8 | 前走1着で元着差欄が空：3件／ハナ・クビなどカテゴリーのみ：5件 | 正常null |
| history.margin_type | 3 | 前走1着で元着差欄が空 | 正常null |
| horse.coat_color | 14 | 元プロフィール全件に保存なし。競走馬ページの冒頭「牡4歳 鹿毛」等に存在。従来収集器は「毛色」のth→tdだけ読む | 14/14直接取得成功・元BACKFILL未修正 |
| **合計** | **37** | 正常なnull 23件＋元取得器の未対応14件 | 原因確定 |

### 確認実績

- BACKFILL：レース日別2,171ファイル（2026-10-12まで）と馬パック2,950ファイルを読み取り専用調査。
- profile kind=horse 保存レコード **101,126件中101,126件の coat_color が null**。元の profile_raw_json でも「毛色」のth-tdキーは0件。
- LIVEのnetkeiba競走馬プロフィール14頭を独立して取得、全馬の毛色を正規表現で実際に抽出。名前・IDを一致させた。
- 前走marginが空の3頭は、ミトノオルフェ、アドマイヤズーム、アンパドゥで、それぞれ前走1着。※着差は直前馬との間隔を表すため、1着の馬にはない。
- ハナ/クビで数値の馬身数を付けなかった他5頭も、元の margin_raw と margin_type を直接照合して確定。
- 距離別成績、馬と今回の騎手との成績のnullは、集計分母が実際に0だった。

### 証跡

- [欠損37件・全14頭・回収毛色の監査用JSON](../audit-results/fuji-20261017-missing37-resolution.json)
- [BACKFILL保存ソースの直接照合Actions](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38848448513)
- [netkeiba馬プロフィール14頭の毛色実取得Actions](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38848634321)

**現行BACKFILL本体の取得コードや過去レコードは変更していない。** 毛色の恒久対応は既存コレクターの取得位置変更と過去プロフィールへの補完を要するため、今回の調査・監査とは明確に分離する。競馬モデルの確定済み特徴量仕様と「正常null」の扱いも変更していない。
