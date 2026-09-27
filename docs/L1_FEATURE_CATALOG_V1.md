# L1 FEATURE CATALOG V1

## 目的

これは「全部モデルへ入れる一覧」ではない。

**KEIBA-MLが将来使ってよい材料の正式台帳**である。

- 正本データ: KEIBA-BACKFILL
- ML側スクレイピング: 禁止
- L1でオッズ: 禁止
- 過去成績: 対象レースより前の日付だけ
- 特徴選び: walk-forwardの中だけで行う

## 今の分類

- AVAILABLE: 現在のKEIBA-MLコードで作れる
- DERIVABLE: 元データがあれば安全に作れる候補
- BACKFILL_REQUIRED: 将来の未実装ソース用。現時点の正規化race-condition項目は実装済みなので0件
- PROHIBITED_L1: L1へ入れてはいけない
- L2_ONLY: 買い方側だけで使う

件数はJSON本体の `counts` を正とする。

## 重要な判断

### 血統ID

父・母・母父などのIDをそのままL1へ入れる方式は、いったん禁止にした。
IDを暗記しているだけの可能性があるため。

血統は今後、

- 種牡馬の過去時点成績
- 距離別成績
- 芝/ダート別成績
- クロス
- 血統の似方

などへ作り直す。

### BACKFILLの履歴充足

内外回り、周回数、レースクラス、grade、年齢/性別/斤量条件、指定区分、raw条件文、着差正規化はソース実装済み。

「項目が実装されているか」と「必要な過去年代まで埋まっているか」は分離する。

- Feature Catalog: 項目/安全性の台帳
- BACKFILL_READINESS_V3: 履歴範囲、開催日分類、pack/parser version、coverageの門番

したがって過去回収中であっても、実装済み項目をBACKFILL_REQUIREDには戻さない。

### 自動で作る特徴

自動生成は許可するが、何でも総当たりにはしない。

- 過去N走の平均・ばらつき・傾向
- 同競馬場 / 同距離 / 同馬場などの条件別成績
- 出走馬内での順位や平均との差
- 意味の近い数値同士の差・比率

さらに、**どの特徴を残すかを決める作業そのものも未来のレースを見てはいけない。**

## ファイル

機械用の正本:

`contracts/l1-feature-catalog-v1.json`

チェック:

`npm run check:feature-catalog`

このチェックは学習を回さず、台帳の形式と危険項目だけを確認する。


## 学習直前のCatalog門番

L1で禁止した事実は、Catalogの `model_keys` と実際のtrainer入力列を照合する。

例:

- `jockey_id`
- `trainer_id`
- 生の血統ID
- final odds / popularity / payout
- 対象レース自身の結果列

禁止列が1つでもモデル入力へ混ざった場合、trainerは学習開始前に失敗させる。
