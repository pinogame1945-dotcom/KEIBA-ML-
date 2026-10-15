#!/usr/bin/env python3
"""Summarize EXACT red->black conversions from completed fixed 3040 lines x17 gates.
No training, no outcome re-fitting, no new data, no extra download.
"""
import csv,gzip,json
from pathlib import Path
from collections import defaultdict,Counter

BASE=Path("research-results/l2-frozen-odds-l1-lenses-v1/run-38694334351")
OUT=BASE/"RED_TO_BLACK_ALL_ODDS_BANDS_SUMMARY_JA.md"
J=BASE/"red_to_black_counts_summary.json"
def read(path):
    with (gzip.open(path,"rt",encoding="utf-8",newline="") if str(path).endswith(".gz") else open(path,encoding="utf-8",newline="")) as f:
        return list(csv.DictReader(f))
def N(r,k):return int(r[k])
def num(n):return f"{n:,}"
def denom(r):
    return f'{r["family"]} / {r["order"]} / 相手{r["k"]}頭 / {r["odds_band"]}倍'
line=read(BASE/"all_3040_lines_x_existing_axes_yearly.csv.gz")
odds=read(BASE/"odds_band_x_existing_axes_yearly.csv")
pooled=[r for r in line if r["year"]=="2023-2025"]
bygate=defaultdict(list); bykey={}
for r in line:
    if r["year"]!="2023-2025":
        bykey[r["year"],r["original_line"],r["gate"]]=r
for r in pooled:
    bygate[r["gate"]].append(r)
    bykey[r["year"],r["original_line"],r["gate"]]=r
expected=3040
assert all(len(xs)==expected for xs in bygate.values())
gates=[x for x in bygate if x!="ALL"]
assert len(gates)==16
converted=defaultdict(list); green_lost=defaultdict(list)
for g in gates:
    for r in bygate[g]:
        before=N(r,"baseline_profit_yen")
        after=N(r,"filtered_profit_yen")
        if before<0 and after>0: converted[g].append(r)
        if before>0 and after<0: green_lost[g].append(r)
unique=set((r["original_line"] for rows in converted.values() for r in rows))
assert all(N(r,"baseline_tickets")>=N(r,"filtered_tickets") for rows in bygate.values() for r in rows)
rzero=[r for r in bygate["ALL"] if N(r,"baseline_profit_yen")<0]
rblack=[r for r in bygate["ALL"] if N(r,"baseline_profit_yen")>0]
rflat=[r for r in bygate["ALL"] if N(r,"baseline_profit_yen")==0]
assert len(rblack)==245 and len(rblack)+len(rzero)+len(rflat)==3040
annual_both={}
for gate,rows in converted.items():
    annual_both[gate]=sum(
        N(bykey[y,r["original_line"],gate],"filtered_profit_yen")>0
        for r in rows for y in ()
    ) if False else sum(all(N(bykey[y,r["original_line"],gate],"filtered_profit_yen")>0 for y in ("2023","2024","2025")) for r in rows)
odds_pool=[r for r in odds if r["year"]=="2023-2025"]
odds_gate=defaultdict(list)
for r in odds_pool:
    if r["gate"]=="ALL":continue
    if N(r,"baseline_profit_yen")<0 and N(r,"filtered_profit_yen")>0:
        odds_gate[r["gate"]].append(r)
odds_unique={(r["bet_type"],r["odds_band"]) for l in odds_gate.values() for r in l}
odds_annual={}
for r in odds:
    odds_annual[r["year"],r["bet_type"],r["odds_band"],r["gate"]]=r
out=[]
out+=["# 赤字→黒字に変わったオッズ帯｜全152買い方×20区間×17評価軸の実測集計","",
      "対象：2023〜2025年、各レースの既存期待値AI1位軸を固定。元の買い目生成ルールを変更せず、評価軸による買い目除外を比較。","",
      "**判定ルール**：元の3年間損益が厳密にマイナス、フィルタ後の同じ3年間損益が厳密にプラス。元損益0円を『赤字』に含めない。購入点数0件は黒字認定しない。","",
      "## 全3,040ラインでの変化","","| 評価軸 | 赤字→黒字ライン | うち2023/24/25すべて最適化後黒字 | 元黒字→赤字ライン |",
      "|---|---:|---:|---:|"]
for g in sorted(gates,key=lambda x:-len(converted[x])):
    out.append(f"| {g} | {len(converted[g])} | {annual_both[g]} | {len(green_lost[g])} |")
out+=["",f"元の3年合計黒字ライン：{len(rblack)}、赤字ライン：{len(rzero)}、損益0円ライン：{len(rflat)}。各評価軸の重複を除いた赤字→黒字の元ライン数：**{len(unique)}**。","",
      "## 7券種×20帯（同じ券種・倍率帯の全買い目を統合した場合）","","| 評価軸 | 赤字→黒字の券種×オッズ帯数 |",
      "|---|---:|"]
for g in sorted(gates,key=lambda x:-len(odds_gate[x])):
    out.append(f"| {g} | {len(odds_gate[g])} |")
out+=["",f"券種×帯（7券種×20区間、最大140種類）のうち1条件以上で赤字→黒字に転換した種類：**{len(odds_unique)}**。","",
      "## 転換の具体例（3年合計利益額上位。的中数・購入点数を隠さない）","",
      "| 評価軸 | 元152方式の馬券とオッズ帯 | 元利益（円） | 削減後利益（円） | 削減後点数 | 削減後的中 | 2023/24/25年の削減後利益（円） |",
      "|---|---|---:|---:|---:|---:|---|"]
top=sorted((r for v in converted.values() for r in v),key=lambda r:(-N(r,"filtered_profit_yen"),-N(r,"filtered_hit_tickets")))
for r in top[:22]:
    parts=r["original_line"].split("|")
    tag=" / ".join(parts)
    yearly="/".join(str(N(bykey[y,r["original_line"],r["gate"]],"filtered_profit_yen")) for y in ("2023","2024","2025"))
    out.append(f'| {r["gate"]} | {tag} | {num(N(r,"baseline_profit_yen"))} | {num(N(r,"filtered_profit_yen"))} | {num(N(r,"filtered_tickets"))} | {num(N(r,"filtered_hit_tickets"))} | {yearly} |')
out+=["","## 券種×倍率帯での具体例（転換後3年利益の順）","","| 券種 | オッズ帯 | 評価軸 | 元の3年利益（円） | 削減後の3年利益（円） | 購入点数 | 的中点数 | 2023/24/25年の削減後利益（円） |","|---|---|---|---:|---:|---:|---:|---|"]
best=sorted([r for rows in odds_gate.values() for r in rows],key=lambda r:-N(r,"filtered_profit_yen"))
for r in best[:25]:
    yearly="/".join(str(N(odds_annual[y,r["bet_type"],r["odds_band"],r["gate"]],"filtered_profit_yen")) for y in ("2023","2024","2025"))
    out.append(f'| {r["bet_type"]} | {r["odds_band"]} | {r["gate"]} | {num(N(r,"baseline_profit_yen"))} | {num(N(r,"filtered_profit_yen"))} | {num(N(r,"filtered_tickets"))} | {num(N(r,"filtered_hit_tickets"))} | {yearly} |')
out+=["","## 解釈上の重要注意","","- **2023〜2025年を見て選んだ最良条件は未知年の証明にならない**。各ラインに対し16通りも試すので多重比較・過剰適合が発生する。","- 上の『2023・24・25年すべて黒字』も同じ期間を見て確認した事後診断であり、将来安定を示す証明ではない。","- 元の全3,040ラインは買い方が重複する。ラインの黒字件数や利益を足しても『実際のユニーク購入馬券の総利益』にはならない。","- 7券種×20帯は152方式を横断してレースごとの物理馬券を重複排除した別集計。","- 過去の最終オッズで振り分けている。締切前に同価格で購入できる保証はない。","- 7王投票・Outsider投票・L1.75信頼度タグの真の値をこの実験で代替推定したわけではない。現在の評価は保存済みの純AI順位、市場順位、Correction Risk、能力スコア、市場補正確率など。","- 2026年は使用していない。",""]
OUT.write_text("\n".join(out)+"\n",encoding="utf-8")
J.write_text(json.dumps({"black":len(rblack),"red":len(rzero),"zero":len(rflat),"by_gate":{g:{"red_to_black":len(converted[g]),"original_black_to_red":len(green_lost[g]),"converted_each_year_positive":annual_both[g],"bet_type_odds_bands_red_to_black":len(odds_gate[g])} for g in gates},"unique_lines_converted_any_gate":len(unique),"unique_bet_type_odds_bands_converted_any_gate":len(odds_unique),"report":str(OUT)},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("RED_BLACK_SUMMARY",J.read_text(),flush=True)
print("RED_BLACK_TOP_LINES",json.dumps([{k:r[k] for k in ("original_line","gate","baseline_profit_yen","filtered_profit_yen","filtered_tickets","filtered_hit_tickets")} for r in top[:12]],ensure_ascii=False))
print("RED_BLACK_TOP_BANDS",json.dumps([{k:r[k] for k in ("bet_type","odds_band","gate","baseline_profit_yen","filtered_profit_yen","filtered_tickets","filtered_hit_tickets")} for r in best[:15]],ensure_ascii=False))
