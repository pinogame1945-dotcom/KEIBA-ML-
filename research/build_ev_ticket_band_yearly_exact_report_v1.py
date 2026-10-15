#!/usr/bin/env python3
"""Loss-inclusive EV-anchor odds-band tables for all 152 betting styles and 3 OOS years.
Uses previously saved ticket-price bands only, no odds requests or training.
All cash, tickets and hits exact, ROI expressed as rational 100*return/stake.
"""
import csv,gzip,json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ROOT=Path("research-results/l2-ticket-odds-band-profit-v1")
YEARS=("2023","2024","2025")
ALL="2023-2025"
LABELS={"WIN":"単勝","PLACE":"複勝","QUINELLA":"馬連","WIDE":"ワイド",
 "EXACTA_FIRST":"馬単1着固定","EXACTA_SECOND":"馬単2着固定",
 "EXACTA_BOTH":"馬単表裏","TRIO":"三連複",
 "TRIFECTA_FIRST":"三連単1着固定","TRIFECTA_SECOND":"三連単2着固定",
 "TRIFECTA_THIRD":"三連単3着固定","TRIFECTA_ANY":"三連単全位置"}
ORDERS={"SINGLE":"軸のみ","AI_PROB":"市場補正AI順",
 "MARKET":"市場人気順","AI_RAW":"純AI順"}
MAIN=(("SINGLE","0","WIN"),("SINGLE","0","PLACE"),
 ("AI_PROB","2","QUINELLA"),("AI_PROB","2","WIDE"),
 ("AI_PROB","2","EXACTA_FIRST"),("AI_PROB","2","EXACTA_SECOND"),
 ("AI_PROB","2","EXACTA_BOTH"),("AI_PROB","2","TRIO"),
 ("AI_PROB","2","TRIFECTA_FIRST"),("AI_PROB","2","TRIFECTA_SECOND"),
 ("AI_PROB","2","TRIFECTA_THIRD"),("AI_PROB","2","TRIFECTA_ANY"))

def yen(value):
 d=Decimal(str(value))
 assert d==d.to_integral_value(),("yen with fraction",value)
 return int(d)

def roi(return_yen,stake_yen):
 return f"100*{return_yen}/{stake_yen}%" if stake_yen else "—"

def read(path):
 with (gzip.open if path.suffix==".gz" else open)(path,"rt",encoding="utf-8",newline="") as f:
  return list(csv.DictReader(f))

def write(path,rows):
 with path.open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def transform(path,tag):
 raw=read(path)
 lookup={}; families=set(); lo={}
 for r in raw:
  assert r["axis"]=="EV"
  ident=(r["order"],r["k"],r["family"],r["odds_band"])
  year=r["year"];assert year in (*YEARS,ALL)
  values=(int(r["tickets"]),int(r["hit_tickets"]),
       yen(r["stake_yen"]),yen(r["return_yen"]),yen(r["profit_yen"]))
  assert values[2]==values[0]*100 and values[4]==values[3]-values[2]
  assert (ident,year) not in lookup,(ident,year)
  lookup[ident,year]=values
  families.add(ident[:3]);lo[ident[3]]=Decimal(r["odds_low"])
 keys=sorted({a for a,_ in lookup},
     key=lambda a:(list(LABELS).index(a[2]),list(ORDERS).index(a[0]),int(a[1]),lo[a[3]]))
 out=[]
 for order,k,family,band in keys:
  ident=(order,k,family,band)
  row={"券種":LABELS[family],"相手選定":ORDERS[order],
   "相手頭数":int(k),"馬券オッズ帯":band,"軸馬":"期待値AI1位",
   "買い方ID":f"EV|{order}|{k}|{family}"}
  each=[]
  for year in (*YEARS,ALL):
   v=lookup.get((ident,year),(0,0,0,0,0))
   n,hit,stake,ret,profit=v
   assert 0<=hit<=n and stake==100*n
   row.update({f"{year}_購入点数":n,f"{year}_的中点数":hit,
      f"{year}_投資円":stake,f"{year}_払戻円":ret,
      f"{year}_損益円":profit,f"{year}_ROI厳密式":roi(ret,stake)})
   if year!=ALL:each.append(v)
  assert tuple(sum(v[i] for v in each) for i in range(5))==lookup.get((ident,ALL),(0,0,0,0,0)),("yearly drift",ident)
  out.append(row)
 assert len(families)==152,(tag,len(families))
 write(ROOT/f"EV軸_全152買い方_オッズ帯別_年度推移_{tag}.csv",out)
 return out,families

def main():
 fine,f=transform(ROOT/"ev-every-ticket-strategy-fine-odds-profit.csv.gz","詳細20区間")
 broad,b=transform(ROOT/"ev-every-ticket-strategy-broad-odds-profit.csv","通常13区間")
 assert f==b
 md=["# 期待値AI1位を軸にした馬券｜全オッズ帯・全検証年の実測損益","",
 "## 前提","",
 "- 対象：2023・2024・2025年。各3,456レース、合計10,368レース。2026年未使用。",
 "- 7券種・合計152買い方について、馬券1点ごとの最終オッズで帯を分ける。軸馬の単勝オッズではない。",
 "- 1点100円。外れ券も購入点数と投資額に含む。全額は円の整数で省略・丸めなし。",
 "- 複勝・ワイドは表示オッズ範囲の下限を用いる。",
 "- ROIの厳密値は 100×払戻円/投資円（%）。この比は丸めずに表記する。",
 "- 同じ買い目が別々の方式に重複することがあるため、異なる方式の利益を直接足し合わせない。",
 "- 以下の12種類の表は各20オッズ帯を省略せず収録。残りの140買い方・各20帯も年度推移CSVに収録。",
 "- 過去の最終オッズを利用した事後検証。購入締切前の価格保証や利益の再現性の証明ではない。",
 ""]
 for order,k,family in MAIN:
  subset=[r for r in fine if r["買い方ID"]==f"EV|{order}|{k}|{family}"]
  assert len(subset)==20,(order,k,family,len(subset))
  md.extend(["## "+LABELS[family]+"｜"+ORDERS[order]+"／相手"+k+"頭","",
   "| 馬券オッズ帯 | 2023:点数/的中 | 2023:払戻/投資(円) | 2024:点数/的中 | 2024:払戻/投資(円) | 2025:点数/的中 | 2025:払戻/投資(円) | 3年:点数/的中 | 3年:払戻/投資(円) | 3年損益(円) |",
   "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
  for r in subset:
   def sizes(y):return str(r[y+"_購入点数"])+"/"+str(r[y+"_的中点数"])
   def cash(y):
    stake=r[y+"_投資円"]
    return str(r[y+"_払戻円"])+"/"+str(stake) if stake else "—"
   md.append("| "+" | ".join([r["馬券オッズ帯"],
      sizes("2023"),cash("2023"),sizes("2024"),cash("2024"),
      sizes("2025"),cash("2025"),sizes(ALL),cash(ALL),
      str(r[ALL+"_損益円"])])+" |")
  md.append("")
 (ROOT/"EV軸_全券種_全20オッズ帯_検証年推移_完全版.md").write_text("\n".join(md)+"\n",encoding="utf-8")
 print("EV_ALL_BANDS_YEARS_COMPLETE",json.dumps({
  "fine_rows":len(fine),"broad_rows":len(broad),"styles":len(f),
  "main_style_full_20_band_tables":len(MAIN),"years":[*YEARS,ALL],
  "cash_exact_yen":True,"roi_unrounded":"100*return/stake",
  "no_training_no_downloads":True,"year_2026_locked":True
 },ensure_ascii=False),flush=True)

if __name__=="__main__":main()
