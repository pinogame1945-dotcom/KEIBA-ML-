#!/usr/bin/env python3
import csv,gzip,json,math,statistics
from collections import defaultdict,Counter
from pathlib import Path

SRC=Path("research-results/l2-trio-split-residual-v4/run-37146919444/race-diagnostics.csv.gz")
OUT=Path("research-results/l2-trio-v4-both-fail-audit")
COURSE={"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}

def bucket(n):
    n=int(n)
    if n<=10:return "<=10"
    if n<=13:return "11-13"
    if n<=16:return "14-16"
    return "17-18"

rows=[]
with gzip.open(SRC,"rt",encoding="utf-8",newline="") as f:
    for r in csv.DictReader(f):
        kd=float(r["king_logloss_delta_vs_market"])
        od=float(r["outsider_logloss_delta_vs_market"])
        rid=str(r["race_id"])
        date=str(r["race_date"])
        x=dict(r)
        x["year"]=int(r["year"])
        x["field_size"]=int(r["field_size"])
        x["market_true_p"]=float(r["market_true_p"])
        x["king_true_p"]=float(r["king_true_p"])
        x["outsider_true_p"]=float(r["outsider_true_p"])
        x["king_delta"]=kd;x["outsider_delta"]=od
        x["both_fail"]=(kd>=0 and od>=0)
        x["month"]=date[5:7]
        x["course_code"]=rid[4:6] if len(rid)>=6 else "??"
        x["course"]=COURSE.get(x["course_code"],x["course_code"])
        x["field_bucket"]=bucket(x["field_size"])
        rows.append(x)

n=len(rows); bf=[r for r in rows if r["both_fail"]]
base=len(bf)/n
def quantile(v,q):
    a=sorted(v)
    if not a:return None
    pos=(len(a)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    if lo==hi:return a[lo]
    return a[lo]*(hi-pos)+a[hi]*(pos-lo)

def desc(rs,key):
    v=[float(r[key]) for r in rs]
    return {
        "n":len(v),"mean":sum(v)/len(v),"median":statistics.median(v),
        "q25":quantile(v,.25),"q75":quantile(v,.75)
    }

slice_rows=[]
for dim in ["year","month","course","field_bucket"]:
    groups=defaultdict(list)
    for r in rows:groups[str(r[dim])].append(r)
    for value,g in groups.items():
        fn=sum(r["both_fail"] for r in g)
        rate=fn/len(g)
        slice_rows.append({
            "dimension":dim,"value":value,"races":len(g),"both_fail":fn,
            "both_fail_pct":100*rate,"lift_vs_all":rate/base if base else None,
            "reportable":int(len(g)>=100)
        })

reportable=[r for r in slice_rows if r["reportable"]]
over=sorted(reportable,key=lambda r:(-r["lift_vs_all"],-r["races"]))[:12]
under=sorted(reportable,key=lambda r:(r["lift_vs_all"],-r["races"]))[:12]

# How wrong were corrections in both-fail races?
worst=sorted(bf,key=lambda r:-(r["king_delta"]+r["outsider_delta"]))[:20]

summary={
    "contract":"L2_TRIO_V4_BOTH_FAIL_AUDIT",
    "source_run":37146919444,
    "definition":"KING logloss delta >= 0 AND OUTSIDER logloss delta >= 0 versus pure TRIO market on the realized winning trio.",
    "races":n,
    "both_fail_races":len(bf),
    "both_fail_pct":100*base,
    "by_year":{str(y):{
        "races":sum(r["year"]==y for r in rows),
        "both_fail":sum(r["year"]==y and r["both_fail"] for r in rows),
        "both_fail_pct":100*sum(r["year"]==y and r["both_fail"] for r in rows)/sum(r["year"]==y for r in rows)
    } for y in sorted(set(r["year"] for r in rows))},
    "field_size_all":desc(rows,"field_size"),
    "field_size_both_fail":desc(bf,"field_size"),
    "market_true_probability_all":desc(rows,"market_true_p"),
    "market_true_probability_both_fail":desc(bf,"market_true_p"),
    "king_delta_both_fail":desc(bf,"king_delta"),
    "outsider_delta_both_fail":desc(bf,"outsider_delta"),
    "most_overrepresented_reportable_slices":over,
    "least_overrepresented_reportable_slices":under,
    "warning":"market_true_p and both_fail are outcome-conditioned diagnostics. They describe failures after the result and must not be used as pre-race router features.",
}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
with (OUT/"slices.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["dimension","value","races","both_fail","both_fail_pct","lift_vs_all","reportable"])
    w.writeheader();w.writerows(slice_rows)
with (OUT/"worst-examples.csv").open("w",encoding="utf-8",newline="") as f:
    fields=["year","race_id","race_date","field_size","market_true_p","king_true_p","outsider_true_p","king_delta","outsider_delta"]
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in worst:w.writerow({k:r[k] for k in fields})
print(json.dumps(summary,ensure_ascii=False))
