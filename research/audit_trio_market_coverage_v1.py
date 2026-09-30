#!/usr/bin/env python3
import argparse,gzip,itertools,json,math
from pathlib import Path
import pandas as pd
from build_l2_bet_kings_dataset_v1 import decode_odds,payout_map,horse_number_map,canonical_numbers

YEARS=(2022,2023,2024,2025)

def readgz(p):
    with gzip.open(p,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()
    root=Path(a.backfill_root); rows=[]
    for y in YEARS:
        for dp in sorted((root/"data/daily").glob(f"{y}-*.jsonl.gz")):
            op=root/"data/odds/daily"/dp.name
            if not op.exists(): continue
            odb={str(x.get("race_id") or ""):x for x in readgz(op)}
            for pack in readgz(dp):
                rid=str((pack.get("race") or {}).get("race_id") or "")
                if not rid or rid not in odb: continue
                pm,present=payout_map(pack)
                if "TRIO" not in present: continue
                hno=horse_number_map(pack)
                nums=sorted(set(int(v) for v in hno.values() if int(v)>0))
                n=len(nums)
                if n<3: continue
                expected=math.comb(n,3)
                om=decode_odds(odb[rid])
                quoted=sum(1 for c in itertools.combinations(nums,3) if ("TRIO",canonical_numbers("TRIO",c)) in om)
                winning=[k for k,v in pm.items() if k[0]=="TRIO" and float(v)>0]
                winner_quoted=sum(1 for k in winning if k in om)
                cov=100.0*quoted/expected if expected else 0.0
                rows.append({
                    "year":y,"race_id":rid,"race_date":dp.name[:10],"field_size":n,
                    "expected_trios":expected,"quoted_trios":quoted,"missing_trios":expected-quoted,
                    "quote_coverage_pct":cov,"winning_trios":len(winning),
                    "winning_trios_quoted":winner_quoted,
                    "all_winning_trios_quoted":int(winner_quoted==len(winning) and len(winning)>0),
                })
    d=pd.DataFrame(rows)
    if d.empty: raise SystemExit("no TRIO audit rows")
    bins=[-0.01,50,75,90,95,99,99.999,100.001]
    labels=["<50","50-75","75-90","90-95","95-99","99-<100","100"]
    d["coverage_band"]=pd.cut(d.quote_coverage_pct,bins=bins,labels=labels,right=False)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"race-coverage.csv",index=False)
    by_year=d.groupby("year",as_index=False).agg(
        races=("race_id","nunique"),
        mean_quote_coverage_pct=("quote_coverage_pct","mean"),
        median_quote_coverage_pct=("quote_coverage_pct","median"),
        complete_market_races=("missing_trios",lambda s:int((s==0).sum())),
        winner_fully_quoted_races=("all_winning_trios_quoted","sum"),
        missing_trios=("missing_trios","sum"),
        expected_trios=("expected_trios","sum"),
        quoted_trios=("quoted_trios","sum"),
    )
    by_year["overall_quote_coverage_pct"]=100.0*by_year.quoted_trios/by_year.expected_trios
    by_year["winner_quote_coverage_pct"]=100.0*by_year.winner_fully_quoted_races/by_year.races
    by_year.to_csv(out/"by-year.csv",index=False)
    bands=d.groupby(["year","coverage_band"],observed=True).size().reset_index(name="races")
    bands.to_csv(out/"coverage-bands.csv",index=False)
    field=d.groupby(["year","field_size"],as_index=False).agg(
        races=("race_id","nunique"),
        median_quote_coverage_pct=("quote_coverage_pct","median"),
        complete_market_races=("missing_trios",lambda s:int((s==0).sum())),
        winner_fully_quoted_races=("all_winning_trios_quoted","sum"),
    )
    field.to_csv(out/"by-field-size.csv",index=False)
    summary={
      "contract":"L2_TRIO_MARKET_COVERAGE_AUDIT_V1_RESULT",
      "years":list(YEARS),
      "races":int(d.race_id.nunique()),
      "by_year":by_year.to_dict(orient="records"),
      "2026_locked":True
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_TRIO_MARKET_COVERAGE_AUDIT_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
if __name__=="__main__": main()
