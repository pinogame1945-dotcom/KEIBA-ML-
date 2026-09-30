#!/usr/bin/env python3
import argparse,gzip,itertools,json,math
from collections import defaultdict
from pathlib import Path
import pandas as pd

def rows_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def num(x):
    try:
        v=float(str(x).replace(",","").strip())
        return v if math.isfinite(v) else None
    except Exception:
        return None

def final_tuple(raw):
    if not isinstance(raw,list) or len(raw)<3:
        return None
    return raw[3:6] if len(raw)>=6 else raw[:3]

def valid_trio_key(key):
    s=str(key)
    return len(s)==6 and s.isdigit() and len({int(s[i:i+2]) for i in (0,2,4)})==3

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--backfill-root",required=True)
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    root=Path(a.backfill_root)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    detail=[]
    key_patterns={}
    rejected_examples=[]
    for race_path in sorted((root/"data/daily").glob("2025-*.jsonl.gz")):
        odds_path=root/"data/odds/daily"/race_path.name
        if not odds_path.exists():
            continue
        odds_by={str(r.get("race_id") or ""):r for r in rows_gz(odds_path)}
        for pack in rows_gz(race_path):
            rid=str((pack.get("race") or {}).get("race_id") or "")
            od=odds_by.get(rid)
            if not od: continue

            started=[]
            all_nums=[]
            for e in pack.get("entries") or []:
                try: hn=int(e.get("horse_number"))
                except Exception: continue
                if hn<=0: continue
                all_nums.append(hn)
                if str(e.get("entry_status") or "STARTED")=="STARTED":
                    started.append(hn)
            started=sorted(set(started))
            all_nums=sorted(set(all_nums))
            if len(started)<3: continue

            expected_started=math.comb(len(started),3)
            expected_all=math.comb(len(all_nums),3)
            g7=((od.get("odds") or {}).get("7") or {})
            summary=((od.get("group_summary") or {}).get("7") or {})
            payload_rows=len(g7) if isinstance(g7,dict) else 0
            summary_rows=int(summary.get("rows") or 0)
            summary_priced=int(summary.get("priced_rows") or 0)

            valid_keys=0; priced=0; invalid_keys=0; invalid_shapes=0
            for k,raw in (g7.items() if isinstance(g7,dict) else []):
                ks=str(k)
                pattern=f"len={len(ks)}|digits={ks.isdigit()}|sample={ks[:2]}"
                key_patterns[pattern]=key_patterns.get(pattern,0)+1
                if not valid_trio_key(k):
                    invalid_keys+=1
                    if len(rejected_examples)<40:
                        rejected_examples.append({"race_id":rid,"field_size":len(started),"key":ks,"raw":raw})
                    continue
                valid_keys+=1
                t=final_tuple(raw)
                if not t:
                    invalid_shapes+=1
                    continue
                p=num(t[0])
                if p is not None and p>0:
                    priced+=1

            detail.append({
                "race_id":rid,
                "date":race_path.name[:10],
                "started_field_size":len(started),
                "all_entry_field_size":len(all_nums),
                "expected_started_trios":expected_started,
                "expected_all_entry_trios":expected_all,
                "group7_summary_rows":summary_rows,
                "group7_payload_rows":payload_rows,
                "group7_summary_priced_rows":summary_priced,
                "decoder_valid_keys":valid_keys,
                "decoder_priced_rows":priced,
                "invalid_keys":invalid_keys,
                "invalid_shapes":invalid_shapes,
                "payload_vs_summary_diff":payload_rows-summary_rows,
                "decoder_vs_payload_diff":priced-payload_rows,
                "source_coverage_started_pct":100.0*payload_rows/expected_started if expected_started else 0.0,
                "decoder_coverage_started_pct":100.0*priced/expected_started if expected_started else 0.0,
            })

    d=pd.DataFrame(detail)
    if d.empty: raise SystemExit("no diagnostic rows")
    d.to_csv(out/"race-detail.csv",index=False)

    agg=d.groupby("started_field_size",as_index=False).agg(
        races=("race_id","nunique"),
        median_expected=("expected_started_trios","median"),
        median_payload_rows=("group7_payload_rows","median"),
        median_decoder_priced=("decoder_priced_rows","median"),
        median_source_coverage_pct=("source_coverage_started_pct","median"),
        median_decoder_coverage_pct=("decoder_coverage_started_pct","median"),
        max_abs_payload_summary_diff=("payload_vs_summary_diff",lambda s:int(abs(s).max())),
        max_abs_decoder_payload_diff=("decoder_vs_payload_diff",lambda s:int(abs(s).max())),
        invalid_keys=("invalid_keys","sum"),
        invalid_shapes=("invalid_shapes","sum"),
    )
    agg.to_csv(out/"by-started-field-size.csv",index=False)

    summary={
      "contract":"TRIO_GROUP7_SOURCE_VS_DECODER_AUDIT_V1",
      "year":2025,
      "races":int(d.race_id.nunique()),
      "payload_rows_equal_summary_rows_races":int((d.group7_payload_rows==d.group7_summary_rows).sum()),
      "decoder_priced_equal_summary_priced_races":int((d.decoder_priced_rows==d.group7_summary_priced_rows).sum()),
      "decoder_priced_equal_payload_rows_races":int((d.decoder_priced_rows==d.group7_payload_rows).sum()),
      "total_payload_rows":int(d.group7_payload_rows.sum()),
      "total_decoder_priced_rows":int(d.decoder_priced_rows.sum()),
      "total_invalid_keys":int(d.invalid_keys.sum()),
      "total_invalid_shapes":int(d.invalid_shapes.sum()),
      "median_source_coverage_pct":float(d.source_coverage_started_pct.median()),
      "median_decoder_coverage_pct":float(d.decoder_coverage_started_pct.median()),
      "key_patterns":dict(sorted(key_patterns.items(), key=lambda kv:(-kv[1],kv[0]))[:30]),
      "rejected_examples":rejected_examples,
      "2026_locked":True
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(agg.to_csv(index=False))

if __name__=="__main__":
    main()
