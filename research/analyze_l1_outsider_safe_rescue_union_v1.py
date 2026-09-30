#!/usr/bin/env python3
import argparse,csv,itertools,json
from collections import Counter,defaultdict
from pathlib import Path

TOPNS=("1","3","6")

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--output-json",required=True)
    p.add_argument("--output-csv",required=True)
    p.add_argument("--output-pairwise-csv",required=True)
    return p.parse_args()

def load(root):
    years=[2021,2022,2023,2024,2025]
    blind={n:set() for n in TOPNS}
    rescues=defaultdict(lambda:{n:set() for n in TOPNS})
    labels={}
    for y in years:
        yr=root/f"y{y}"
        s=json.loads((yr/"summary.json").read_text(encoding="utf-8"))
        for n in TOPNS:
            blind[n].update(map(str,s["seven_blind_race_ids_by_topn"][n]))
        for p in sorted(yr.glob("outsider_*.json")):
            x=json.loads(p.read_text(encoding="utf-8"))
            cand=x["candidate"]; labels[cand]=x.get("label_ja") or cand
            for n in TOPNS:
                rescues[cand][n].update(map(str,x["topn"][n]["safe_rescue_race_ids"]))
    return years,blind,rescues,labels

def best_by_size(cands,sets):
    out=[]
    prev=0
    for k in range(1,len(cands)+1):
        best_count=-1; best_combo=None; ties=0
        for combo in itertools.combinations(cands,k):
            u=set()
            for c in combo: u.update(sets[c])
            count=len(u)
            if count>best_count:
                best_count=count; best_combo=combo; ties=1
            elif count==best_count:
                ties+=1
                if combo<best_combo: best_combo=combo
        out.append({
            "size":k,
            "rescue_count":best_count,
            "marginal_vs_prev_best":best_count-prev,
            "candidates":list(best_combo),
            "tie_count":ties,
        })
        prev=best_count
    return out

def main():
    a=args(); root=Path(a.root)
    years,blind,rescues,labels=load(root)
    cands=sorted(rescues)
    if len(cands)!=13: raise SystemExit(f"expected 13 candidates, got {len(cands)}")

    result={
        "contract":"L1_OUTSIDER_SAFE_RESCUE_UNION_V1",
        "years":years,
        "candidate_count":len(cands),
        "topn":{},
        "labels":labels,
        "2026_sealed":True,
    }
    flat_rows=[]
    pair_rows=[]
    for n in TOPNS:
        denom=len(blind[n])
        sets={c:set(rescues[c][n]) for c in cands}
        union=set().union(*(sets[c] for c in cands))
        freq=Counter()
        for rid in blind[n]:
            freq[sum(rid in sets[c] for c in cands)]+=1
        individual={}
        for c in cands:
            others=set().union(*(sets[o] for o in cands if o!=c))
            exclusive=sets[c]-others
            individual[c]={
                "rescue_count":len(sets[c]),
                "rescue_rate_on_blind":len(sets[c])/denom if denom else None,
                "exclusive_rescue_count":len(exclusive),
                "exclusive_rescue_rate_on_blind":len(exclusive)/denom if denom else None,
            }
            flat_rows.append({
                "topn":n,"candidate":c,"label_ja":labels[c],
                **individual[c],
            })

        pairs={}
        for a1,a2 in itertools.combinations(cands,2):
            inter=sets[a1]&sets[a2]; u=sets[a1]|sets[a2]
            key=f"{a1}+{a2}"
            pairs[key]={
                "intersection":len(inter),
                "union":len(u),
                "jaccard":len(inter)/len(u) if u else 1.0,
            }
            pair_rows.append({
                "topn":n,"candidate_a":a1,"candidate_b":a2,
                "label_a":labels[a1],"label_b":labels[a2],
                "intersection":len(inter),"union":len(u),
                "jaccard":len(inter)/len(u) if u else 1.0,
            })

        best=best_by_size(cands,sets)
        for row in best:
            row["coverage_rate_on_blind"]=row["rescue_count"]/denom if denom else None
            row["labels"]=[labels[c] for c in row["candidates"]]

        result["topn"][n]={
            "seven_blind_count":denom,
            "all_13_union_rescue_count":len(union),
            "all_13_union_rescue_rate":len(union)/denom if denom else None,
            "remaining_unrescued_count":denom-len(union),
            "remaining_unrescued_rate":(denom-len(union))/denom if denom else None,
            "rescued_by_outsider_count_distribution":{str(k):v for k,v in sorted(freq.items())},
            "individual":individual,
            "best_combinations_by_size":best,
            "pairwise":pairs,
        }

    p=Path(a.output_json); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    with open(a.output_csv,"w",newline="",encoding="utf-8-sig") as fh:
        fields=["topn","candidate","label_ja","rescue_count","rescue_rate_on_blind","exclusive_rescue_count","exclusive_rescue_rate_on_blind"]
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(flat_rows)
    with open(a.output_pairwise_csv,"w",newline="",encoding="utf-8-sig") as fh:
        fields=["topn","candidate_a","candidate_b","label_a","label_b","intersection","union","jaccard"]
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(pair_rows)

    print("SAFE_RESCUE_UNION_OK")
    print(json.dumps({
        n:{
            "blind":result["topn"][n]["seven_blind_count"],
            "all13":result["topn"][n]["all_13_union_rescue_count"],
            "rate":result["topn"][n]["all_13_union_rescue_rate"],
            "best3":result["topn"][n]["best_combinations_by_size"][2],
        } for n in TOPNS
    },ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
