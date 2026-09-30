#!/usr/bin/env python3
import argparse, gzip, json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from emit_l1_to_l2 import load_validation_frame

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--schema",required=True)
    p.add_argument("--meta",required=True)
    p.add_argument("--valid-start",required=True)
    p.add_argument("--valid-end",required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def main():
    a=parse_args()
    schema=json.loads(Path(a.schema).read_text(encoding="utf-8"))
    meta=json.loads(Path(a.meta).read_text(encoding="utf-8"))
    feature_order=list(schema.get("feature_order") or [])
    category_levels=dict(schema.get("category_levels") or {})
    if not feature_order:
        raise ValueError("schema feature_order is empty")
    booster=lgb.Booster(model_file=a.model)
    frame,ids=load_validation_frame(a.dataset,a.valid_start,a.valid_end,feature_order,category_levels)
    best_iteration=int(meta.get("best_iteration") or booster.best_iteration or -1)
    probs=np.asarray(booster.predict(frame,num_iteration=best_iteration),dtype=float)
    margins=np.asarray(booster.predict(frame,raw_score=True,num_iteration=best_iteration),dtype=float)

    by={}
    for i,ident in enumerate(ids):
        by.setdefault(ident["race_id"],[]).append(i)
    ranks=np.empty(len(ids),dtype=int)
    for rid,indices in by.items():
        local=np.asarray(indices,dtype=int)
        order=np.argsort(-probs[local],kind="stable")
        for rank,pos in enumerate(order,start=1):
            ranks[local[pos]]=rank

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    writer=gzip.open if str(out).endswith(".gz") else open
    with writer(out,"wt",encoding="utf-8") as fh:
        for i,ident in enumerate(ids):
            row={
                "race_id":ident["race_id"],
                "horse_id":ident["horse_id"],
                "horse_number":ident.get("horse_number"),
                "raw_win_probability":float(probs[i]),
                "raw_margin_logit":float(margins[i]),
                "predicted_rank":int(ranks[i]),
            }
            fh.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    print(json.dumps({"rows":len(ids),"races":len(by),"output":str(out)},separators=(",",":")))

if __name__=="__main__":
    main()
