#!/usr/bin/env python3
"""Frozen 2026-10-17 ORIGINAL 13 candidate score and unchanged 7King support; research only."""
import argparse,csv,gzip,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import lightgbm as lgb
from emit_l1_to_l2 import load_validation_frame,sigmoid
from prepare_2026_7king_13_outsider_research_signal_v1 import canonical_candidates,load_consensus,CONTRACT
DAY='2026-10-17';RACES=36;HORSES=471

def demand(x,msg):
    if not x:raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def score(models,inputs,out):
    demand(not out.exists(),'SCORES_ALREADY_EXIST')
    names=canonical_candidates()
    idx=json.loads((models/'index.json').read_text())
    manifest=json.loads((inputs/'manifest.json').read_text())
    demand(idx.get('outsider_candidate_count')==13 and idx.get('2026_candidate_approval_status')=='UNRELEASED_RESEARCH_ONLY' and
      idx.get('target_2026_results_read') is False and set(idx['candidates'])==set(names),'MODEL_INDEX_DRIFT')
    demand(manifest.get('target_races')==RACES and manifest.get('target_horses')==HORSES and
      manifest.get('candidate_source_feature_datasets_ready')==13 and manifest.get('2026_target_results_used') is False and
      manifest.get('actual_prediction_scores_produced') is False,'INPUT_MANIFEST_DRIFT')
    out.mkdir(parents=True)
    universe=None
    results={}
    for name in names:
        dataset=inputs/(name+'.jsonl.gz')
        demand(dataset.is_file() and sha(dataset)==manifest['ready_candidates'][name]['output_sha256'],'INPUT_SHA_DRIFT:'+name)
        rows=list(read(dataset))
        pairs={(str(r.get('race_id')),str(r.get('horse_id'))) for r in rows}
        demand(len(rows)==len(pairs)==HORSES and len({p[0] for p in pairs})==RACES and
          all(r.get('target')=={} and r.get('leakage_policy')=='STRICT_PRIOR_DATE_ONLY' and
              r.get('prediction_phase')=='FINAL' and r.get('feature_sets')==['BASE'] and
              (r.get('features') or {}).get('race_date')==DAY for r in rows), 'TARGET_ROWS_DRIFT:'+name)
        if universe is None:universe=pairs
        demand(pairs==universe,'OUTSIDER_INPUT_UNIVERSE_MISMATCH:'+name)
        paths={}
        for kind in ('model.txt','model.json','schema.json'):
            filename=name+'__2026-candidate__'+kind
            file=models/filename
            demand(file.is_file() and sha(file)==idx['candidates'][name]['files'][filename]['sha256'],'MODEL_FILE_SHA_DRIFT:'+filename)
            paths[kind]=file
        meta=json.loads(paths['model.json'].read_text());schema=json.loads(paths['schema.json'].read_text())
        order=schema['feature_order'];rounds=meta['best_iteration']
        demand(meta.get('contract')=='KEIBA_ORIGINAL_13_OUTSIDER_2026_CANDIDATE_REFIT_V1' and
          meta.get('candidate_name')==name and meta.get('deployment_status')=='CANDIDATE_2026_BLIND_UNRELEASED' and
          meta.get('training_years')==[2024,2025] and meta.get('2026_outcomes_used') is False and
          meta.get('2026_validation_used') is False and meta.get('training_rule_approval_ref') is None and
          meta.get('model_sha256')==sha(paths['model.txt']) and meta.get('schema_sha256')==sha(paths['schema.json']) and
          schema.get('leakage_policy')=='STRICT_PRIOR_DATE_ONLY' and schema.get('training_years')==[2024,2025] and
          schema.get('prediction_phase')=='FINAL' and len(order)==meta.get('feature_count'),'RESEARCH_MODEL_GATE_DRIFT:'+name)
        booster=lgb.Booster(model_file=str(paths['model.txt']))
        demand(booster.feature_name()==order and type(rounds) is int and 0<rounds<=booster.current_iteration(),'MODEL_ORDER_OR_ROUNDS_DRIFT:'+name)
        frame,ids=load_validation_frame(dataset,DAY,DAY,order,schema.get('category_levels') or {})
        demand(len(ids)==HORSES and {(z['race_id'],z['horse_id']) for z in ids}==universe,'MODEL_FRAME_UNIVERSE_DRIFT:'+name)
        raw=np.asarray(booster.predict(frame,num_iteration=rounds,raw_score=True))
        demand(len(raw)==HORSES and bool(np.isfinite(raw).all()),'NONFINITE_OUTSIDER_SCORES:'+name)
        path=out/(name+'.jsonl.gz')
        with gzip.open(path,'wt',encoding='utf-8') as f:
            for identity,value in zip(ids,raw):
                f.write(json.dumps({'race_id':identity['race_id'],'horse_id':identity['horse_id'],
                    'horse_number':identity['horse_number'],'raw_margin_logit':float(value),
                    'raw_win_probability':sigmoid(float(value)),'candidate':name,
                    'deployment_status':'CANDIDATE_2026_BLIND_UNRELEASED',
                    'ready_for_betting':False,'2026_target_results_used':False},allow_nan=False)+'\n')
        results[name]={'score_sha256':sha(path),'horse_rows':HORSES,'model_sha256':sha(paths['model.txt'])}
        print('ORIGINAL_OUTSIDER_2026_REAL_SCORE '+name+' 36R_471H',flush=True)
    (out/'index.json').write_text(json.dumps({'contract':'ORIGINAL_13_OUTSIDER_2026_1017_UNRELEASED_SCORES_V1',
        'races':RACES,'horses':HORSES,'candidate_count':13,'candidates':results,
        'ready_for_betting':False,'2026_results_used':False},indent=2)+'\n')
    print('13_OUTSIDER_2026_10_17_REAL_SCORES_DONE',flush=True)

def join(council,ballots,output):
    demand(not output.exists(),'RESEARCH_SIGNAL_OVERWRITE_REFUSED')
    names=canonical_candidates()
    kings=load_consensus(council)
    allvotes=defaultdict(dict)
    files={p.stem:p for p in ballots.glob('*.csv')}
    demand(set(files)==set(names),'EXACTLY_13_CANDIDATE_BALLOTS_REQUIRED')
    for candidate in names:
        with files[candidate].open('r',encoding='utf-8-sig',newline='') as f:
            rd=csv.DictReader(f)
            demand(set(rd.fieldnames or [])=={'race_id','candidate','top1_horse_id','top2_horse_id','top3_horse_id'},'FROZEN_BALLOT_COLUMNS_CHANGED')
            n=0
            for row in rd:
                rid=row['race_id'];hs=[row['top'+str(i)+'_horse_id'] for i in (1,2,3)]
                demand(row['candidate']==candidate and rid in kings and rid.startswith('2026') and
                  candidate not in allvotes[rid],'DUPLICATED_OR_BAD_BALLOT')
                demand(len(set(hs))==3 and all(hs) and
                  set(hs).issubset({str(z['horse_id']) for z in kings[rid]['horses']}),'TOP3_INVALID_HORSE:'+rid)
                allvotes[rid][candidate]=hs;n+=1
            demand(n==RACES,'NOT_36_BALLOTS:'+candidate)
    demand(len(allvotes)==RACES and all(set(v)==set(names) for v in allvotes.values()) and
      sum(len(kings[r]['horses']) for r in allvotes)==HORSES,'13_X_36_VOTE_UNIVERSE_DRIFT')
    output.parent.mkdir(parents=True,exist_ok=True)
    op=gzip.open if str(output).endswith('.gz') else open
    with op(output,'wt',encoding='utf-8') as f:
        for rid in sorted(allvotes):
            original=kings[rid];horse_rows=[]
            for h in sorted(original['horses'],key=lambda z:z['consensus_rank']):
                hid=str(h['horse_id']);top1=top2=top3=0
                for candidate in names:
                    ballot=allvotes[rid][candidate]
                    if hid==ballot[0]:top1+=1
                    elif hid==ballot[1]:top2+=1
                    elif hid==ballot[2]:top3+=1
                horse_rows.append({'horse_id':hid,'king_rank':h['consensus_rank'],
                  'king_borda_score':h['borda_score'],'king_probability_mean':h['probability_mean'],
                  'outsider_support_count':top1+top2+top3,'outsider_score':3*top1+2*top2+top3,
                  'outsider_top1_votes':top1,'outsider_top2_votes':top2,'outsider_top3_votes':top3,
                  'p3_calibrated':None,'calibration_status':'NOT_TRAINED_FOR_2026'})
            demand([h['king_rank'] for h in horse_rows]==list(range(1,len(horse_rows)+1)),'KING_RERANK_FORBIDDEN')
            f.write(json.dumps({'contract':CONTRACT,'race_id':rid,'prediction_year':2026,
              'horse_count':len(horse_rows),'candidate_count':13,'ranking_policy':'SEVEN_KING_UNCHANGED',
              'outsider_policy':'SIGNAL_ONLY_NO_RERANK','calibration_available':False,
              'model_release_authorized':False,'ready_for_betting':False,'l175_l21_l3_invoked':False,
              '2026_results_used':False,'horses':horse_rows},ensure_ascii=False,allow_nan=False)+'\n')
    print('ORIGINAL13_SEVENKING_1017_RESEARCH_ONLY_SIGNAL_PASS 36R_471H',flush=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('score')
    for name in ('models','inputs','scores'):a.add_argument('--'+name,required=True,type=Path)
    b=sub.add_parser('join')
    for name in ('council','ballots','output'):b.add_argument('--'+name,required=True,type=Path)
    v=p.parse_args()
    if v.command=='score':score(v.models,v.inputs,v.scores)
    else:join(v.council,v.ballots,v.output)
if __name__=='__main__':main()
