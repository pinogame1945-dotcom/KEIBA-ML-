#!/usr/bin/env python3
import argparse,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path

MODES={"DAY_TREND","RACE_SHAPE","GATE_COURSE","JOCKEY","FIELD_COMPOSITION","CHIMERA","ROTATION","CLASS_MOVE","WEIGHT_CHANGE","BODY_CHANGE","DISTANCE_TRANSITION","COURSE_TRANSITION","OPPONENT_GAP"}
STYLE_KEYS=[
    "style_recent_avg_first_ratio",
    "style_recent_avg_last_ratio",
    "style_recent_avg_position_gain",
    "style_recent_front_rate",
    "style_recent_finish_front_rate",
    "style_recent_improve_rate",
    "style_previous_first_ratio",
    "style_previous_last_ratio",
    "style_previous_position_gain",
]
DAY_STYLE_KEYS=[
    "style_recent_avg_first_ratio",
    "style_recent_front_rate",
    "style_recent_improve_rate",
]
RACE_BASE_KEYS=[
    "race_date","venue_code","meeting_no","meeting_day","race_no",
    "discipline","surface","distance_m","direction","weather","track_condition",
]
CLASS_CODE={"NEWCOMER":0,"MAIDEN":1,"ONE_WIN":2,"TWO_WIN":3,"THREE_WIN":4,"OPEN":5}
GRADE_CODE={"NONE":0,"UNKNOWN":0,"L":1,"G3":2,"JPN3":2,"G2":3,"JPN2":3,"G1":4,"JPN1":4}
WEIGHT_CODE={"UNKNOWN":0,"WEIGHT_FOR_AGE":1,"SET_WEIGHT":2,"SPECIAL_WEIGHT":3,"HANDICAP":4}
LAYOUTS=("INNER","OUTER","NORMAL")
FIELD_NUMERIC=("age","carried_weight","body_weight","body_weight_diff")

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",required=True)
    p.add_argument("--mode",required=True,choices=sorted(MODES))
    p.add_argument("--output",required=True)
    return p.parse_args()

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def finite(v):
    if v is None or (isinstance(v,str) and not v.strip()): return None
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def bnum(v):
    if v is True: return 1.0
    if v is False: return 0.0
    return None

def mean(vals):
    x=[float(v) for v in vals if v is not None]
    return sum(x)/len(x) if x else None

def std(vals):
    x=[float(v) for v in vals if v is not None]
    return statistics.pstdev(x) if len(x)>=2 else (0.0 if len(x)==1 else None)

def minv(vals):
    x=[float(v) for v in vals if v is not None]
    return min(x) if x else None

def maxv(vals):
    x=[float(v) for v in vals if v is not None]
    return max(x) if x else None

def target_win(row):
    t=row.get("target") or {}
    v=t.get("is_win")
    if v is not None: return bool(v)
    try: return int(float(t.get("finish_position")))==1
    except (TypeError,ValueError): return False

def race_context(f):
    out={k:f.get(k) for k in RACE_BASE_KEYS}
    out.update({
        "nh_field_size":finite(f.get("backfill_field_size")),
        "nh_course_laps":finite(f.get("backfill_course_laps")),
        "nh_class_code":CLASS_CODE.get(str(f.get("backfill_race_class_normalized") or "").upper()),
        "nh_grade_code":GRADE_CODE.get(str(f.get("backfill_grade") or "").upper()),
        "nh_age_min":finite(f.get("backfill_age_min")),
        "nh_age_max":finite(f.get("backfill_age_max")),
        "nh_weight_rule_code":WEIGHT_CODE.get(str(f.get("backfill_weight_rule") or "").upper()),
        "nh_mixed":bnum(f.get("backfill_mixed")),
        "nh_international":bnum(f.get("backfill_international")),
        "nh_special_designated":bnum(f.get("backfill_special_designated")),
        "nh_designated":bnum(f.get("backfill_designated")),
    })
    layout=str(f.get("backfill_course_layout") or "").upper()
    for name in LAYOUTS:
        out["nh_layout_"+name.lower()]=1.0 if layout==name else 0.0
    return out

def gate_fraction(f, fallback_size=None):
    gate=finite(f.get("gate"))
    size=finite(f.get("backfill_field_size")) or finite(fallback_size)
    if gate is None or size is None or size<=1: return None
    return (gate-1.0)/(size-1.0)

def add_gate_course(out,f,field_size):
    out["gate"]=finite(f.get("gate"))
    out["horse_number"]=finite(f.get("horse_number"))
    gf=gate_fraction(f,field_size)
    out["nh_gate_fraction"]=gf
    out["nh_inner_third"]=1.0 if gf is not None and gf<=1/3 else 0.0 if gf is not None else None
    out["nh_outer_third"]=1.0 if gf is not None and gf>=2/3 else 0.0 if gf is not None else None

def add_shape(out,row,field_rows):
    f=row.get("features") or {}
    for key in STYLE_KEYS:
        vals=[finite((x.get("features") or {}).get(key)) for x in field_rows]
        own=finite(f.get(key)); m=mean(vals); s=std(vals)
        stem="nh_shape_"+key.removeprefix("style_")
        out[stem+"_own"]=own
        out[stem+"_field_mean"]=m
        out[stem+"_field_std"]=s
        out[stem+"_diff"]=own-m if own is not None and m is not None else None
    fronts=[finite((x.get("features") or {}).get("style_recent_front_rate")) for x in field_rows]
    fronts=[x for x in fronts if x is not None]
    out["nh_shape_front_heavy_share"]=sum(x>=0.5 for x in fronts)/len(fronts) if fronts else None
    early=[finite((x.get("features") or {}).get("style_recent_avg_first_ratio")) for x in field_rows]
    early=[x for x in early if x is not None]
    out["nh_shape_early_mean"]=mean(early)
    out["nh_shape_early_std"]=std(early)

def add_jockey(out,f):
    count=0
    for key,value in f.items():
        if not str(key).startswith("actor_jockey_"): continue
        v=finite(value)
        if v is None: continue
        out["nh_jockey_"+str(key)[len("actor_jockey_"):]]=v
        count+=1
    out["nh_jockey_feature_count"]=float(count)

def add_field_composition(out,row,field_rows):
    f=row.get("features") or {}
    for key in FIELD_NUMERIC:
        vals=[finite((x.get("features") or {}).get(key)) for x in field_rows]
        own=finite(f.get(key)); m=mean(vals); s=std(vals)
        stem="nh_field_"+key
        out[stem+"_own"]=own
        out[stem+"_mean"]=m
        out[stem+"_std"]=s
        out[stem+"_min"]=minv(vals)
        out[stem+"_max"]=maxv(vals)
        out[stem+"_diff"]=own-m if own is not None and m is not None else None
        out[stem+"_z"]=(own-m)/s if own is not None and m is not None and s not in (None,0) else None
    sexes=[str((x.get("features") or {}).get("sex") or "") for x in field_rows]
    n=len(sexes)
    for label,name in (("牡","male"),("牝","female"),("セ","gelding")):
        out[f"nh_field_{name}_share"]=sum(s==label for s in sexes)/n if n else None
        out[f"nh_field_is_{name}"]=1.0 if str(f.get("sex") or "")==label else 0.0

def add_rotation(out,f):
    days=finite(f.get("days_since_last_start"))
    starts=finite(f.get("prior_starts"))
    recent=finite(f.get("recent_window_starts"))
    out["nh_rotation_days_since_last_start"]=days
    out["nh_rotation_prior_starts"]=starts
    out["nh_rotation_recent_window_starts"]=recent
    out["nh_rotation_within_14d"]=1.0 if days is not None and days<=14 else 0.0 if days is not None else None
    out["nh_rotation_within_28d"]=1.0 if days is not None and days<=28 else 0.0 if days is not None else None
    out["nh_rotation_29_60d"]=1.0 if days is not None and 29<=days<=60 else 0.0 if days is not None else None
    out["nh_rotation_61_89d"]=1.0 if days is not None and 61<=days<=89 else 0.0 if days is not None else None
    out["nh_rotation_90d_plus"]=1.0 if days is not None and days>=90 else 0.0 if days is not None else None
    out["nh_rotation_180d_plus"]=1.0 if days is not None and days>=180 else 0.0 if days is not None else None

def _class_code_from_features(f):
    return CLASS_CODE.get(str(f.get("backfill_race_class_normalized") or "").upper())

def _grade_code_from_features(f):
    return GRADE_CODE.get(str(f.get("backfill_grade") or "").upper())

def _history_last(history):
    return history[-1] if history else None

def _history_prev(history):
    return history[-2] if len(history)>=2 else None

def add_class_move(out,f,history):
    prev=_history_last(history)
    cur_class=_class_code_from_features(f)
    cur_grade=_grade_code_from_features(f)
    prev_class=prev.get("class_code") if prev else None
    prev_grade=prev.get("grade_code") if prev else None
    out["nh_class_current"]=cur_class
    out["nh_class_previous"]=prev_class
    out["nh_class_delta"]=cur_class-prev_class if cur_class is not None and prev_class is not None else None
    out["nh_class_up"]=1.0 if cur_class is not None and prev_class is not None and cur_class>prev_class else 0.0 if cur_class is not None and prev_class is not None else None
    out["nh_class_down"]=1.0 if cur_class is not None and prev_class is not None and cur_class<prev_class else 0.0 if cur_class is not None and prev_class is not None else None
    out["nh_class_same"]=1.0 if cur_class is not None and prev_class is not None and cur_class==prev_class else 0.0 if cur_class is not None and prev_class is not None else None
    out["nh_grade_current"]=cur_grade
    out["nh_grade_previous"]=prev_grade
    out["nh_grade_delta"]=cur_grade-prev_grade if cur_grade is not None and prev_grade is not None else None

def _field_numeric(field_rows,key):
    return [finite((x.get("features") or {}).get(key)) for x in field_rows]

def add_weight_change(out,f,field_rows,history):
    prev=_history_last(history)
    cur=finite(f.get("carried_weight"))
    prv=prev.get("carried_weight") if prev else None
    vals=_field_numeric(field_rows,"carried_weight")
    m=mean(vals); sd=std(vals)
    out["nh_weight_current"]=cur
    out["nh_weight_previous"]=prv
    out["nh_weight_delta"]=cur-prv if cur is not None and prv is not None else None
    out["nh_weight_field_mean"]=m
    out["nh_weight_field_std"]=sd
    out["nh_weight_vs_field"]=cur-m if cur is not None and m is not None else None
    out["nh_weight_field_z"]=(cur-m)/sd if cur is not None and m is not None and sd not in (None,0) else None
    rule=WEIGHT_CODE.get(str(f.get("backfill_weight_rule") or "").upper())
    out["nh_weight_rule_code"]=rule

def add_body_change(out,f,field_rows,history):
    prev=_history_last(history)
    cur=finite(f.get("body_weight"))
    src_diff=finite(f.get("body_weight_diff"))
    prv=prev.get("body_weight") if prev else None
    delta=cur-prv if cur is not None and prv is not None else src_diff
    vals=_field_numeric(field_rows,"body_weight")
    m=mean(vals); sd=std(vals)
    out["nh_body_current"]=cur
    out["nh_body_previous"]=prv
    out["nh_body_delta"]=delta
    out["nh_body_source_diff"]=src_diff
    out["nh_body_delta_pct"]=delta/prv if delta is not None and prv not in (None,0) else None
    out["nh_body_field_mean"]=m
    out["nh_body_field_std"]=sd
    out["nh_body_vs_field"]=cur-m if cur is not None and m is not None else None
    out["nh_body_field_z"]=(cur-m)/sd if cur is not None and m is not None and sd not in (None,0) else None

def add_distance_transition(out,f,history):
    prev=_history_last(history)
    prev2=_history_prev(history)
    cur=finite(f.get("distance_m"))
    prv=finite(f.get("previous_distance_m"))
    if prv is None and prev:
        prv=prev.get("distance_m")
    prv2=prev2.get("distance_m") if prev2 else None
    delta=finite(f.get("distance_change_m"))
    if delta is None and cur is not None and prv is not None:
        delta=cur-prv
    prev_delta=(prv-prv2) if prv is not None and prv2 is not None else None
    out["nh_dist_current"]=cur
    out["nh_dist_previous"]=prv
    out["nh_dist_previous2"]=prv2
    out["nh_dist_delta"]=delta
    out["nh_dist_abs_delta"]=abs(delta) if delta is not None else None
    out["nh_dist_extended"]=1.0 if delta is not None and delta>0 else 0.0 if delta is not None else None
    out["nh_dist_shortened"]=1.0 if delta is not None and delta<0 else 0.0 if delta is not None else None
    out["nh_dist_same"]=1.0 if delta is not None and delta==0 else 0.0 if delta is not None else None
    out["nh_dist_consecutive_extension"]=1.0 if delta is not None and prev_delta is not None and delta>0 and prev_delta>0 else 0.0 if delta is not None and prev_delta is not None else None
    out["nh_dist_consecutive_shortening"]=1.0 if delta is not None and prev_delta is not None and delta<0 and prev_delta<0 else 0.0 if delta is not None and prev_delta is not None else None
    out["nh_dist_return_to_two_back"]=1.0 if cur is not None and prv2 is not None and cur==prv2 else 0.0 if cur is not None and prv2 is not None else None

def add_course_transition(out,f,history):
    prev=_history_last(history)
    if not prev:
        for key in ("venue_changed","surface_changed","direction_changed","layout_changed","same_venue","same_surface","same_direction","same_layout"):
            out["nh_course_"+key]=None
        return
    cur_venue=str(f.get("venue_code") or "")
    cur_surface=str(f.get("surface") or "")
    cur_direction=str(f.get("direction") or "")
    cur_layout=str(f.get("backfill_course_layout") or "").upper()
    pvenue=str(prev.get("venue") or "")
    psurface=str(prev.get("surface") or "")
    pdirection=str(prev.get("direction") or "")
    playout=str(prev.get("layout") or "")
    def changed(a,b):
        return 1.0 if a and b and a!=b else 0.0 if a and b else None
    def same(a,b):
        return 1.0 if a and b and a==b else 0.0 if a and b else None
    out["nh_course_venue_changed"]=changed(cur_venue,pvenue)
    out["nh_course_surface_changed"]=changed(cur_surface,psurface)
    out["nh_course_direction_changed"]=changed(cur_direction,pdirection)
    out["nh_course_layout_changed"]=changed(cur_layout,playout)
    out["nh_course_same_venue"]=same(cur_venue,pvenue)
    out["nh_course_same_surface"]=same(cur_surface,psurface)
    out["nh_course_same_direction"]=same(cur_direction,pdirection)
    out["nh_course_same_layout"]=same(cur_layout,playout)

def add_opponent_gap(out,row,field_rows):
    f=row.get("features") or {}
    keys=sorted(k for k in f if str(k).startswith("opponent_"))
    for key in keys:
        own=finite(f.get(key))
        if own is None: continue
        vals=[finite((x.get("features") or {}).get(key)) for x in field_rows]
        m=mean(vals); sd=std(vals)
        stem="nh_opp_"+str(key)[len("opponent_"):]
        out[stem+"_own"]=own
        out[stem+"_field_mean"]=m
        out[stem+"_field_std"]=sd
        out[stem+"_diff"]=own-m if m is not None else None
        out[stem+"_z"]=(own-m)/sd if m is not None and sd not in (None,0) else None

def horse_signature(row):
    f=row.get("features") or {}
    return {
        "race_date":str(f.get("race_date") or "")[:10],
        "venue":str(f.get("venue_code") or ""),
        "surface":str(f.get("surface") or ""),
        "direction":str(f.get("direction") or ""),
        "layout":str(f.get("backfill_course_layout") or "").upper(),
        "distance_m":finite(f.get("distance_m")),
        "class_code":_class_code_from_features(f),
        "grade_code":_grade_code_from_features(f),
        "carried_weight":finite(f.get("carried_weight")),
        "body_weight":finite(f.get("body_weight")),
    }

def day_features(row,field_rows,prior_winners):
    f=row.get("features") or {}
    out={}
    field_size=len(field_rows)
    gf=gate_fraction(f,field_size)
    gate_vals=[x.get("gate_fraction") for x in prior_winners if x.get("gate_fraction") is not None]
    gm=mean(gate_vals)
    out["nh_day_prior_races"]=float(len(prior_winners))
    out["nh_day_has_prior"]=1.0 if prior_winners else 0.0
    out["nh_day_gate_fraction"]=gf
    out["nh_day_winner_gate_mean_fraction"]=gm
    out["nh_day_gate_match"]=-abs(gf-gm) if gf is not None and gm is not None else None
    out["nh_day_inner_win_rate"]=sum(x<=1/3 for x in gate_vals)/len(gate_vals) if gate_vals else None
    out["nh_day_outer_win_rate"]=sum(x>=2/3 for x in gate_vals)/len(gate_vals) if gate_vals else None
    for key in DAY_STYLE_KEYS:
        vals=[x.get(key) for x in prior_winners if x.get(key) is not None]
        m=mean(vals); own=finite(f.get(key))
        stem="nh_day_"+key.removeprefix("style_")
        out[stem+"_winner_mean"]=m
        out[stem+"_match"]=-abs(own-m) if own is not None and m is not None else None
    return out

def winner_signature(row,field_size):
    f=row.get("features") or {}
    out={"gate_fraction":gate_fraction(f,field_size)}
    for key in DAY_STYLE_KEYS:
        out[key]=finite(f.get(key))
    return out

def build_features(mode,row,field_rows,prior_winners,horse_history):
    f=row.get("features") or {}
    hid=str(row.get("horse_id") or "")
    history=horse_history.get(hid,[])
    out=race_context(f)
    if mode in {"DAY_TREND","CHIMERA"}:
        add_gate_course(out,f,len(field_rows))
        out.update(day_features(row,field_rows,prior_winners))
    if mode in {"RACE_SHAPE","CHIMERA"}:
        add_shape(out,row,field_rows)
    if mode in {"GATE_COURSE","CHIMERA"}:
        add_gate_course(out,f,len(field_rows))
    if mode in {"JOCKEY","CHIMERA"}:
        add_jockey(out,f)
    if mode in {"FIELD_COMPOSITION","CHIMERA"}:
        add_field_composition(out,row,field_rows)
    if mode=="ROTATION":
        add_rotation(out,f)
    if mode=="CLASS_MOVE":
        add_class_move(out,f,history)
    if mode=="WEIGHT_CHANGE":
        add_weight_change(out,f,field_rows,history)
    if mode=="BODY_CHANGE":
        add_body_change(out,f,field_rows,history)
    if mode=="DISTANCE_TRANSITION":
        add_distance_transition(out,f,history)
    if mode=="COURSE_TRANSITION":
        add_course_transition(out,f,history)
    if mode=="OPPONENT_GAP":
        add_opponent_gap(out,row,field_rows)
    return out

def validate_safe(features):
    # Strong allow-list: this derived research dataset may contain only race context,
    # gate/horse-number, and explicitly derived nonhorse features.
    # This is safer than substring-blocking names such as jockey_recent_win_rate or
    # style_recent_finish_front_rate, which are legitimate nonhorse/shape signals.
    allowed_exact=set(RACE_BASE_KEYS)|{"gate","horse_number"}
    bad=[key for key in features if key not in allowed_exact and not str(key).startswith("nh_")]
    market=[key for key in features if any(token in str(key).lower() for token in ("odds","popularity","payout"))]
    if bad or market:
        merged=sorted(set(bad+market))
        raise ValueError("forbidden/unexpected feature leaked: "+",".join(merged[:20]))

def process_day(day_rows,mode,writer,meta,horse_history):
    races=defaultdict(list)
    for row in day_rows:
        races[str(row.get("race_id") or "")].append(row)
    ordered=[]
    for rid,rows in races.items():
        f=rows[0].get("features") or {}
        venue=str(f.get("venue_code") or "")
        race_no=finite(f.get("race_no"))
        ordered.append((venue, race_no if race_no is not None else 999.0, rid, rows))
    ordered.sort(key=lambda x:(x[0],x[1],x[2]))
    state=defaultdict(list)
    count=0
    for venue,_race_no,rid,rows in ordered:
        prior=state[venue]
        for row in rows:
            features=build_features(mode,row,rows,prior,horse_history)
            validate_safe(features)
            record={
                "ml_dataset_version":int(row.get("ml_dataset_version") or 0),
                "feature_schema_version":int(row.get("feature_schema_version") or 0),
                "leakage_policy":row.get("leakage_policy"),
                "race_id":rid,
                "horse_id":str(row.get("horse_id") or ""),
                "prediction_phase":"FINAL",
                "feature_sets":["BASE"],
                "history_windows":meta["history_windows"],
                "small_sample_policy":meta["small_sample_policy"],
                "features":features,
                "target":row.get("target") or {},
            }
            writer.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
            count+=1
        winners=[r for r in rows if target_win(r)]
        for w in winners:
            prior.append(winner_signature(w,len(rows)))
    # Transition state is committed only after every row from the date has been built.
    # This preserves STRICT_PRIOR_DATE_ONLY even in the unlikely case that a horse
    # appears more than once on the same calendar date.
    seen=set()
    for row in day_rows:
        hid=str(row.get("horse_id") or "")
        if not hid or hid in seen: continue
        seen.add(hid)
        history=horse_history[hid]
        history.append(horse_signature(row))
        if len(history)>3:
            del history[:-3]
    return count

def main():
    a=args()
    inputs=[Path(x.strip()) for x in a.inputs.split(",") if x.strip()]
    if not inputs: raise SystemExit("no inputs")
    mode=a.mode.upper()
    first=None
    for p in inputs:
        with open_text(p) as f:
            for line in f:
                if line.strip():
                    first=json.loads(line); break
        if first: break
    if not first: raise SystemExit("empty inputs")
    if int(first.get("ml_dataset_version") or 0)!=3: raise SystemExit("dataset version mismatch")
    if int(first.get("feature_schema_version") or 0)!=8: raise SystemExit("feature schema mismatch")
    if first.get("leakage_policy")!="STRICT_PRIOR_DATE_ONLY": raise SystemExit("leakage policy mismatch")
    meta={
        "history_windows":first.get("history_windows") or {},
        "small_sample_policy":first.get("small_sample_policy") or {},
    }
    if not meta["history_windows"] or not meta["small_sample_policy"]:
        raise SystemExit("source contract missing history/small-sample policy")

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    writer=gzip.open(out,"wt",encoding="utf-8") if str(out).endswith(".gz") else open(out,"w",encoding="utf-8")
    rows_written=0
    last_date=None
    day=[]
    horse_history=defaultdict(list)
    try:
        for path in inputs:
            with open_text(path) as fh:
                for line in fh:
                    if not line.strip(): continue
                    row=json.loads(line)
                    date=str((row.get("features") or {}).get("race_date") or "")[:10]
                    if not date: continue
                    if last_date is not None and date<last_date:
                        raise ValueError(f"input date order regressed: {date} < {last_date}")
                    if last_date is not None and date!=last_date:
                        rows_written+=process_day(day,mode,writer,meta,horse_history)
                        day=[]
                    day.append(row); last_date=date
        if day:
            rows_written+=process_day(day,mode,writer,meta,horse_history)
    finally:
        writer.close()
    print("L1_NONHORSE_DATASET_OK")
    print(json.dumps({
        "mode":mode,"rows":rows_written,"output":str(out),
        "ordinary_horse_form":False,"odds":False,
        "same_day_rule":"same venue, prior race number only"
    },ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
