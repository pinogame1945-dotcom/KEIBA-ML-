#!/usr/bin/env bash
set -euo pipefail

alias_name="${1:?usage: rebuild-seven-score-local-v1.sh ALIAS YEAR OUT_DIR}"
year="${2:?year required}"
out_dir="${3:?out dir required}"

case "$alias_name" in
  core4) candidate="core4_no_pedigree" ;;
  pedlegacy) candidate="no_auto_full_pedigree_legacy" ;;
  condition) candidate="jockey_trainer_condition" ;;
  full) candidate="no_auto_full" ;;
  light) candidate="no_auto_jockey" ;;
  pedv1) candidate="no_auto_full_pedigree_v1" ;;
  condrc) candidate="jockey_trainer_condition_plus_raceclass" ;;
  *) echo "::error::unknown seven-king alias=$alias_name" >&2; exit 2 ;;
esac

if [[ "$year" == "2026" ]]; then
  echo "::error::2026 sealed" >&2
  exit 3
fi
if [[ ! "$year" =~ ^20[0-9][0-9]$ ]]; then
  echo "::error::invalid year=$year" >&2
  exit 4
fi

generation_id="eb13d4096519167b"
snapshot_ref="pino1945/keiba-ml-snapshot-${generation_id}"
session_script="scripts/kaggle-dataset-session-v2.sh"
local_snapshot_root="${SEVEN_SCORE_LOCAL_SNAPSHOT_ROOT:-}"
if [[ -n "$local_snapshot_root" ]]; then
  session_dir="$local_snapshot_root"
  expected_generation=""
else
  session_dir="out/local-rebuild/snapshot-session-${generation_id}"
  expected_generation="$generation_id"
fi
work="out/local-rebuild/${alias_name}-y${year}"
mkdir -p "$session_dir" "$work" "$out_dir"

if [[ -z "$local_snapshot_root" ]]; then
  if [[ ! -s "$session_dir/.files.json" || ! -s "$session_dir/.dataset-ref" ]]; then
    bash "$session_script" init "$snapshot_ref" "$session_dir"
  fi
  bash "$session_script" fetch "$session_dir" manifest.json
fi

manifest="$(find "$session_dir" -maxdepth 2 -type f -name 'manifest.json' -print -quit)"
test -n "$manifest"; test -s "$manifest"

python - "$manifest" "$expected_generation" "$year" "$work/requested-files.txt" <<'PY'
import json,sys
manifest,generation,year,out=sys.argv[1:]
m=json.load(open(manifest,encoding="utf-8"))
actual=(m.get("generation") or {}).get("generation_id")
if generation and actual!=generation:
    raise SystemExit(f"snapshot generation mismatch expected={generation} actual={actual}")
if m.get("contract")!="L1_YEARLY_SUPERSET_SNAPSHOT_V1":
    raise SystemExit(f"bad snapshot contract={m.get('contract')}")
wanted={int(year)-2,int(year)-1,int(year)}
rows={int(x["year"]):x for x in m.get("years",[]) if "year" in x and "file" in x}
missing=sorted(wanted-set(rows))
if missing:
    raise SystemExit(f"snapshot manifest missing years={missing}")
with open(out,"w",encoding="utf-8") as f:
    for y in sorted(wanted):
        f.write(str(rows[y]["file"])+"\n")
PY

mapfile -t snapshot_files < "$work/requested-files.txt"
if [[ -z "$local_snapshot_root" ]]; then
  bash "$session_script" fetch "$session_dir" "${snapshot_files[@]}"
else
  for logical in "${snapshot_files[@]}"; do
    test -s "$session_dir/$logical" || {
      echo "::error::local snapshot file missing logical=$logical root=$session_dir" >&2
      exit 7
    }
  done
  echo "LOCAL_SNAPSHOT_REUSE_OK root=$session_dir files=${#snapshot_files[@]}"
fi

python - "$manifest" "$session_dir" "$work/requested-files.txt" <<'PY'
import gzip,hashlib,json,os,sys
manifest,root,list_path=sys.argv[1:]
m=json.load(open(manifest,encoding="utf-8"))
by_name={str(x["file"]):x for x in m.get("years",[]) if "file" in x}
requested=[x.strip() for x in open(list_path,encoding="utf-8") if x.strip()]
for logical in requested:
    candidates=[os.path.join(root,logical)]
    if logical.endswith(".gz"):
        candidates.append(os.path.join(root,logical[:-3]))
    else:
        candidates.append(os.path.join(root,logical+".gz"))
    path=next((p for p in candidates if os.path.isfile(p) and os.path.getsize(p)>0),None)
    if path is None:
        raise SystemExit(f"restored snapshot missing logical={logical}")
    meta=by_name[logical]
    if path.endswith(".gz"):
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
        if meta.get("sha256") and h.hexdigest()!=meta["sha256"]:
            raise SystemExit(f"snapshot sha256 mismatch logical={logical}")
        opener=lambda p=path:gzip.open(p,"rt",encoding="utf-8")
    else:
        opener=lambda p=path:open(p,"rt",encoding="utf-8")
    rows=0
    with opener() as f:
        for line in f:
            if line.strip(): rows+=1
    if int(meta.get("rows",rows))!=rows:
        raise SystemExit(f"snapshot row mismatch logical={logical} expected={meta.get('rows')} actual={rows}")
print("LOCAL_REBUILD_SNAPSHOT_VALIDATION_OK")
PY

inputs="$(python - "$session_dir" "$work/requested-files.txt" <<'PY'
import os,sys
root,list_path=sys.argv[1:]
out=[]
for logical in [x.strip() for x in open(list_path,encoding="utf-8") if x.strip()]:
    candidates=[os.path.join(root,logical)]
    if logical.endswith(".gz"): candidates.append(os.path.join(root,logical[:-3]))
    else: candidates.append(os.path.join(root,logical+".gz"))
    path=next((p for p in candidates if os.path.isfile(p) and os.path.getsize(p)>0),None)
    if not path: raise SystemExit(f"snapshot missing {logical}")
    out.append(path)
print(",".join(out))
PY
)"

python - "$candidate" "$year" "$work/candidate.json" "$work/env.json" <<'PY'
import json,sys
name,year,cand_out,env_out=sys.argv[1:]
cfg=json.load(open("research/l1-raceclass-absorption-walkforward-v1.json",encoding="utf-8"))
row=next((x for x in cfg["candidates"] if x["name"]==name),None)
if row is None:
    raise SystemExit(f"candidate missing: {name}")
y=int(year)
cand={"name":name,"feature_sets":row["feature_sets"]}
for k in ("actor_prefixes","auto_slices","pedigree_slices"):
    cand[k]=row.get(k) or []
json.dump([cand],open(cand_out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
env={
  "train_start":f"{y-2}-01-01","train_end":f"{y-1}-12-31",
  "valid_start":f"{y}-01-01","valid_end":f"{y}-12-31",
  "feature_sets_json":json.dumps(cand["feature_sets"],separators=(",",":")),
  "actor_prefixes_json":json.dumps(cand["actor_prefixes"],separators=(",",":")),
  "auto_slices_json":json.dumps(cand["auto_slices"],separators=(",",":")),
  "pedigree_slices_json":json.dumps(cand["pedigree_slices"],separators=(",",":")),
}
json.dump(env,open(env_out,"w",encoding="utf-8"),ensure_ascii=False)
PY

eval "$(python - "$work/env.json" <<'PY'
import json,shlex,sys
e=json.load(open(sys.argv[1],encoding="utf-8"))
for k,v in e.items():
    print(f"{k.upper()}={shlex.quote(str(v))}")
PY
)"

arena="$work/arena"
python research/run_feature_arena.py \
  --inputs "$inputs" \
  --out-dir "$arena" \
  --train-start "$TRAIN_START" \
  --train-end "$TRAIN_END" \
  --valid-start "$VALID_START" \
  --valid-end "$VALID_END" \
  --train-race-class ALL \
  --valid-race-class ALL \
  --prediction-phase FINAL \
  --feature-selection none \
  --candidates-json "$work/candidate.json" \
  --source-ref local-rebuild \
  --ml-source-sha "${GITHUB_SHA:-local}" \
  --keep-projections

projection="$(find "$arena/projections" -type f -name '*.jsonl.gz' -print -quit)"
model="$(find "$arena/models" -type f -name '*.txt' -print -quit)"
meta="$(find "$arena/models" -type f -name '*.json' -print -quit)"
schema="$(find "$arena/schemas" -type f -name '*-schema.json' -print -quit)"
for x in "$projection" "$model" "$meta" "$schema"; do
  test -n "$x"; test -s "$x"
done

python research/emit_l1_to_l2.py \
  --dataset "$projection" \
  --model "$model" \
  --schema "$schema" \
  --meta "$meta" \
  --output "$out_dir/score.jsonl.gz" \
  --candidate-name "$candidate" \
  --feature-sets-json "$FEATURE_SETS_JSON" \
  --actor-prefixes-json "$ACTOR_PREFIXES_JSON" \
  --auto-slices-json "$AUTO_SLICES_JSON" \
  --pedigree-slices-json "$PEDIGREE_SLICES_JSON" \
  --valid-start "$VALID_START" \
  --valid-end "$VALID_END" \
  --chunk-size 64

python - "$out_dir/score.jsonl.gz" "$year" "$candidate" <<'PY'
import gzip,json,sys
path,year,candidate=sys.argv[1:]
rows=0;races=set()
with gzip.open(path,"rt",encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        x=json.loads(line); rows+=1
        if x.get("contract")!="L1_TO_L2_OUTPUT_CONTRACT_V1":
            raise SystemExit("bad rebuilt score contract")
        if str(x.get("race_date") or "")[:4]!=year:
            raise SystemExit("rebuilt score year drift")
        races.add(str(x.get("race_id") or ""))
if not rows or not races:
    raise SystemExit("empty rebuilt score")
print(f"SEVEN_SCORE_LOCAL_REBUILD_OK candidate={candidate} year={year} rows={rows} races={len(races)}")
PY
