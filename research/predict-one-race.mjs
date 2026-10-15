#!/usr/bin/env node
import {mkdir,writeFile} from "node:fs/promises";
import path from "node:path";
import {spawnSync} from "node:child_process";
import {runOneRace} from "../src/single-race-engine.mjs";

function options(args){
  const out={};
  for(let i=0;i<args.length;i++){
    const key=args[i];
    if(!["--race","--source-root","--history-from","--feature-sets","--phase",
          "--registry","--features-out","--results-out"].includes(key))
      throw new Error("unknown flag "+key);
    if(i+1===args.length||args[i+1].startsWith("--"))
      throw new Error("missing value for "+key);
    out[key.slice(2)]=args[++i];
  }
  if(!out.race)throw new Error("Usage: node research/predict-one-race.mjs --race RACE_ID_OR_URL --source-root BACKFILL_CHECKOUT [--history-from YYYY-MM-DD] [--features-out out.json] [--registry models/one-race-registry.json] [--results-out ranks.json]");
  return out;
}
const opts=options(process.argv.slice(2));
const result=await runOneRace({
  sourceRoot:path.resolve(opts["source-root"]??"../KEIBA-BACKFILL"),
  raceId:opts.race,
  historyFrom:opts["history-from"]??"2021-01-01",
  featureSets:opts["feature-sets"],
  predictionPhase:opts.phase??"EARLY",
});
const featureOut=path.resolve(opts["features-out"]??"out/one-race-features.json");
await mkdir(path.dirname(featureOut),{recursive:true});
await writeFile(featureOut,JSON.stringify(result,null,2)+"\n");
console.log("ONE_RACE_FEATURES_READY",JSON.stringify({
  race_id:result.race_id,date:result.race_date,starters:result.features_by_horse.length,
  history_days:result.history.days,history_races:result.history.races,
  phase:result.phase,file:featureOut
}));
const registry=path.resolve(opts.registry??"models/one-race-registry.json");
const args=[
  "research/score-one-race.py","--input",featureOut,
  "--registry",registry,
  "--output",path.resolve(opts["results-out"]??"out/one-race-rankings.json"),
];
const executed=spawnSync("python3",args,{encoding:"utf8",stdio:"inherit"});
if(executed.error)throw executed.error;
if(executed.status!==0)process.exitCode=executed.status??1;
