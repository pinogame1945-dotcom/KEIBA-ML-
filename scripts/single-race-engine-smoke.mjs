import assert from "node:assert/strict";
import {mkdtemp,mkdir,writeFile,readFile,rm} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {gzipSync} from "node:zlib";
import {spawnSync} from "node:child_process";
import {parseRaceId,runOneRace,stripTargetOutcome,verifyRaceManifestPath}
  from "../src/single-race-engine.mjs";
import {createMlDatasetProcessor} from "../src/ml-dataset.mjs";
import {applyPredictionPhase,selectFeatureFamilies} from "../src/l1-feature-contract.mjs";

assert.equal(parseRaceId("202305010101"),"202305010101");
assert.equal(parseRaceId("https://race.netkeiba.com/race/shutuba.html?race_id=202305010101"),"202305010101");
assert.throws(()=>parseRaceId("https://evil.com/?race_id=202305010101"),/netkeiba/);
assert.throws(()=>verifyRaceManifestPath("/data","../bad"),/invalid/);
const dir=await mkdtemp(join(tmpdir(),"keiba-one-race-test-"));
try {
  await mkdir(join(dir,"data","daily"),{recursive:true});
  const historic={
    race:{race_id:"202305010101",actual_date:"2023-01-01",scheduled_date:"2023-01-01",
      venue_code:"05",race_no:1,distance_m:1600,surface:"TURF",discipline:"FLAT",
      field_size:2,weather:"SUN",track_condition:"GOOD"},
    entries:[
      {horse_id:"1111111111",horse_name:"Alpha",horse_number:1,gate:1,sex:"MALE",age:4,carried_weight:56,entry_status:"ACTIVE"},
      {horse_id:"2222222222",horse_name:"Beta",horse_number:2,gate:2,sex:"FEMALE",age:4,carried_weight:54,entry_status:"ACTIVE"}],
    results:[
      {horse_id:"1111111111",result_status:"FINISHED",official_finish_position:1,last_3f:34.7},
      {horse_id:"2222222222",result_status:"FINISHED",official_finish_position:2,last_3f:35.5}],
    payouts:[{bet_type:"WIN",combination:"1",payout_yen:200}],laps:[],
  };
  const target={
    race:{...historic.race,race_id:"202305010102",race_no:2,scheduled_date:"2023-01-02",actual_date:"2023-01-02",
      weather:"RAIN",track_condition:"SLOPPY",finish_time_ms:9999},
    entries:[
      {...historic.entries[1],horse_number:1,gate:1,body_weight:490,win_odds:3.2,popularity:1},
      {...historic.entries[0],horse_number:2,gate:2,body_weight:510,win_odds:1.2,popularity:2}],
    results:[{horse_id:"1111111111",result_status:"FINISHED",official_finish_position:1},
             {horse_id:"2222222222",result_status:"FINISHED",official_finish_position:2}],
    payouts:[{bet_type:"WIN",payout_yen:100}],laps:[{lap_seconds:10.0}],
  };
  const fp=(date)=>"data/daily/"+date+".jsonl.gz";
  for(const [date,data] of [["2023-01-01",historic],["2023-01-02",target]]){
    await writeFile(join(dir,fp(date)),gzipSync(Buffer.from(JSON.stringify(data)+"\n")));
  }
  const manifest={days:{
    "2023-01-01":{status:"SUCCESS",file:fp("2023-01-01")},
    "2023-01-02":{status:"SUCCESS",file:fp("2023-01-02")}
  }};
  await writeFile(join(dir,"data/manifest.json"),JSON.stringify(manifest));
  const features="BASE,OPPONENT,NETWORK,LAP,STYLE,DISTANCE,BACKFILL,AUTO,ACTOR,TIME_PACE";
  const run=()=>runOneRace({sourceRoot:dir,raceId:"202305010102",
    historyFrom:"2023-01-01",featureSets:features});
  await assert.rejects(()=>runOneRace({sourceRoot:dir,raceId:"202605010111",historyFrom:"2023-01-01"}),/2026_SEALED/);
  const result=await run();
  assert.equal(result.race_id,"202305010102");
  assert.equal(result.phase,"EARLY");
  assert.equal(result.features_by_horse.length,2);
  assert.equal(result.history.days,1);
  assert.equal(result.history.races,1);
  const [a,b]=result.features_by_horse;
  assert.deepEqual([a.horse_number,b.horse_number],[1,2]);
  assert.equal(a.horse_id,"2222222222");
  assert.equal(a.features.previous_finish_position,2);
  assert.equal(b.features.previous_finish_position,1);
  assert.ok("lap_recent_races_measured" in a.features);
  assert.ok("distx_same_band_starts" in a.features);
  assert.equal("weather" in a.features,false);
  assert.equal("track_condition" in a.features,false);
  assert.equal("body_weight" in a.features,false);
  for(const row of result.features_by_horse){
    const txt=JSON.stringify(row);
    assert.ok(!txt.includes("payout_yen"));
    assert.ok(!txt.includes("official_finish_position"));
    assert.equal(row.target,undefined);
    assert.equal(row.market_outcome,undefined);
  }
  // Shared BASE features must match the training builder on equivalent
  // history, after EARLY phase gating. Same-day target results are not read.
  const proc=createMlDatasetProcessor({includeAutoFeatures:true,includeBackfillFeatures:true,
    includeActorFeatures:true,includeOpponentRelationshipFeatures:true,includeTimePaceFeatures:true});
  proc.processDay([historic],"2023-01-01");
  const archiveTrain=proc.processDay([target],"2023-01-02");
  const comparison=new Map(archiveTrain.map(x=>[x.horse_id,
    applyPredictionPhase(selectFeatureFamilies(x.features,result.features_by_horse[0].feature_sets),"EARLY")]));
  for(const row of result.features_by_horse){
    for(const [key,value] of Object.entries(comparison.get(row.horse_id))){
      if(key.startsWith("lap_")||key.startsWith("style_")||key.startsWith("distx_"))continue;
      assert.deepEqual(row.features[key],value,"feature drift "+key+" for "+row.horse_id);
    }
  }
  assert.throws(()=>proc.predictDay([stripTargetOutcome(target)],"2023-01-02"),/AS_OF_LEAKAGE/);
  const fresh=createMlDatasetProcessor();
  fresh.processDay([historic],"2023-01-01");
  assert.throws(()=>fresh.predictDay([target],"2023-01-02"),/post-race arrays/);
  assert.deepEqual(result.features_by_horse, (await run()).features_by_horse);
  await assert.rejects(()=>runOneRace({sourceRoot:dir,raceId:"202305010102",historyFrom:"2023-01-01",predictionPhase:"FINAL"}),/UNVERIFIED/);

  // Real interface test: two archival model scores and one deterministic
  // combination. Scores are synthetic fixtures, never presented as AI output.
  const modelDir=join(dir,"models");
  await mkdir(modelDir,{recursive:true});
  const mBase={historical_oof:true,prediction_phase:result.phase,
    feature_schema_version:result.feature_schema_version,
    feature_sets:a.feature_sets,history_windows:a.history_windows,
    small_sample_policy:a.small_sample_policy,
    split:{train_end:"2022-09-30",valid_end:"2022-12-31"}};
  await writeFile(join(modelDir,"meta.json"),JSON.stringify(mBase));
  await writeFile(join(modelDir,"m1.csv"),"race_id,horse_id,score\n202305010102,1111111111,0.8\n202305010102,2222222222,0.2\n");
  await writeFile(join(modelDir,"m2.csv"),"race_id,horse_id,score\n202305010102,1111111111,0.4\n202305010102,2222222222,0.6\n");
  const rg={models:[
    {name:"A",kind:"archived_oof",score_path:"m1.csv",metadata_path:"meta.json",score_field:"score"},
    {name:"B",kind:"archived_oof",score_path:"m2.csv",metadata_path:"meta.json",score_field:"score"},
    {name:"NO_PROMOTION",kind:"unconfigured"}],
    ensembles:[{name:"A_B",members:["A","B"],method:"rank_average"}]};
  await writeFile(join(modelDir,"registry.json"),JSON.stringify(rg));
  const featurePath=join(dir,"features.json");
  await writeFile(featurePath,JSON.stringify(result));
  const procPy=spawnSync("python3",["research/score-one-race.py",
    "--input",featurePath,"--registry",join(modelDir,"registry.json")],{encoding:"utf8"});
  assert.equal(procPy.status,0,procPy.stderr);
  const scored=JSON.parse(procPy.stdout);
  assert.equal(scored.scored_models,3);
  assert.deepEqual(scored.model_results.map(x=>x.status),["SCORED","SCORED","UNAVAILABLE","SCORED"]);
  assert.equal(scored.model_results[0].ranking[0].horse_id,"1111111111");
  assert.equal(scored.model_results[1].ranking[0].horse_id,"2222222222");
  assert.equal(scored.model_results[3].ranking[0].horse_id,"1111111111");

  const combined=spawnSync("python3",[
    "research/analyze-one-race.py",
    "--race","https://race.netkeiba.com/race/shutuba.html?race_id=202305010102",
    "--source-root",dir,"--history-from","2023-01-01",
    "--registry",join(modelDir,"registry.json"),"--output-dir",join(dir,"combined"),
    "--require-scored",
  ],{encoding:"utf8"});
  assert.equal(combined.status,0,combined.stderr);
  const combinedReport=JSON.parse(await readFile(join(dir,"combined","analysis.json"),"utf8"));
  assert.equal(combinedReport.race_id,"202305010102");
  assert.equal(combinedReport.starters,2);
  assert.equal(combinedReport.models.filter(x=>x.status==="SCORED").length,3);
  assert.equal(combinedReport.horses.length,2);
  assert.equal(combinedReport.horses[0].model_ranks.A.rank,2);
  assert.equal(combinedReport.horses[0].model_ranks.B.rank,1);
  assert.equal(combinedReport.horses[1].model_ranks.A_B.rank,1);
  assert.equal(combinedReport.saved_seven_kings.status,"NOT_REQUESTED");

  const metadata=JSON.parse(await readFile(join(modelDir,"meta.json"),"utf8"));
  metadata.split.valid_end="2023-01-02";
  await writeFile(join(modelDir,"meta.json"),JSON.stringify(metadata));
  const rejected=spawnSync("python3",["research/score-one-race.py",
    "--input",featurePath,"--registry",join(modelDir,"registry.json")],{encoding:"utf8"});
  const blocked=JSON.parse(rejected.stdout);
  assert.equal(blocked.scored_models,0);
  assert.match(blocked.model_results[0].reason,/AS_OF_LEAKAGE/);

  delete manifest.days["2023-01-01"];
  await writeFile(join(dir,"data/manifest.json"),JSON.stringify(manifest));
  await assert.rejects(run(),/HISTORY_DAY_NOT_VERIFIED/);
  console.log("SINGLE_RACE_ENGINE_SMOKE_PASS: redaction, feature parity, registry, ensemble, cutoff, completeness");
} finally {
  await rm(dir,{recursive:true,force:true});
}
