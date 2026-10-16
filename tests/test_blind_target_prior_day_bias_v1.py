#!/usr/bin/env python3
import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from compute_blind_target_prior_day_bias_v1 import calculate


def target():
    return {"contract":"KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1",
      "race_id":"202505050812",
      "used_target_labels":False,
      "used_results_payouts_final_odds":False,
      "source_prediction_phase":"FINAL_PRESTART_PROJECTED_FROM_HISTORICAL_SNAPSHOT",
      "rows":[{"_race_id":"202505050812",
               "_horse_id":str(2024000000+i),
               "_race_date":"2025-11-30",
               "venue_code":"TOKYO","surface":"TURF",
               "race_no":12,"horse_number":i,"field_size":17,
               "style_recent_avg_first_ratio":.20+i/100}
              for i in range(1,18)]}

def make_row(rid,number,finish,*,race_no,venue="TOKYO",surface="TURF",
             date="2025-11-30"):
    return {"race_id":rid,"horse_id":str(2023000000+number),
      "race_date":date,
      "features":{"race_date":date,"venue_code":venue,"surface":surface,
                  "race_no":race_no,"horse_number":number,"field_size":3,
                  "distance_m":1600,"style_recent_avg_first_ratio":.15+number/10},
      "target":{"finish_position":finish,"finish_time_ms":96000,
                "last_3f":35.1+number}}

def write(tmp,rows):
    path=Path(tmp)/"2025.jsonl"
    path.write_text("".join(json.dumps(x)+"\n" for x in rows))
    return path

class BlindDayBias(unittest.TestCase):
    def test_original_bias_uses_only_prior_same_track_and_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[make_row("202505050810",i,i,race_no=10) for i in (1,2,3)]
            rows+=[make_row("202505050811",i,i,race_no=11,surface="DIRT") for i in (1,2,3)]
            rows+=[make_row("202505050812",i,i,race_no=12) for i in (1,2,3)]
            rows+=[make_row("202505050813",i,i,race_no=13) for i in (1,2,3)]
            out=calculate(write(tmp,rows),target())
            self.assertEqual(out["source_prior_race_count"],1)
            self.assertEqual(out["source_prior_race_ids"],["202505050810"])
            self.assertEqual(len(out["horse_scores"]),17)
            self.assertTrue(all(x["features"]["daybias_prior_races"]==1.0
                               for x in out["horse_scores"]))
            self.assertFalse(out["target_result_or_final_odds_used"])
            self.assertFalse(out["target_race_start_timestamp_independently_verified"])
            self.assertTrue(all("fit_front_ewma" in x["features"]
                                for x in out["horse_scores"]))
            changed=copy.deepcopy(rows)
            for x in changed:
                if x["race_id"]=="202505050812":x["target"]["finish_position"]=0
            same=calculate(write(tmp,changed),target())
            self.assertEqual(same["horse_scores"],out["horse_scores"])

    def test_missing_prior_races_is_unknown_not_fake_bias(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=write(tmp,[make_row("202505050812",1,1,race_no=12)])
            out=calculate(p,target())
            self.assertEqual(out["source_prior_race_count"],0)
            self.assertTrue(all(x["features"]["fit_front_mean"] is None
                                for x in out["horse_scores"]))

    def test_target_identity_and_stage_rejected_if_wrong(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=write(tmp,[make_row("202505050810",1,1,race_no=10)])
            altered=target();altered["rows"][0]["race_no"]=9
            with self.assertRaisesRegex(ValueError,"TARGET_META_DRIFT"):
                calculate(p,altered)
            altered=target();altered["used_target_labels"]=True
            with self.assertRaisesRegex(ValueError,"POSTRACE_TARGET_INPUT"):
                calculate(p,altered)

if __name__=="__main__":
    unittest.main()
