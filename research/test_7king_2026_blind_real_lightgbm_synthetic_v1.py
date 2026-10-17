#!/usr/bin/env python3
"""Real LightGBM smoke for the 2026-only ORIGINAL L1->L2 scoring adapter.

TRAINING/TARGET ROWS ARE SYNTHETIC. No 2026 horse, odds, result or real
model is downloaded, opened or saved. Exactly 786 fake target rows are
scored to exercise production chunk/SHAP/rank normalization logic.
"""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from score_7king_2026_blind_from_approved_models_v1 import (
    CANDIDATES, model_gate, score_king,
)

FEATURES=["f1","f2","f3","f4"]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class SevenKingBlindSyntheticScorerSmoke(unittest.TestCase):
    def test_actual_lightgbm_prediction_ranks_shap_and_normalization(self):
        rng=np.random.default_rng(20261017)
        with tempfile.TemporaryDirectory(prefix="fake-7king-2026-") as td:
            root=Path(td)
            folder=root/"dummy-model"
            folder.mkdir()
            synthetic=pd.DataFrame({
                name:rng.normal(0,1,250).tolist() for name in FEATURES
            })
            labelled=(synthetic["f1"]+0.5*synthetic["f2"]>0).astype(int)
            booster=lgb.train(
                {"objective":"binary","metric":"binary_logloss",
                 "num_leaves":7,"min_data_in_leaf":5,"learning_rate":0.07,
                 "num_threads":2,"verbosity":-1,
                 "deterministic":True,"force_col_wise":True},
                lgb.Dataset(synthetic,label=labelled),
                num_boost_round=12,
            )
            model=folder/"model.txt"
            booster.save_model(str(model))
            schema={
                "contract":"L1_FEATURE_SCHEMA_V1","model_version":"SYNTHETIC_ONLY",
                "feature_order":FEATURES,"category_levels":{},
                "prediction_phase":"FINAL","ability_uses_odds":False,
                "ml_dataset_version":3,"feature_schema_version":8,
                "leakage_policy":"STRICT_PRIOR_DATE_ONLY",
                "training_years":[2024,2025],
            }
            sch=folder/"schema.json"
            sch.write_text(json.dumps(schema),encoding="utf-8")
            meta={
                "contract":"KEIBA_7KING_2026_BLIND_MODEL_RELEASE_V1",
                "deployment_status":"APPROVED_2026_BLIND",
                "candidate_name":"core4_no_pedigree",
                "prediction_year":2026,"training_years":[2024,2025],
                "train_start":"2024-01-01","train_end":"2025-12-31",
                "snapshot_generation":"eb13d4096519167b",
                "2026_outcomes_used":False,"ability_uses_odds":False,
                "feature_selection":"none","prediction_phase":"FINAL",
                "training_rule_id":"UNIT_TEST_SYNTHETIC_ONLY",
                "training_rule_approval_ref":"NOT_A_REAL_APPROVAL_UNIT_TEST_ONLY",
                "model_version":"SYNTHETIC_ONLY",
                "model_sha256":sha(model),"schema_sha256":sha(sch),
                "best_iteration":12,"feature_sets":["BASE"],
            }
            (folder/"model.json").write_text(json.dumps(meta),encoding="utf-8")
            fake_input=root/"fake-targets.jsonl"
            expected=set()
            per_race={}
            with fake_input.open("w",encoding="utf-8") as out:
                for race in range(60):
                    rid="2026"+str(race+1).zfill(8)
                    runners=14 if race<6 else 13
                    per_race[rid]=runners
                    for h in range(runners):
                        hid=str(2018000000+race*20+h)
                        expected.add((rid,hid))
                        value={
                            name:float(rng.normal()) for name in FEATURES
                        }
                        record={"race_id":rid,"horse_id":hid,
                                "features":{"race_date":"2026-10-17",**value}}
                        out.write(json.dumps(record)+"\n")
            self.assertEqual(len(expected),786)
            with patch.dict(CANDIDATES,{"core4":("core4_no_pedigree",4)}):
                gates=model_gate("core4",folder)
                output=root/"fake-scores.jsonl.gz"
                summary=score_king("core4",gates,fake_input,expected,output)
            self.assertEqual(summary["horse_rows"],786)
            self.assertEqual(summary["races"],60)
            self.assertEqual(summary["best_iteration"],12)
            self.assertEqual(summary["model_sha256"],sha(model))
            records=[]
            with gzip.open(output,"rt",encoding="utf-8") as src:
                for line in src:records.append(json.loads(line))
            self.assertEqual(len(records),786)
            by_race={}
            for x in records:
                self.assertEqual(x["contract"],"L1_TO_L2_OUTPUT_CONTRACT_V1")
                self.assertEqual(x["prediction_phase"],"FINAL")
                self.assertEqual(x["ml_dataset_version"],3)
                self.assertEqual(x["feature_schema_version"],8)
                self.assertEqual(x["leakage_policy"],"STRICT_PRIOR_DATE_ONLY")
                self.assertTrue(np.isfinite(x["raw_win_probability"]))
                self.assertTrue(np.isfinite(x["raw_margin_logit"]))
                self.assertTrue(np.isfinite(x["bias_logit"]))
                self.assertLess(abs(x["raw_margin_logit"]-
                    x["bias_logit"]-sum(x["family_contribution_logit"].values())),1e-5)
                self.assertNotIn("finish_position",x)
                self.assertNotIn("payout",x)
                by_race.setdefault(x["race_id"],[]).append(x)
            self.assertEqual(len(by_race),60)
            for rid,rs in by_race.items():
                self.assertEqual(len(rs),per_race[rid])
                self.assertEqual(
                    sorted(x["predicted_rank"] for x in rs),
                    list(range(1,len(rs)+1)))
                self.assertAlmostEqual(
                    sum(x["race_normalized_win_probability"] for x in rs),1,places=12)

if __name__=="__main__":
    unittest.main()
