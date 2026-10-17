#!/usr/bin/env python3
"""All targets synthetic: 60 race 786 horse 690->649 safe adapter gate."""
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from prepare_2026_original_13_outsider_649_cached_source_v1 import prepare,source,ORIGINAL_OUTSIDER_MIN_SOURCE

class OriginalOutsider2026SourceSafety(unittest.TestCase):
    def fixtures(self,root,inject_target=False):
        backfill=root/"bf"
        folder=backfill/"ml_features/delivery/690"
        folder.mkdir(parents=True)
        feature_keys=list(ORIGINAL_OUTSIDER_MIN_SOURCE)+[f"f{i:03d}" for i in range(690-len(ORIGINAL_OUTSIDER_MIN_SOURCE))]
        schema=root/"schema.json"
        schema.write_text(json.dumps({"feature_order":feature_keys[:649]}))
        old=root/"2025.jsonl"
        old.write_text(json.dumps({
            "ml_dataset_version":3,"feature_schema_version":8,
            "leakage_policy":"STRICT_PRIOR_DATE_ONLY",
            "features":{"race_date":"2025-10-01"},
            "history_windows":{"race":[1,2,3]},
            "small_sample_policy":{"feature_shrinkage":1.0}
        })+"\n")
        entries={}
        for day,locations in [
            ("2026-10-10",[("05","03"),("08","03")]),
            ("2026-10-17",[("04","01"),("05","06"),("08","06")]),
        ]:
            for venue,d in locations:
                for i in range(1,13):
                    rid=f"2026{venue}04{d}{i:02d}"
                    # 24*13+3=315, 36*13+3=471.
                    is_bonus=(day=="2026-10-10" and venue=="05" and i<=3) or (
                        day=="2026-10-17" and venue=="04" and i<=3)
                    n=14 if is_bonus else 13
                    rows=[]
                    for j in range(n):
                        row={"_race_id":rid,"_horse_id":str(2000000000+len(entries)*24+j),
                             "_race_date":day}
                        row.update({k:0.5 for k in feature_keys})
                        rows.append(row)
                    if inject_target and not entries:
                        rows[0]["target"]={"is_win":True}
                    doc={"contract":"KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1",
                         "race_id":rid,"race_date":day,"feature_count":690,
                         "starters":n,"readiness":"ASOF_UNVERIFIED_DO_NOT_BET",
                         "used_target_results_or_final_odds":False,"rows":rows}
                    path=folder/day/(rid+".json")
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(json.dumps(doc))
                    entries[rid]={"date":day,"starters":n,
                        "path":f"ml_features/delivery/690/{day}/{rid}.json",
                        "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        (folder/"index.json").write_text(json.dumps({
            "contract":"KEIBA_BACKFILL_TO_ML_690_INDEX_V1",
            "feature_count":690,"total_races":60,"total_horses":786,
            "readiness":"ASOF_UNVERIFIED_DO_NOT_BET","races":entries}))
        return backfill,schema,old

    def test_786_rows_partitioned_and_no_2026_labels(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            bf,schema,old=self.fixtures(root)
            dest=root/"result"
            summary=prepare(bf,schema,old,dest)
            self.assertEqual(summary["total_horses"],786)
            self.assertEqual(summary["source_690_sha256_checked"],60)
            self.assertFalse(summary["independent_feature_asof_certified"])
            self.assertEqual(summary["targets"]["2026-10-10"]["horses"],315)
            self.assertEqual(summary["targets"]["2026-10-17"]["horses"],471)
            for day,expected in (("2026-10-10",315),("2026-10-17",471)):
                rows=list(source(dest/f"target-{day}.jsonl.gz"))
                self.assertEqual(len(rows),expected)
                self.assertTrue(all(len(r["features"])==650 and
                        r["target"]=={} and r["features"]["race_date"]==day
                        and r["prediction_phase"]=="FINAL"
                        for r in rows))

    def test_core4_missing_original_style_or_backfill_is_rejected(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            bf,schema,old=self.fixtures(root)
            data=json.loads(schema.read_text())
            data["feature_order"]=[f"onlycore4_{i}" for i in range(649)]
            schema.write_text(json.dumps(data))
            dest=root/"result"
            with self.assertRaisesRegex(ValueError,
                    "ORIGINAL_13_OUTSIDER_NEEDS_FEATURES_NOT_IN_SEVEN_KING_649"):
                prepare(bf,schema,old,dest)
            self.assertFalse(dest.exists())

    def test_tampered_690_sha_is_hard_stop_before_writing(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            bf,schema,old=self.fixtures(root)
            x=next((bf/"ml_features/delivery/690/2026-10-10").glob("*.json"))
            x.write_text(x.read_text()+" ")
            dest=root/"result"
            with self.assertRaisesRegex(ValueError,"690_SOURCE_SHA256_MISMATCH"):
                prepare(bf,schema,old,dest)
            self.assertFalse(dest.exists())

    def test_target_leak_is_rejected_even_with_matching_sha(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            bf,schema,old=self.fixtures(root,inject_target=True)
            dest=root/"result"
            with self.assertRaisesRegex(ValueError,"690_NOT_SUPERSET_OF_ORIGINAL_CORE4_649"):
                prepare(bf,schema,old,dest)
            self.assertFalse(dest.exists())

if __name__=="__main__":
    unittest.main()
