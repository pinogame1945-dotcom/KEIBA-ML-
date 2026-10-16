#!/usr/bin/env python3
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from export_original_core4_prior_ability_year_v1 import build

def row(year,race,horse,finish):
    return {"race_id":f"{year}0101{race:04d}",
            "horse_id":f"{year}123{race:03d}{horse:03d}",
            "race_date":f"{year}-01-05",
            "features":{"race_date":f"{year}-01-05","venue_code":"TOKYO",
                        "opponent_rating":horse/3,"network_hidden":5,
                        "race_no":race,"horse_number":horse},
            "target":{"finish_position":finish,"finish_time_ms":70000,
                      "last_3f":35.2}}

class PriorAbilityTrainingAudit(unittest.TestCase):
    def test_original_read_year_preserves_legitimate_dead_heat(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"snapshot-2023.jsonl.gz"
            rows=[row(2023,1,1,1),row(2023,1,2,1),row(2023,1,3,3),
                  row(2023,2,1,2),row(2023,2,2,1),row(2023,2,3,3)]
            with gzip.open(p,"wt") as f:
                for r in rows: f.write(json.dumps(r)+"\n")
            d=build(p,2023,len(rows),"2023-01-05","2023-01-05",tmp)
            self.assertEqual(d["training_horse_rows"],6)
            self.assertEqual(d["training_races"],2)
            self.assertEqual(d["multi_winner_races_preserved_per_original"],1)
            self.assertIn("opponent_rating",d["feature_names"])
            self.assertNotIn("network_hidden",d["feature_names"])
            self.assertTrue((Path(tmp)/"original-core4-train-2023.csv.gz").is_file())

    def test_wrong_manifest_counts_must_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"bad.jsonl"
            p.write_text(json.dumps(row(2024,1,1,1))+"\n")
            with self.assertRaisesRegex(ValueError,"MANIFEST_COUNT_OR_DATE_DRIFT"):
                build(p,2024,2,"2024-01-05","2024-01-05",tmp)

    def test_target_year_not_allowed_as_training(self):
        with self.assertRaisesRegex(ValueError,"ABILITY_YEAR_MUST_BE_2023_OR_2024"):
            build("not-a-file",2025,1,"2025-01-05","2025-01-05","/tmp")

if __name__=="__main__":
    unittest.main()
