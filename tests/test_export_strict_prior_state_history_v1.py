#!/usr/bin/env python3
"""Historical real-data exporter safety: TARGET and future outcome never read."""
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from export_strict_prior_state_history_v1 import export


def row(rid,date,hid,finish):
    return {"race_id":rid,"horse_id":hid,"race_date":date,
      "features":{"race_date":date,"distance_m":1600,
                  "network_expected_pairwise_score":.5,"body_weight":470,
                  "carried_weight":55},
      "target":{"finish_position":finish,"finish_time_ms":95000,
                "last_3f":35.0,"margin_lengths":.2}}


class StrictHistoricalState(unittest.TestCase):
    def test_user_target_and_same_day_outcomes_not_passed_into_original_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"2025.jsonl"
            rows=[row("202505050801","2025-11-29","2020000001",1),
                  row("202505050801","2025-11-29","2020000002",2),
                  row("202505050812","2025-11-30","2020000001",999),
                  row("202505050811","2025-11-30","2020000002",999),
                  row("202505050813","2025-12-01","2020000003",999)]
            p.write_text("".join(json.dumps(x)+"\n" for x in rows))
            dest,proof=export(p,2025,5,"2025-11-29","2025-12-01",tmp)
            self.assertEqual(proof["source_rows_before_cutoff"],2)
            self.assertEqual(proof["original_valid_finish_rows"],2)
            self.assertFalse(proof["selected_race_outcome_accessed"])
            self.assertFalse(proof["date_on_or_after_cutoff_outcome_accessed"])
            self.assertTrue(dest.is_file())
            import pandas as pd
            data=pd.read_csv(dest)
            self.assertTrue(data["_race_date"].eq("2025-11-29").all())
            self.assertEqual(len(data),2)

    def test_drift_and_2026_seal_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"2025.jsonl"
            p.write_text(json.dumps(row("202505050801","2025-01-01","2020000001",1))+"\n")
            with self.assertRaisesRegex(ValueError,"MANIFEST_ROWS_OR_DATES_DRIFT"):
                export(p,2025,2,"2025-01-01","2025-01-01",tmp)
            with self.assertRaisesRegex(ValueError,"YEARS_ONLY_2019_TO_2025"):
                export(p,2026,1,"2026-01-01","2026-01-01",tmp)

if __name__=="__main__":unittest.main()
