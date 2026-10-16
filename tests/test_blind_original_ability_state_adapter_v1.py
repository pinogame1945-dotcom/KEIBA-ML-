#!/usr/bin/env python3
"""Verify original state/regime functions accept prior compact CSVs unchanged."""
import json,sys,tempfile,unittest
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from run_l1_state_transition_v1 import read_compact_state_source
from run_blind_202505050812_original_ability_v1 import original_state_regime

def row(year,h,race):
    rid=f"{year}0101{race:04d}"
    return {"race_id":rid,"horse_id":f"202000000{h}",
            "race_date":f"{year}-01-{race:02d}",
            "features":{"race_date":f"{year}-01-{race:02d}","distance_m":1600,
                        "body_weight":470+h,"carried_weight":57,
                        "network_expected_pairwise_score":0.3},
            "target":{"finish_position":h,"finish_time_ms":90000+h*1000,
                      "last_3f":34+h,"margin_lengths":h*.1}}

class FrozenStateAdapter(unittest.TestCase):
    def test_full_2019_25_compact_source_through_unchanged_state_and_regime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths={}
            for y in range(2019,2026):
                raw=root/f"{y}.jsonl"
                raw.write_text("".join(json.dumps(row(y,h,10))+"\n" for h in (1,2,3)))
                compact=read_compact_state_source(raw,y)
                csv=root/f"compact-{y}.csv.gz"
                compact.to_csv(csv,index=False,compression="gzip")
                paths[y]=csv
            state,regimes,cols,coverage=original_state_regime(paths)
            self.assertEqual(sorted(state),list(range(2019,2026)))
            self.assertEqual(sorted(regimes),list(range(2019,2026)))
            self.assertEqual(len(cols),66)
            self.assertEqual(len(state[2025]),3)
            self.assertEqual(len(regimes[2025]),3)
            self.assertEqual(len(coverage),7)
            self.assertTrue(regimes[2025]["regime_speed_resid_lag1"].notna().all())

    def test_2025_same_day_target_or_future_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths={}
            for y in range(2019,2026):
                raw=root/f"{y}.jsonl"
                day=30 if y==2025 else 10
                raw.write_text("".join(json.dumps(row(y,h,day))+"\n" for h in (1,2,3)))
                compact=read_compact_state_source(raw,y)
                csv=root/f"compact-{y}.csv.gz"
                compact.to_csv(csv,index=False,compression="gzip")
                paths[y]=csv
            with self.assertRaisesRegex(ValueError,"2025_STATE_TARGET_OR_FUTURE_RESULT_LEAK"):
                original_state_regime(paths)

if __name__=="__main__":unittest.main()
