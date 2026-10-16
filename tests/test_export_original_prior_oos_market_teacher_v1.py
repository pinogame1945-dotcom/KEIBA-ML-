#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from export_original_prior_oos_market_teacher_v1 import export

def data(year):
    return [
      {"race_id":f"{year}05050001","horse_id":"2021000001",
       "race_date":f"{year}-01-11",
       "features":{"race_date":f"{year}-01-11"},
       "target":{"finish_position":1},
       "market_outcome":{"final_win_odds":2.2,"final_popularity":1}},
      {"race_id":f"{year}05050001","horse_id":"2021000002",
       "race_date":f"{year}-01-11",
       "features":{"race_date":f"{year}-01-11"},
       "target":{"finish_position":2},
       "market_outcome":{"final_win_odds":11.4,"final_popularity":4}}
    ]

class OriginalPastOOSMarket(unittest.TestCase):
    def test_only_past_2021_2024_market_teacher(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/"prior.jsonl"
            source.write_text("".join(json.dumps(x)+"\n" for x in data(2024)))
            dest=Path(root)/"prior-market.csv.gz"
            export(source,2024,dest)
            obj=pd.read_csv(dest,compression="gzip")
            self.assertEqual(len(obj),2)
            self.assertEqual(obj["market_rank"].astype(int).tolist(),[1,4])
            self.assertTrue(obj["final_win_odds"].gt(0).all())
            self.assertEqual(set(obj["_horse_id"].astype(str)),
                             {"2021000001","2021000002"})

    def test_2025_selected_and_2026_outcomes_blocked(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/"bad.jsonl"
            source.write_text("")
            for year in (2025,2026):
                with self.subTest(year=year):
                    with self.assertRaisesRegex(ValueError,"ONLY_PRIOR_2021_24"):
                        export(source,year,Path(root)/"out.csv.gz")

if __name__=="__main__": unittest.main()
