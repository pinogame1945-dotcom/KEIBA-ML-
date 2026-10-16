import sys,unittest,tempfile,csv,gzip
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_ev_rank1to6_stabilized_capture_v1 as cap
import run_l2_ev_rank1to6_stability_union_v1 as union
import run_l2_ev_rank1to6_physical_union_v1 as original

class SixRankOfficialStabilityTest(unittest.TestCase):
    def test_exact_original_17_gates(self):
        self.assertEqual(len(cap.GATES),17)
        self.assertEqual(tuple(cap.RULES),("ALL_YEAR_BLACK","STABLE_MEDIUM","STABLE_STRICT"))
        self.assertEqual(cap.official.CONFIG["STABLE_MEDIUM"],
                         dict(n=500,hits=30,per_year_hits=8,max1=.4,max3=.7,minus3_roi=90))
        self.assertEqual(cap.official.CONFIG["STABLE_STRICT"],
                         dict(n=800,hits=50,per_year_hits=12,max1=.3,max3=.55,minus3_roi=100))
    def test_six_frozen_pruners_and_previous_study(self):
        self.assertEqual([original.EXPECT[i][0] for i in range(1,7)],
                         ["MARKET_10","MARKET_25","PROB_25",
                          "MARKET_75","LEARNED_10","LEARNED_10"])
        self.assertEqual(set(union.PREVIOUS_OFFICIAL),{2,3,4,5,6})
        self.assertEqual(union.PREVIOUS_OFFICIAL[5]["STABLE_MEDIUM"],(6376,746870))
        self.assertEqual(sum(union.PREVIOUS_OFFICIAL[r]["STABLE_STRICT"][0] for r in range(2,7)),0)
        self.assertEqual(tuple(union.YEARS),(2023,2024,2025))
    def test_source_rank1_saved_20_pruner(self):
        from pathlib import Path
        f=Path("research-results/l2-ev-rank1-same20-pruners-v1/run-38919159628/rank-1/summary.json")
        import json
        a=json.loads(f.read_text())
        self.assertEqual(a["adopted_policy"],"MARKET_10")
        self.assertEqual(a["adopted_portfolio"]["n"],145224)
    def test_stage_csv_parse_counts(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"sample.csv.gz"
            with gzip.open(p,"wt",encoding="utf-8",newline="") as f:
                w=csv.writer(f)
                w.writerow(["year","race_id","bet_type","selection","payout_yen"])
                w.writerow([2023,"r1","QUINELLA","1,2",0])
                w.writerow([2024,"r2","TRIO","1,2,3",1240])
            ref={}
            d=union.stage_record(p,1,"STABLE_MEDIUM",ref)
            self.assertEqual(len(ref),2)
            self.assertEqual(d[2023]["tickets"],1)
            self.assertEqual(d[2024]["payout"],1240)

if __name__=="__main__":unittest.main()
