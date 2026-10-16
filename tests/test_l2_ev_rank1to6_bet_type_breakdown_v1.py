import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_ev_rank1to6_bet_type_breakdown_v1 as report

class BreakdownTest(unittest.TestCase):
    def test_expected_existing_unions(self):
        self.assertEqual([report.EXPECTED[x][0] for x in report.STAGES],
                         [653734,130958,7325,949])
        self.assertEqual(report.YEARS,(2023,2024,2025))
    def test_unique_physical_ticket_aggregation(self):
        x=[
            ((2023,"raceA","QUINELLA","1,2"),500),
            ((2023,"raceA","QUINELLA","1,3"),0),
            ((2023,"raceB","QUINELLA","1,2"),0),
        ]
        a=report.metric({"stage":"TEST","year":"2023","ticket_type":"QUINELLA"},x)
        self.assertEqual((a["tickets"],a["hits"],a["races_bought"],a["races_hit"]),(3,1,2,1))
        self.assertEqual((a["stake_yen"],a["payout_yen"],a["profit_yen"]),(300,500,200))
    def test_real_winning_payout_multiplier_band(self):
        self.assertEqual(report.payout_band(900),"10倍未満")
        self.assertEqual(report.payout_band(1000),"10〜30倍未満")
        self.assertEqual(report.payout_band(100000),"1000倍以上")
    def test_selected_original_recipe_is_not_physical_ticket_count(self):
        base=dict(stage="ALL_YEAR_BLACK",ev_rank=1,partner_order="AI_PROB",
                  original_ticket_template="TRIO",source_line="same_original_line",
                  historical_ticket_odds_band="20-30",horse_gate="ALL")
        x=report.profile_rows([dict(base),dict(base,horse_gate="GATE_X")])
        self.assertEqual(x[0]["accepted_line_gate_pairs"],2)
        self.assertEqual(x[0]["selected_unique_source_lines"],1)

if __name__=="__main__":unittest.main()
