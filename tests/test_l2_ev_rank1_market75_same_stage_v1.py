import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_ev_rank1_market75_same_stage_v1 as study
import run_l2_fixed_ticket_pruning_3040_v1 as frozen

class SameStageTests(unittest.TestCase):
    def test_selected_method_preapproved_not_reselected(self):
        self.assertEqual(study.POLICY,"MARKET_75")
        self.assertIn(study.POLICY,frozen.POLICIES)
        self.assertEqual(len(study.ev.LINE_KEYS),3040)
        self.assertEqual(study.ev.RANKS,(2,3,4,5,6))

    def test_original_ev1_baseline_exact(self):
        path="research-results/l2-ticket-odds-band-profit-v1/EV軸_全152買い方_オッズ帯別_年度推移_詳細20区間.csv"
        a=study.read_original(path)
        self.assertEqual(len(a),3040)
        self.assertTrue(all(set(r)==set(study.YEARS) for r in a.values()))

    def test_common_january_excluded(self):
        self.assertFalse(study.ev.pruning_eval_eligible(2023,"2023-01-29"))
        self.assertTrue(study.ev.pruning_eval_eligible(2023,"2023-02-04"))
        self.assertTrue(study.ev.pruning_eval_eligible(2024,"2024-01-06"))

if __name__=="__main__":unittest.main()
