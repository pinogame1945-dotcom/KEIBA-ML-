"""Read-only synthetic validation of 7 independent experts / 13 ballots."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import check_l1_seven_king_outsider_handoff_v1 as r

RID="202605040301"
DATE="2026-10-10"
HORSES=["2024100001","2024100002","2024100003"]


def source(root):
    obj={"contract":r.BACKFILL_CONTRACT,"race_id":RID,"race_date":DATE,
         "readiness":"ASOF_UNVERIFIED_DO_NOT_BET",
         "used_target_results_or_final_odds":False,
         "source_capture_asof_independently_verified":False,
         "feature_count":690,"starters":3,
         "rows":[{"_race_id":RID,"_race_date":DATE,"_horse_id":hid,
                  **{f"example_feature_{i}":None for i in range(690)}}
                 for hid in HORSES]}
    p=root/"race.json"
    p.write_text(json.dumps(obj))
    return p


def expert(root,alias,wrong_year=False):
    p=root/f"{alias}.jsonl"
    p.write_text("".join(json.dumps({
        "contract":r.EXPERT_CONTRACT,
        "race_id":("202505040301" if wrong_year else RID),
        "horse_id":h,"predicted_rank":i+1},ensure_ascii=False)+"\\n"
        for i,h in enumerate(HORSES)))
    return p


def ballots(root):
    p=root/"ballots.csv"
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["race_id","candidate",
                         "top1_horse_id","top2_horse_id","top3_horse_id"])
        w.writeheader()
        for cand in r.OUTSIDERS:
            w.writerow({"race_id":RID,"candidate":cand,
                        "top1_horse_id":HORSES[0],
                        "top2_horse_id":HORSES[1],
                        "top3_horse_id":HORSES[2]})
    return p


class TestSevenKingHandoff(unittest.TestCase):
    def test_missing_real_experts_stays_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report=r.preflight(source(root),{},None)
            self.assertEqual(report["status"],"BLOCKED_ORIGINAL_7KING_OUTSIDER_2026_INPUTS")
            self.assertEqual(len(report["missing_experts"]),7)
            self.assertFalse(report["models_trained_or_scored"])

    def test_7_experts_and_13_ballots_are_identity_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inp=source(root)
            experts={a:expert(root,a) for a in r.EXPERTS}
            report=r.preflight(inp,experts,ballots(root))
            self.assertEqual(report["status"],"READY_FOR_FROZEN_REPLAY_INPUT_REVIEW")
            self.assertEqual(report["outsider_ballots_verified"],13)
            self.assertFalse(report["original_council_executed"])

    def test_historical_scores_rejected_as_2026(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inp=source(root)
            with self.assertRaisesRegex(ValueError,"OTHER_RACE_SCORE_FORBIDDEN"):
                r.preflight(inp,{"core4":expert(root,"core4",True)},None)

    def test_mismatched_horse_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inp=source(root)
            x=expert(root,"core4")
            t=x.read_text().replace(HORSES[0],"2024999999")
            x.write_text(t)
            with self.assertRaisesRegex(ValueError,"EXPERT_HORSE_ID_MISMATCH"):
                r.preflight(inp,{"core4":x},None)

    def test_wrong_690_population_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inp=source(root)
            doc=json.loads(inp.read_text())
            doc["rows"][0].pop("example_feature_0")
            inp.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,"690_COLUMN_COUNT_DRIFT"):
                r.preflight(inp,{},None)

if __name__=="__main__":
    unittest.main()
