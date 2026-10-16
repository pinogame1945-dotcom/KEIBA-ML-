#!/usr/bin/env python3
"""Scope the existing ORIGINAL blind research pipeline to 2025 Arima Kinen.

Only changes the target identity/date/cardinality guards in the ephemeral Actions
checkout. No model hyperparameters, training folds, p3 coverage, L2.1 rules,
L3 V3 stakes or stored historical predictions are altered.
Never run this on the canonical development or freeze branches.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD_ID="202505050812"
NEW_ID="202506050811"
OLD_DATE="2025-11-30"
NEW_DATE="2025-12-28"
TARGET_SCRIPTS=(
    "research/export_strict_prior_state_history_v1.py",
    "research/compute_blind_target_prior_day_bias_v1.py",
    "research/compute_blind_target_seq_regime_v1.py",
    "research/run_blind_202505050812_original_ability_v1.py",
    "research/rebuild_blind_original_2025_risk_from_prior_oos_v1.py",
    "research/bridge_blind_original_l1_to_l175_v1.py",
    "research/run_fresh_202505050812_historical_final_l21_v3_v1.py",
    "scripts/publish_fresh_historical_202505050812_mobile_v1.py",
)
NUMERIC_HORSE_COUNT=re.compile(r"(?<![A-Za-z0-9_])17(?![A-Za-z0-9_])")

def replace_once(text,old,new,path):
    count=text.count(old)
    if count!=1:
        raise ValueError(f"ARIMA_GUARD_EXPECTED_ONCE:{path}:{old}:{count}")
    return text.replace(old,new)

def main():
    updated={}
    for name in TARGET_SCRIPTS:
        p=ROOT/name
        original=p.read_text(encoding="utf-8")
        if OLD_ID not in original:
            raise ValueError(f"ARIMA_SOURCE_TARGET_ID_MISSING:{name}")
        value=original.replace(OLD_ID,NEW_ID).replace(OLD_DATE,NEW_DATE)
        # Integer-only participant bounds, not identifiers, hashes or variable names.
        value=NUMERIC_HORSE_COUNT.sub("16",value)
        if name=="research/compute_blind_target_prior_day_bias_v1.py":
            value=replace_once(value,"target_race_no==12","target_race_no==11",name)
        if name=="research/run_blind_202505050812_original_ability_v1.py":
            value=replace_once(value,"range(1,18)","range(1,17)",name)
        if name=="research/bridge_blind_original_l1_to_l175_v1.py":
            value=replace_once(value,"range(1,18)","range(1,17)",name)
        if name=="research/run_fresh_202505050812_historical_final_l21_v3_v1.py":
            value=replace_once(value,"len(entries)==18","len(entries)==16",name)
            value=replace_once(value,"len(scratched)==1","len(scratched)==0",name)
            value=replace_once(value,'"race_entry_count":18','"race_entry_count":16',name)
        if name=="scripts/publish_fresh_historical_202505050812_mobile_v1.py":
            value=replace_once(value,'"race_entry_count":18','"race_entry_count":16',name)
        if OLD_ID in value or OLD_DATE in value:
            raise ValueError(f"ARIMA_OLD_TARGET_STILL_PRESENT:{name}")
        if original==value:
            raise ValueError(f"ARIMA_TARGET_UNCHANGED:{name}")
        p.write_text(value,encoding="utf-8")
        updated[name]=len(value)
    # Reassert essential original modules are NOT patched.
    for name in ("research/l21_github_frozen_prebet.py","research/l3_v3_operational.py",
                 "contracts/KEIBA_L2_1_FIXED_PURCHASE_V1.json",
                 "research/run_market_fixed_anchor_residual_v2.py"):
        if not (ROOT/name).is_file():
            raise ValueError(f"ARIMA_IMMUTABLE_SOURCE_MISSING:{name}")
    print("ARIMA_TARGET_ONLY_EPHEMERAL_SOURCE_PATCH_OK",NEW_ID,NEW_DATE,
          "started=16 scratched=0",len(updated),updated,flush=True)

if __name__=="__main__":main()
