"""One-shot execution of all established KEIBA runtime components."""
import argparse
from . import analyze_to_json


def main():
    parser = argparse.ArgumentParser(description="KEIBA one-shot race runtime")
    parser.add_argument("--race", action="append", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_to_json(args.race, args.manifest, args.output)
    print("KEIBA_ONE_SHOT", result["status"], len(result["races"]))
    for race in result["races"]:
        print(race["race_id"], race["status"], race.get("blocked_reason") or "")


if __name__ == "__main__":
    main()
