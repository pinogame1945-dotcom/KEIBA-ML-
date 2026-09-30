#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--files-json",required=True)
    p.add_argument("--wanted",action="append",required=True)
    return p.parse_args()

def main():
    a=ap()
    rows=json.loads(Path(a.files_json).read_text(encoding="utf-8"))
    names=[str(x.get("name") or "") for x in rows]
    for wanted in a.wanted:
        if wanted in names:
            print(wanted)
            return
    nested=[x for x in names if any(x.endswith("/"+w) for w in a.wanted)]
    if len(nested)==1:
        print(nested[0])
        return
    raise SystemExit(f"requested file missing/ambiguous wanted={a.wanted} names={names}")

if __name__=="__main__":
    main()
