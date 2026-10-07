#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def norm_logical(name):
    name=str(name or "").strip()
    return name[:-3] if name.endswith(".gz") else name

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--files-json",required=True)
    p.add_argument("--generation-id",required=True)
    p.add_argument("--years",default="2019,2020,2021,2022,2023,2024,2025")
    p.add_argument("--output",required=True)
    a=p.parse_args()

    m=json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    if (m.get("generation") or {}).get("generation_id")!=a.generation_id:
        raise SystemExit("snapshot generation mismatch")

    raw=json.loads(Path(a.files_json).read_text(encoding="utf-8"))
    remotes=[str(x.get("name") or "").strip() for x in raw if str(x.get("name") or "").strip()]
    if len(remotes)!=len(set(remotes)):
        raise SystemExit("duplicate remote names in metadata")
    if "manifest.json" not in remotes:
        raise SystemExit("manifest.json missing from Kaggle metadata")

    by_year={int(x["year"]):x for x in (m.get("years") or [])}
    years=[int(x) for x in a.years.split(",") if x.strip()]
    lines=[]
    for y in years:
        if y not in by_year:
            raise SystemExit(f"manifest year missing: {y}")
        row=by_year[y]
        logical=str(row.get("file") or "").strip()
        if not logical:
            raise SystemExit(f"manifest file missing y={y}")

        exact=[r for r in remotes if r==logical]
        if len(exact)==1:
            remote=exact[0]
            mode="EXACT"
        else:
            key=norm_logical(logical)
            compatible=[r for r in remotes if norm_logical(r)==key]
            if len(compatible)!=1:
                raise SystemExit(
                    f"metadata resolution failed y={y} logical={logical} "
                    f"compatible={compatible}; stop as metadata drift"
                )
            remote=compatible[0]
            mode="METADATA_COMPRESSION_EQUIVALENT"

        lines.append([
            str(y),logical,remote,mode,str(row.get("sha256") or ""),
            str(row.get("rows") or ""),str(row.get("min_date") or ""),str(row.get("max_date") or "")
        ])

    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(
        "\n".join("\t".join(x) for x in lines)+"\n",
        encoding="utf-8"
    )
    for x in lines:
        print("RESOLVED_MANIFEST_REMOTE",json.dumps({
            "year":int(x[0]),"manifest_name":x[1],"remote_name":x[2],"mode":x[3]
        },separators=(",",":")))

if __name__=="__main__":
    main()
