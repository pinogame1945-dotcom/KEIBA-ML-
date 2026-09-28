#!/usr/bin/env python3
import argparse,hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--ref",required=True)
    p.add_argument("--title",required=True)
    p.add_argument("--source-dir",required=True)
    return p.parse_args()

def run(*xs,capture=False,check=True):
    p=subprocess.run([str(x) for x in xs],text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None)
    if check and p.returncode:
        raise RuntimeError((p.stderr or p.stdout or "").strip())
    return p

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    a=args()
    if os.environ.get("KAGGLE_STORAGE_PREFLIGHT_OK")!="1":
        raise SystemExit("storage preflight marker missing")
    src=Path(a.source_dir)
    files=sorted(p for p in src.iterdir() if p.is_file())
    if not files: raise SystemExit("source dir empty")
    manifest={"contract":"KEIBA_L15_7KING_STAGE_V1","ref":a.ref,
              "files":[{"name":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in files]}
    with tempfile.TemporaryDirectory() as td:
        st=Path(td)
        for p in files: shutil.copy2(p,st/p.name)
        (st/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (st/"dataset-metadata.json").write_text(json.dumps({
            "id":a.ref,"title":a.title[:50],"licenses":[{"name":"other"}],
            "description":"Private KEIBA-ML seven-king L1.5 staging dataset; capacity checked before compute."
        },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

        status=run("kaggle","datasets","status",a.ref,capture=True,check=False)
        if status.returncode==0:
            print("Dataset already exists; verifying manifest only.")
        else:
            run("kaggle","datasets","create","-p",str(st),"--quiet","--keep-tabular","--dir-mode","skip")

        ok=False
        with tempfile.TemporaryDirectory() as vd:
            v=Path(vd)
            for i in range(1,61):
                m=v/"manifest.json"
                if m.exists(): m.unlink()
                r=run("kaggle","datasets","download",a.ref,"-f","manifest.json","-p",str(v),
                      "--unzip","--quiet","--force",capture=True,check=False)
                if r.returncode==0 and m.is_file():
                    if m.read_bytes()!=(st/"manifest.json").read_bytes():
                        raise SystemExit("existing/remote manifest mismatch")
                    ok=True
                    print("KAGGLE_EXACT_PERSIST_VERIFIED",a.ref)
                    break
                if i<60: time.sleep(10)
        if not ok: raise SystemExit("remote dataset did not become readable")
    print("dataset_ref="+a.ref)

if __name__=="__main__": main()
