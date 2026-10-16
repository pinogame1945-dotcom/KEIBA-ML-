#!/usr/bin/env python3
"""GitHub public archive downloader: metadata before bytes; verify Git blob SHA.

No clone, no unverified guessed/fallback URL, no artifact/cache upload.
A source that changed between metadata and download fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_RE=re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MAX_RETRIES=3
HTTP_TRANSIENT={429,502,503,504}

def request_bytes(url, *,max_bytes):
    req=urllib.request.Request(url,headers={"User-Agent":"KEIBA-ML-readonly-archive",
        "Accept":"application/vnd.github+json" if url.startswith("https://api.github.com/") else "application/octet-stream"})
    for attempt in range(MAX_RETRIES+1):
        try:
            with urllib.request.urlopen(req,timeout=30) as result:
                length=result.headers.get("Content-Length")
                if length and int(length)>max_bytes:
                    raise ValueError("HTTP response exceeds verified size limit")
                raw=result.read(max_bytes+1)
                if len(raw)>max_bytes:raise ValueError("HTTP response too large")
                return raw
        except urllib.error.HTTPError as ex:
            if ex.code not in HTTP_TRANSIENT or attempt>=MAX_RETRIES:
                raise ValueError(f"GitHub HTTP {ex.code} (404 never retried)") from ex
        except (urllib.error.URLError,socket.timeout,TimeoutError):
            if attempt>=MAX_RETRIES:raise
        time.sleep(2**attempt)
    raise RuntimeError("no response")

def metadata(repo,ref,path):
    if not REPO_RE.fullmatch(repo):raise ValueError("invalid GitHub repository")
    if not path or path.startswith("/") or any(x in ("",".","..") for x in path.split("/")):
        raise ValueError("invalid archive path")
    if ref.startswith("-") or not ref:raise ValueError("invalid ref")
    url=("https://api.github.com/repos/"+repo+"/contents/"+
         urllib.parse.quote(path,safe="/")+"?ref="+urllib.parse.quote(ref,safe=""))
    data=json.loads(request_bytes(url,max_bytes=3_000_000))
    if not isinstance(data,dict) or data.get("type")!="file":
        raise ValueError("archive metadata is not a regular file: "+path)
    if data.get("path")!=path:
        raise ValueError("ARCHIVE_METADATA_PATH_DRIFT "+path)
    if not re.fullmatch(r"[a-f0-9]{40}",str(data.get("sha") or "")):
        raise ValueError("Git SHA missing in metadata")
    return data

def git_blob_sha(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()

def obtain(repo,ref,path,dest,cap):
    if cap<1 or cap>100_000_000:raise ValueError("unsafe download cap")
    dest=Path(dest).resolve()
    dest.parent.mkdir(parents=True,exist_ok=True)
    info=metadata(repo,ref,path)
    size=info.get("size")
    if not isinstance(size,int) or size<1 or size>cap:
        raise ValueError(f"ARCHIVE_STORAGE_GUARD path={path} bytes={size}")
    if dest.is_file():
        blob=dest.read_bytes()
        if len(blob)==size and git_blob_sha(blob)==info["sha"]:
            print("ARCHIVE_LOCAL_VERIFIED_REUSE",path,size,flush=True)
            return
    source=info.get("download_url")
    if not isinstance(source,str) or not source.startswith(
        "https://raw.githubusercontent.com/"+repo+"/"):
        raise ValueError("metadata has no approved immutable raw source for "+path)
    blob=request_bytes(source,max_bytes=size)
    if len(blob)!=size or git_blob_sha(blob)!=info["sha"]:
        raise ValueError("ARCHIVE_METADATA_DRIFT_SHA_OR_SIZE "+path)
    tmp=dest.with_name(dest.name+".partial."+str(os.getpid()))
    try:
        tmp.write_bytes(blob)
        os.replace(tmp,dest)
    finally:
        tmp.unlink(missing_ok=True)
    print("ARCHIVE_VERIFIED_DOWNLOAD",repo,path,size,info["sha"],flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True)
    p.add_argument("--ref",required=True)
    p.add_argument("--path",required=True)
    p.add_argument("--dest",required=True)
    p.add_argument("--max-bytes",type=int,default=50_000_000)
    a=p.parse_args()
    obtain(a.repo,a.ref,a.path,a.dest,a.max_bytes)

if __name__=="__main__":
    main()
