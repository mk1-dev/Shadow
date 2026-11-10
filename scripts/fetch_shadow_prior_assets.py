"""Fetch the pinned official FSD source/checkpoint outside the Shadow worktree."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

FSD_COMMIT = "b22927c5a2ec25c878a15ecf76344dde474fe503"
FSD_SHA256 = "a408c3e9f6cb4c6b304c5cfe742346e97f6434c73e9116b78a362a7180cd58f2"
FSD_URL = f"https://media.githubusercontent.com/media/IMU-Group/FSD/{FSD_COMMIT}/ckpt/FSD_best.ckpt"


def checksum(path):
    with Path(path).open("rb") as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--proxy", help="Optional HTTP(S) proxy for Git and the weight download")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        raise ValueError("Keep third-party source and weights outside the Shadow repository")
    root.mkdir(parents=True, exist_ok=True)
    source = root/"source"
    git = ["git"]+(["-c", "http.proxy="+args.proxy] if args.proxy else [])
    if not source.exists():
        subprocess.run(git+["clone", "--no-checkout", "--filter=blob:none",
            "https://github.com/IMU-Group/FSD.git", str(source)], check=True,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
        subprocess.run(git+["-C", str(source), "-c", "filter.lfs.smudge=", "-c", "filter.lfs.required=false",
            "checkout", FSD_COMMIT], check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    actual = subprocess.check_output(git+["-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != FSD_COMMIT:
        raise ValueError("Existing FSD source is not the pinned version")
    checkpoint = root/"FSD_best.ckpt"
    if not checkpoint.exists():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": args.proxy}) if args.proxy
                                             else urllib.request.ProxyHandler())
        temporary = checkpoint.with_suffix(".partial")
        with opener.open(FSD_URL, timeout=120) as response, temporary.open("wb") as handle:
            for block in iter(lambda: response.read(4*1024*1024), b""):
                handle.write(block)
        if checksum(temporary) != FSD_SHA256:
            raise ValueError("Downloaded FSD checkpoint differs from the official SHA256")
        temporary.replace(checkpoint)
    if checksum(checkpoint) != FSD_SHA256:
        raise ValueError("Existing FSD checkpoint differs from the official SHA256")
    record = {"repository": "https://github.com/IMU-Group/FSD", "commit": actual,
        "checkpoint": str(checkpoint), "checkpoint_sha256": FSD_SHA256,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "target_domain_shadow_labels_used": False, "external_supervised_shadow_prior": True}
    (root/"assets.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
