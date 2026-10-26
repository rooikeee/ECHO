#!/usr/bin/env python3
"""Restore pinned upstream archives and replay the versioned Ascend patches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd=None, env=None):
    subprocess.run(command, cwd=cwd, env=env, check=True)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def archive_for(entry, cache):
    path = cache / entry["archive"]
    if not path.exists():
        url = entry["repository"].replace("https://github.com/", "https://codeload.github.com/")
        url += "/tar.gz/" + entry["commit"]
        temporary = path.with_suffix(path.suffix + ".partial")
        print(f"Downloading {entry['repository']} at {entry['commit']}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if digest(temporary) != entry["archive_sha256"]:
            raise RuntimeError(f"Archive SHA256 mismatch: {temporary}")
        temporary.replace(path)
    if digest(path) != entry["archive_sha256"]:
        raise RuntimeError(f"Archive SHA256 mismatch: {path}")
    return path


def extract(archive, destination):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing source tree: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".source-") as temporary:
        with tarfile.open(archive, "r:gz") as source:
            source.extractall(temporary, filter="data")
        entries = list(Path(temporary).iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise RuntimeError(f"Expected one source root in {archive}")
        entries[0].rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT,
                        help="destination for sources; existing trees are never overwritten")
    parser.add_argument("--use-checked-in-ascend", action="store_true",
                        help="keep the versioned vllm-ascend tree; prepare only vLLM and dependencies")
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "build/reference")
    parser.add_argument("--skip-catlass", action="store_true", help="source/patch verification only; cannot build kernels")
    parser.add_argument("--references", action="store_true",
                        help="also extract pristine original/PR trees used by reference builds")
    args = parser.parse_args()
    lock = json.loads((ROOT / "vllm-ascend.lock.json").read_text())
    destination = args.destination.resolve()
    cache = args.archive_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if args.use_checked_in_ascend and not (destination / "vllm-ascend/CMakeLists.txt").is_file():
        raise FileNotFoundError("--use-checked-in-ascend requires the versioned vllm-ascend source")
    for name in (("vllm",) if args.use_checked_in_ascend else ("vllm", "vllm-ascend")):
        if (destination / name).exists():
            raise FileExistsError(destination / name)
    if not args.skip_catlass and (destination / "vllm-ascend/csrc/third_party/catlass").exists():
        raise FileExistsError("CATLASS already exists; refusing to overwrite it")
    archives = {name: archive_for(lock[name], cache) for name in ("vllm", "vllm_ascend")}
    extract(archives["vllm"], destination / "vllm")
    if not args.use_checked_in_ascend:
        extract(archives["vllm_ascend"], destination / "vllm-ascend")
    patch_env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(destination.parent))
    for patch in lock["patch_order"]:
        # The checked-in Ascend source already contains all Ascend-side changes.
        include = ["--include=vllm/*"] if args.use_checked_in_ascend else []
        run(["git", "apply", "--check", *include, str(ROOT / patch)], cwd=destination, env=patch_env)
        run(["git", "apply", *include, str(ROOT / patch)], cwd=destination, env=patch_env)
    if not args.skip_catlass:
        dependency = lock["catlass"]
        target = destination / "vllm-ascend/csrc/third_party/catlass"
        target.mkdir(parents=True, exist_ok=True)
        run(["git", "init", str(target)])
        run(["git", "remote", "add", "origin", dependency["repository"]], cwd=target)
        run(["git", "fetch", "--depth=1", "origin", dependency["commit"]], cwd=target)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=target)
    if args.references:
        base = destination / "build/lim-copysfa-optimization/versions"
        extract(archives["vllm_ascend"], base / "pr")
        extract(archive_for(lock["original_vllm_ascend"], cache), base / "original")
    (destination / "build").mkdir(exist_ok=True)
    receipt = {"status": "sources_prepared_not_built", "destination": str(destination),
               "lock_sha256": digest(ROOT / "vllm-ascend.lock.json"),
               "patch_sha256": {p: digest(ROOT / p) for p in lock["patch_order"]},
               "catlass_prepared": not args.skip_catlass, "references": args.references,
               "checked_in_ascend": args.use_checked_in_ascend}
    (destination / "build/vllm-source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
