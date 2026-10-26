#!/usr/bin/env python3
"""Download the pinned full GLM-5.2 ModelSlim W4A8 checkpoint."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request

REPO = "Eco-Tech/GLM-5.2-w4a8"
REVISION = "29e0e32383ce2a95f6cbac7322f3b830a70301eb"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    directory = ROOT / "build/models/GLM-5.2-w4a8"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory.parent / "glm52-w4a8-manifest.json"
    status_path = directory.parent / "glm52-w4a8-download-status.json"

    def status(state, **kwargs):
        data = {"status": state, "repo": REPO, "revision": REVISION, "pid": os.getpid(),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model_dir": str(directory), **kwargs}
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(status_path)
        print(json.dumps(data), flush=True)

    with (directory.parent / "glm52-w4a8-download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not manifest_path.exists():
            query = urllib.parse.urlencode({"Revision": REVISION, "Recursive": "true"})
            with urllib.request.urlopen(f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?{query}",
                                        timeout=120) as response:
                manifest = json.load(response)
            if manifest.get("Code") != 200 or not manifest.get("Data", {}).get("Files"):
                raise RuntimeError("Pinned model manifest unavailable")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        files = [entry for entry in json.loads(manifest_path.read_text())["Data"]["Files"]
                 if entry["Type"] == "blob" and "/" not in entry["Path"] and entry["Path"] != ".gitattributes"]
        chosen = [entry for entry in files if not args.metadata_only or not entry["Path"].endswith(".safetensors")]
        total = sum(entry["Size"] for entry in files)
        remaining = sum(entry["Size"] for entry in chosen
                        if not (directory / entry["Path"]).is_file() or
                        (directory / entry["Path"]).stat().st_size != entry["Size"])
        if shutil.disk_usage(directory).free < remaining + 20 * 1024**3:
            raise RuntimeError("Insufficient space for checkpoint and download buffers")
        os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / "build/npu-cache/modelscope"))
        os.environ.setdefault("MODELSCOPE_HOME", str(ROOT / "build/npu-cache/modelscope-home"))
        from modelscope_hub.compat import snapshot_download
        status("downloading_metadata" if args.metadata_only else "downloading", expected_bytes=total,
               selected_files=len(chosen), selected_bytes=sum(entry["Size"] for entry in chosen))
        try:
            snapshot_download(REPO, revision=REVISION, local_dir=str(directory),
                              allow_file_pattern=[entry["Path"] for entry in chosen], max_workers=args.workers)
            for entry in chosen:
                path = directory / entry["Path"]
                if not path.is_file() or path.stat().st_size != entry["Size"]:
                    raise RuntimeError(f"Incomplete download: {path}")
            config = json.loads((directory / "config.json").read_text())
            if config.get("model_type") != "glm_moe_dsa" or config.get("num_hidden_layers") != 78:
                raise RuntimeError("Checkpoint is not the requested full GLM-5.2 architecture")
            index = json.loads((directory / "quant_model_weights.safetensors.index.json").read_text())
            shards = set(index["weight_map"].values())
            if not args.metadata_only:
                for shard in shards:
                    if not (directory / shard).is_file():
                        raise RuntimeError(f"Missing shard: {shard}")
            status("metadata_ready" if args.metadata_only else "complete", expected_bytes=total,
                   weight_shards=len(shards), model_layers=config["num_hidden_layers"],
                   indexer_heads=config["index_n_heads"], attention_heads=config["num_attention_heads"],
                   sha256_validation="ModelScope SDK", inference_validation="not_run")
        except Exception as exc:
            status("failed", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
