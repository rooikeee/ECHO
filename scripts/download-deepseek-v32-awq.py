#!/usr/bin/env python3
"""Resume the pinned AWQ snapshot download; the SDK checks each file's SHA256."""

import fcntl
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "build/models/DeepSeek-V3.2-AWQ"
MANIFEST = ROOT / "build/models/deepseek-v32-awq-modelscope-manifest.json"
STATUS = ROOT / "build/models/deepseek-v32-awq-download-status.json"
REPO = "QuantTrio/DeepSeek-V3.2-AWQ"
REVISION = "3a6e8af042a84d4dfd5b09dc6dbdfcd4569beeef"


def write_status(state, **details):
    report = {
        "status": state,
        "repo": REPO,
        "source": "modelscope.cn",
        "revision": REVISION,
        "model_dir": str(MODEL_DIR),
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **details,
    }
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(STATUS)
    print(json.dumps(report), flush=True)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR.parent / "deepseek-v32-awq-download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not MANIFEST.exists():
            query = urllib.parse.urlencode({"Revision": REVISION, "Recursive": "true"})
            url = f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?{query}"
            with urllib.request.urlopen(url, timeout=120) as response:
                manifest = json.load(response)
            if manifest.get("Code") != 200 or not manifest.get("Data", {}).get("Files"):
                raise RuntimeError("ModelScope did not return a nonempty pinned file manifest")
            temporary = MANIFEST.with_suffix(".partial")
            temporary.write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
            temporary.replace(MANIFEST)
        files = [
            item for item in json.loads(MANIFEST.read_text())["Data"]["Files"]
            if item["Type"] == "blob" and "/" not in item["Path"]
            and item["Path"] != ".gitattributes"
        ]
        total = sum(item["Size"] for item in files)
        remaining = sum(
            max(0, item["Size"] - (MODEL_DIR / item["Path"]).stat().st_size)
            if (MODEL_DIR / item["Path"]).exists() else item["Size"]
            for item in files
        )
        if shutil.disk_usage(MODEL_DIR).free < remaining + 20 * 1024**3:
            raise RuntimeError("Insufficient free space for weights and download buffers")
        os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / "build/npu-cache/modelscope"))
        os.environ.setdefault("MODELSCOPE_HOME", str(ROOT / "build/npu-cache/modelscope-home"))
        os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "4")
        from modelscope_hub.compat import snapshot_download

        write_status("downloading", expected_files=len(files), expected_bytes=total)
        try:
            snapshot_download(
                REPO,
                revision=REVISION,
                local_dir=str(MODEL_DIR),
                allow_file_pattern=[item["Path"] for item in files],
                max_workers=4,
            )
            for item in files:
                path = MODEL_DIR / item["Path"]
                if not path.is_file() or path.stat().st_size != item["Size"]:
                    raise RuntimeError(f"Incomplete file: {path}")
            config = json.loads((MODEL_DIR / "config.json").read_text())
            quant = config["quantization_config"]
            if quant["quant_method"] != "awq" or quant["bits"] != 4:
                raise RuntimeError("Downloaded configuration is not 4-bit AWQ")
            index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
            shards = set(index["weight_map"].values())
            if len(shards) != 121:
                raise RuntimeError(f"Unexpected shard count: {len(shards)}")
            for shard in shards:
                if not (MODEL_DIR / shard).is_file():
                    raise RuntimeError(f"Missing indexed shard: {shard}")
            write_status(
                "complete", files=len(files), bytes=total, weight_shards=len(shards),
                quantization=quant, sha256_validation="passed by ModelScope SDK",
                inference_validation="not yet run",
            )
        except Exception as exc:
            write_status("failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
