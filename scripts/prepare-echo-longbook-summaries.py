"""Prepare distinct English book summaries for the synthetic-KV serving test."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="longbook_sum_eng.jsonl")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    rows = []
    with args.data.open() as source:
        for line in source:
            entry = json.loads(line)
            ids = tokenizer.encode(entry["context"] + "\n" + entry["input"])
            original_length = len(ids)
            if len(ids) > 64000:
                ids = ids[:32000] + ids[-32000:]
            rows.append(
                {
                    "task": "longbook_sum_eng",
                    "source_id": entry["id"],
                    "prompt_token_ids": ids,
                    "prompt_tokens": len(ids),
                    "original_prompt_tokens": original_length,
                    "output_tokens": 256,
                }
            )
            if len(rows) == args.count:
                break
    if len(rows) != args.count or len({r["source_id"] for r in rows}) != args.count:
        raise ValueError("Insufficient distinct source samples")
    rows.sort(key=lambda r: r["prompt_tokens"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "samples": len(rows),
                "sha256": digest,
                "truncation": "head 32000 + tail 32000",
                "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
