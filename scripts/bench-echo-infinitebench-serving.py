"""Synthetic-KV HTTP serving benchmark; reports actual tokens and queue occupancy."""

import argparse
import asyncio
import json
import pathlib
import random
import re
import statistics
import time

import aiohttp


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=pathlib.Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--rate", type=float, default=float("inf"))
    p.add_argument("--port", type=int, default=8310)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(l) for l in args.data.open()][: args.limit]
    rng = random.Random(1024)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=14400)) as session:

        async def request(i, row):
            start = time.perf_counter()
            first = None
            last = start
            times = []
            tokens = 0
            error = None
            body = {
                "model": "glm52",
                "prompt": row["prompt_token_ids"],
                "max_tokens": row["output_tokens"],
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "seed": 1024,
            }
            try:
                async with session.post(f"http://127.0.0.1:{args.port}/v1/completions", json=body) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}: " + (await resp.text())[:1000])
                    async for raw in resp.content:
                        line = raw.decode().strip()
                        if not line.startswith("data: "):
                            continue
                        if line == "data: [DONE]":
                            break
                        data = json.loads(line[6:])
                        now = time.perf_counter()
                        if data.get("error"):
                            raise RuntimeError(str(data["error"]))
                        for choice in data.get("choices", []):
                            if choice.get("text"):
                                if first is None:
                                    first = now
                                else:
                                    times.append(now - last)
                                last = now
                        if data.get("usage"):
                            tokens = data["usage"]["completion_tokens"]
            except Exception as exc:
                error = repr(exc)
            end = time.perf_counter()
            result = {
                "index": i,
                "source_id": row["source_id"],
                "input_tokens": row["prompt_tokens"],
                "output_tokens": tokens,
                "success": error is None and tokens == row["output_tokens"],
                "error": error,
                "latency_s": end - start,
                "ttft_s": None if first is None else first - start,
                "tpot_s": None if first is None or tokens < 2 else (last - first) / (tokens - 1),
                "stream_intervals_s": times,
            }
            print(json.dumps(result), flush=True)
            return result

        start = time.perf_counter()
        tasks = []
        samples = []

        async def monitor():
            while True:
                try:
                    async with session.get(
                        f"http://127.0.0.1:{args.port}/metrics", timeout=aiohttp.ClientTimeout(total=3)
                    ) as response:
                        text = await response.text()
                    sample = {"elapsed_s": time.perf_counter() - start}
                    for key in ["num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"]:
                        vals = re.findall(r"^vllm:" + key + r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)$", text, re.M)
                        if vals:
                            sample[key] = sum(float(v) for v in vals)
                    samples.append(sample)
                except Exception:
                    pass
                await asyncio.sleep(1)

        monitor_task = asyncio.create_task(monitor())
        for i, row in enumerate(rows):
            if i and args.rate != float("inf"):
                await asyncio.sleep(rng.expovariate(args.rate))
            tasks.append(asyncio.create_task(request(i, row)))
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
        monitor_task.cancel()
        await asyncio.gather(monitor_task, return_exceptions=True)
    good = [r for r in results if r["success"]]
    out = {
        "variant": args.variant,
        "synthetic_kv": True,
        "accuracy_validated": False,
        "concurrency_limit": "See saved launch configuration and runtime memory plan",
        "input_context_tokens": 64000,
        "mtp": 3,
        "timing_scope": "complete request, including speculative rounds; not target-phase kernel time",
        "requests": len(rows),
        "successful": len(good),
        "duration_s": duration,
        "request_rate": str(args.rate),
        "output_tokens_per_s": sum(r["output_tokens"] for r in good) / duration,
        "requests_per_s": len(good) / duration,
        "requests_detail": results,
        "scheduler_samples": samples,
    }
    for metric in ["ttft_s", "tpot_s", "latency_s"]:
        vals = [r[metric] for r in good if r[metric] is not None]
        if vals:
            out["mean_" + metric] = statistics.mean(vals)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("requests_detail", "scheduler_samples")}), flush=True)
    if len(good) != len(rows):
        raise SystemExit(1)


asyncio.run(main())
