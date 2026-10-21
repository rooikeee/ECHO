#!/usr/bin/env python3
"""Compare KV management Graph replay before/after optimization on one NPU.

This excludes indexer, sparse attention and model forward: it is a component
benchmark, not an end-to-end throughput claim.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.echo_recall import EchoRecallCache


def load_before(path):
    spec = importlib.util.spec_from_file_location("echo_recall_before_optimization", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.EchoRecallCache


def measure(cache_type, batch, case, args):
    cache = cache_type(host_size=8320, capacity=4096)
    graph = None
    try:
        torch.manual_seed(42)
        context = torch.arange(128, 128 + args.history_tokens, dtype=torch.int32, device="npu")
        keys = torch.randn(args.history_tokens, 1, 512, dtype=torch.bfloat16, device="npu")
        ropes = torch.randn(args.history_tokens, 1, 64, dtype=torch.bfloat16, device="npu")
        cache.write(context, keys, ropes)
        torch.npu.synchronize()
        cache.check_errors()
        mapped = cache.host_to_device[context.long()]
        if not torch.all((mapped >= 0) & (mapped < cache.capacity)).item():
            raise AssertionError(f"Context mapping out of bounds: {mapped.cpu().tolist()}")
        mapped_cpu = mapped.cpu().long()
        if mapped_cpu.unique().numel() != args.history_tokens:
            print({"miss_count": cache.miss_count.item(),
                   "hosts_head": cache.recall_hosts[:32].cpu().tolist(),
                   "hosts_tail": cache.recall_hosts[args.history_tokens - 32:args.history_tokens].cpu().tolist(),
                   "slots_head": cache.recall_slots[:32].cpu().tolist(),
                   "slots_tail": cache.recall_slots[args.history_tokens - 32:args.history_tokens].cpu().tolist()}, flush=True)
            raise AssertionError(f"Duplicate context slots: unique={mapped_cpu.unique().numel()}, "
                                 f"head={mapped_cpu[:20].tolist()}, tail={mapped_cpu[-20:].tolist()}")
        torch.testing.assert_close(cache.device_to_host[mapped.long()].cpu(), context.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(cache.key[mapped.long()].cpu(), keys.cpu(), rtol=0, atol=0)
        loc = torch.arange(128 + args.history_tokens, 128 + args.history_tokens + batch,
                           dtype=torch.int32, device="npu")
        new_keys = torch.randn(batch, 1, 512, dtype=torch.bfloat16, device="npu")
        new_ropes = torch.randn(batch, 1, 64, dtype=torch.bfloat16, device="npu")
        ids = torch.full((batch, 2048), -1, dtype=torch.int32, device="npu")
        ids[:, :args.history_tokens].copy_(context.expand(batch, -1))

        def run():
            if case != "recall_all_hit":
                loc.add_(batch)
                cache.write(loc, new_keys, new_ropes)
            if case != "write_only":
                if case == "write_plus_recall":
                    ids[:, args.history_tokens].copy_(loc)
                return cache.recall(ids)
            return None

        run()
        torch.npu.synchronize()
        cache.check_errors()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()
        for _ in range(5):
            graph.replay()
        torch.npu.synchronize()
        cache.check_errors()

        samples = []
        wall_samples = []
        for _ in range(args.trials):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            wall = time.perf_counter()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            wall_samples.append((time.perf_counter() - wall) * 1000 / args.iterations)
            samples.append(start.elapsed_time(end) / args.iterations)
            cache.check_errors()

        if output is not None:
            torch.testing.assert_close(
                cache.key[output[0, :args.history_tokens].long()].cpu(),
                keys.cpu(), rtol=0, atol=0,
            )
            if case == "write_plus_recall":
                torch.testing.assert_close(
                    cache.key[output[:, args.history_tokens].long()].cpu(),
                    new_keys.cpu(), rtol=0, atol=0,
                )
            if cache.miss_count.item() != 0:
                raise AssertionError("Expected an all-hit recall workload")
        if case != "recall_all_hit":
            torch.npu.synchronize()
            selected = loc.cpu().long()
            torch.testing.assert_close(cache.host.tensor[selected, :512], new_keys[:, 0].cpu(),
                                       rtol=0, atol=0)
        return {
            "device_ms_per_replay_median": statistics.median(samples),
            "device_ms_per_replay_samples": samples,
            "wall_ms_per_replay_median": statistics.median(wall_samples),
            "graph_replay_and_content_checks_passed": True,
        }
    finally:
        torch.npu.synchronize()
        del graph
        cache.close()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-path", type=Path,
                        default=root / "build/echo-recall-optimization/echo_recall_before.py")
    parser.add_argument("--output", type=Path,
                        default=root / "build/echo-recall-optimization/planner-metrics.json")
    parser.add_argument("--history-tokens", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--case", choices=("write_only", "recall_all_hit", "write_plus_recall"))
    parser.add_argument("--version", choices=("before", "after"))
    args = parser.parse_args()
    if not 1 <= args.history_tokens < 2048 or args.iterations < 1 or args.trials < 1:
        parser.error("Invalid history length or repeat count")
    if 128 + args.history_tokens + 2 * (10 + args.iterations * args.trials) >= 8320:
        parser.error("Repeat count would exceed logical ID space")
    torch.npu.set_device(0)
    before = load_before(args.before_path)
    results = {
        "scope": "KV management only; not model or sparse attention throughput",
        "before": "device_first_full_scan_sort", "after": "epoch_hit_fastpath_bounded_lru",
        "host_size": 8320, "cache_capacity": 4096, "history_tokens": args.history_tokens,
        "iterations": args.iterations, "trials": args.trials, "measurements": [],
    }
    for batch in (1, 2):
        for case in ([args.case] if args.case else ("write_only", "recall_all_hit", "write_plus_recall")):
            row = {"batch": batch, "case": case}
            for name, cache_type in (("before", before), ("after", EchoRecallCache)):
                if args.version and name != args.version:
                    continue
                print(f"Measuring {name}: batch={batch}, case={case}", flush=True)
                row[name] = measure(cache_type, batch, case, args)
            if "before" in row and "after" in row:
                row["device_time_speedup"] = (
                    row["before"]["device_ms_per_replay_median"]
                    / row["after"]["device_ms_per_replay_median"]
                )
            results["measurements"].append(row)
            print(json.dumps(row, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
