#!/usr/bin/env python3
"""Profile LIM alone on the controlled MTP3 request-union workload."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch_npu

from echo_indexer_prefetch import EchoMTPIndexerPrefetch
from echo_profile_workload import controlled_resident_sources


def load_helpers(root):
    path = root / "scripts/collect-vllm-sparse-profile-npu.py"
    spec = importlib.util.spec_from_file_location("lim_profile_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize(path):
    rows = [row for filename in path.rglob("kernel_details.csv")
            for row in csv.DictReader(filename.open(newline=""))
            if row["Name"] == "FusedLightningIndexerManage"]
    if not rows:
        raise RuntimeError(f"No LIM tasks in {path}")
    metrics = {}
    for name in rows[0]:
        if name != "Duration(us)" and not name.startswith(("aic", "aiv", "cube_")):
            continue
        values = [float(row[name]) for row in rows if row[name] not in ("", "N/A")]
        if values:
            metrics[name] = {"mean": statistics.mean(values), "min": min(values),
                             "max": max(values), "stdev": statistics.pstdev(values)}
    return {"calls": len(rows), "metrics": metrics, "raw_rows": rows}


def profile(graph, reset, path, metric, repeats):
    experimental = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=getattr(torch_npu.profiler.AiCMetrics, metric))
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=experimental,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(path)),
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=repeats, repeat=1),
    ) as prof:
        for _ in range(repeats + 1):
            reset()
            graph.replay()
            torch.npu.synchronize()
            prof.step()
    return summarize(path)


def check_outputs(cache, outputs, expected_cache, expected_outputs):
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    for i in (0, 1, 2, 5):
        torch.testing.assert_close(outputs[i], expected_outputs[i], rtol=0, atol=0)
    for b, count in enumerate(expected_outputs[5].cpu().tolist()):
        for i in (3, 4):
            torch.testing.assert_close(outputs[i][b, :count], expected_outputs[i][b, :count],
                                       rtol=0, atol=0)


@torch.inference_mode()
def run(args, root, manifest):
    helper = load_helpers(root)
    upstream = helper.load_upstream(root)
    torch.npu.set_device(0)
    torch.manual_seed(20261023 + args.batch)
    batch, q, context = args.batch, 4, args.context
    blocks = context // 128
    case = upstream._build_case(q_values=[q] * batch, states=[-1] * batch,
                                offload_len=context - 128, cache_tokens=args.hot_tokens, heads=args.indexer_heads)
    case.index_key_cache = torch.randn(batch * blocks, 128, 1, 128,
                                       dtype=torch.bfloat16, device="npu")
    case.index_key_dequant_scale = torch.zeros(batch * blocks, 128, 1, device="npu")
    offsets = torch.arange(batch, dtype=torch.int32, device="npu").view(-1, 1) * blocks
    case.index_block_table.add_(offsets)
    case.route_block_table.copy_(case.index_block_table.repeat_interleave(q, dim=0))
    cache = case.cache_seed.clone()
    outputs = upstream._make_outputs(case)
    upstream._call_op(case, cache, outputs)
    mapping = cache.cpu()
    selected = []
    for b, row in enumerate(case.req_entries):
        ids = torch.nonzero(mapping[row] >= 0).flatten()
        inverse = torch.full((args.hot_tokens,), -1, dtype=torch.int64)
        inverse[mapping[row, ids].long()] = ids
        selected.append(inverse[outputs[1][b * q:(b + 1) * q, 0].cpu().long()])
    resident = controlled_resident_sources(
        torch.stack(selected), context - 128, args.hot_tokens, args.misses_per_request,
        20261025 + batch, request_union=True)
    seed = torch.full_like(case.cache_seed.cpu(), -1)
    for b, row in enumerate(case.req_entries):
        seed[row, resident[b]] = torch.arange(args.hot_tokens, dtype=torch.int32)
    case.cache_seed.copy_(seed.to(case.query.device))

    def reset():
        cache.copy_(case.cache_seed)

    def baseline():
        upstream._call_op(case, cache, outputs)

    reset()
    baseline()
    expected_cache = cache.clone()
    expected_outputs = tuple(t.clone() for t in outputs)
    assert outputs[5].cpu().tolist() == [args.misses_per_request] * batch
    if args.write_reference:
        if args.write_reference.exists():
            raise FileExistsError(args.write_reference)
        torch.save({"cache_seed": case.cache_seed.cpu(), "query": case.query.cpu(),
                    "cache": expected_cache.cpu(), "outputs": [t.cpu() for t in expected_outputs]},
                   args.write_reference)
        manifest["reference_written"] = str(args.write_reference)
    if args.reference:
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
        torch.testing.assert_close(case.cache_seed.cpu(), reference["cache_seed"], rtol=0, atol=0)
        torch.testing.assert_close(case.query.cpu(), reference["query"], rtol=0, atol=0)
        check_outputs(cache.cpu(), [t.cpu() for t in outputs], reference["cache"], reference["outputs"])
        expected_cache = reference["cache"].to(cache.device)
        expected_outputs = tuple(t.to(cache.device) for t in reference["outputs"])
        manifest["cross_version_reference"] = str(args.reference)
    manifest["misses_per_query"] = outputs[2].cpu().tolist()
    manifest["union_misses_per_request"] = outputs[5].cpu().tolist()
    host_kv = host_rope = prefetch = graph = None
    try:
        if any(mode != "baseline" for mode in args.modes):
            host_kv = helper.HostBuffer((batch * blocks, 128, 512))
            host_rope = helper.HostBuffer((batch * blocks, 128, 64))
            table = torch.arange(batch * blocks, dtype=torch.int32, device="npu").view(batch, blocks)
            prefetch = EchoMTPIndexerPrefetch(case, host_kv, host_rope, table)
        for mode in args.modes:
            call = baseline if mode == "baseline" else lambda: prefetch.indexer(cache, outputs)
            if prefetch is not None:
                prefetch.budgets.fill_(0 if mode == "context-budget-0" else 64)
                prefetch.thresholds.fill_(float("inf"))
            reset()
            call()
            reset()
            graph = helper.capture(call)
            watched = (cache, *outputs, case.query, case.actual_seq_lengths_key,
                       *((*prefetch.buffers, prefetch.context) if prefetch is not None else ()))
            pointers = [t.data_ptr() for t in watched]
            for _ in range(3):
                reset()
                graph.replay()
                torch.npu.synchronize()
                check_outputs(cache, outputs, expected_cache, expected_outputs)
                assert pointers == [t.data_ptr() for t in watched]
            record = {"mode": mode, "correctness": "three_exact_graph_replays_fixed_addresses",
                      **helper.measure(graph, reset, args.repeats), "profiles": {}}
            if mode != "baseline":
                record["prefetched_per_query"] = prefetch.counts.cpu().tolist()
            for metric in args.metrics:
                record["profiles"][metric] = profile(
                    graph, reset, args.output / mode / metric, metric, args.profile_repeats)
            if args.external_profiler:
                reset()
                torch.npu.synchronize()
                marker = torch_npu.npu.mstx.range_start("lim_profile")
                if args.external_profiler == "graph":
                    graph.replay()
                else:
                    call()
                torch.npu.synchronize()
                torch_npu.npu.mstx.range_end(marker)
                check_outputs(cache, outputs, expected_cache, expected_outputs)
                record["external_profiler_execution"] = args.external_profiler
            # Exercise persistent cache state on the captured graph as well.
            reset()
            graph.replay()
            torch.npu.synchronize()
            persistent_seed = cache.clone()
            baseline()
            persistent_cache = cache.clone()
            persistent_outputs = tuple(t.clone() for t in outputs)
            cache.copy_(persistent_seed)
            graph.replay()
            torch.npu.synchronize()
            check_outputs(cache, outputs, persistent_cache, persistent_outputs)
            assert not bool(outputs[5].any())
            record["persistent_all_hit_replay"] = "passed"
            if args.profile_all_hit and mode == "baseline":
                def reset_all_hit():
                    cache.copy_(persistent_seed)

                record["all_hit_control"] = {
                    "note": "same scoring tensors; cache after one miss400 pass; includes all-hit metadata paths",
                    **helper.measure(graph, reset_all_hit, args.repeats),
                    "profile": profile(graph, reset_all_hit, args.output / "baseline-all-hit" / "PipeUtilization",
                                       "PipeUtilization", args.profile_repeats),
                }
                check_outputs(cache, outputs, persistent_cache, persistent_outputs)
            graph = None
            manifest["results"].append(record)
            (args.output / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({k: v for k, v in record.items()
                              if k not in ("profiles", "all_hit_control")}), flush=True)
    finally:
        graph = None
        torch.npu.synchronize()
        for host in (host_kv, host_rope):
            if host is not None:
                host.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--context", type=int, default=40960)
    parser.add_argument("--hot-tokens", type=int, default=8192)
    parser.add_argument("--indexer-heads", type=int, choices=[32, 64], default=64)
    parser.add_argument("--misses-per-request", type=int, default=400)
    parser.add_argument("--modes", nargs="+", default=["baseline", "context-budget-0", "prefetch-64-history"],
                        choices=["baseline", "context-budget-0", "prefetch-64-history"])
    parser.add_argument("--metrics", nargs="*", default=["PipeUtilization", "ResourceConflictRatio", "Memory"],
                        choices=["PipeUtilization", "ResourceConflictRatio", "Memory", "ArithmeticUtilization", "MemoryUB"])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--profile-all-hit", action="store_true")
    parser.add_argument("--write-reference", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--external-profiler", choices=["graph", "eager"],
                        help="mark one invocation for msprof op; eager is diagnostic only")
    args = parser.parse_args()
    if args.batch < 1 or args.repeats < 1 or args.profile_repeats < 1:
        parser.error("batch and repeat counts must be positive")
    if (args.context % 128 or args.hot_tokens % 256 or
            not 2048 <= args.hot_tokens <= 16128 or args.context - 128 <= args.hot_tokens):
        parser.error("invalid context/cache alignment or capacity")
    if not 0 <= args.misses_per_request <= 8192:
        parser.error("misses-per-request must be in [0,8192]")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "scripts/echo_profile_workload.py",
               root / "scripts/echo_indexer_prefetch.py",
               root / "patches/vllm-ascend-echo-indexer-prefetch.patch"]
    manifest = {"status": "running", "coverage": "isolated_LIM_graph_not_model_or_attention",
                "batch": args.batch, "mtp": 3, "queries_per_request": 4,
                "context": args.context, "hot_tokens": args.hot_tokens,
                "indexer_heads": args.indexer_heads,
                "controlled_union_misses_per_request": args.misses_per_request,
                "input_note": "same synthetic scoring seed and controlled residency as union400 chain profile",
                "timing_note": "cache reset outside graph; prefetch ready/count reset inside graph; PMU values overlap",
                "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "command": sys.argv, "metrics": args.metrics, "repeats": args.repeats,
                "profile_repeats": args.profile_repeats,
                "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "results": []}
    try:
        run(args, root, manifest)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
