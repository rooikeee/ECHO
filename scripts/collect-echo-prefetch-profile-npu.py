#!/usr/bin/env python3
"""Profile synthetic LIM -> bounded prefetch consumer -> CopySFA graphs."""

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch
import torch_npu

from echo_indexer_prefetch import EchoIndexerPrefetch, EchoMTPIndexerPrefetch
from echo_profile_workload import controlled_resident_sources


def load_profile():
    path = Path(__file__).with_name("collect-vllm-sparse-profile-npu.py")
    spec = importlib.util.spec_from_file_location("echo_chain_profile_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize_chain(path):
    """Pair LIM and CopySFA tasks; exclude graph-external cache reset kernels."""
    pairs = []
    for filename in path.rglob("kernel_details.csv"):
        with filename.open(newline="") as source:
            tasks = list(csv.DictReader(source))
        tasks.sort(key=lambda row: float(row["Start Time(us)"]))
        pending = None
        between = []
        for task in tasks:
            name = (task.get("Name", "") + task.get("Type", "")).lower()
            if "fusedlightningindexermanage" in name:
                pending, between = task, []
            elif "fusedscattercopysparseflashattention" in name and pending is not None:
                end = float(task["Start Time(us)"]) + float(task["Duration(us)"])
                pairs.append({
                    "lim_us": float(pending["Duration(us)"]),
                    "consumer_kernel_sum_us": sum(float(row["Duration(us)"]) for row in between),
                    "consumer_kernel_calls": len(between),
                    "copy_sfa_us": float(task["Duration(us)"]),
                    "lim_to_sfa_span_us": end - float(pending["Start Time(us)"]),
                })
                pending, between = None, []
            elif pending is not None:
                between.append(task)
    if not pairs:
        raise RuntimeError(f"No complete LIM -> CopySFA pairs in {path}")
    means = {key: statistics.mean(row[key] for row in pairs) for key in pairs[0]}
    means["profile_chain_calls"] = len(pairs)
    means["target_kernel_sum_us"] = (
        means["lim_us"] + means["consumer_kernel_sum_us"] + means["copy_sfa_us"]
    )
    return means


@torch.inference_mode()
def run_batch(args, helper, upstream, batch):
    directory = args.output / f"batch-{batch}"
    directory.mkdir()
    # Match the independent-request Indexer inputs of the original experiment.
    torch.manual_seed(20261023 + batch)
    offload = args.context - 128
    q = 1 + args.mtp
    total_queries = batch * q
    case = upstream._build_case(q_values=[q] * batch, states=[-1] * batch,
                                offload_len=offload, cache_tokens=args.hot_tokens, heads=args.indexer_heads)
    blocks = args.context // 128
    case.index_key_cache = torch.randn(batch * blocks, 128, 1, 128,
                                       dtype=torch.bfloat16, device="npu")
    case.index_key_dequant_scale = torch.zeros(batch * blocks, 128, 1, device="npu")
    offsets = torch.arange(batch, dtype=torch.int32, device="npu").view(-1, 1) * blocks
    case.index_block_table.add_(offsets)
    case.route_block_table.copy_(case.index_block_table.repeat_interleave(q, dim=0))
    controlled_sources = None
    if args.misses_per_query is not None or args.misses_per_request is not None:
        # An untimed exact LIM pass identifies the selected sources. Only cache
        # residency is changed; scoring tensors and the timed operators are not.
        scratch_cache = case.cache_seed.clone()
        scratch_outputs = upstream._make_outputs(case)
        upstream._call_op(case, scratch_cache, scratch_outputs)
        scratch_mapping = scratch_cache.cpu()
        selected = []
        for b, row in enumerate(case.req_entries):
            mapping = scratch_mapping[row]
            ids = torch.nonzero(mapping >= 0).flatten()
            inverse = torch.full((args.hot_tokens,), -1, dtype=torch.int64)
            inverse[mapping[ids].long()] = ids
            slots = scratch_outputs[1][b * q:(b + 1) * q, 0].cpu().long()
            selected.append(inverse[slots])
        controlled_sources = controlled_resident_sources(
            torch.stack(selected), offload, args.hot_tokens,
            args.misses_per_request if args.misses_per_request is not None else args.misses_per_query,
            20261025 + batch, request_union=args.misses_per_request is not None)
        controlled_cache = torch.full_like(case.cache_seed.cpu(), -1)
        for b, row in enumerate(case.req_entries):
            controlled_cache[row, controlled_sources[b]] = torch.arange(args.hot_tokens, dtype=torch.int32)
        case.cache_seed.copy_(controlled_cache.to(case.query.device))
        del scratch_cache, scratch_outputs
    cache = case.cache_seed.clone()
    outputs = upstream._make_outputs(case)
    dram_table = torch.arange(batch * blocks, dtype=torch.int32, device="npu").view(batch, blocks)
    host_kv = helper.HostBuffer((batch * blocks, 128, 512))
    host_rope = helper.HostBuffer((batch * blocks, 128, 64))
    graph = None
    try:
        owner = EchoMTPIndexerPrefetch if args.mtp else EchoIndexerPrefetch
        prefetch = owner(case, host_kv, host_rope, dram_table, args.capacity)
        consumed = prefetch.allocate_outputs(outputs)
        hbm_blocks = args.hot_tokens // 128 + 1
        hbm_table = torch.arange(batch * hbm_blocks, dtype=torch.int32, device="npu").view(batch, -1)
        kv_seed = torch.randn(batch * hbm_blocks, 128, 1, 512,
                              dtype=torch.bfloat16, device="npu") * .1
        rope_seed = torch.randn(batch * hbm_blocks, 128, 1, 64,
                                dtype=torch.bfloat16, device="npu") * .1
        if controlled_sources is not None:
            for b in range(batch):
                physical_host = b * blocks * 128 + controlled_sources[b]
                hbm_start = b * hbm_blocks * 128
                kv_seed.view(-1, 512)[hbm_start:hbm_start + args.hot_tokens].copy_(
                    host_kv.tensor.view(-1, 512)[physical_host].to(case.query.device))
                rope_seed.view(-1, 64)[hbm_start:hbm_start + args.hot_tokens].copy_(
                    host_rope.tensor.view(-1, 64)[physical_host].to(case.query.device))
        kv, rope = kv_seed.clone(), rope_seed.clone()
        local_heads = args.attention_heads // args.tp
        kernel_heads = max(4 if args.native_four_heads else 8, local_heads)
        query = torch.randn(total_queries, kernel_heads, 512, dtype=torch.bfloat16, device="npu") * .1
        query_rope = torch.randn(total_queries, kernel_heads, 64, dtype=torch.bfloat16, device="npu") * .1
        if local_heads < kernel_heads:
            query[:, local_heads:].zero_()
            query_rope[:, local_heads:].zero_()
        out = torch.empty_like(query)
        actual_kv = torch.full((batch,), args.hot_tokens + 128, dtype=torch.int32, device="npu")

        def sfa(data):
            sources, slots, misses, miss_sources, miss_slots, counts = data
            torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention(
                query_rope, query, case.actual_seq_lengths_query, actual_kv,
                case.num_cache_tokens, slots, sources, misses,
                miss_sources, miss_slots, counts, hbm_table, dram_table,
                rope, kv, host_rope.device_view, host_kv.device_view,
                1 / math.sqrt(576), out,
            )

        def baseline():
            upstream._call_op(case, cache, outputs)
            sfa(outputs)

        def with_prefetch():
            prefetch.indexer(cache, outputs)
            prefetch.consume(outputs, hbm_table, kv, rope)
            sfa(consumed)

        def reset():
            cache.copy_(case.cache_seed)
            kv.copy_(kv_seed)
            rope.copy_(rope_seed)
            if mode == "prefetch-64-unfiltered":
                prefetch.thresholds.fill_(float("-inf"))

        mode = "baseline"
        reset()
        baseline()
        torch.npu.synchronize()
        expected_cache, expected_kv, expected_rope, expected_out = (
            tensor.clone() for tensor in (cache, kv, rope, out)
        )
        expected_outputs = [tensor.clone() for tensor in outputs]
        if args.attention_reference:
            cpu_kv = expected_kv.cpu().reshape(-1, 512).float()
            cpu_rope = expected_rope.cpu().reshape(-1, 64).float()
            cpu_query, cpu_query_rope = query.cpu().float(), query_rope.cpu().float()
            cpu_slots, cpu_table = outputs[1].cpu(), hbm_table.cpu()
            reference = torch.empty_like(out, device="cpu", dtype=torch.float32)
            for b in range(batch):
                for route in range(q):
                    row = b * q + route
                    tail = torch.arange(args.hot_tokens, args.hot_tokens + 128 - (q - 1 - route))
                    slots = torch.cat((cpu_slots[row, 0].long(), tail))
                    physical = cpu_table[b, slots // 128].long() * 128 + slots % 128
                    keys, ropes = cpu_kv[physical], cpu_rope[physical]
                    scores = (cpu_query[row] @ keys.T + cpu_query_rope[row] @ ropes.T) / math.sqrt(576)
                    reference[row] = scores.softmax(-1) @ keys
            torch.testing.assert_close(expected_out.cpu().float(), reference, rtol=.02, atol=.002)
        original_misses = outputs[2].cpu().tolist()
        original_union_misses = outputs[5].cpu().tolist()
        if args.misses_per_query is not None:
            assert original_misses == [args.misses_per_query] * total_queries
            assert original_union_misses == [q * args.misses_per_query] * batch
        if args.misses_per_request is not None:
            assert original_union_misses == [args.misses_per_request] * batch
        rows = []
        modes = args.modes
        watched = (cache, kv, rope, out, *outputs, *consumed, *prefetch.buffers, prefetch.context)
        addresses = [tensor.data_ptr() for tensor in watched]
        for mode in modes:
            prefetch.budgets.fill_(0 if mode == "context-budget-0" else args.capacity)
            prefetch.thresholds.fill_(float("inf"))
            call = baseline if mode == "baseline" else with_prefetch
            # First call seeds the history. The next capture sees the same query.
            reset()
            call()
            torch.npu.synchronize()
            reset()
            graph = helper.capture(call)
            for _ in range(3):
                reset()
                graph.replay()
                torch.npu.synchronize()
                for actual, reference in zip((cache, kv, rope),
                                              (expected_cache, expected_kv, expected_rope)):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                torch.testing.assert_close(out.float(), expected_out.float(), rtol=.02, atol=.002)
                for index in (0, 1, 2, 5):
                    torch.testing.assert_close(outputs[index], expected_outputs[index], rtol=0, atol=0)
                for b, count in enumerate(original_union_misses):
                    for index in (3, 4):
                        torch.testing.assert_close(outputs[index][b, :count],
                                                   expected_outputs[index][b, :count], rtol=0, atol=0)
                if mode != "baseline":
                    assert bool((prefetch.counts <= prefetch.budgets).all())
                    assert bool((prefetch.counts >= 0).all())
                    available = prefetch.counts.view(batch, q).sum(1).repeat_interleave(q)
                    assert bool((prefetch.used <= available).all())
                    if mode == "context-budget-0":
                        assert not bool(prefetch.counts.any())
                    torch.testing.assert_close(consumed[2], outputs[2] - prefetch.used, rtol=0, atol=0)
                    if args.mtp:
                        torch.testing.assert_close(consumed[5], outputs[5] - prefetch.union_used, rtol=0, atol=0)
                assert addresses == [tensor.data_ptr() for tensor in watched]
            copied = [0] * total_queries if mode == "baseline" else prefetch.counts.cpu().tolist()
            useful = [0] * total_queries if mode == "baseline" else prefetch.used.cpu().tolist()
            union_used = ([0] * batch if mode == "baseline" else
                          (prefetch.union_used if args.mtp else prefetch.used).cpu().tolist())
            row = {
                "batch": batch, "mtp": args.mtp, "queries_per_request": q,
                "total_queries": total_queries, "mode": mode,
                "original_misses_per_query": original_misses,
                "original_union_misses_per_request": original_union_misses,
                "prefetched_per_query": copied, "covered_misses_per_query": useful,
                "useful_unique_sources_per_request": union_used,
                "remaining_misses_per_query": [count - used for count, used in zip(original_misses, useful)],
                "prefetch_host_read_bytes": sum(copied) * 576 * 2,
                "copy_sfa_host_read_bytes": sum(count - used for count, used in zip(original_misses, useful)) * 576 * 2,
                "correctness": "exact_cache_and_raw_indexer_outputs_attention_rtol_0.02_atol_0.002",
                "independent_attention_reference": args.attention_reference,
                **helper.measure(graph, reset, args.repeats),
            }
            if args.profile_repeats:
                trace_path = directory / mode
                helper.profile(graph, reset, trace_path, args.profile_repeats)
                row["stages"] = summarize_chain(trace_path)
            if args.mtp and mode != "baseline":
                query_seed = case.query.clone()
                weight_seed = case.index_weights.clone()
                records = []

                def compare_dynamic(name):
                    reset()
                    baseline()
                    torch.npu.synchronize()
                    references = [tensor.clone() for tensor in (cache, kv, rope, out, *outputs)]
                    reset()
                    prefetch.thresholds.fill_(float("-inf"))
                    graph.replay()
                    torch.npu.synchronize()
                    for actual, reference in zip((cache, kv, rope), references[:3]):
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    torch.testing.assert_close(out.float(), references[3].float(), rtol=.02, atol=.002)
                    for index in (0, 1, 2, 5):
                        torch.testing.assert_close(outputs[index], references[4 + index], rtol=0, atol=0)
                    for b, count in enumerate(outputs[5].cpu().tolist()):
                        for index in (3, 4):
                            torch.testing.assert_close(outputs[index][b, :count],
                                                       references[4 + index][b, :count], rtol=0, atol=0)
                    torch.testing.assert_close(consumed[2], outputs[2] - prefetch.used, rtol=0, atol=0)
                    torch.testing.assert_close(consumed[5], outputs[5] - prefetch.union_used, rtol=0, atol=0)
                    assert bool(((prefetch.counts >= 0) & (prefetch.counts <= prefetch.budgets)).all())
                    assert bool(torch.isfinite(prefetch.thresholds).all())
                    if name == "zero-budget":
                        assert not bool(prefetch.counts.any())
                    if name == "mixed-first-fill":
                        assert not bool(prefetch.counts[:q].any())
                    for t, count in enumerate(prefetch.counts.cpu().tolist()):
                        ids = prefetch.sources[t, :count].cpu().long()
                        assert ids.unique().numel() == count
                        assert bool(((ids >= 0) & (ids < int(case.offload_seq_lengths_key[t // q]))).all())
                        if count:
                            physical = (t // q) * blocks * 128 + ids
                            torch.testing.assert_close(prefetch.kv[t, :count].cpu(),
                                                       host_kv.tensor.view(-1, 512)[physical], rtol=0, atol=0)
                            torch.testing.assert_close(prefetch.rope[t, :count].cpu(),
                                                       host_rope.tensor.view(-1, 64)[physical], rtol=0, atol=0)
                    assert addresses == [tensor.data_ptr() for tensor in watched]
                    records.append({"scenario": name, "copied": int(prefetch.counts.sum()),
                                    "unique_useful": int(prefetch.union_used.sum())})

                case.query.neg_()
                compare_dynamic("changed-query")
                case.offload_seq_lengths_key.fill_(offload - 128)
                case.actual_seq_lengths_key.fill_(args.context - 128)
                prefetch.budgets.fill_(7)
                compare_dynamic("shorter-length-budget-7")
                prefetch.budgets.zero_()
                compare_dynamic("zero-budget")
                case.offload_seq_lengths_key.fill_(offload)
                case.actual_seq_lengths_key.fill_(args.context)
                prefetch.budgets.fill_(args.capacity)
                case.query.copy_(query_seed.view(batch, q, 64, 128)[:, :1].expand(-1, q, -1, -1).reshape_as(case.query))
                case.index_weights.copy_(weight_seed.view(batch, q, 64)[:, :1].expand(-1, q, -1).reshape_as(case.index_weights))
                compare_dynamic("duplicate-candidates-across-routes")
                case.request_state[0] = -2
                compare_dynamic("mixed-first-fill")
                case.request_state.fill_(-1)
                case.query.copy_(query_seed)
                case.index_weights.copy_(weight_seed)
                compare_dynamic("steady-again")
                # Advance without reset, using exactly the preceding cache state.
                previous = [tensor.clone() for tensor in (cache, kv, rope)]
                baseline()
                torch.npu.synchronize()
                reference = [tensor.clone() for tensor in (cache, kv, rope, out)]
                for target, seed in zip((cache, kv, rope), previous):
                    target.copy_(seed)
                graph.replay()
                torch.npu.synchronize()
                for actual, expected in zip((cache, kv, rope), reference[:3]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(out.float(), reference[3].float(), rtol=.02, atol=.002)
                assert not bool(outputs[2].any()) and not bool(prefetch.union_used.any())
                assert addresses == [tensor.data_ptr() for tensor in watched]
                records.append({"scenario": "persistent-all-hit", "remaining_misses": 0})
                row["dynamic_graph_checks"] = records
            rows.append(row)
            (directory / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            del graph
            graph = None
        return rows
    finally:
        del graph
        gc.collect()
        torch.npu.synchronize()
        host_rope.close()
        host_kv.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[16])
    parser.add_argument("--mtp", type=int, choices=[0, 3], default=3)
    parser.add_argument("--modes", nargs="+", choices=["baseline", "context-budget-0",
                        "prefetch-64-history", "prefetch-64-unfiltered"],
                        default=["baseline", "prefetch-64-history"])
    parser.add_argument("--context", type=int, default=40960)
    parser.add_argument("--hot-tokens", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--indexer-heads", type=int, choices=[32, 64], default=64)
    parser.add_argument("--attention-heads", type=int, choices=[64, 128], default=128)
    parser.add_argument("--native-four-heads", action="store_true",
                        help="use the native four-head path instead of the PR's eight-head padding")
    parser.add_argument("--attention-reference", action="store_true",
                        help="check the initial attention output against CPU float attention")
    parser.add_argument("--capacity", type=int, choices=[64], default=64)
    miss_group = parser.add_mutually_exclusive_group()
    miss_group.add_argument("--misses-per-query", type=int,
                        help="construct exact per-query misses using query-exclusive sources; not a real trace")
    miss_group.add_argument("--misses-per-request", type=int,
                           help="construct exact request-union misses across the four MTP3 rows; not a real trace")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile-repeats", type=int, default=10)
    args = parser.parse_args()
    if (args.context % 128 or args.hot_tokens % 256 or
            not 2048 <= args.hot_tokens <= 16128 or args.context - 128 <= args.hot_tokens):
        parser.error("need aligned context and hot tokens in [2048,16128], with context-128 > hot tokens")
    if args.tp <= 0 or args.attention_heads % args.tp or args.attention_heads // args.tp < 1:
        parser.error("need supported local attention head count")
    if args.repeats < 1 or args.profile_repeats < 0 or any(batch < 1 for batch in args.batches):
        parser.error("invalid batch/replay counts")
    if args.misses_per_query is not None and not 0 <= args.misses_per_query <= 2048:
        parser.error("misses-per-query must be in [0,2048]")
    if args.misses_per_request is not None and not 0 <= args.misses_per_request <= (1 + args.mtp) * 2048:
        parser.error("misses-per-request must be in [0,Q*2048]")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    native_root = Path(os.environ.get("ECHO_BENCH_SOURCE", root / "vllm-ascend"))
    native_sources = [native_root / "csrc/attention/fused_lightning_indexer_manage/op_kernel/fused_lightning_indexer_manage_union.h",
                      native_root / "csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/sparse_flash_attention_impl/fused_scatter_copy_sparse_flash_attention_service_vector_mla.h",
                      native_root / "csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/fused_scatter_copy_sparse_flash_attention_tiling.cpp",
                      native_root / "vllm_ascend/attention/sfa_kv_offload.py"]
    manifest = {
        "source_root": str(native_root), "variant": os.environ.get("ECHO_BENCH_VARIANT", "working_tree"),
        "native_source_sha256": {str(p.relative_to(native_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in native_sources},
        "status": "running", "coverage": "synthetic_single_layer_chained_graph_not_model_inference",
        "context_tokens": args.context, "offload_prefix_tokens": args.context - 128,
        "hot_tokens_per_request": args.hot_tokens, "topk": 2048, "tail_tokens": 128,
        "indexer_heads": args.indexer_heads, "indexer_head_dim": 128, "dtype": "bfloat16",
        "attention_heads_per_rank": args.attention_heads // args.tp, "tp_shape_assumption": args.tp,
        "attention_kernel_heads": max(4 if args.native_four_heads else 8, args.attention_heads // args.tp),
        "actual_execution_devices": 1, "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "capacity": args.capacity, "batches": args.batches,
        "mtp": args.mtp, "queries_per_request": 1 + args.mtp,
        "mtp_scope": "target verification only; excludes three draft forwards, acceptance/rejection and model layers",
        "modes": args.modes,
        "controlled_misses_per_query": args.misses_per_query,
        "controlled_union_misses_per_request": args.misses_per_request,
        "residency_note": ("constructed from untimed exact LIM selections; sample distinct sources from each request union; "
                           "query-level counts follow selected-source membership; resident HBM KV matches Host; not real trace"
                           if args.misses_per_request is not None else
                           "constructed from untimed exact LIM selections; only query-exclusive sources miss; "
                           "request union misses=Q*target; resident HBM KV matches registered Host source; not real trace"
                           if args.misses_per_query is not None else "original prefix residency seed"),
        "history_note": "same fixed query replay; warmed previous-step cutoff, optimistic prediction, not real model accuracy",
        "reset_note": "restore cache mapping and KV/RoPE on same stream outside event timing; readiness/count/source reset inside prefetch graph",
        "timing_note": "events cover complete graph including pre-LIM ready resets; stage sums exclude those resets and inter-task gaps; span begins at LIM; profiled execution can be perturbed",
        "host_source": "aclrtHostRegister_mmap_device_alias_not_MemFabric",
        "warmup_replays_per_measurement": 5, "event_replays": args.repeats,
        "profiler_active_steps": args.profile_repeats, "metrics": [],
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), root / "scripts/echo_indexer_prefetch.py",
                                       root / "patches/vllm-ascend-echo-indexer-prefetch.patch",
                                       root / "scripts/echo_profile_workload.py")},
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        torch.npu.set_device(0)
        helper = load_profile()
        upstream = helper.load_upstream(root)
        for batch in args.batches:
            manifest["metrics"].extend(run_batch(args, helper, upstream, batch))
            result_path.write_text(json.dumps(manifest, indent=2) + "\n")
            gc.collect()
            torch.npu.empty_cache()
        manifest["status"] = "completed_operator_chain_only"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        result_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
