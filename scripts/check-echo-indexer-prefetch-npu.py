#!/usr/bin/env python3
"""Validate bounded Indexer Host prefetch -> exact fallback -> CopySFA graphs."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch
import torch_npu  # noqa: F401

from echo_indexer_prefetch import EchoIndexerPrefetch


def load_profile():
    path = Path(__file__).with_name("collect-vllm-sparse-profile-npu.py")
    spec = importlib.util.spec_from_file_location("echo_prefetch_profile", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def call_indexer(case, cache, outputs, context=None):
    torch.ops._C_ascend.npu_fused_lightning_indexer_manage(
        case.index_weights, case.query_dequant_scale, case.query,
        case.index_key_dequant_scale, case.index_key_cache, case.index_block_table,
        case.actual_seq_lengths_query, case.actual_seq_lengths_key,
        case.offload_seq_lengths_key, case.num_cache_tokens, case.request_state,
        case.req_pool_entries, cache, *outputs, prefetch_context=context,
    )


@torch.inference_mode()
def run(profile, upstream, batch, context, capacity):
    offload = context - 128
    hot = 2048
    case = upstream._build_case(q_values=[1] * batch, states=[-1] * batch,
                                offload_len=offload, cache_tokens=hot, heads=64)
    cache = case.cache_seed.clone()
    baseline_cache = cache.clone()
    outputs = upstream._make_outputs(case)
    baseline_outputs = upstream._make_outputs(case)
    blocks = context // 128
    # Exercise nonidentity Host and HBM page tables and independent requests.
    dram_table = torch.stack([torch.randperm(blocks) + b * blocks for b in range(batch)]).int().to("npu")
    hbm_blocks = hot // 128 + 1
    hbm_table = torch.stack([torch.randperm(hbm_blocks) + b * hbm_blocks for b in range(batch)]).int().to("npu")
    host_kv = profile.HostBuffer((batch * blocks, 128, 512))
    host_rope = profile.HostBuffer((batch * blocks, 128, 64))
    try:
        prefetch = EchoIndexerPrefetch(case, host_kv, host_rope, dram_table, capacity)
        consumed = prefetch.allocate_outputs(outputs)
        kv_seed = torch.randn(batch * hbm_blocks, 128, 1, 512, dtype=case.query.dtype, device="npu") * .1
        rope_seed = torch.randn(batch * hbm_blocks, 128, 1, 64, dtype=case.query.dtype, device="npu") * .1
        kv, baseline_kv = kv_seed.clone(), kv_seed.clone()
        rope, baseline_rope = rope_seed.clone(), rope_seed.clone()
        query = torch.randn(batch, 16, 512, dtype=case.query.dtype, device="npu") * .1
        query_rope = torch.randn(batch, 16, 64, dtype=case.query.dtype, device="npu") * .1
        out, baseline_out = torch.empty_like(query), torch.empty_like(query)
        actual_kv = torch.full((batch,), hot + 128, dtype=torch.int32, device="npu")

        def sfa(data, target_kv, target_rope, target):
            sources, slots, misses, miss_sources, miss_slots, counts = data
            torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention(
                query_rope, query, case.actual_seq_lengths_query, actual_kv,
                case.num_cache_tokens, slots, sources, misses,
                miss_sources, miss_slots, counts, hbm_table, dram_table,
                target_rope, target_kv, host_rope.device_view, host_kv.device_view,
                1 / math.sqrt(576), target,
            )

        def reset():
            cache.copy_(case.cache_seed)
            kv.copy_(kv_seed)
            rope.copy_(rope_seed)

        def fused():
            prefetch.indexer(cache, outputs)
            prefetch.consume(outputs, hbm_table, kv, rope)
            sfa(consumed, kv, rope, out)

        # Seed the first prediction from the preceding query's exact cutoff.
        keys = case.index_key_cache.float().view(-1, 128)
        cutoffs = []
        for b in range(batch):
            table = case.index_block_table[b].long()
            logical = keys.view(-1, 128, 128)[table].reshape(-1, 128)[:offload]
            score = (case.query[b].float() @ logical.T).relu()
            score = (score * case.index_weights[b].float()[:, None]).sum(0)
            cutoffs.append(score.topk(2048).values[-1])
        prediction = torch.stack(cutoffs)
        query_seed = case.query.clone()
        reset()
        fused()
        torch.npu.synchronize()
        reset()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            fused()
        watched = (cache, *outputs, *consumed, prefetch.context, prefetch.ready,
                   prefetch.counts, prefetch.sources, prefetch.kv, prefetch.rope)
        addresses = [tensor.data_ptr() for tensor in watched]
        records = []
        scenarios = [
            ("saturate", -1, offload, capacity, float("-inf")),
            ("disabled-threshold", -1, offload, capacity, float("inf")),
            ("zero-budget", -1, offload, 0, float("-inf")),
            ("equal-threshold", -1, offload, capacity, 0.0),
            ("shorter-small-budget", -1, offload - 128, min(7, capacity), float("-inf")),
            ("predicted-threshold", -1, offload, capacity, None),
            ("automatic-history", -1, offload, capacity, "history"),
            ("first-fill", -2, offload, capacity, float("-inf")),
            ("mixed-first-fill", -1, offload, capacity, float("-inf")),
            ("steady-again", -1, offload, capacity, float("-inf")),
        ]
        for step, (name, state, length, budget, threshold) in enumerate(scenarios):
            case.query.copy_(query_seed if step % 2 else -query_seed)
            if name == "equal-threshold":
                case.query.zero_()
            case.request_state.fill_(state)
            if name == "mixed-first-fill":
                case.request_state[0] = -2
            case.offload_seq_lengths_key.fill_(length)
            prefetch.budgets.fill_(budget)
            if threshold is None:
                prefetch.thresholds.copy_(prediction)
            elif threshold != "history":
                prefetch.thresholds.fill_(threshold)
            baseline_cache.copy_(case.cache_seed)
            baseline_kv.copy_(kv_seed)
            baseline_rope.copy_(rope_seed)
            call_indexer(case, baseline_cache, baseline_outputs)
            sfa(baseline_outputs, baseline_kv, baseline_rope, baseline_out)
            reset()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(cache, baseline_cache, rtol=0, atol=0)
            for index in (0, 1, 2, 5):
                torch.testing.assert_close(outputs[index], baseline_outputs[index], rtol=0, atol=0)
            for b in range(batch):
                valid = int(outputs[5][b])
                for index in (3, 4):
                    torch.testing.assert_close(outputs[index][b, :valid],
                                               baseline_outputs[index][b, :valid], rtol=0, atol=0)
            torch.testing.assert_close(kv, baseline_kv, rtol=0, atol=0)
            torch.testing.assert_close(rope, baseline_rope, rtol=0, atol=0)
            torch.testing.assert_close(out.float(), baseline_out.float(), rtol=.02, atol=.002)
            copied = prefetch.counts.cpu().tolist()
            used = prefetch.used.cpu().tolist()
            assert bool(torch.isfinite(prefetch.thresholds).all()), "TopK cutoff feedback missing"
            for b, count in enumerate(copied):
                assert 0 <= count <= budget
                if name in ("saturate", "shorter-small-budget", "steady-again"):
                    assert count == budget, (name, b, count, budget)
                if (name in ("disabled-threshold", "zero-budget", "equal-threshold", "first-fill") or
                        (name == "mixed-first-fill" and b == 0)):
                    assert count == 0, (name, count)
                ids = prefetch.sources[b, :count].cpu().long()
                assert ids.unique().numel() == count
                assert bool(((ids >= 0) & (ids < length)).all())
                row = case.req_entries[b]
                assert bool((case.cache_seed[row].cpu()[ids] < 0).all())
                if count:
                    physical = dram_table[b].cpu().long()[ids // 128] * 128 + ids % 128
                    torch.testing.assert_close(prefetch.kv[b, :count].cpu(),
                                               host_kv.tensor.view(-1, 512)[physical], rtol=0, atol=0)
                    torch.testing.assert_close(prefetch.rope[b, :count].cpu(),
                                               host_rope.tensor.view(-1, 64)[physical], rtol=0, atol=0)
                assert int(consumed[2][b]) == int(outputs[2][b]) - used[b]
            assert addresses == [tensor.data_ptr() for tensor in watched]
            records.append({"scenario": name, "copied": copied, "useful": used})
            print(json.dumps({"batch": batch, "context": context, **records[-1]}), flush=True)
        # Replay without resetting persistent cache: the same query is now all-hit.
        baseline_cache.copy_(cache)
        baseline_kv.copy_(kv)
        baseline_rope.copy_(rope)
        call_indexer(case, baseline_cache, baseline_outputs)
        sfa(baseline_outputs, baseline_kv, baseline_rope, baseline_out)
        graph.replay()
        torch.npu.synchronize()
        assert not bool(outputs[2].any())
        assert not bool(prefetch.used.any())
        torch.testing.assert_close(cache, baseline_cache, rtol=0, atol=0)
        torch.testing.assert_close(kv, baseline_kv, rtol=0, atol=0)
        torch.testing.assert_close(rope, baseline_rope, rtol=0, atol=0)
        torch.testing.assert_close(out.float(), baseline_out.float(), rtol=.02, atol=.002)
        print(f"PASS batch={batch}, context={context}: capture + {len(scenarios) + 1} changing/persistent replays", flush=True)
        del graph
        return records
    finally:
        # Successful runs destroy the graph before unregistering Host aliases.
        # On an exception the graph object, if created, must also be destroyed.
        if "graph" in locals():
            del graph
        torch.npu.synchronize()
        host_rope.close()
        host_kv.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--context", type=int, default=40960)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.context < 4352 or args.context % 128:
        parser.error("context must be a multiple of 128 and >=4352")
    torch.npu.set_device(0)
    profile = load_profile()
    upstream = profile.load_upstream(Path(__file__).resolve().parents[1])
    results = {batch: run(profile, upstream, batch, args.context, args.capacity)
               for batch in args.batches}
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
