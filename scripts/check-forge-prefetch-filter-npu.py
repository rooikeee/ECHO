#!/usr/bin/env python3
"""Check ordered 910 prefetch filtering, partial tiles, and negative sentinels."""

import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch_npu  # noqa: F401

from echo_indexer_prefetch import COMPACT_ROUTES_FLAG, PREFETCH_SCAN_TILES_SHIFT, EchoMTPIndexerPrefetch


def load_helpers():
    path = Path(__file__).with_name("collect-vllm-sparse-profile-npu.py")
    spec = importlib.util.spec_from_file_location("forge_filter_helpers", path)
    helpers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helpers
    spec.loader.exec_module(helpers)
    return helpers


@torch.inference_mode()
def main():
    torch.npu.set_device(0)
    torch.manual_seed(20261102)
    helpers = load_helpers()
    upstream = helpers.load_upstream(Path(__file__).resolve().parents[1])
    batch, queries, context, hot = 2, 4, 4352, 2048
    offload = context - 128
    case = upstream._build_case(
        q_values=[queries] * batch,
        states=[-1] * batch,
        offload_len=offload,
        cache_tokens=hot,
        heads=64,
    )
    # Exercise both negative encodings without changing residency.
    seed = case.cache_seed.cpu()
    columns = torch.arange(seed.shape[1])
    seed[(seed < 0) & (columns[None, :] % 2 == 0)] = torch.iinfo(torch.int32).min
    case.cache_seed.copy_(seed.to(case.query.device))
    cache = case.cache_seed.clone()
    baseline_cache = cache.clone()
    outputs, expected = upstream._make_outputs(case), upstream._make_outputs(case)
    blocks = context // 128
    table = (
        torch.stack([torch.randperm(blocks) + b * blocks for b in range(batch)])
        .int()
        .to("npu")
    )
    host_kv = helpers.HostBuffer((batch * blocks, 128, 512))
    host_rope = helpers.HostBuffer((batch * blocks, 128, 64))
    graph = None
    try:
        prefetch = EchoMTPIndexerPrefetch(case, host_kv, host_rope, table)
        query_seed = case.query.clone()

        def call():
            cache.copy_(case.cache_seed)
            prefetch.indexer(cache, outputs, compact_routes=True)

        graph = helpers.capture(call)
        watched = (cache, *outputs, *prefetch.buffers, prefetch.context)
        addresses = [tensor.data_ptr() for tensor in watched]
        checks = 0
        lengths = (offload, offload - 1, offload - 63, offload - 65)
        scenarios = [(length, tiles) for tiles in (64, 1, 5) for length in lengths]
        for length, scan_tiles in scenarios:
            case.offload_seq_lengths_key.fill_(length)
            prefetch.compact_context[:, 12].fill_(
                prefetch.capacity | COMPACT_ROUTES_FLAG | (scan_tiles << PREFETCH_SCAN_TILES_SHIFT))
            for budget, threshold, zero_query in (
                (64, float("-inf"), False),
                (7, float("-inf"), False),
                (0, float("-inf"), False),
                (64, 0.0, True),
                (64, 1e30, False),
            ):
                case.query.copy_(query_seed)
                if zero_query:
                    case.query.zero_()
                baseline_cache.copy_(case.cache_seed)
                upstream._call_op(case, baseline_cache, expected)
                prefetch.budgets.fill_(budget)
                for _ in range(2):
                    # Cutoff publication updates history; force this test's
                    # threshold before each replay, outside captured work.
                    prefetch.thresholds.fill_(threshold)
                    graph.replay()
                    torch.npu.synchronize()
                    torch.testing.assert_close(cache, baseline_cache, rtol=0, atol=0)
                    for index in (0, 1, 2, 5):
                        torch.testing.assert_close(
                            outputs[index], expected[index], rtol=0, atol=0
                        )
                    counts = prefetch.counts.cpu().tolist()
                    sources = prefetch.sources.cpu()
                    for b, row in enumerate(case.req_entries):
                        scan_end = min(length, scan_tiles * 512)
                        ids = torch.nonzero(seed[row, :scan_end] < 0).flatten()[:budget]
                        if threshold != float("-inf"):
                            ids = ids[:0]
                        for route in range(queries):
                            q = b * queries + route
                            assert counts[q] == len(ids), (
                                length,
                                budget,
                                threshold,
                                q,
                                counts[q],
                            )
                            torch.testing.assert_close(
                                sources[q, : len(ids)].long(), ids, rtol=0, atol=0
                            )
                            if len(ids):
                                physical = (
                                    table[b].cpu().long()[ids // 128] * 128 + ids % 128
                                )
                                torch.testing.assert_close(
                                    prefetch.kv[q, : len(ids)].cpu(),
                                    host_kv.tensor.view(-1, 512)[physical],
                                    rtol=0,
                                    atol=0,
                                )
                                torch.testing.assert_close(
                                    prefetch.rope[q, : len(ids)].cpu(),
                                    host_rope.tensor.view(-1, 64)[physical],
                                    rtol=0,
                                    atol=0,
                                )
                    route_tags = prefetch.copy_routes.cpu().long() & 0xffffffff
                    miss_sources = outputs[0].cpu().view(batch, queries, -1)
                    miss_counts = outputs[2].cpu().view(batch, queries)
                    flat_staging = sources.reshape(-1)
                    for b in range(batch):
                        available = set()
                        for q in range(b * queries, (b + 1) * queries):
                            available.update(sources[q, :counts[q]].tolist())
                        owners = {}
                        for route in range(queries):
                            q = b * queries + route
                            for position in range(int(miss_counts[b, route])):
                                source = int(miss_sources[b, route, position])
                                tag = int(route_tags[q, position])
                                staging = tag & 0x7fffffff
                                assert bool(staging) == (source in available)
                                if staging:
                                    assert b * queries * prefetch.capacity <= staging - 1 < (b + 1) * queries * prefetch.capacity
                                    assert int(flat_staging[staging - 1]) == source
                                owners[source] = owners.get(source, 0) + int((tag & 0x80000000) == 0)
                        assert all(count == 1 for count in owners.values())
                    assert addresses == [tensor.data_ptr() for tensor in watched]
                    checks += 1
        print(
            json.dumps(
                {
                    "status": "pass",
                    "graph_replays": checks,
                    "batch": batch,
                    "queries_per_request": queries,
                    "checks": "ordered IDs, INT_MIN/-1, unaligned lengths, strict threshold, exact Host payload, packed staging tags, one owner per miss",
                }
            )
        )
    finally:
        if graph is not None:
            del graph
        torch.npu.synchronize()
        host_rope.close()
        host_kv.close()


if __name__ == "__main__":
    main()
