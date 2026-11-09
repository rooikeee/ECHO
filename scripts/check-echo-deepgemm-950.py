#!/usr/bin/env python3
"""950-only validation/benchmark: MTP=3, captured resets/metadata/fused scoring.

CPU checks live in test-echo-deepgemm.py. This script never substitutes a 910
backend. HBM is a copy diagnostic; --host-ddr requires a real MemFabric DVA.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def load_attention_helpers():
    path = Path(__file__).resolve().parents[1] / "deepgemm-ascend/tests/test_attention.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("echo_dg_attention_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_pool(torch, tokens, token_bytes, host_ddr):
    count = tokens * token_bytes
    if not host_ddr:
        return torch.empty((tokens, token_bytes), dtype=torch.uint8, device="npu"), None
    from memfabric_hybrid import offload
    alignment = 2 * 1024 * 1024
    size = (count + alignment - 1) // alignment * alignment + alignment
    config = offload.OffloadConfig()
    config.device_id = torch.npu.current_device()
    config.reserve_size = config.alloc_size = size
    config.world_size, config.rank_id = 1, 0
    config.scene, config.flags = offload.Scene.LOCAL, offload.OFFLOAD_FLAG_GIANT_PAGE
    if offload.initialize(config) != 0:
        raise RuntimeError("MemFabric initialization failed")
    raw = offload.empty([size], dtype=torch.int8, pin_memory=True)
    offset = (-raw.data_ptr()) % alignment
    view = raw[offset:offset + count].view(torch.uint8).reshape(tokens, token_bytes)
    dva = int(offload.get_dva(view.data_ptr()))
    if not dva:
        raise RuntimeError("MemFabric returned a null DVA")
    return view, dva


def run(torch, dg, args, requests, source, source_dva, helpers):
    from deep_gemm.echo import EchoPrefetchBuffers
    torch.manual_seed(9011 + requests)
    rows, length, pages_per_req = requests * 4, args.seq_len, (args.seq_len + 127) // 128
    q, sfq, _ = helpers.quantize_mqa_tensor(torch.randn(rows, 1, 32, 128, device="npu", dtype=torch.bfloat16), False)
    sfq = sfq.reshape(rows * 32, 2).t().contiguous().t()
    kv, sfkv, _ = helpers.quantize_mqa_tensor(
        torch.randn(requests * pages_per_req, 128, 128, device="npu", dtype=torch.bfloat16), False)
    fused = torch.cat((kv.contiguous().view(torch.uint8).reshape(-1, 128 * 128),
                       sfkv.contiguous().view(torch.uint8).reshape(-1, 128 * 4)), 1).view(-1, 128, 1, 132)
    page_table = torch.randperm(requests * pages_per_req, device="npu", dtype=torch.int32).reshape(requests, -1)
    page_table = page_table.repeat_interleave(4, 0).contiguous()
    indices = torch.arange(requests, device="npu", dtype=torch.int32).repeat_interleave(4)
    lengths = torch.tensor([length - 3 + i for i in range(4)] * requests, device="npu", dtype=torch.int32).view(-1, 1)
    weights = torch.randn(rows, 32, device="npu", dtype=torch.bfloat16)
    thresholds = torch.full((rows,), -float("inf"), device="npu")
    stable = torch.full((rows,), length - 4, device="npu", dtype=torch.int32)
    active = torch.tensor([rows], device="npu", dtype=torch.int32)
    budgets = torch.full((rows,), args.capacity, device="npu", dtype=torch.int32)
    host_ids = (indices[:, None] * length + torch.arange(length, device="npu", dtype=torch.int32)[None] + 1).contiguous()
    mapping = torch.full((source.shape[0],), -1, device="npu", dtype=torch.int32)
    written = torch.ones_like(mapping)
    buffers = EchoPrefetchBuffers.allocate(rows, length, args.capacity, source, source_dva=source_dva,
        source_kind="MemFabric-DDR" if args.host_ddr else "HBM-diagnostic")
    operands = (q,sfq,fused,weights,lengths,page_table,indices,thresholds,stable,active,budgets,host_ids,mapping,written)
    launch = lambda: buffers.run(*operands)
    # Complete JIT and runtime setup outside capture; buffer addresses stay fixed.
    for _ in range(2):
        launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    watched = (*operands,buffers.logits,buffers.counts,buffers.tokens,buffers.hosts,buffers.staging,buffers.metadata)
    addresses = [t.data_ptr() for t in watched]
    source_cpu = source.cpu()
    scenarios = ("cold", "all-hit", "mixed", "shorter", "zero-budget", "empty-threshold",
                 "invalid-host", "inactive", "all-inactive", "request-reset", "cold")
    records = []
    for scenario in scenarios:
        mapping.fill_(-1); written.fill_(1); thresholds.fill_(-float("inf")); stable.fill_(length - 4)
        budgets.fill_(args.capacity); active.fill_(rows)
        lengths.copy_(torch.tensor([length - 3 + i for i in range(4)] * requests, device="npu", dtype=torch.int32).view(-1, 1))
        host_ids.copy_((indices[:, None] * length + torch.arange(length, device="npu", dtype=torch.int32)[None] + 1))
        if scenario == "all-hit": mapping.fill_(0)
        elif scenario == "mixed": mapping[::2] = 0; written[::7] = 0
        elif scenario == "shorter":
            end = max(4, length // 2)
            lengths.copy_(torch.tensor([end - 3 + i for i in range(4)] * requests, device="npu", dtype=torch.int32).view(-1, 1))
            stable.fill_(end - 2); budgets.fill_(min(7, args.capacity))
        elif scenario == "zero-budget": budgets.zero_()
        elif scenario == "empty-threshold": thresholds.fill_(float("inf"))
        elif scenario == "invalid-host": host_ids[:, ::2] = -1; host_ids[:, 1::4] = 0; host_ids[:, 3::4] = source.shape[0]
        elif scenario == "inactive": active.fill_(max(0, rows - 2))
        elif scenario == "all-inactive": active.zero_(); lengths.zero_()
        elif scenario == "request-reset":
            # Mutate the immutable-for-one-launch mapping between launches.
            host_ids.copy_(host_ids.flip(1)); mapping[1::3] = 5
        before = (mapping.cpu().clone(), written.cpu().clone())
        # Compare exact logits and the selected subset; saturated selection may
        # differ across schedules, but every staged byte/token must be valid.
        meta = dg.get_paged_mqa_logits_metadata(lengths,32,indices)
        reference = dg.fp8_fp4_paged_mqa_logits((q,sfq),fused,weights,lengths,page_table,meta,length,indices,False)
        for _ in range(2):
            graph.replay(); torch.npu.synchronize()
            actual = buffers.logits.cpu(); ref = reference.cpu(); lens = lengths.cpu().flatten()
            for row,n in enumerate(lens.tolist()):
                torch.testing.assert_close(actual[row,:n], ref[row,:n], rtol=0, atol=0)
                assert bool(torch.isneginf(actual[row,n:]).all())
            ids, mp, wr = host_ids.cpu(), mapping.cpu(), written.cpu()
            threshold_cpu, stable_cpu, budget_cpu = thresholds.cpu(), stable.cpu(), budgets.cpu()
            counts,tokens,hosts,staging = (t.cpu() for t in (buffers.counts,buffers.tokens,buffers.hosts,buffers.staging))
            for row,n in enumerate(lens.tolist()):
                eligible = []
                if row < int(active.cpu()[0]):
                    for token in range(min(n,int(stable_cpu[row]))):
                        h = int(ids[row,token])
                        if (bool(actual[row,token] > threshold_cpu[row]) and 0 < h < source.shape[0]
                                and int(wr[h]) != 0 and int(mp[h]) == -1):
                            eligible.append(token)
                expected_count = min(len(eligible),max(0,min(args.capacity,int(budget_cpu[row]))))
                count = int(counts[row]); assert count == expected_count, (scenario,row,count,expected_count)
                selected = tokens[row,:count].tolist()
                assert len(set(selected)) == count and set(selected).issubset(eligible)
                for i,token in enumerate(selected):
                    h = int(ids[row,token]); assert int(hosts[row,i]) == h
                    torch.testing.assert_close(staging[row,i],source_cpu[h],rtol=0,atol=0)
                assert bool((tokens[row,count:] == -1).all()) and bool((hosts[row,count:] == -1).all())
            torch.testing.assert_close(mapping.cpu(),before[0],rtol=0,atol=0)
            torch.testing.assert_close(written.cpu(),before[1],rtol=0,atol=0)
            assert [t.data_ptr() for t in watched] == addresses
        records.append({"scenario":scenario,"misses":int(buffers.counts.sum().cpu()),"replays":2})
    # Optional target-phase timing covers captured resets + metadata + fused MQA
    # and staging, and excludes downstream TopK/attention/speculative acceptance.
    timing = baseline_timing = None
    if args.repeats:
        baseline_outputs = {}
        def baseline_launch():
            metadata = dg.get_paged_mqa_logits_metadata(lengths,32,indices)
            baseline_outputs["logits"] = dg.fp8_fp4_paged_mqa_logits(
                (q,sfq),fused,weights,lengths,page_table,metadata,length,indices,False)
        baseline_launch(); torch.npu.synchronize()
        baseline_graph = torch.npu.NPUGraph()
        with torch.npu.graph(baseline_graph):
            baseline_launch()
        def measure(g):
            start,end = torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            for _ in range(5):g.replay()
            start.record()
            for _ in range(args.repeats):g.replay()
            end.record();end.synchronize()
            return start.elapsed_time(end) * 1000 / args.repeats
        baseline_timing, timing = measure(baseline_graph), measure(graph)
    return {"requests":requests,"queries_per_request":4,"mtp":3,"source":buffers.source_kind,
            "seq_len":length,"token_bytes":args.token_bytes,"capacity":args.capacity,"checks":records,
            "baseline_mqa_with_metadata_graph_us":baseline_timing,
            "target_mqa_prefetch_graph_us":timing,"complete_speculative_round_us":None,
            "timing_scope":"prefetch includes resets, schedule and staging; baseline includes schedule and logits only"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests",type=int,nargs="+",default=[1,2])
    parser.add_argument("--seq-len",type=int,default=2048)
    parser.add_argument("--capacity",type=int,default=64)
    parser.add_argument("--token-bytes",type=int,default=1152)
    parser.add_argument("--host-ddr",action="store_true")
    parser.add_argument("--device",type=int,default=0)
    parser.add_argument("--repeats",type=int,default=0)
    parser.add_argument("--output",type=Path,default=Path("build/deepgemm-echo/950-validation.json"))
    args=parser.parse_args()
    if min(args.requests)<=0 or args.seq_len<4 or not 0<args.capacity<=2048 or not 0<args.token_bytes<=65536 or args.repeats<0:
        parser.error("positive requests, seq-len>=4, capacity<=2048, token-bytes<=65536 and repeats>=0 required")
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(args.device)
    if "950" not in torch.npu.get_device_name():
        raise RuntimeError("This validation requires Ascend 950; this machine is not a target")
    import deep_gemm as dg
    if not hasattr(dg._C,"echo_paged_mqa_prefetch_out"):
        raise RuntimeError("Rebuild this repository's deepgemm-ascend package first")
    helpers=load_attention_helpers()
    source,dva=source_pool(torch,max(args.requests)*args.seq_len+1,args.token_bytes,args.host_ddr)
    for start in range(0,source.numel(),1<<20):
        stop=min(start+(1<<20),source.numel())
        source.view(-1)[start:stop].copy_(((torch.arange(stop-start,device=source.device,dtype=torch.int32)+start)%251).to(torch.uint8))
    with torch.inference_mode():
        results=[run(torch,dg,args,b,source,dva,helpers) for b in args.requests]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=="__main__":main()
