#!/usr/bin/env python3
"""DS-V3.2-shaped operator profiling, NOT full-model AWQ inference."""

import argparse
import csv
import ctypes
import gc
import importlib.util
import json
import math
import mmap
import os
from pathlib import Path
import statistics
import sys

import torch
import torch_npu


class HostBuffer:
    """Own registered CPU storage until all referring graphs are destroyed."""

    def __init__(self, shape):
        count = math.prod(shape)
        self.nbytes = math.ceil(count * 2 / mmap.PAGESIZE) * mmap.PAGESIZE
        self.storage = mmap.mmap(-1, self.nbytes)
        self.tensor = torch.frombuffer(self.storage, dtype=torch.bfloat16, count=count).view(shape)
        self.tensor.normal_(std=0.1)
        self.lib = ctypes.CDLL("libascendcl.so")
        self.lib.aclrtHostRegister.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.aclrtHostRegister.restype = ctypes.c_int
        self.lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self.lib.aclrtHostUnregister.restype = ctypes.c_int
        pointer = ctypes.c_void_p()
        result = self.lib.aclrtHostRegister(self.tensor.data_ptr(), self.nbytes, 0, ctypes.byref(pointer))
        if result:
            raise RuntimeError(f"aclrtHostRegister failed: {result}")
        # The native adapter forwards a DRAM tensor's pointer to the device.
        # This non-owning view describes the device alias; NEVER read it on CPU.
        self.alias_storage = (ctypes.c_byte * (count * 2)).from_address(pointer.value)
        self.device_view = torch.frombuffer(self.alias_storage, dtype=torch.bfloat16, count=count).view(shape)

    def close(self):
        torch.npu.synchronize()
        result = self.lib.aclrtHostUnregister(self.tensor.data_ptr())
        if result:
            raise RuntimeError(f"aclrtHostUnregister failed: {result}")


def load_upstream(root):
    path = root / "vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_lightning_indexer_manage.py"
    spec = importlib.util.spec_from_file_location("profile_indexer_upstream", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture(call):
    call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    graph.replay()
    torch.npu.synchronize()
    return graph


def measure(graph, reset, repeats):
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
    for _ in range(5):
        reset()
        graph.replay()
    torch.npu.synchronize()
    for start, end in zip(starts, ends):
        reset()
        start.record()
        graph.replay()
        end.record()
    torch.npu.synchronize()
    values = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    ordered = sorted(values)
    return {
        "replays": repeats,
        "event_mean_us": statistics.mean(values),
        "event_median_us": statistics.median(values),
        "event_min_us": min(values),
        "event_p90_us": ordered[math.ceil(len(values) * 0.9) - 1],
    }


def profile(graph, reset, path, repeats):
    experimental = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(path)),
        experimental_config=experimental,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=repeats, repeat=1),
    ) as prof:
        for index in range(repeats + 1):
            reset()
            graph.replay()
            if index == repeats:
                torch.npu.synchronize()
            prof.step()
        torch.npu.synchronize()
    kernels = extract_kernels(path)
    if not kernels:
        raise RuntimeError(f"Profiler did not export target kernel tasks: {path}")
    summary = {"profile_kernel_calls": len(kernels)}
    for key in (
        "Duration(us)", "aic_time(us)", "aicore_time(us)", "aiv_time(us)",
        "aic_mac_ratio", "aiv_vec_ratio", "aic_mte2_ratio", "aiv_mte2_ratio",
        "aiv_mte3_ratio", "cube_utilization(%)",
    ):
        values = [float(row[key]) for row in kernels if row.get(key) not in (None, "", "N/A")]
        if values:
            summary[key] = statistics.mean(values)
    return summary


def extract_kernels(path):
    found = []
    for filename in path.rglob("kernel_details.csv"):
        with filename.open(newline="") as source:
            for row in csv.DictReader(source):
                name = " ".join(str(row.get(key, "")) for key in ("Name", "Type", "Op Name", "OP Type"))
                if "indexer" in name.lower() or "scatter" in name.lower() or "sparseflash" in name.lower():
                    found.append(row)
    return found


@torch.inference_mode()
def run_batch(args, upstream, batch):
    directory = args.output / f"batch-{batch}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261023 + batch)
    offload_len = args.context - 128
    case = upstream._build_case(
        q_values=[1] * batch, states=[-1] * batch,
        offload_len=offload_len, cache_tokens=args.hot_tokens, heads=64,
    )
    # Each request owns independent physical index pages, as in serving.
    blocks = args.context // 128
    case.index_key_cache = torch.randn(batch * blocks, 128, 1, 128, dtype=torch.bfloat16, device="npu")
    case.index_key_dequant_scale = torch.zeros(batch * blocks, 128, 1, device="npu")
    offsets = torch.arange(batch, dtype=torch.int32, device="npu").view(-1, 1) * blocks
    case.index_block_table.add_(offsets)
    case.route_block_table.copy_(case.index_block_table)
    cache = case.cache_seed.clone()
    outputs = upstream._make_outputs(case)

    def indexer():
        upstream._call_op(case, cache, outputs)

    def reset_indexer():
        cache.copy_(case.cache_seed)

    reset_indexer()
    indexer()
    torch.npu.synchronize()
    expected = [tensor.clone() for tensor in outputs]
    expected_cache = cache.clone()
    index_graph = capture(indexer)
    for _ in range(3):
        reset_indexer()
        index_graph.replay()
        torch.npu.synchronize()
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    native = upstream._native_topk(case)
    for request in range(batch):
        # topk_src_ids is source-tagged for hits; recover hits from cache mapping.
        slots = outputs[1][request, 0].to(torch.int64)
        mapping = cache[case.req_entries[request]]
        resident_sources = torch.nonzero(mapping >= 0).flatten()
        inverse = torch.full((args.hot_tokens,), -1, dtype=torch.int64, device="npu")
        inverse[mapping[resident_sources].long()] = resident_sources
        recovered = inverse[slots]
        torch.testing.assert_close(recovered.sort().values, native[request].long().sort().values, rtol=0, atol=0)
    rows = [{"batch": batch, "operator": "indexer_manage", "scenario": "random_steady", **measure(index_graph, reset_indexer, args.repeats)}]
    rows[0]["misses_per_request"] = outputs[2].cpu().tolist()
    if args.profile_repeats:
        rows[0]["profiler"] = profile(index_graph, reset_indexer, directory / "indexer", args.profile_repeats)
    print(json.dumps(rows[-1]), flush=True)

    host_rope = HostBuffer((batch * blocks, 128, 64))
    host_kv = HostBuffer((batch * blocks, 128, 512))
    sfa_graph = None
    try:
        heads = 128 // args.tp
        query = torch.randn(batch, heads, 512, dtype=torch.bfloat16, device="npu") * 0.1
        query_rope = torch.randn(batch, heads, 64, dtype=torch.bfloat16, device="npu") * 0.1
        # CopySFA consumes sparse TopK plus the recent 128-token tail, not dense 40k.
        hbm_blocks = args.hot_tokens // 128 + 1
        hbm_table = torch.arange(batch * hbm_blocks, dtype=torch.int32, device="npu").view(batch, hbm_blocks)
        dram_table = torch.arange(batch * blocks, dtype=torch.int32, device="npu").view(batch, blocks)
        hbm_rope = torch.randn(batch * hbm_blocks, 128, 1, 64, dtype=torch.bfloat16, device="npu") * 0.1
        hbm_kv = torch.randn(batch * hbm_blocks, 128, 1, 512, dtype=torch.bfloat16, device="npu") * 0.1
        hbm_rope_seed = hbm_rope.clone()
        hbm_kv_seed = hbm_kv.clone()
        actual_kv = torch.full((batch,), args.hot_tokens + 128, dtype=torch.int32, device="npu")
        attention_out = torch.empty_like(query)
        slots = torch.arange(2048, dtype=torch.int32, device="npu").view(1, 1, -1).repeat(batch, 1, 1)
        sources = torch.full_like(slots, -1)
        miss_counts = torch.zeros(batch, dtype=torch.int32, device="npu")
        unused_ids = torch.full((batch, 32768), -1, dtype=torch.int32, device="npu")
        unused_counts = torch.zeros_like(miss_counts)

        def sfa():
            torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention.default(
                query_rope, query, case.actual_seq_lengths_query, actual_kv,
                case.num_cache_tokens, slots, sources, miss_counts,
                unused_ids, unused_ids, unused_counts, hbm_table, dram_table,
                hbm_rope, hbm_kv, host_rope.device_view, host_kv.device_view,
                1 / math.sqrt(576), attention_out,
            )

        def reset_sfa():
            hbm_rope.copy_(hbm_rope_seed)
            hbm_kv.copy_(hbm_kv_seed)

        sfa_graph = capture(sfa)
        for misses in args.misses:
            # Scatter a deterministic random subset across the full 40k host span.
            generator = torch.Generator().manual_seed(7100 + batch + misses)
            source_cpu = torch.full((batch, 1, 2048), -1, dtype=torch.int32)
            for request in range(batch):
                source_cpu[request, 0, :misses] = torch.randperm(offload_len, generator=generator)[:misses].int()
            sources.copy_(source_cpu)
            miss_counts.fill_(misses)
            reset_sfa()
            sfa()
            torch.npu.synchronize()
            expected_out = attention_out.clone()
            expected_rope = hbm_rope.clone()
            expected_kv = hbm_kv.clone()
            # Check actual host reads/writeback and an independent attention reference.
            for request in range(batch):
                base = request * hbm_blocks * 128
                if misses:
                    ids = source_cpu[request, 0, :misses].long() + request * args.context
                    torch.testing.assert_close(
                        hbm_kv.view(-1, 512)[base:base + misses].cpu(),
                        host_kv.tensor.view(-1, 512)[ids], rtol=0, atol=0,
                    )
                    torch.testing.assert_close(
                        hbm_rope.view(-1, 64)[base:base + misses].cpu(),
                        host_rope.tensor.view(-1, 64)[ids], rtol=0, atol=0,
                    )
                indices = torch.cat((torch.arange(2048), torch.arange(args.hot_tokens, args.hot_tokens + 128))).to("npu") + base
                selected_kv = hbm_kv.view(-1, 512)[indices].float()
                selected_rope = hbm_rope.view(-1, 64)[indices].float()
                scores = (query[request].float() @ selected_kv.T + query_rope[request].float() @ selected_rope.T) / math.sqrt(576)
                reference = scores.softmax(-1) @ selected_kv
                torch.testing.assert_close(attention_out[request].float(), reference, rtol=0.02, atol=0.002)
            for _ in range(3):
                reset_sfa()
                sfa_graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(attention_out, expected_out, rtol=0, atol=0)
                torch.testing.assert_close(hbm_rope, expected_rope, rtol=0, atol=0)
                torch.testing.assert_close(hbm_kv, expected_kv, rtol=0, atol=0)
            row = {
                "batch": batch, "operator": "copy_sfa", "scenario": f"miss-{misses}",
                "misses_per_request": [misses] * batch,
                "logical_host_read_bytes": batch * misses * 576 * 2,
                **measure(sfa_graph, reset_sfa, args.repeats),
            }
            rows.append(row)
            if args.profile_repeats:
                row["profiler"] = profile(sfa_graph, reset_sfa, directory / f"copy-sfa-miss-{misses}", args.profile_repeats)
            print(json.dumps(row), flush=True)
        (directory / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        return rows
    finally:
        del sfa_graph, index_graph
        gc.collect()
        torch.npu.synchronize()
        host_kv.close()
        host_rope.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--context", type=int, default=40960)
    parser.add_argument("--hot-tokens", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--misses", type=int, nargs="+", default=[0, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.context % 128 or args.hot_tokens % 256 or not 2048 <= args.hot_tokens <= 16128:
        parser.error("context must be 128-aligned and hot tokens 256-aligned in [2048, 16128]")
    if args.context - 128 <= args.hot_tokens or 128 % args.tp or 128 // args.tp < 8:
        parser.error("need a long offload context and supported per-TP attention head count")
    if args.repeats < 1 or args.profile_repeats < 0 or any(not 0 <= n <= 2048 for n in args.misses):
        parser.error("invalid replay/miss counts")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "coverage": "synthetic_single_layer_operator_graph_not_full_model_awq",
        "context_tokens": args.context, "indexer_heads": 64, "indexer_head_dim": 128,
        "attention_heads_per_rank": 128 // args.tp, "tp_shape_assumption": args.tp,
        "actual_execution_devices": 1, "hot_tokens_per_request": args.hot_tokens,
        "topk": 2048, "tail_tokens": 128,
        "host_source": "aclrtHostRegister_mmap_device_alias_nonowning_views_not_MemFabric",
        "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "full_model_blocker": "current vLLM Ascend supported_quantization excludes AWQ",
        "timing_note": "event timings include replay submission gaps; use profiler Duration(us) for kernel latency; counter collection may perturb kernels",
        "status": "running", "metrics": [],
    }
    manifest_path = args.output / "result.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        torch.npu.set_device(0)
        root = Path(__file__).resolve().parents[1]
        upstream = load_upstream(root)
        for batch in args.batches:
            manifest["metrics"].extend(run_batch(args, upstream, batch))
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            gc.collect()
            torch.npu.empty_cache()
        manifest["kernel_details"] = extract_kernels(args.output)
        manifest["status"] = "completed_operator_only"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
