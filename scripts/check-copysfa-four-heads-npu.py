#!/usr/bin/env python3
"""Compare native GLM TP16 heads with the original padded CopySFA path."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch
import torch_npu  # noqa: F401


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/collect-vllm-sparse-profile-npu.py"
    spec = importlib.util.spec_from_file_location("copy4_helpers", path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    helper.load_upstream(root)
    torch.npu.set_device(0)
    torch.manual_seed(20261025)
    batch, q, hot, context = 16, 4, 12288, 64000
    host_kv = helper.HostBuffer((batch * (context // 128), 128, 512))
    host_rope = helper.HostBuffer((batch * (context // 128), 128, 64))
    graphs = []
    records = []
    try:
        kv_seed = torch.randn(batch * (hot // 128 + 1), 128, 1, 512, device="npu", dtype=torch.bfloat16) * .1
        rope_seed = torch.randn(batch * (hot // 128 + 1), 128, 1, 64, device="npu", dtype=torch.bfloat16) * .1
        kv, rope = kv_seed.clone(), rope_seed.clone()
        hbm_table = torch.arange(batch * (hot // 128 + 1), device="npu", dtype=torch.int32).view(batch, -1)
        dram_table = torch.arange(batch * (context // 128), device="npu", dtype=torch.int32).view(batch, -1)
        slots = torch.arange(2048, device="npu", dtype=torch.int32).view(1, 1, -1).repeat(batch * q, 1, 1)
        sources = slots.clone()
        query4 = torch.randn(batch * q, 4, 512, device="npu", dtype=torch.bfloat16) * .1
        rope4 = torch.randn(batch * q, 4, 64, device="npu", dtype=torch.bfloat16) * .1
        query8 = torch.zeros(batch * q, 8, 512, device="npu", dtype=torch.bfloat16)
        rope8 = torch.zeros(batch * q, 8, 64, device="npu", dtype=torch.bfloat16)
        query8[:, :4].copy_(query4)
        rope8[:, :4].copy_(rope4)
        misses = torch.zeros(batch * q, device="npu", dtype=torch.int32)
        union_sources = torch.full((batch, 32768), -1, device="npu", dtype=torch.int32)
        union_slots = torch.full_like(union_sources, -1)
        union_counts = torch.zeros(batch, device="npu", dtype=torch.int32)
        ends = torch.arange(1, batch + 1, device="npu", dtype=torch.int32) * q
        lengths = torch.full((batch,), hot + 128, device="npu", dtype=torch.int32)
        cache_tokens = torch.full_like(lengths, hot)
        out4, out8 = torch.empty_like(query4), torch.empty_like(query8)

        def call(query, qrope, output):
            torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention(
                qrope, query, ends, lengths, cache_tokens, slots, sources, misses,
                union_sources, union_slots, union_counts, hbm_table, dram_table,
                rope, kv, host_rope.device_view, host_kv.device_view, 1 / math.sqrt(576), output)

        def reset():
            kv.copy_(kv_seed)
            rope.copy_(rope_seed)

        for query, qrope, output in ((query4, rope4, out4), (query8, rope8, out8)):
            graphs.append(helper.capture(lambda query=query, qrope=qrope, output=output: call(query, qrope, output)))

        def padded_serving():
            query8.zero_()
            rope8.zero_()
            query8[:, :4].copy_(query4)
            rope8[:, :4].copy_(rope4)
            call(query8, rope8, out8)

        graphs.append(helper.capture(padded_serving))
        pointers = [t.data_ptr() for t in (query4, query8, slots, sources, misses, kv, rope, out4, out8)]
        for count in (0, 1, 31, 32, 63, 64, 127, 128, 400, 511, 512, 513, 2048):
            misses.fill_(count)
            union_counts.fill_(count)
            if count:
                union_sources[:, :count].copy_(torch.arange(count, device="npu", dtype=torch.int32))
                union_slots[:, :count].copy_(union_sources[:, :count])
            for _ in range(2):
                reset()
                graphs[1].replay()
                expected_kv, expected_rope = kv.clone(), rope.clone()
                reset()
                graphs[0].replay()
                torch.npu.synchronize()
                torch.testing.assert_close(kv, expected_kv, rtol=0, atol=0)
                torch.testing.assert_close(rope, expected_rope, rtol=0, atol=0)
                torch.testing.assert_close(out4.float(), out8[:, :4].float(), rtol=.02, atol=.002)
                assert pointers == [t.data_ptr() for t in (query4, query8, slots, sources, misses, kv, rope, out4, out8)]
            record = {"misses_per_row": count, "union_misses_per_request": count,
                      "note": "four rows share misses; intentional high-overlap control", "correctness": "passed"}
            if count in (0, 400):
                record["native4"] = helper.measure(graphs[0], reset, 50)
                record["padded8"] = helper.measure(graphs[1], reset, 50)
                record["padded8_with_preparation"] = helper.measure(graphs[2], reset, 50)
            records.append(record)
            print(json.dumps(record), flush=True)
        args.output.write_text(json.dumps({"status": "passed", "mtp": 3, "batch": batch, "cases": records}, indent=2) + "\n")
    finally:
        graphs.clear()
        torch.npu.synchronize()
        host_kv.close()
        host_rope.close()


if __name__ == "__main__":
    main()
