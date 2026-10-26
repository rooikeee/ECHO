#!/usr/bin/env python3
"""Validate MTP3 slot expansion at vector/tail/fallback boundaries."""

import argparse
import importlib.util
import json
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
    root = Path(__file__).resolve().parents[1]
    path = root / "vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_lightning_indexer_manage.py"
    spec = importlib.util.spec_from_file_location("lim_occurrence_upstream", path)
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    torch.npu.set_device(0)
    result = {"mtp": 3, "status": "running", "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for count in (1, 31, 32, 63, 64, 65, 127, 128, 511, 512, 513, 2048, 4095, 4096, 4097):
            case = upstream._build_occurrence_boundary_case(4, count)
            expected_cache, expected = upstream._run_and_assert(case)
            assert int(expected[2].sum()) == count
            assert int(expected[5][0]) == count
            cache = case.cache_seed.clone()
            outputs = upstream._make_outputs(case)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                upstream._call_op(case, cache, outputs)
            pointers = [t.data_ptr() for t in (cache, *outputs)]
            for _ in range(3):
                cache.copy_(case.cache_seed)
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
                for i in (0, 1, 2, 5):
                    torch.testing.assert_close(outputs[i], expected[i], rtol=0, atol=0)
                for i in (3, 4):
                    torch.testing.assert_close(outputs[i][0, :count], expected[i][0, :count], rtol=0, atol=0)
                assert pointers == [t.data_ptr() for t in (cache, *outputs)]
            graph.replay()
            torch.npu.synchronize()
            assert int(outputs[5][0]) == 0
            result["cases"].append({"occurrences": count, "reference": "native_topk_and_residency_checks",
                                    "graph_replays": 4, "persistent_all_hit": True})
            print(f"PASS MTP3 occurrences={count}", flush=True)
            graph = None
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
