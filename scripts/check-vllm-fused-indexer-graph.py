#!/usr/bin/env python3
"""Compare repeated fused-indexer Graph replay against eager execution."""

import importlib.util
from pathlib import Path
import sys

import torch
import torch_npu  # noqa: F401


@torch.inference_mode()
def main():
    root = Path(__file__).resolve().parents[1]
    path = root / (
        "vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/"
        "test_fused_lightning_indexer_manage.py"
    )
    spec = importlib.util.spec_from_file_location("upstream_indexer_test", path)
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    torch.npu.set_device(0)
    for batch in (1, 2):
        case = upstream._build_case(
            q_values=[1] * batch,
            states=[upstream.REQUEST_STATE_STEADY] * batch,
            offload_len=8320,
            cache_tokens=8192,
        )
        cache = case.cache_seed.clone()
        outputs = upstream._make_outputs(case)
        upstream._call_op(case, cache, outputs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            upstream._call_op(case, cache, outputs)
        addresses = [t.data_ptr() for t in (cache, *outputs)]
        for state in (-2, -1, -3, -1):
            case.request_state.fill_(state)
            case.query.mul_(-1)
            expected_cache = case.cache_seed.clone()
            expected_outputs = upstream._make_outputs(case)
            upstream._call_op(case, expected_cache, expected_outputs)
            cache.copy_(case.cache_seed)
            for actual, blank in zip(outputs, upstream._make_outputs(case)):
                actual.copy_(blank)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(cache.cpu(), expected_cache.cpu(), rtol=0, atol=0)
            for actual, expected in zip(outputs, expected_outputs):
                torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
            assert addresses == [t.data_ptr() for t in (cache, *outputs)]
        print(f"PASS batch={batch}: 4 state/input-changing replays", flush=True)


if __name__ == "__main__":
    main()
