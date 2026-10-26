#!/usr/bin/env python3
"""Check the isolated runtime; this does not validate ECHO or model inference."""

import importlib.metadata
import json
import os

import torch
import torch_npu  # noqa: F401


def main():
    results = []
    torch.npu.set_device(0)
    for batch_size in (1, 2, 4, 8):
        source = torch.zeros((batch_size, 16), device="npu")
        output = torch.empty_like(source)
        output.copy_(source * 2 + 1)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        addresses = (source.data_ptr(), output.data_ptr())
        with torch.npu.graph(graph):
            output.copy_(source * 2 + 1)
        for value in (0, 3, -2, 7):
            source.fill_(value)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(
                output.cpu(), torch.full((batch_size, 16), value * 2 + 1.0)
            )
            assert addresses == (source.data_ptr(), output.data_ptr())
        results.append({"batch_size": batch_size, "replays": 4})
    print(json.dumps({
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "cann": os.environ.get("ASCEND_HOME_PATH"),
        "torch": torch.__version__,
        "torch_npu": importlib.metadata.version("torch-npu"),
        "runtime_graph_tests": results,
    }, indent=2))


if __name__ == "__main__":
    main()
