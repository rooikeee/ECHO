#!/usr/bin/env python3
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import torch_npu
import sgl_kernel_npu


def main():
    if not torch.npu.is_available():
        raise RuntimeError("No visible NPU; check the driver and device permissions")
    torch.npu.set_device(0)
    torch.manual_seed(42)
    left = torch.randn(32, 64).to(torch.bfloat16)
    right = torch.randn(64, 32).to(torch.bfloat16)
    expected = (left.float() @ right.float()).to(torch.bfloat16)
    actual = (left.npu() @ right.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    torch.npu.synchronize()
    report = {
        "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "device": torch.npu.get_device_name(0),
        "cann": os.environ["ASCEND_HOME_PATH"],
        "atb": os.environ["ATB_HOME_PATH"],
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch", "torch-npu", "triton-ascend", "sglang", "sgl-kernel-npu"
            )
        },
        "bf16_matmul": "passed",
        "lightning_indexer_api": hasattr(torch_npu, "npu_lightning_indexer"),
    }
    result_path = Path(__file__).resolve().parents[1] / "build" / "npu-environment.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
