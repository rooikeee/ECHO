"""Validate shared cache addressing with active/padded MTP rows across Graph replay."""

import importlib.util
import sys
from pathlib import Path

import torch
from vllm_ascend.utils import adapt_patch

adapt_patch(True)
adapt_patch(False)

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.shared_hot_layout import (
    SharedHotLayout,
)


def test_graph_switches_real_and_dummy_rows_without_aliasing():
    path = (
        Path(__file__).resolve().parents[1]
        / "vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_copy_sfa_offload_metadata.py"
    )
    spec = importlib.util.spec_from_file_location("shared_metadata_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    torch.npu.set_device(0)
    builder = mod.make_builder()
    layout = SharedHotLayout(2, 4, 8192, 128)
    builder.shared_hot_layout = layout
    cache = torch.full((layout.allocation_rows * layout.stride, 2), -7.0, device="npu")
    values = torch.arange(16, device="npu", dtype=torch.float32).reshape(8, 2)
    cm = mod.common([4, 8], [10371, 10371], pools=(0, 1))
    metadata = mod.populate(builder, cm)
    slots = metadata.copy_sfa_device_slots[:8]
    pointer = slots.data_ptr()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        cache.index_copy_(0, slots, values)
    for pools, generations in [
        ((0, 1), (11, 12)),
        ((1, 0), (12, 11)),
        ((0, 1), (-1, 12)),
        ((1, 0), (13, -1)),
        ((0, 1), (14, 15)),
    ]:
        cm = mod.common([4, 8], [10371, 10371], pools=pools, generations=generations)
        metadata = mod.populate(builder, cm)
        assert metadata.copy_sfa_device_slots.data_ptr() == pointer
        indices = slots.cpu().long()
        assert indices.unique().numel() == 8
        assert indices.min() >= 0 and indices.max() < cache.shape[0]
        table = metadata.copy_sfa_hbm_block_table.cpu().long()
        assert table.min() >= 0 and (table.max() + 1) * 128 <= cache.shape[0]
        for _ in range(3):
            cache.fill_(-7)
            graph.replay()
            torch.npu.synchronize()
            expected = torch.full(cache.shape, -7.0)
            expected[indices] = values.cpu()
            torch.testing.assert_close(cache.cpu(), expected, atol=0, rtol=0)
