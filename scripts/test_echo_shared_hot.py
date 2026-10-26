from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from echo_decode_bench_connector import EchoSharedDecodeBenchConnector
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.shared_hot_layout import (
    SharedHotLayout,
)


def test_shared_layout_keeps_request_and_graph_padding_disjoint():
    layout = SharedHotLayout(16, 32, 12288, 128)
    real = layout.token_bases(np.arange(16))
    dummy = layout.token_bases(np.arange(32, 64))
    assert len(set(real.tolist() + dummy.tolist())) == 48
    assert real[-1] + layout.stride <= dummy[0]
    assert dummy[-1] + layout.dummy_stride <= layout.allocation_rows * layout.stride
    for draft_step in range(4):
        assert layout.token_bases(np.array([5] * 4)).tolist() == [5 * layout.stride] * 4
    with pytest.raises(ValueError, match="exhausted"):
        layout.token_bases([16])


def test_resident_admission_waits_and_resumes_after_request_finishes():
    c = object.__new__(EchoSharedDecodeBenchConnector)
    c.resident_capacity = 1
    c.resident_owners = {"A"}
    c.connector_scheduler = Mock()
    c.connector_scheduler.get_num_new_matched_tokens.return_value = (63999, False)
    a = SimpleNamespace(request_id="A")
    b = SimpleNamespace(request_id="B")
    assert c.get_num_new_matched_tokens(b, 0) == (None, False)
    assert c.get_num_new_matched_tokens(a, 0) == (63999, False)
    c.request_finished_all_groups(a, ([],))
    assert c.get_num_new_matched_tokens(b, 0) == (63999, False)
