"""CPU checks for synthetic Ascend cache setup; NPU replay is checked separately."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from echo_decode_bench_connector import EchoDecodeBenchConnector
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnectorScheduler,
)


def test_cache_layouts_and_noop_transfer():
    connector = object.__new__(EchoDecodeBenchConnector)
    connector.connector_worker = Mock()
    caches = {
        "ordinary": (torch.ones(3, 2), torch.ones(3, 1)),
        "offload": (None, None, torch.ones(3, 2), torch.ones(3, 1), torch.ones(2, 2), torch.ones(2, 1)),
        "indexer": torch.ones(3, 4),
    }
    tensors = [t for c in caches.values() for t in ((c,) if isinstance(c, torch.Tensor) else c) if t is not None]
    pointers = [t.data_ptr() for t in tensors]
    with patch("echo_decode_bench_connector.get_tp_group") as tp, patch.object(torch.npu, "synchronize") as sync:
        connector.register_kv_caches(caches)
        tp.return_value.barrier.assert_called_once()
        sync.assert_called_once()
    assert all(torch.count_nonzero(t) == 0 for t in tensors)
    tensors[0].fill_(7)
    connector.start_load_kv(None)
    assert torch.all(tensors[0] == 7), "fake transfer must not erase ongoing decode state"
    assert pointers == [t.data_ptr() for t in tensors]


def test_external_prompt_accounting():
    scheduler = object.__new__(DecodeBenchConnectorScheduler)
    scheduler._filled_requests = set()
    req = SimpleNamespace(request_id="one", num_tokens=100001)
    assert scheduler.get_num_new_matched_tokens(req, 0) == (100000, False)
    assert scheduler.get_num_new_matched_tokens(req, 128) == (99872, False)
    scheduler._filled_requests.add("one")
    assert scheduler.get_num_new_matched_tokens(req, 0) == (0, False)
    scheduler.request_finished(req)
    assert scheduler.get_num_new_matched_tokens(req, 0) == (100000, False)
