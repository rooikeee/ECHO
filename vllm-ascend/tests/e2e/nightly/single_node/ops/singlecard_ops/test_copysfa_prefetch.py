# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Input contract for the optional in-CopySFA prefetch consumer.

Numerical, changing-Graph, and actual staging-source checks are exercised by
scripts/collect-echo-prefetch-profile-npu.py --prefetch-consumer copysfa
--profile-scope operators --check-staging-source --attention-reference.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend import vllm_ascend_C  # noqa: F401


@pytest.mark.parametrize(
    "invalid", ["partial", "map_dtype", "map_rank", "capacity", "rope_shape", "payload_dtype", "cpu", "noncontiguous"]
)
def test_copysfa_rejects_invalid_prefetch_inputs(invalid):
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    batch, rows, slots = 1, 4, 8192
    ckv = torch.zeros(65, 128, 1, 512, dtype=torch.bfloat16, device=device)
    kpe = torch.zeros(65, 128, 1, 64, dtype=torch.bfloat16, device=device)
    query = torch.zeros(rows, 4, 512, dtype=torch.bfloat16, device=device)
    query_rope = torch.zeros(rows, 4, 64, dtype=torch.bfloat16, device=device)
    lengths = torch.tensor([rows], dtype=torch.int32, device=device)
    kv_lengths = torch.tensor([slots + 128], dtype=torch.int32, device=device)
    cache = torch.tensor([slots], dtype=torch.int32, device=device)
    topk = torch.zeros(rows, 1, 2048, dtype=torch.int32, device=device)
    misses = torch.zeros(rows, dtype=torch.int32, device=device)
    union = torch.zeros(batch, 32768, dtype=torch.int32, device=device)
    counts = torch.zeros(batch, dtype=torch.int32, device=device)
    table = torch.arange(65, dtype=torch.int32, device=device).view(1, -1)
    args = (
        query_rope,
        query,
        lengths,
        kv_lengths,
        cache,
        topk,
        topk,
        misses,
        union,
        union,
        counts,
        table,
        table,
        kpe,
        ckv,
        kpe.view(65, 128, 64),
        ckv.view(65, 128, 512),
        0.1,
        torch.empty_like(query),
    )
    options = {
        "prefetch_map": torch.zeros(rows, 16384, dtype=torch.int32, device=device),
        "prefetch_kv": torch.empty(rows, 64, 512, dtype=query.dtype, device=device),
        "prefetch_rope": torch.empty(rows, 64, 64, dtype=query.dtype, device=device),
    }
    if invalid == "partial":
        options.pop("prefetch_rope")
    elif invalid == "map_dtype":
        options["prefetch_map"] = options["prefetch_map"].long()
    elif invalid == "map_rank":
        options["prefetch_map"] = options["prefetch_map"].view(-1)
    elif invalid == "capacity":
        options["prefetch_kv"] = options["prefetch_kv"][:, :0]
    elif invalid == "rope_shape":
        options["prefetch_rope"] = options["prefetch_rope"][:, :, :32].contiguous()
    elif invalid == "payload_dtype":
        options["prefetch_kv"] = options["prefetch_kv"].float()
    elif invalid == "cpu":
        options["prefetch_map"] = options["prefetch_map"].cpu()
    elif invalid == "noncontiguous":
        options["prefetch_map"] = options["prefetch_map"][:, ::2]
    with pytest.raises(RuntimeError, match="[Pp]refetch"):
        torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention(*args, **options)
