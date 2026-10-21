#!/usr/bin/env python3
"""Check the collective and graph path before loading the full model."""
import datetime
import os

import torch
import torch.distributed as dist
import torch_npu

rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(rank)
dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=90))
size = dist.get_world_size()
value = torch.empty((2, 7168), device=f"npu:{rank}", dtype=torch.bfloat16)
expected = size * (size + 1) // 2
for _ in range(5):
    value.fill_(rank + 1)
    dist.all_reduce(value)
    torch.npu.synchronize()
    assert (value.cpu() == expected).all()
print(f"rank {rank}: eager HCCL passed", flush=True)
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    dist.all_reduce(value)
for _ in range(3):
    value.fill_(rank + 1)
    graph.replay()
    torch.npu.synchronize()
    assert (value.cpu() == expected).all()
print(f"rank {rank}: graph HCCL passed", flush=True)
graph.reset()
dist.destroy_process_group()
