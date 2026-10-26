"""CPU-only construction of explicitly controlled operator-profile residency."""

import torch


def controlled_resident_sources(topk, offload_tokens, hot_tokens, target_misses, seed,
                                request_union=False):
    """Control either request-union misses or misses in each query row.

    Union mode samples from the exact union, preserving any shared membership.
    Query mode removes only exclusive sources, giving union misses=Q*target.
    Neither mode is a measured model access pattern.
    """
    if topk.device.type != "cpu" or topk.ndim != 3:
        raise ValueError("topk must be a CPU [B,Q,K] tensor")
    capacity = topk.shape[1] * topk.shape[2] if request_union else topk.shape[2]
    if not 0 <= target_misses <= capacity:
        raise ValueError("target_misses exceeds selection capacity")
    if not 0 < hot_tokens < offload_tokens:
        raise ValueError("hot_tokens must be in (0,offload_tokens)")
    generator = torch.Generator().manual_seed(seed)
    residents = []
    for request in topk:
        frequency = torch.zeros(offload_tokens, dtype=torch.int32)
        for route in request:
            if bool(((route < 0) | (route >= offload_tokens)).any()) or route.unique().numel() != route.numel():
                raise ValueError("each TopK row must contain distinct legal sources")
            frequency[route.long()] += 1
        selected = frequency > 0
        missing = torch.zeros_like(selected)
        if request_union:
            union = torch.nonzero(selected).flatten()
            if union.numel() < target_misses:
                raise ValueError("target_misses exceeds the distinct request union")
            choice = union[torch.randperm(union.numel(), generator=generator)[:target_misses]]
            missing[choice] = True
        else:
            for route in request:
                exclusive = route.long()[frequency[route.long()] == 1]
                if exclusive.numel() < target_misses:
                    raise ValueError("not enough query-exclusive sources to control the requested miss count")
                choice = exclusive[torch.randperm(exclusive.numel(), generator=generator)[:target_misses]]
                missing[choice] = True
        resident = torch.nonzero(selected & ~missing).flatten()
        if resident.numel() > hot_tokens:
            raise ValueError("hot cache cannot hold the required TopK hits")
        filler = torch.nonzero(~selected).flatten()
        needed = hot_tokens - resident.numel()
        if filler.numel() < needed:
            raise ValueError("not enough unselected sources to fill the hot cache")
        filler = filler[torch.randperm(filler.numel(), generator=generator)[:needed]]
        resident = torch.cat((resident, filler))
        resident = resident[torch.randperm(hot_tokens, generator=generator)]
        mask = torch.zeros_like(selected)
        mask[resident] = True
        correct = (int((selected & ~mask).sum()) == target_misses if request_union else
                   bool(((~mask[request.long()]).sum(1) == target_misses).all()))
        if not correct:
            raise AssertionError("controlled residency construction failed")
        residents.append(resident)
    return torch.stack(residents)
