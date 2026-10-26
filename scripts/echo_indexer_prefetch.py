"""Experimental ECHO decode prefetch for the pinned Ascend fusion operators.

The owner must outlive every referring graph. Do not use its internal descriptor
directly or under torch.compile: pointer targets are not dispatcher-visible.
Host inputs must own registered device-alias views, never ordinary CPU tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _match(Sources, Misses, Prefetched, Counts, States, Match,
           P: tl.constexpr, PP: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    p = tl.arange(0, PP)
    source = tl.load(Sources + b * 2048 + i)
    candidate = tl.load(Prefetched + b * P + p, p < P, -1)
    count = tl.load(Counts + b)
    misses = tl.load(Misses + b)
    steady = tl.load(States + b) == -1
    equal = ((source[:, None] == candidate[None, :]) &
             (p[None, :] < count) & (i[:, None] < misses) &
             (source[:, None] >= 0) & steady)
    index = tl.max(tl.where(equal, p[None, :], -1), axis=1)
    tl.store(Match + b * 2048 + i, index)


@triton.jit
def _commit(Match, Slots, StagingKV, StagingRope, HbmKV, HbmRope,
            HbmTable, P: tl.constexpr, TABLE_WIDTH: tl.constexpr):
    b = tl.program_id(0)
    p = tl.program_id(1)
    i = tl.arange(0, 2048)
    match = tl.load(Match + b * 2048 + i)
    position = tl.max(tl.where(match == p, i, -1), axis=0)
    if position >= 0:
        slot = tl.load(Slots + b * 2048 + position)
        block = tl.load(HbmTable + b * TABLE_WIDTH + slot // 128)
        row = block * 128 + slot % 128
        d = tl.arange(0, 512)
        kv = tl.load(StagingKV + (b * P + p) * 512 + d)
        tl.store(HbmKV + row * 512 + d, kv)
        r = tl.arange(0, 64)
        rope = tl.load(StagingRope + (b * P + p) * 64 + r)
        tl.store(HbmRope + row * 64 + r, rope)


@triton.jit
def _classify(Misses, Match, Remaining, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    missing = i < tl.load(Misses + b)
    matched = tl.load(Match + b * 2048 + i) >= 0
    tl.store(Remaining + b * 2048 + i, (missing & ~matched).to(tl.int32))


@triton.jit
def _partition_indices(Misses, Remaining, Prefix, Permutation, OutMisses, Used):
    b = tl.program_id(0)
    i = tl.arange(0, 2048)
    old_misses = tl.load(Misses + b)
    remaining = tl.load(Remaining + b * 2048 + i) != 0
    prefix = tl.load(Prefix + b * 2048 + i)
    new_misses = tl.load(Prefix + b * 2048 + 2047)
    destination = tl.where(remaining, prefix - 1, new_misses + i - prefix)
    tl.store(Permutation + b * 2048 + i, destination.to(tl.int64))
    tl.store(OutMisses + b, new_misses)
    tl.store(Used + b, old_misses - new_misses)


@triton.jit
def _finish_metadata(Sources, Slots, Misses, States, Counts,
                     OutMissSources, OutMissSlots, OutCounts, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    new_misses = tl.load(Misses + b)
    changed = (tl.load(States + b) == -1) & (tl.load(Counts + b) > 0)
    source = tl.load(Sources + b * 2048 + i)
    slot = tl.load(Slots + b * 2048 + i)
    tl.store(OutMissSources + b * 32768 + i, source, changed & (i < new_misses))
    tl.store(OutMissSlots + b * 32768 + i, slot, changed & (i < new_misses))
    if tl.program_id(1) == 0:
        tl.store(OutCounts + b, new_misses, changed)


class EchoIndexerPrefetch:
    """Fixed-address bounded prefetch state; base consumer is single-query only.

    thresholds starts disabled and receives the current TopK cutoff after each
    Indexer call for prediction on the next step. It is not an accuracy
    threshold: unpredicted misses fall back to CopySFA. The caller may override
    thresholds/budgets in place between replays.
    """

    def __init__(self, case, host_kv, host_rope, dram_table, capacity=64):
        self.host_owners = (host_kv, host_rope)
        host_kv, host_rope = host_kv.device_view, host_rope.device_view
        batch = case.index_block_table.shape[0]
        device = case.query.device
        dtype = case.query.dtype
        total_queries = case.query.shape[0]
        if not batch <= total_queries <= 14 * batch or not 1 <= capacity <= 2048:
            raise ValueError("Requires B <= T <= 14B and capacity in [1, 2048].")
        if (host_kv.shape[1:] != (128, 512) or
                host_rope.shape != (host_kv.shape[0], 128, 64) or
                host_kv.dtype != dtype or host_rope.dtype != dtype or
                not host_kv.is_contiguous() or not host_rope.is_contiguous()):
            raise ValueError("Host aliases must be contiguous matching-dtype paged CKV/KPE views.")
        if (dram_table.ndim != 2 or dram_table.shape[0] != batch or
                dram_table.dtype != torch.int32 or dram_table.device != device or
                not dram_table.is_contiguous()):
            raise ValueError("DRAM block table must be contiguous NPU int32 [B, blocks].")
        self.capacity = capacity
        self.case = case
        self.offload_lengths = case.offload_seq_lengths_key
        self.host_kv = host_kv
        self.host_rope = host_rope
        self.dram_table = dram_table
        self.thresholds = torch.full((total_queries,), float("inf"), dtype=torch.float32, device=device)
        self.budgets = torch.full((total_queries,), capacity, dtype=torch.int32, device=device)
        self.counts = torch.zeros(total_queries, dtype=torch.int32, device=device)
        self.sources = torch.full((total_queries, capacity), -1, dtype=torch.int32, device=device)
        self.kv = torch.empty(total_queries, capacity, 512, dtype=dtype, device=device)
        self.rope = torch.empty(total_queries, capacity, 64, dtype=dtype, device=device)
        self.ready = torch.zeros(total_queries, case.cache_seed.shape[1], dtype=torch.int32, device=device)
        self.matches = torch.empty(total_queries, 1, 2048, dtype=torch.int32, device=device)
        self.remaining = torch.empty(total_queries, 2048, dtype=torch.int32, device=device)
        self.prefix = torch.empty_like(self.remaining)
        self.permutation = torch.empty(total_queries, 2048, dtype=torch.int64, device=device)
        self.used = torch.zeros(total_queries, dtype=torch.int32, device=device)
        fields = [self.thresholds.data_ptr(), self.budgets.data_ptr(),
                  self.counts.data_ptr(), self.sources.data_ptr(), self.kv.data_ptr(),
                  self.ready.data_ptr(), host_kv.data_ptr(), host_rope.data_ptr(),
                  dram_table.data_ptr(), self.rope.data_ptr(), dram_table.shape[1],
                  host_kv.shape[0], capacity, case.offload_seq_lengths_key.data_ptr()]
        self.context = torch.tensor([fields] * total_queries, dtype=torch.int64, device=device)
        self.buffers = (self.thresholds, self.budgets, self.counts, self.sources,
                        self.kv, self.rope, self.ready, dram_table,
                        case.offload_seq_lengths_key)
        self.outputs = None

    def reset(self):
        """Capture this before Indexer; no CPU branch or allocation per replay."""
        self.ready.zero_()
        self.counts.zero_()
        self.sources.fill_(-1)

    def indexer(self, cache, outputs):
        """Reset replay-local state and run exact Indexer with bounded prefetch."""
        if cache.shape != self.case.cache_seed.shape:
            raise ValueError("Cache shape must match the bound prefetch state.")
        case = self.case
        if case.offload_seq_lengths_key is not self.offload_lengths:
            raise ValueError("Update sequence lengths in place; do not replace their storage.")
        self.reset()
        torch.ops._C_ascend.npu_fused_lightning_indexer_manage(
            case.index_weights, case.query_dequant_scale, case.query,
            case.index_key_dequant_scale, case.index_key_cache, case.index_block_table,
            case.actual_seq_lengths_query, case.actual_seq_lengths_key,
            case.offload_seq_lengths_key, case.num_cache_tokens, case.request_state,
            case.req_pool_entries, cache, *outputs, prefetch_context=self.context,
        )

    def allocate_outputs(self, outputs):
        """Call before capture, not inside a graph or hot decode loop."""
        self.outputs = tuple(torch.empty_like(tensor) for tensor in outputs)
        return self.outputs

    def consume(self, outputs, hbm_table, hbm_kv, hbm_rope):
        """Commit useful prefetch, then partition remaining Host misses first."""
        if self.outputs is None:
            raise RuntimeError("allocate_outputs must run before capture.")
        if self.case.query.shape[0] != self.case.index_block_table.shape[0]:
            raise ValueError("Multi-query consume requires EchoMTPIndexerPrefetch.")
        sources, slots, misses, miss_sources, miss_slots, counts = outputs
        out_sources, out_slots, out_misses, out_miss_sources, out_miss_slots, out_counts = self.outputs
        out_miss_sources.copy_(miss_sources)
        out_miss_slots.copy_(miss_slots)
        out_counts.copy_(counts)
        batch = sources.shape[0]
        _match[(batch, 16)](sources, misses, self.sources, self.counts,
                           self.case.request_state, self.matches,
                           self.capacity, triton.next_power_of_2(self.capacity), 128)
        _commit[(batch, self.capacity)](self.matches, slots, self.kv, self.rope,
                                       hbm_kv, hbm_rope, hbm_table,
                                       self.capacity, hbm_table.shape[1])
        _classify[(batch, 16)](misses, self.matches, self.remaining, 128)
        # This Triton version lowers scans through host-initialized locks;
        # the native out= scan keeps GLOBAL-mode capture free of host copies.
        torch.cumsum(self.remaining, dim=1, dtype=torch.int32, out=self.prefix)
        _partition_indices[(batch,)](misses, self.remaining, self.prefix,
                                     self.permutation, out_misses, self.used)
        # Discrete Triton scatter also uses host-initialized locks on this
        # compiler. Native fixed-output scatter avoids that capture limitation.
        out_sources.view(batch, 2048).scatter_(1, self.permutation, sources.view(batch, 2048))
        out_slots.view(batch, 2048).scatter_(1, self.permutation, slots.view(batch, 2048))
        _finish_metadata[(batch, 16)](out_sources, out_slots, out_misses,
                                      self.case.request_state, self.counts,
                                      out_miss_sources, out_miss_slots, out_counts, 128)
        return self.outputs


@triton.jit
def _mtp_match_rows(Sources, Misses, Prefetched, Counts, States, Match, Remaining,
                    P: tl.constexpr, Q: tl.constexpr, WIDTH: tl.constexpr,
                    UNION: tl.constexpr, PP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row if UNION else row // Q
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    p = tl.arange(0, PP)
    misses = tl.load(Misses + row)
    # Skip candidate comparisons for padding, but define every prefix-scan cell.
    result = tl.full((BLOCK,), -1, tl.int32)
    if tl.program_id(1) * BLOCK < misses and tl.load(States + b) == -1:
        source = tl.load(Sources + row * WIDTH + i, i < misses, -1)
        candidate = tl.load(Prefetched + b * Q * P + p, p < Q * P, -1)
        count = tl.load(Counts + b * Q + p // P, p < Q * P, 0)
        equal = ((source[:, None] == candidate[None, :]) &
                 (p[None, :] < Q * P) & (p[None, :] % P < count[None, :]) &
                 (i[:, None] < misses) & (source[:, None] >= 0))
        # Lowest pooled candidate wins, including duplicates across query rows.
        result = tl.min(tl.where(equal, p[None, :], 2147483647), axis=1)
        result = tl.where(result == 2147483647, -1, result)
    tl.store(Match + row * WIDTH + i, result)
    tl.store(Remaining + row * WIDTH + i, ((i < misses) & (result < 0)).to(tl.int32))


@triton.jit
def _mtp_commit_union(Match, Slots, StagingKV, StagingRope, HbmKV, HbmRope,
                      HbmTable, P: tl.constexpr, Q: tl.constexpr,
                      TABLE_WIDTH: tl.constexpr, SCAN: tl.constexpr):
    b, p = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, SCAN)
    match = tl.load(Match + b * 32768 + i)
    position = tl.max(tl.where(match == p, i, -1), axis=0)
    if position >= 0:
        slot = tl.load(Slots + b * 32768 + position)
        block = tl.load(HbmTable + b * TABLE_WIDTH + slot // 128)
        row = block * 128 + slot % 128
        d, r = tl.arange(0, 512), tl.arange(0, 64)
        candidate = b * Q * P + p
        kv = tl.load(StagingKV + candidate * 512 + d)
        rope = tl.load(StagingRope + candidate * 64 + r)
        tl.store(HbmKV + row * 512 + d, kv)
        tl.store(HbmRope + row * 64 + r, rope)


@triton.jit
def _mtp_partition(Misses, Remaining, Prefix, Permutation, OutMisses, Used,
                   WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    prefix = tl.load(Prefix + row * WIDTH + i)
    remaining = tl.load(Remaining + row * WIDTH + i) != 0
    count = tl.load(Prefix + row * WIDTH + WIDTH - 1)
    destination = tl.where(remaining, prefix - 1, count + i - prefix)
    tl.store(Permutation + row * WIDTH + i, destination.to(tl.int64))
    if tl.program_id(1) == 0:
        tl.store(OutMisses + row, count)
        tl.store(Used + row, tl.load(Misses + row) - count)


class EchoMTPIndexerPrefetch(EchoIndexerPrefetch):
    """Uniform four-query MTP3 target verification; budget remains per query.

    Pool candidates within a request, commit each union source exactly once,
    then partition both query-level and request-union miss lists. Target query
    rows retain independent predictions; first-fill requests never speculate.
    """

    def __init__(self, case, host_kv, host_rope, dram_table, capacity=64):
        if any(q != 4 for q in case.q_values) or capacity != 64:
            raise ValueError("MTP3 target verification requires four rows and capacity=64.")
        super().__init__(case, host_kv, host_rope, dram_table, capacity)
        batch, device = len(case.q_values), case.query.device
        self.union_matches = torch.empty(batch, 32768, dtype=torch.int32, device=device)
        self.union_remaining = torch.empty_like(self.union_matches)
        self.union_prefix = torch.empty_like(self.union_matches)
        self.union_permutation = torch.empty(batch, 32768, dtype=torch.int64, device=device)
        self.union_used = torch.zeros(batch, dtype=torch.int32, device=device)
        self.union_scan = triton.next_power_of_2(max(4 * 2048, max(case.cache_tokens)))
        if self.union_scan > 32768:
            raise ValueError("Request union exceeds supported capacity.")
        self.buffers += (self.union_matches, self.union_remaining, self.union_prefix,
                         self.union_permutation, self.union_used)

    def consume(self, outputs, hbm_table, hbm_kv, hbm_rope):
        if self.outputs is None:
            raise RuntimeError("allocate_outputs must run before capture.")
        sources, slots, misses, union_sources, union_slots, union_counts = outputs
        out_sources, out_slots, out_misses, out_union_sources, out_union_slots, out_union_counts = self.outputs
        batch, queries = union_sources.shape[0], sources.shape[0]
        pp = triton.next_power_of_2(4 * self.capacity)
        _mtp_match_rows[(batch, 1024)](
            union_sources, union_counts, self.sources, self.counts, self.case.request_state,
            self.union_matches, self.union_remaining, self.capacity, 4, 32768, True, pp, 32)
        _mtp_commit_union[(batch, 4 * self.capacity)](
            self.union_matches, union_slots, self.kv, self.rope, hbm_kv, hbm_rope,
            hbm_table, self.capacity, 4, hbm_table.shape[1], self.union_scan)
        _mtp_match_rows[(queries, 64)](
            sources, misses, self.sources, self.counts, self.case.request_state,
            self.matches, self.remaining, self.capacity, 4, 2048, False, pp, 32)
        torch.cumsum(self.remaining, dim=1, dtype=torch.int32, out=self.prefix)
        torch.cumsum(self.union_remaining, dim=1, dtype=torch.int32, out=self.union_prefix)
        _mtp_partition[(queries, 16)](misses, self.remaining, self.prefix,
            self.permutation, out_misses, self.used, 2048, 128)
        _mtp_partition[(batch, 256)](union_counts, self.union_remaining, self.union_prefix,
            self.union_permutation, out_union_counts, self.union_used, 32768, 128)
        out_sources.view(queries, 2048).scatter_(1, self.permutation, sources.view(queries, 2048))
        out_slots.view(queries, 2048).scatter_(1, self.permutation, slots.view(queries, 2048))
        out_union_sources.scatter_(1, self.union_permutation, union_sources)
        out_union_slots.scatter_(1, self.union_permutation, union_slots)
        return self.outputs
