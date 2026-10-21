"""Exact, device-planned KV recall from mapped host memory.

All launches use fixed capacities. The owner must outlive every graph referring
to its mapped memory. Calls and graph replay must be ordered on one NPU stream;
host-side writes are only permitted after synchronizing that stream.
"""

import ctypes
import mmap
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def _acl():
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.aclrtHostRegister.restype = ctypes.c_int
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    return lib


class _MappedPointer:
    def __init__(self, owner):
        self.owner = owner
        self.dtype = owner.tensor.dtype

    def data_ptr(self):
        if self.owner.closed:
            raise RuntimeError("ECHO mapped host buffer is closed")
        return self.owner.device_address


class MappedHostBuffer:
    """Page-aligned, registered CPU allocation with an NPU-visible pointer."""

    def __init__(self, shape, dtype):
        import torch_npu  # noqa: F401

        torch.npu.init()
        self.device = torch.npu.current_device()
        self.closed = True
        count = 1
        for dimension in shape:
            if dimension <= 0:
                raise ValueError("ECHO host buffer dimensions must be positive")
            count *= dimension
        nbytes = count * torch.empty((), dtype=dtype).element_size()
        aligned_bytes = triton.cdiv(nbytes, mmap.PAGESIZE) * mmap.PAGESIZE
        self.allocation = mmap.mmap(-1, aligned_bytes)
        self.tensor = torch.frombuffer(
            self.allocation, dtype=dtype, count=count
        ).view(shape)
        pointer = ctypes.c_void_p()
        result = _acl().aclrtHostRegister(
            self.tensor.data_ptr(), aligned_bytes, 0, ctypes.byref(pointer)
        )
        if result != 0:
            raise RuntimeError(
                f"ECHO aclrtHostRegister failed ({result}); "
                "mapped host access is required, there is no eager fallback"
            )
        self.device_address = pointer.value
        self.closed = False
        self.pointer = _MappedPointer(self)

    def close(self):
        """Only close once no captured graph can be replayed with this buffer."""
        if not self.closed:
            with torch.npu.device(self.device):
                torch.npu.synchronize()
                result = _acl().aclrtHostUnregister(self.tensor.data_ptr())
            if result != 0:
                raise RuntimeError(f"ECHO aclrtHostUnregister failed ({result})")
            self.closed = True


@triton.jit
def _write_hot(
    loc, slots, key, rope, hot_key, hot_rope,
    HOST_SIZE: tl.constexpr, CAPACITY: tl.constexpr,
    KEY_DIM: tl.constexpr, ROPE_DIM: tl.constexpr,
    KEY_STRIDE: tl.constexpr, ROPE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    raw_index = tl.load(loc + row)
    slot = tl.load(slots + row)
    valid = (raw_index > 0) & (raw_index < HOST_SIZE) & (slot >= 0) & (slot < CAPACITY)
    if valid:
        offsets = tl.arange(0, BLOCK)
        k = tl.load(key + row * KEY_STRIDE + offsets, offsets < KEY_DIM, 0)
        r = tl.load(rope + row * ROPE_STRIDE + offsets, offsets < ROPE_DIM, 0)
        tl.store(hot_key + slot * KEY_DIM + offsets, k, offsets < KEY_DIM)
        tl.store(hot_rope + slot * ROPE_DIM + offsets, r, offsets < ROPE_DIM)


@triton.jit
def _offload_hot(
    loc, slots, hot_key, hot_rope, host, written,
    HOST_SIZE: tl.constexpr, CAPACITY: tl.constexpr,
    KEY_DIM: tl.constexpr, ROPE_DIM: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    raw_index = tl.load(loc + row)
    slot = tl.load(slots + row)
    valid = (raw_index > 0) & (raw_index < HOST_SIZE) & (slot >= 0) & (slot < CAPACITY)
    index = tl.where(valid, raw_index, 0).to(tl.int32)
    if valid:
        offsets = tl.arange(0, BLOCK)
        k = tl.load(hot_key + slot * KEY_DIM + offsets, offsets < KEY_DIM, 0)
        r = tl.load(hot_rope + slot * ROPE_DIM + offsets, offsets < ROPE_DIM, 0)
        tl.store(host + index * (KEY_DIM + ROPE_DIM) + offsets, k, offsets < KEY_DIM)
        tl.store(host + index * (KEY_DIM + ROPE_DIM) + KEY_DIM + offsets, r, offsets < ROPE_DIM)
        tl.store(written + index, 1)


@triton.jit
def _begin_plan(clock, errors, count):
    tl.store(clock, tl.load(clock) + 1)
    tl.store(errors, 0)
    tl.store(count, 0)


@triton.jit
def _latch_errors(errors, sticky_errors):
    tl.store(sticky_errors, tl.load(sticky_errors) | tl.load(errors))


@triton.jit
def _mark_required(indices, required, written, h2d, priority, clock, flags,
                   N: tl.constexpr, HOST_SIZE: tl.constexpr, BLOCK: tl.constexpr,
                   REQUIRE_WRITTEN: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw_index = tl.load(indices + offsets, offsets < N, -1)
    valid = (offsets < N) & (raw_index > 0) & (raw_index < HOST_SIZE)
    index = tl.where(valid, raw_index, 0).to(tl.int32)
    now = tl.load(clock)
    # Epoch tags avoid clearing the entire logical pool; duplicates store the
    # same epoch and priority, without atomics or CPU miss-count readback.
    tl.store(required + index, now, valid)
    slot = tl.load(h2d + index, valid, -1)
    safe_slot = tl.where(slot >= 0, slot, 0)
    tl.store(priority + safe_slot, now, valid & (slot >= 0))
    invalid = (offsets < N) & ((raw_index < -1) | (raw_index >= HOST_SIZE))
    flag = tl.full((), 0, tl.int32)
    if REQUIRE_WRITTEN:
        exists = tl.load(written + index, valid, 1) != 0
        flag = tl.where(tl.sum(invalid.to(tl.int32), 0) > 0, 1, 0)
        flag = flag | tl.where(tl.sum((valid & ~exists).to(tl.int32), 0) > 0, 2, 0)
    flag = flag | tl.where(tl.sum((valid & (slot < 0)).to(tl.int32), 0) > 0, 8, 0)
    tl.store(flags + tl.program_id(0), flag)


@triton.jit
def _plan_misses(required, h2d, d2h, priority, clock, flags, errors,
                 sticky_errors, count, recall_hosts, recall_slots, candidates,
                 free_slots, free_count_ptr,
                 HOST_SIZE: tl.constexpr, CAPACITY: tl.constexpr,
                 FLAGS: tl.constexpr, BLOCK: tl.constexpr,
                 FLAG_BLOCK: tl.constexpr):
    f = tl.arange(0, FLAG_BLOCK)
    status = tl.load(flags + f, f < FLAGS, 0)
    error = tl.where(tl.sum((status & 1 != 0).to(tl.int32), 0) > 0, 1, 0)
    error = error | tl.where(tl.sum((status & 2 != 0).to(tl.int32), 0) > 0, 2, 0)
    has_miss = tl.sum((status & 8 != 0).to(tl.int32), 0) > 0
    tl.store(errors, error)
    # This branch is evaluated by the captured device kernel on every replay.
    # All-hit calls do not scan the host ID space or allocate/evict slots.
    if (error == 0) & has_miss:
        now = tl.load(clock)
        offsets = tl.arange(0, BLOCK)
        misses = tl.full((), 0, tl.int32)
        hits = tl.full((), 0, tl.int32)
        for base in range(0, HOST_SIZE, BLOCK):
            host = base + offsets
            host_mask = (host > 0) & (host < HOST_SIZE)
            needed = tl.load(required + host, host_mask, 0) == now
            mapped = tl.load(h2d + host, host_mask, -1)
            missing = host_mask & needed & (mapped < 0)
            rank = misses + tl.cumsum(missing.to(tl.int32), 0) - 1
            active = missing & (rank < CAPACITY)
            safe_rank = tl.where(active, rank, 0)
            tl.store(recall_hosts + safe_rank, host, active)
            misses += tl.sum(missing.to(tl.int32), 0)
            hits += tl.sum((host_mask & needed & (mapped >= 0)).to(tl.int32), 0)
        if misses + hits > CAPACITY:
            tl.store(errors, 4)
        elif tl.load(sticky_errors) == 0:
            tl.store(count, misses)
            free_count = tl.load(free_count_ptr)
            take = tl.minimum(misses, free_count)
            for base in range(0, take, BLOCK):
                rank = base + offsets
                active = rank < take
                source = tl.where(active, free_count - take + rank, 0)
                slot = tl.load(free_slots + source, active, 0)
                safe_rank = tl.where(active, rank, 0)
                tl.store(recall_slots + safe_rank, slot, active)
            tl.store(free_count_ptr, free_count - take)
            if misses > take:
                for base in range(0, CAPACITY, BLOCK):
                    slot = base + offsets
                    slot_mask = slot < CAPACITY
                    resident = tl.load(d2h + slot, slot_mask, -1)
                    safe_host = tl.where(resident > 0, resident, 0)
                    protected = (resident > 0) & (tl.load(required + safe_host, slot_mask, 0) == now)
                    available = slot_mask & (resident > 0) & ~protected
                    ages = tl.load(priority + slot, slot_mask, -1)
                    tl.store(candidates + slot, tl.where(available, ages, 0x3FFFFFFFFFFFFFFF), slot_mask)
                # Only real evictions need victim selection. Preserve LRU
                # ordering with bounded reductions instead of a full sort.
                for rank in range(take, misses):
                    oldest = tl.full((), 0x3FFFFFFFFFFFFFFF, tl.int64)
                    victim = tl.full((), CAPACITY, tl.int32)
                    for base in range(0, CAPACITY, BLOCK):
                        slot = base + offsets
                        ages = tl.load(candidates + slot, slot < CAPACITY, 0x3FFFFFFFFFFFFFFF)
                        age = tl.min(ages, 0)
                        candidate = tl.min(tl.where(ages == age, slot, CAPACITY), 0)
                        better = (age < oldest) | ((age == oldest) & (candidate < victim))
                        victim = tl.where(better, candidate, victim)
                        oldest = tl.minimum(oldest, age)
                    tl.store(recall_slots + rank, victim)
                    tl.store(candidates + victim, 0x3FFFFFFFFFFFFFFF)


@triton.jit
def _plan_small_write(loc, h2d, d2h, priority, clock, errors, sticky_errors,
                      count, output, free_slots, free_count_ptr,
                      N: tl.constexpr, HOST_SIZE: tl.constexpr,
                      CAPACITY: tl.constexpr, INPUT_BLOCK: tl.constexpr,
                      BLOCK: tl.constexpr):
    rows = tl.arange(0, INPUT_BLOCK)
    raw = tl.load(loc + rows, rows < N, 0)
    valid = (rows < N) & (raw > 0) & (raw < HOST_SIZE)
    ids = tl.where(valid, raw, 0).to(tl.int32)
    mapped = tl.load(h2d + ids, valid, -1)
    misses = tl.sum((valid & (mapped < 0)).to(tl.int32), 0)
    now = tl.load(clock)
    safe_mapped = tl.where(mapped >= 0, mapped, 0)
    tl.store(priority + safe_mapped, now, valid & (mapped >= 0))
    error = tl.full((), 0, tl.int32)
    tl.store(errors, error)
    tl.store(count, misses)
    if (error == 0) & (tl.load(sticky_errors) == 0):
        free_count = tl.load(free_count_ptr)
        for row in tl.static_range(N):
            raw_id = tl.load(loc + row)
            valid_id = (raw_id > 0) & (raw_id < HOST_SIZE)
            host = tl.where(valid_id, raw_id, 0).to(tl.int32)
            target = tl.load(h2d + host, valid_id, -1)
            if valid_id & (target < 0):
                if free_count > 0:
                    free_count -= 1
                    target = tl.load(free_slots + free_count)
                else:
                    oldest = tl.full((), 0x3FFFFFFFFFFFFFFF, tl.int64)
                    target = tl.full((), CAPACITY, tl.int32)
                    for base in range(0, CAPACITY, BLOCK):
                        slot = base + tl.arange(0, BLOCK)
                        slot_mask = slot < CAPACITY
                        resident = tl.load(d2h + slot, slot_mask, -1)
                        protected = tl.full((BLOCK,), False, tl.int1)
                        for incoming in tl.static_range(N):
                            raw_incoming = tl.load(loc + incoming)
                            protected = protected | ((raw_incoming > 0) & (raw_incoming < HOST_SIZE) & (resident == raw_incoming))
                        ages = tl.load(priority + slot, slot_mask, -1)
                        ages = tl.where(slot_mask & ~protected, ages, 0x3FFFFFFFFFFFFFFF)
                        age = tl.min(ages, 0)
                        candidate = tl.min(tl.where(ages == age, slot, CAPACITY), 0)
                        better = (age < oldest) | ((age == oldest) & (candidate < target))
                        target = tl.where(better, candidate, target)
                        oldest = tl.minimum(oldest, age)
                old_host = tl.load(d2h + target)
                safe_old = tl.where(old_host > 0, old_host, 0)
                tl.store(h2d + safe_old, -1, old_host > 0)
                tl.store(h2d + host, target)
                tl.store(d2h + target, host)
                tl.store(priority + target, now)
            tl.store(output + row, target)
        tl.store(free_count_ptr, free_count)
    else:
        tl.store(output + rows, -1, rows < N)


@triton.jit
def _assign_slots(count, h2d, d2h, priority, recall_hosts, recall_slots,
                  errors, clock, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    if tl.load(errors) == 0:
        rank = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = (rank < CAPACITY) & (rank < tl.load(count))
        host = tl.load(recall_hosts + rank, active, 0)
        slot = tl.load(recall_slots + rank, active, 0)
        old = tl.load(d2h + slot, active, -1)
        safe_old = tl.where(old > 0, old, 0)
        tl.store(h2d + safe_old, -1, active & (old > 0))
        tl.store(h2d + host, slot, active)
        tl.store(d2h + slot, host, active)
        tl.store(priority + slot, tl.load(clock), active)


@triton.jit
def _copy_misses(host, key, rope, recall_hosts, recall_slots, count, errors,
                 KEY_DIM: tl.constexpr, ROPE_DIM: tl.constexpr,
                 WORKERS: tl.constexpr, BLOCK: tl.constexpr):
    if tl.load(errors) == 0:
        for row in range(tl.program_id(0), tl.load(count), WORKERS):
            index = tl.load(recall_hosts + row)
            slot = tl.load(recall_slots + row)
            offsets = tl.arange(0, BLOCK)
            k = tl.load(host + index * (KEY_DIM + ROPE_DIM) + offsets, offsets < KEY_DIM, 0)
            r = tl.load(host + index * (KEY_DIM + ROPE_DIM) + KEY_DIM + offsets, offsets < ROPE_DIM, 0)
            tl.store(key + slot * KEY_DIM + offsets, k, offsets < KEY_DIM)
            tl.store(rope + slot * ROPE_DIM + offsets, r, offsets < ROPE_DIM)


@triton.jit
def _resolve(indices, h2d, slots, errors, sticky_errors,
             N: tl.constexpr, HOST_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw_index = tl.load(indices + offsets, offsets < N, -1)
    error = tl.load(errors) | tl.load(sticky_errors)
    if error != 0:
        tl.store(sticky_errors, error)
    valid = (raw_index > 0) & (raw_index < HOST_SIZE) & (error == 0)
    index = tl.where(valid, raw_index, 0).to(tl.int32)
    slot = tl.load(h2d + index, (offsets < N) & valid, -1)
    tl.store(slots + offsets, slot, offsets < N)


@triton.jit
def _tag_release(loc, required, clock,
                 N: tl.constexpr, HOST_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw_host = tl.load(loc + offsets, offsets < N, -1)
    valid = (offsets < N) & (raw_host > 0) & (raw_host < HOST_SIZE)
    host = tl.where(valid, raw_host, 0).to(tl.int32)
    tl.store(required + host, tl.load(clock), valid)


@triton.jit
def _release_slots(required, clock, h2d, d2h, written, priority,
                   free_slots, free_count_ptr,
                   HOST_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    free_count = tl.load(free_count_ptr)
    now = tl.load(clock)
    for base in range(0, HOST_SIZE, BLOCK):
        host = base + offsets
        valid_host = (host > 0) & (host < HOST_SIZE)
        safe_host = tl.where(valid_host, host, 0)
        selected = valid_host & (tl.load(required + safe_host) == now)
        safe_host = tl.where(selected, host, 0)
        slot = tl.load(h2d + safe_host, selected, -1)
        resident = selected & (slot >= 0)
        safe_slot = tl.where(resident, slot, 0)
        rank = free_count + tl.cumsum(resident.to(tl.int32), 0) - 1
        safe_rank = tl.where(resident, rank, 0)
        tl.store(free_slots + safe_rank, slot, resident)
        free_count += tl.sum(resident.to(tl.int32), 0)
        tl.store(d2h + safe_slot, -1, resident)
        tl.store(priority + safe_slot, -1, resident)
        tl.store(h2d + safe_host, -1, selected)
        tl.store(written + safe_host, 0, selected)
    tl.store(free_count_ptr, free_count)


class EchoRecallCache:
    """One layer's complete host KV and bounded NPU cache, without DeepGEMM.

    Logical ID 0 is padding, -1 is an unselected entry, and -1 in the mapping
    means a miss. Cache slots are zero-based. Output slots are NOT attention
    sparse indices: the attention backend must adapt their addressing.
    """

    def __init__(self, host_size, capacity, key_dim=512, rope_dim=64,
                 dtype=torch.bfloat16, device="npu"):
        if host_size < 2 or capacity < 1 or key_dim < 1 or rope_dim < 1:
            raise ValueError("Invalid ECHO recall dimensions")
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("ECHO recall requires a floating-point KV dtype")
        if torch.device(device).type != "npu":
            raise ValueError("ECHO mapped recall is an NPU-only implementation")
        self.host_size = host_size
        self.capacity = capacity
        self.key_dim = key_dim
        self.rope_dim = rope_dim
        self.dtype = dtype
        self.device = device
        self.key = torch.zeros((capacity, 1, key_dim), dtype=dtype, device=device)
        self.rope = torch.zeros((capacity, 1, rope_dim), dtype=dtype, device=device)
        self.host_to_device = torch.full((host_size,), -1, dtype=torch.int32, device=device)
        self.device_to_host = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        self.written = torch.zeros(host_size, dtype=torch.int32, device=device)
        self.priority = torch.full((capacity,), -1, dtype=torch.int64, device=device)
        self.clock = torch.zeros(1, dtype=torch.int64, device=device)
        self.errors = torch.zeros(1, dtype=torch.int32, device=device)
        self.sticky_errors = torch.zeros_like(self.errors)
        self.required = torch.zeros(host_size, dtype=torch.int64, device=device)
        self._miss_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.plan_flags = torch.empty(
            triton.cdiv(max(host_size, capacity), 256), dtype=torch.int32, device=device,
        )
        self.candidates = torch.empty(capacity, dtype=torch.int64, device=device)
        self.free_slots = torch.arange(capacity, dtype=torch.int32, device=device)
        self.free_count = torch.full((1,), capacity, dtype=torch.int32, device=device)
        self.recall_hosts = torch.empty(capacity, dtype=torch.int32, device=device)
        self.recall_slots = torch.empty_like(self.recall_hosts)
        self.resolved_slots = torch.empty_like(self.recall_hosts)
        self.write_slots = torch.empty_like(self.recall_hosts)
        # Register last, so failed device allocations cannot leak registrations.
        with torch.npu.device(self.key.device):
            self.host = MappedHostBuffer((host_size, key_dim + rope_dim), dtype)

    @property
    def miss_count(self):
        return self._miss_count

    def write(self, logical_ids, key, rope):
        self._validate_ids(logical_ids)
        if not logical_ids.is_contiguous() or logical_ids.ndim != 1:
            raise ValueError("ECHO write IDs must be a contiguous vector")
        if key.shape != (logical_ids.numel(), 1, self.key_dim):
            raise ValueError("ECHO write key shape does not match logical IDs")
        if rope.shape != (logical_ids.numel(), 1, self.rope_dim):
            raise ValueError("ECHO write RoPE shape does not match logical IDs")
        if not key.is_contiguous() or not rope.is_contiguous():
            raise ValueError("ECHO write KV must be contiguous")
        if key.dtype != self.dtype or rope.dtype != self.dtype:
            raise ValueError("ECHO write KV dtype must match the cache")
        if key.device != self.key.device or rope.device != self.key.device:
            raise ValueError("ECHO write KV must be on the cache's NPU")
        for start in range(0, logical_ids.numel(), self.capacity):
            stop = min(start + self.capacity, logical_ids.numel())
            ids = logical_ids[start:stop]
            slots = self.write_slots[:stop - start]
            if ids.numel() <= 8:
                _begin_plan[(1,)](self.clock, self.errors, self._miss_count)
                _plan_small_write[(1,)](
                    ids, self.host_to_device, self.device_to_host, self.priority,
                    self.clock, self.errors, self.sticky_errors, self._miss_count,
                    slots, self.free_slots, self.free_count,
                    ids.numel(), self.host_size, self.capacity,
                    triton.next_power_of_2(ids.numel()),
                    min(512, triton.next_power_of_2(self.capacity)),
                )
            else:
                self._plan_slots(ids, require_written=False)
                _resolve[(triton.cdiv(ids.numel(), 256),)](
                    ids, self.host_to_device, slots, self.errors, self.sticky_errors,
                    ids.numel(), self.host_size, 256,
                )
            _write_hot[(ids.numel(),)](
                ids, slots, key[start:stop], rope[start:stop], self.key, self.rope,
                self.host_size, self.capacity, self.key_dim, self.rope_dim,
                key.stride(0), rope.stride(0),
                triton.next_power_of_2(max(self.key_dim, self.rope_dim)),
            )
            # Same-stream ordering backs up the hot KV before any slot reuse.
            self._offload(ids, slots)
            _latch_errors[(1,)](self.errors, self.sticky_errors)

    def _offload(self, logical_ids, slots):
        _offload_hot[(logical_ids.numel(),)](
            logical_ids, slots, self.key, self.rope, self.host.pointer, self.written,
            self.host_size, self.capacity, self.key_dim, self.rope_dim,
            triton.next_power_of_2(max(self.key_dim, self.rope_dim)),
        )

    def recall(self, logical_ids):
        self._validate_ids(logical_ids)
        if not logical_ids.is_contiguous():
            raise ValueError("ECHO recall IDs must be contiguous")
        if logical_ids.numel() == 0:
            raise ValueError("ECHO recall needs a nonempty fixed-capacity input")
        self._plan_slots(logical_ids, require_written=True)
        workers = min(self.capacity, 32)
        _copy_misses[(workers,)](
            self.host.pointer, self.key, self.rope, self.recall_hosts,
            self.recall_slots, self._miss_count, self.errors,
            self.key_dim, self.rope_dim, workers,
            triton.next_power_of_2(max(self.key_dim, self.rope_dim)),
        )
        # A graph-owned temporary is needed when input capacity exceeds cache
        # capacity (e.g. duplicate Top-K IDs). No shape depends on device data.
        output = (
            self.resolved_slots[:logical_ids.numel()].view(logical_ids.shape)
            if logical_ids.numel() <= self.capacity
            else torch.empty_like(logical_ids, dtype=torch.int32)
        )
        _resolve[(triton.cdiv(logical_ids.numel(), 256),)](
            logical_ids, self.host_to_device, output, self.errors, self.sticky_errors,
            logical_ids.numel(), self.host_size, 256,
        )
        return output

    def _plan_slots(self, logical_ids, require_written):
        _begin_plan[(1,)](self.clock, self.errors, self._miss_count)
        blocks = triton.cdiv(logical_ids.numel(), 256)
        flags = (
            self.plan_flags[:blocks] if blocks <= self.plan_flags.numel()
            else torch.empty(blocks, dtype=torch.int32, device=self.key.device)
        )
        _mark_required[(triton.cdiv(logical_ids.numel(), 256),)](
            logical_ids, self.required, self.written, self.host_to_device,
            self.priority, self.clock, flags,
            logical_ids.numel(), self.host_size, 256,
            REQUIRE_WRITTEN=require_written,
        )
        _plan_misses[(1,)](
            self.required, self.host_to_device, self.device_to_host, self.priority,
            self.clock, flags, self.errors, self.sticky_errors, self._miss_count,
            self.recall_hosts, self.recall_slots, self.candidates,
            self.free_slots, self.free_count,
            self.host_size, self.capacity, blocks,
            min(512, triton.next_power_of_2(max(self.host_size, self.capacity))),
            triton.next_power_of_2(blocks),
        )
        _assign_slots[(triton.cdiv(self.capacity, 256),)](
            self._miss_count, self.host_to_device, self.device_to_host,
            self.priority, self.recall_hosts,
            self.recall_slots, self.errors, self.clock, self.capacity, 256,
        )

    def release(self, logical_ids):
        self._validate_ids(logical_ids)
        if not logical_ids.is_contiguous():
            raise ValueError("ECHO release IDs must be contiguous")
        if logical_ids.numel():
            _begin_plan[(1,)](self.clock, self.errors, self._miss_count)
            _tag_release[(triton.cdiv(logical_ids.numel(), 256),)](
                logical_ids, self.required, self.clock,
                logical_ids.numel(), self.host_size, 256,
            )
            _release_slots[(1,)](
                self.required, self.clock, self.host_to_device, self.device_to_host,
                self.written, self.priority, self.free_slots, self.free_count,
                self.host_size, min(512, triton.next_power_of_2(self.host_size)),
            )

    def clear(self, preserve_written=False):
        self.host_to_device.fill_(-1)
        self.device_to_host.fill_(-1)
        if not preserve_written:
            self.written.zero_()
        self.priority.fill_(-1)
        self.clock.zero_()
        self.errors.zero_()
        self.sticky_errors.zero_()
        self.required.zero_()
        self._miss_count.zero_()
        self.free_slots.copy_(torch.arange(self.capacity, dtype=torch.int32, device=self.key.device))
        self.free_count.fill_(self.capacity)

    def check_errors(self):
        """Graph-external diagnostic; never call inside a captured region."""
        value = self.sticky_errors.item()
        if value:
            raise RuntimeError(
                f"ECHO recall failed (flags={value}: "
                "1=invalid ID, 2=unwritten KV, 4=insufficient cache capacity)"
            )

    def close(self):
        self.host.close()

    def _validate_ids(self, ids):
        if self.host.closed:
            raise RuntimeError("ECHO recall cache is closed")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("ECHO logical IDs must be int32 or int64")
        if ids.device != self.key.device:
            raise ValueError("ECHO logical IDs must be on the cache's NPU")


@triton.jit
def _map_topk(topk, block_table, cu_q, seq_lens, output,
              QUERIES: tl.constexpr, TOPK: tl.constexpr, BATCH: tl.constexpr,
              TABLE_STRIDE: tl.constexpr, PAGE_SIZE: tl.constexpr,
              BLOCK_BS: tl.constexpr, BLOCK_K: tl.constexpr):
    qid = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    requests = tl.arange(0, BLOCK_BS)
    ends = tl.load(cu_q + requests, requests < BATCH, 0x7FFFFFFF)
    batch = tl.sum((qid >= ends).to(tl.int32), 0)
    valid_batch = batch < BATCH
    safe_batch = tl.minimum(batch, BATCH - 1)
    begin = tl.load(cu_q + tl.maximum(safe_batch - 1, 0), valid_batch & (batch > 0), 0)
    end = tl.load(cu_q + safe_batch, valid_batch, 0)
    length = tl.load(seq_lens + safe_batch, valid_batch, 0)
    # NSA supplies logical causal positions. Preserve that filtering before
    # addressing attention with physical slots in independent query sequences.
    causal_position = length - (end - begin) + qid - begin
    index = tl.load(topk + qid * TOPK + columns, columns < TOPK, -1)
    page_column = index // PAGE_SIZE
    valid = valid_batch & (index >= 0) & (index < length)
    valid = valid & (index <= causal_position) & (page_column < TABLE_STRIDE)
    safe_column = tl.where(valid, page_column, 0)
    page = tl.load(block_table + safe_batch * TABLE_STRIDE + safe_column, valid, 0)
    host = page * PAGE_SIZE + index % PAGE_SIZE
    # Page zero is the framework's reserved padding page.
    host = tl.where(valid & (page > 0), host, -1)
    tl.store(output + qid * TOPK + columns, host, columns < TOPK)


def map_topk_to_host(topk, block_table, cu_query_lengths, kv_lengths, page_size):
    if topk.ndim == 3 and topk.shape[1] == 1:
        topk = topk[:, 0]
    if topk.ndim != 2:
        raise ValueError("ECHO NSA Top-K must have shape [queries, 1, K] or [queries, K]")
    topk = topk.contiguous()
    output = torch.empty_like(topk, dtype=torch.int32)
    # An idle DP rank can have no local queries during eager prefill. Do not
    # launch a zero-grid kernel or form batch index -1 for empty metadata.
    if topk.shape[0] == 0:
        return output
    if cu_query_lengths.numel() == 0:
        raise ValueError("ECHO nonempty queries require local request metadata")
    _map_topk[(topk.shape[0], triton.cdiv(topk.shape[1], 256))](
        topk, block_table, cu_query_lengths, kv_lengths, output,
        topk.shape[0], topk.shape[1], cu_query_lengths.numel(),
        block_table.stride(0), page_size,
        triton.next_power_of_2(cu_query_lengths.numel()), 256,
    )
    return output


def recalled_sparse_attention(cache, query, query_rope, host_ids, scale,
                              all_history_fits=False):
    """Run NSA over recalled slots, with one independent sequence per query.

    Original causal filtering happens in map_topk_to_host. RoPE is already
    applied. Readdressing selected KV does not change the dot products or
    softmax domain. No positional bias is supported by this adapter.
    """
    import torch_npu

    topk = host_ids.shape[-1]
    if query.shape[0] == 0:
        return torch.empty_like(query)
    if cache.capacity % 128 or cache.capacity < topk:
        raise ValueError("ECHO cache must be a multiple of 128 and at least NSA Top-K")
    if query.shape[0] != host_ids.shape[0] or query_rope.shape[0] != query.shape[0]:
        raise ValueError("ECHO query and selected KV row counts do not match")
    pages = cache.capacity // 128
    # The backend may prove this from CPU prefill lengths. Otherwise bound the
    # worst-case union by query_count * Top-K, without reading device results.
    max_queries = query.shape[0] if all_history_fits else cache.capacity // topk
    outputs = []
    for start in range(0, query.shape[0], max_queries):
        stop = min(start + max_queries, query.shape[0])
        rows = stop - start
        slots = cache.recall(host_ids[start:stop].contiguous())
        output, _, _ = torch_npu.npu_sparse_flash_attention(
            query=query[start:stop].contiguous(),
            key=cache.key.view(pages, 128, 1, cache.key_dim),
            value=cache.key.view(pages, 128, 1, cache.key_dim),
            query_rope=query_rope[start:stop].contiguous(),
            key_rope=cache.rope.view(pages, 128, 1, cache.rope_dim),
            sparse_indices=slots.view(rows, 1, topk),
            scale_value=scale,
            actual_seq_lengths_query=torch.arange(1, rows + 1, dtype=torch.int32, device=query.device),
            actual_seq_lengths_kv=torch.full((rows,), cache.capacity, dtype=torch.int32, device=query.device),
            block_table=torch.arange(pages, dtype=torch.int32, device=query.device).expand(rows, pages).contiguous(),
            sparse_block_size=1,
            layout_query="TND", layout_kv="PA_BSND",
            sparse_mode=3, attention_mode=2, return_softmax_lse=False,
        )
        outputs.append(output)
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
