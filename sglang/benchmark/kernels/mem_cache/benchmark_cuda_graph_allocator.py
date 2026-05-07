import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

import torch

# Allow running this benchmark without installing sglang as a package.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.srt.mem_cache.allocator import (  # type: ignore[reportMissingImports]  # noqa: E402
    CudaGraphTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)


def _parse_int_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer value.")
    return values


def _parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    for value in values:
        if not 0.0 < value <= 1.0:
            raise argparse.ArgumentTypeError("free ratios must be in (0, 1].")
    return values


def _bench_wall_time_us(fn: Callable[[], None], rep: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1e6 / rep


def _make_perm(pool_size: int, num_free: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randperm(pool_size, device="cuda", generator=generator)[:num_free]


def _num_free_from_ratio(pool_size: int, free_ratio: float) -> int:
    return max(1, min(pool_size, int(round(pool_size * free_ratio))))


def _prepare_token_allocator(
    pool_size: int,
    free_ratio: float,
    free_perm: torch.Tensor,
    need_sort: bool,
) -> TokenToKVPoolAllocator:
    allocator = TokenToKVPoolAllocator(
        pool_size,
        torch.bfloat16,
        "cuda",
        None,
        need_sort,
    )

    num_free = _num_free_from_ratio(pool_size, free_ratio)
    if num_free == pool_size:
        return allocator

    allocated = allocator.alloc(pool_size)
    if allocated is None or allocated.numel() != pool_size:
        raise RuntimeError("Failed to reserve the full pool for token allocator setup.")
    allocator.free(allocated[free_perm].contiguous())

    if allocator.available_size() != num_free:
        raise RuntimeError(
            f"Token allocator setup mismatch: expected {num_free}, got {allocator.available_size()}."
        )
    return allocator


def _prepare_cuda_graph_allocator(
    pool_size: int,
    free_ratio: float,
    free_perm: torch.Tensor,
) -> CudaGraphTokenToKVPoolAllocator:
    allocator = CudaGraphTokenToKVPoolAllocator(
        pool_size,
        torch.bfloat16,
        "cuda",
        None,
        False,
    )

    num_free = _num_free_from_ratio(pool_size, free_ratio)
    if num_free == pool_size:
        return allocator

    full_size = torch.tensor([pool_size], dtype=torch.int32, device="cuda")
    full_alloc_buf = torch.zeros((pool_size,), dtype=torch.int32, device="cuda")
    free_size = torch.tensor([num_free], dtype=torch.int32, device="cuda")

    allocator.alloc(full_size, full_alloc_buf)
    allocator.free(full_alloc_buf[free_perm].contiguous(), free_size)

    available_size = int(allocator.available_size().item())
    if available_size != num_free:
        raise RuntimeError(
            f"CudaGraph allocator setup mismatch: expected {num_free}, got {available_size}."
        )
    return allocator


# The full set of device-resident tensors that together encode the internal
# state of a CudaGraphTokenToKVPoolAllocator. Any alloc/free call mutates
# these tensors in place, so we clone/copy them to snapshot and restore.
_CUDA_GRAPH_ALLOCATOR_STATE_FIELDS = (
    "free_pages",
    "free_stack",
    "available_size_tensor",
    "last_alloc_size_tensor",
    "alloc_counter",
)


def _snapshot_cuda_graph_allocator(
    allocator: CudaGraphTokenToKVPoolAllocator,
) -> dict[str, torch.Tensor]:
    """Clone every persistent tensor of the allocator so we can restore it later."""
    torch.cuda.synchronize()
    return {
        name: getattr(allocator, name).detach().clone()
        for name in _CUDA_GRAPH_ALLOCATOR_STATE_FIELDS
    }


def _restore_cuda_graph_allocator(
    allocator: CudaGraphTokenToKVPoolAllocator,
    snapshot: dict[str, torch.Tensor],
) -> None:
    """Overwrite the allocator's persistent tensors from a previously taken snapshot.

    We use copy_ (instead of reassigning attributes) so that any kernel launch
    that baked in raw data_ptr values (notably a captured CUDA graph) keeps
    pointing at the same storage.
    """
    torch.cuda.synchronize()
    for name in _CUDA_GRAPH_ALLOCATOR_STATE_FIELDS:
        getattr(allocator, name).copy_(snapshot[name])
    # Invalidate the cached alloc-buffer data_ptr so that the next real alloc
    # call re-runs its initialisation logic.
    allocator._last_alloc_buffer_data_ptr = None
    torch.cuda.synchronize()


def _assert_cuda_graph_available_size(
    allocator: CudaGraphTokenToKVPoolAllocator,
    expected: int,
    context: str,
) -> None:
    actual = int(allocator.available_size().item())
    if actual != expected:
        raise RuntimeError(
            f"CudaGraph allocator state drift detected {context}: "
            f"expected available_size={expected}, got {actual}."
        )


def _bench_token_allocator(
    pool_size: int,
    alloc_size: int,
    free_ratio: float,
    rep: int,
    warmup: int,
    need_sort: bool,
    seed: int,
) -> float:
    num_free = _num_free_from_ratio(pool_size, free_ratio)
    if alloc_size > num_free:
        raise ValueError(
            f"alloc_size={alloc_size} exceeds available free slots={num_free} "
            f"for pool_size={pool_size}, free_ratio={free_ratio}."
        )

    free_perm = _make_perm(pool_size, num_free, seed)
    allocator = _prepare_token_allocator(pool_size, free_ratio, free_perm, need_sort)

    def run() -> None:
        allocated = allocator.alloc(alloc_size)
        if allocated is None:
            raise RuntimeError("Token allocator returned None during benchmark.")
        allocator.free(allocated)

    return _bench_wall_time_us(run, rep=rep, warmup=warmup)


def _bench_cuda_graph_allocator_direct(
    pool_size: int,
    alloc_size: int,
    free_ratio: float,
    rep: int,
    warmup: int,
    seed: int,
) -> float:
    num_free = _num_free_from_ratio(pool_size, free_ratio)
    if alloc_size > num_free:
        raise ValueError(
            f"alloc_size={alloc_size} exceeds available free slots={num_free} "
            f"for pool_size={pool_size}, free_ratio={free_ratio}."
        )

    free_perm = _make_perm(pool_size, num_free, seed)
    allocator = _prepare_cuda_graph_allocator(pool_size, free_ratio, free_perm)
    need_size = torch.tensor([alloc_size], dtype=torch.int32, device="cuda")
    alloc_buf = torch.zeros((pool_size,), dtype=torch.int32, device="cuda")

    # Snapshot the steady-state so warmup / rep loops cannot drift the pool.
    snapshot = _snapshot_cuda_graph_allocator(allocator)
    _assert_cuda_graph_available_size(
        allocator, num_free, context="before direct benchmark"
    )

    def run() -> None:
        allocator.alloc(need_size, alloc_buf)
        allocator.free(alloc_buf, need_size)

    try:
        elapsed = _bench_wall_time_us(run, rep=rep, warmup=warmup)
    finally:
        _restore_cuda_graph_allocator(allocator, snapshot)

    _assert_cuda_graph_available_size(
        allocator, num_free, context="after direct benchmark restore"
    )
    return elapsed


def _bench_cuda_graph_allocator_replay(
    pool_size: int,
    alloc_size: int,
    free_ratio: float,
    rep: int,
    warmup: int,
    seed: int,
) -> float:
    num_free = _num_free_from_ratio(pool_size, free_ratio)
    if alloc_size > num_free:
        raise ValueError(
            f"alloc_size={alloc_size} exceeds available free slots={num_free} "
            f"for pool_size={pool_size}, free_ratio={free_ratio}."
        )

    free_perm = _make_perm(pool_size, num_free, seed)
    allocator = _prepare_cuda_graph_allocator(pool_size, free_ratio, free_perm)
    need_size = torch.tensor([alloc_size], dtype=torch.int32, device="cuda")
    alloc_buf = torch.zeros((pool_size,), dtype=torch.int32, device="cuda")

    # torch.cuda.graph capture actually launches the kernels, so it mutates the
    # allocator's persistent tensors. Snapshot the steady-state BEFORE capture
    # and restore it right after so the captured graph starts every replay
    # from the intended (num_free) fragmentation configuration.
    pre_capture_snapshot = _snapshot_cuda_graph_allocator(allocator)
    _assert_cuda_graph_available_size(
        allocator, num_free, context="before capture"
    )

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        allocator.alloc(need_size, alloc_buf)
        allocator.free(alloc_buf, need_size)
    torch.cuda.synchronize()

    _restore_cuda_graph_allocator(allocator, pre_capture_snapshot)
    _assert_cuda_graph_available_size(
        allocator, num_free, context="after capture restore"
    )

    # Each replay also writes into the same tensors; while a well-formed
    # alloc+free pair should be state-neutral, we still snapshot once more
    # and restore after the timed section as a defensive guard against any
    # asymmetric side-effects.
    replay_snapshot = _snapshot_cuda_graph_allocator(allocator)

    def run() -> None:
        graph.replay()

    try:
        elapsed = _bench_wall_time_us(run, rep=rep, warmup=warmup)
    finally:
        _restore_cuda_graph_allocator(allocator, replay_snapshot)

    _assert_cuda_graph_available_size(
        allocator, num_free, context="after replay benchmark restore"
    )
    return elapsed


def run_benchmarks(
    pool_sizes: list[int],
    alloc_sizes: list[int],
    free_ratios: list[float],
    rep: int,
    warmup: int,
    need_sort: bool,
    include_cuda_graph_python_dispatch: bool,
    capture_graph: bool,
) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    rows: list[dict[str, object]] = []
    seed = 1

    for pool_size in pool_sizes:
        for free_ratio in free_ratios:
            num_free = _num_free_from_ratio(pool_size, free_ratio)
            for alloc_size in alloc_sizes:
                if alloc_size > num_free:
                    continue

                token_us = _bench_token_allocator(
                    pool_size=pool_size,
                    alloc_size=alloc_size,
                    free_ratio=free_ratio,
                    rep=rep,
                    warmup=warmup,
                    need_sort=need_sort,
                    seed=seed,
                )
                rows.append(
                    {
                        "allocator": "TokenToKVPoolAllocator",
                        "mode": "python_dispatch",
                        "pool_size": pool_size,
                        "free_ratio": free_ratio,
                        "free_slots": num_free,
                        "alloc_size": alloc_size,
                        "avg_us": token_us,
                    }
                )

                if include_cuda_graph_python_dispatch:
                    cuda_graph_direct_us = _bench_cuda_graph_allocator_direct(
                        pool_size=pool_size,
                        alloc_size=alloc_size,
                        free_ratio=free_ratio,
                        rep=rep,
                        warmup=warmup,
                        seed=seed,
                    )
                    rows.append(
                        {
                            "allocator": "CudaGraphTokenToKVPoolAllocator",
                            "mode": "python_dispatch",
                            "pool_size": pool_size,
                            "free_ratio": free_ratio,
                            "free_slots": num_free,
                            "alloc_size": alloc_size,
                            "avg_us": cuda_graph_direct_us,
                            "speedup_vs_token": token_us / cuda_graph_direct_us,
                        }
                    )

                if capture_graph:
                    cuda_graph_replay_us = _bench_cuda_graph_allocator_replay(
                        pool_size=pool_size,
                        alloc_size=alloc_size,
                        free_ratio=free_ratio,
                        rep=rep,
                        warmup=warmup,
                        seed=seed,
                    )
                    rows.append(
                        {
                            "allocator": "CudaGraphTokenToKVPoolAllocator",
                            "mode": "cuda_graph_replay",
                            "pool_size": pool_size,
                            "free_ratio": free_ratio,
                            "free_slots": num_free,
                            "alloc_size": alloc_size,
                            "avg_us": cuda_graph_replay_us,
                            "speedup_vs_token": token_us / cuda_graph_replay_us,
                        }
                    )
                seed += 1

    return rows


def print_results(rows: list[dict[str, object]]) -> None:
    print(
        "allocator,mode,pool_size,free_ratio,free_slots,alloc_size,avg_us,speedup_vs_token"
    )
    for row in rows:
        speedup = row.get("speedup_vs_token")
        speedup_str = "" if speedup is None else f"{float(speedup):.3f}"
        print(
            f"{row['allocator']},{row['mode']},{row['pool_size']},{float(row['free_ratio']):.4f},"
            f"{row['free_slots']},{row['alloc_size']},{float(row['avg_us']):.3f},{speedup_str}"
        )


def save_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "allocator",
        "mode",
        "pool_size",
        "free_ratio",
        "free_slots",
        "alloc_size",
        "avg_us",
        "speedup_vs_token",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TokenToKVPoolAllocator vs CudaGraphTokenToKVPoolAllocator "
            "under matched pool size, allocation size, and fragmentation."
        )
    )
    parser.add_argument(
        "--pool-sizes",
        type=_parse_int_list,
        default=[300_000],
        help="Comma-separated pool sizes. Default: 131072,262144",
    )
    parser.add_argument(
        "--alloc-sizes",
        type=_parse_int_list,
        default=[64, 256, 1024, 4096],
        help="Comma-separated allocation sizes. Default: 64,256,1024,4096",
    )
    parser.add_argument(
        "--free-ratios",
        type=_parse_float_list,
        default=[0.1, 0.5, 0.9],
        help="Comma-separated steady-state free ratios in (0, 1]. Default: 0.1,0.5,0.9",
    )
    parser.add_argument("--rep", type=int, default=500, help="Benchmark repetitions.")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations.")
    parser.add_argument(
        "--need-sort",
        action="store_true",
        help="Benchmark TokenToKVPoolAllocator with need_sort=True.",
    )
    parser.add_argument(
        "--no-capture-graph",
        action="store_true",
        help="Skip replay benchmarking via torch.cuda.CUDAGraph.",
    )
    parser.add_argument(
        "--include-cuda-graph-python-dispatch",
        action="store_true",
        help="Also benchmark CudaGraphTokenToKVPoolAllocator without CUDA graph replay.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional csv output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_benchmarks(
        pool_sizes=args.pool_sizes,
        alloc_sizes=args.alloc_sizes,
        free_ratios=args.free_ratios,
        rep=args.rep,
        warmup=args.warmup,
        need_sort=args.need_sort,
        include_cuda_graph_python_dispatch=args.include_cuda_graph_python_dispatch,
        capture_graph=not args.no_capture_graph,
    )
    print(f"# device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print_results(rows)
    if args.csv is not None:
        save_csv(rows, args.csv)
        print(f"Saved benchmark csv to {args.csv}")


if __name__ == "__main__":
    main()
# python benchmark/kernels/mem_cache/benchmark_cuda_graph_allocator.py --need-sort --free-ratios 0.9 > alloc_bench.csv
