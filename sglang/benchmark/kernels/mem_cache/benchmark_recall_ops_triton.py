import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
import triton

# Allow running this benchmark without installing sglang as a package.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.srt.mem_cache.recall_ops import (  # type: ignore[reportMissingImports]  # noqa: E402
    I32_MAX,
    _recall_update_kernel,
    free_update,
    get_free_loc,
    protect,
    recall_update,
    recall_update_extend,
    set_mla_kv_buffer_cuda_graph,
    set_update,
)
from sglang.srt.mem_cache.allocator import (  # type: ignore[reportMissingImports]  # noqa: E402
    free_kernel,
    update_alloc_kernel,
)


def _bench_us(fn: Callable[[], None], rep: int, warmup: int) -> float:
    ms = triton.testing.do_bench(fn, rep=rep, warmup=warmup)
    return float(ms * 1000.0)


def _make_rng(seed: int) -> torch.Generator:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g


def _bench_protect(rep: int, warmup: int, quick: bool) -> List[Tuple[str, Dict[str, int], float]]:
    pool_size = 131072 if quick else 262144
    cfgs = [(8, 128), (32, 256)] if quick else [(8, 128), (32, 256), (64, 512)]
    rng = _make_rng(1)
    rows = []

    priority = torch.full((pool_size + 1,), -1, dtype=torch.int32, device="cuda")
    priority[0] = I32_MAX
    fifo_counter = torch.tensor([1], dtype=torch.int32, device="cuda")

    for bsz, topk in cfgs:
        protected_index = torch.randint(
            1,
            pool_size,
            (bsz, topk),
            dtype=torch.int32,
            device="cuda",
            generator=rng,
        )

        def run():
            protect(priority, protected_index, fifo_counter)

        us = _bench_us(run, rep=rep, warmup=warmup)
        rows.append(("protect", {"pool_size": pool_size, "bsz": bsz, "topk": topk}, us))
    return rows


def _bench_get_free_loc(rep: int, warmup: int, quick: bool) -> List[Tuple[str, Dict[str, int], float]]:
    pool_size = 131072 if quick else 262144
    free_sizes = [64, 256] if quick else [64, 256, 1024]
    max_priority = 10_000_000
    rng = _make_rng(2)
    rows = []

    priority_template = torch.randint(
        0,
        max_priority + 1,
        (pool_size + 1,),
        dtype=torch.int32,
        device="cuda",
        generator=rng,
    )
    priority_template[0] = I32_MAX
    work_priority = priority_template.clone()
    loc = torch.zeros((pool_size + 1,), dtype=torch.int32, device="cuda")

    for free_size_v in free_sizes:
        free_size = torch.tensor([free_size_v], dtype=torch.int32, device="cuda")

        # Correctness check: the selected indices must correspond to the
        # free_size_v smallest priority values (ties broken arbitrarily).
        work_priority.copy_(priority_template)
        get_free_loc(work_priority, loc, free_size)
        picked = loc[:free_size_v].long()
        picked_values = priority_template[picked].sort().values
        ref_values, _ = torch.topk(priority_template, free_size_v, largest=False, sorted=True)
        ref_values = ref_values.sort().values
        assert torch.equal(picked_values, ref_values), (
            f"get_free_loc correctness failed at free_size={free_size_v}: "
            f"picked {picked_values[:8].tolist()} vs ref {ref_values[:8].tolist()}"
        )

        # Kernel-only timing: exclude the caller-side clone of the priority
        # buffer. The clone is a 4*(pool_size+1) bytes D->D memcpy (~1 MiB at
        # pool_size=262144) whose cost is bound by HBM bandwidth and is
        # orthogonal to the selection kernel. Re-seed `work_priority` once
        # before the timed loop; the kernel does not mutate `priority`, so
        # every iteration sees the same input.
        work_priority.copy_(priority_template)

        def run_kernel():
            get_free_loc(work_priority, loc, free_size)

        us_kernel = _bench_us(run_kernel, rep=rep, warmup=warmup)
        rows.append(
            ("get_free_loc", {"pool_size": pool_size, "free_size": free_size_v}, us_kernel)
        )
    return rows


def _bench_free_update(rep: int, warmup: int, quick: bool) -> List[Tuple[str, Dict[str, int], float]]:
    device_pool_size = 131072 if quick else 262144
    host_pool_size = 524288 if quick else 1048576
    free_sizes = [64, 256] if quick else [64, 256, 1024]
    rng = _make_rng(3)
    rows = []

    # free_update expects sort_loc[idx] to be a valid device index (>0).
    # Keep shape [device_pool_size + 1] to mirror runtime buffers, but fill the
    # active candidate region with indices in [1, device_pool_size].
    sort_loc_template = torch.zeros((device_pool_size + 1,), dtype=torch.int32, device="cuda")
    sort_loc_template[:device_pool_size] = (
        torch.randperm(device_pool_size, device="cuda", generator=rng, dtype=torch.int64).to(torch.int32)
        + 1
    )

    device_to_host_template = torch.full(
        (device_pool_size + 1,), I32_MAX, dtype=torch.int32, device="cuda"
    )
    host_to_device_template = torch.full(
        (host_pool_size + 1,), I32_MAX, dtype=torch.int32, device="cuda"
    )
    host_indices = torch.randint(
        1,
        host_pool_size,
        (device_pool_size,),
        dtype=torch.int32,
        device="cuda",
        generator=rng,
    )
    device_idx = torch.arange(1, device_pool_size + 1, dtype=torch.int32, device="cuda")
    device_to_host_template[1:] = host_indices
    host_to_device_template[host_indices.long()] = device_idx

    priority_template = torch.randint(
        0,
        1_000_000,
        (device_pool_size + 1,),
        dtype=torch.int32,
        device="cuda",
        generator=rng,
    )
    priority_template[0] = I32_MAX

    free_index_device = torch.zeros((device_pool_size,), dtype=torch.int32, device="cuda")
    sort_loc = sort_loc_template.clone()
    device_to_host = device_to_host_template.clone()
    host_to_device = host_to_device_template.clone()
    priority = priority_template.clone()

    for free_size_v in free_sizes:
        free_size = torch.tensor([free_size_v], dtype=torch.int32, device="cuda")

        def run():
            sort_loc.copy_(sort_loc_template)
            device_to_host.copy_(device_to_host_template)
            host_to_device.copy_(host_to_device_template)
            priority.copy_(priority_template)
            free_update(
                free_size,
                free_index_device,
                sort_loc,
                device_to_host,
                priority,
                host_to_device,
            )

        us = _bench_us(run, rep=rep, warmup=warmup)
        rows.append(
            (
                "free_update",
                {
                    "device_pool_size": device_pool_size,
                    "host_pool_size": host_pool_size,
                    "free_size": free_size_v,
                },
                us,
            )
        )
    return rows


def _bench_free_kernel(rep: int, warmup: int, quick: bool) -> List[Tuple[str, Dict[str, int], float]]:
    pool_size = 131072 if quick else 262144
    free_sizes = [64, 256] if quick else [64, 256, 1024]
    rng = _make_rng(9)
    rows = []

    free_pages_template = torch.zeros((pool_size + 1,), dtype=torch.bool, device="cuda")
    free_index_buf = (
        torch.randperm(pool_size, device="cuda", generator=rng, dtype=torch.int64).to(torch.int32) + 1
    )
    block = 256

    for free_size_v in free_sizes:
        free_size = torch.tensor([free_size_v], dtype=torch.int32, device="cuda")

        def run():
            free_pages_template.zero_()
            free_kernel[(triton.cdiv(pool_size, block),)](
                free_index_buf,
                free_size,
                free_pages_template,
                BLOCK=block,
            )

        us = _bench_us(run, rep=rep, warmup=warmup)
        rows.append(("free_kernel", {"pool_size": pool_size, "free_size": free_size_v}, us))
    return rows


def _bench_update_alloc_kernel(
    rep: int, warmup: int, quick: bool
) -> List[Tuple[str, Dict[str, int], float]]:
    pool_size = 131072 if quick else 262144
    need_sizes = [64, 256, 1024] if quick else [64, 256, 1024, 4096]
    free_ratios = [0.1, 0.5] if quick else [0.1, 0.5, 0.9]
    rng = _make_rng(8)
    rows = []

    free_pages_template = torch.zeros((pool_size + 1,), dtype=torch.bool, device="cuda")
    alloc_counter = torch.zeros((1,), dtype=torch.uint32, device="cuda")
    device_pool_loc_alloc = torch.zeros((pool_size,), dtype=torch.int32, device="cuda")

    for free_ratio in free_ratios:
        free_pages_template.fill_(0)
        num_free = max(need_sizes) if quick else max(need_sizes)
        num_free = max(num_free, int(pool_size * free_ratio))
        num_free = min(num_free, pool_size)
        free_indices = torch.randperm(pool_size, device="cuda", generator=rng)[:num_free] + 1
        free_pages_template[free_indices] = 1

        for need_size_v in need_sizes:
            need_size = torch.tensor([need_size_v], dtype=torch.int32, device="cuda")
            block = 256

            def run_kernel():
                alloc_counter.zero_()
                device_pool_loc_alloc.zero_()
                update_alloc_kernel[(triton.cdiv(pool_size, block),)](
                    need_size,
                    alloc_counter,
                    device_pool_loc_alloc,
                    free_pages_template[1:],
                    pool_size,
                    BLOCK=block,
                )

            us_kernel = _bench_us(run_kernel, rep=rep, warmup=warmup)
            rows.append(
                (
                    "update_alloc_kernel",
                    {
                        "pool_size": pool_size,
                        "need_size": need_size_v,
                        "free_ratio_pct": int(free_ratio * 100),
                    },
                    us_kernel,
                )
            )
    return rows


def _bench_set_update(rep: int, warmup: int, quick: bool) -> List[Tuple[str, Dict[str, int], float]]:
    host_pool_size = 524288 if quick else 1048576
    device_pool_size = 131072 if quick else 262144
    need_sizes = [1024, 4096] if quick else [1024, 4096, 16384]
    rng = _make_rng(4)
    rows = []

    for need_size in need_sizes:
        loc = torch.randint(
            1,
            host_pool_size,
            (need_size,),
            dtype=torch.int32,
            device="cuda",
            generator=rng,
        )
        # Keep some zero entries to represent padding path.
        if need_size >= 8:
            loc[: need_size // 8] = 0

        device_pool_loc = torch.randint(
            1,
            device_pool_size,
            (need_size,),
            dtype=torch.int32,
            device="cuda",
            generator=rng,
        )
        if need_size >= 8:
            device_pool_loc[: need_size // 8] = 0

        device_token_to_host = torch.full(
            (device_pool_size + 1,), I32_MAX, dtype=torch.int32, device="cuda"
        )
        host_token_to_device = torch.full(
            (host_pool_size + 1,), I32_MAX, dtype=torch.int32, device="cuda"
        )

        def run():
            device_token_to_host.fill_(I32_MAX)
            host_token_to_device.fill_(I32_MAX)
            set_update(loc, device_pool_loc, device_token_to_host, host_token_to_device)

        us = _bench_us(run, rep=rep, warmup=warmup)
        rows.append(
            (
                "set_update",
                {
                    "need_size": need_size,
                    "host_pool_size": host_pool_size,
                    "device_pool_size": device_pool_size,
                },
                us,
            )
        )
    return rows


def _bench_set_mla_kv_buffer(rep: int, warmup: int, quick: bool) -> List[Tuple[str, Dict[str, int], float]]:
    device_pool_size = 131072 if quick else 262144
    need_sizes = [1024, 4096] if quick else [1024, 4096, 16384]
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    rng = _make_rng(5)
    rows = []

    kv_buffer = torch.randn(
        (device_pool_size + 1, kv_lora_rank + qk_rope_head_dim),
        device="cuda",
        dtype=torch.float16,
        generator=rng,
    )

    for need_size_v in need_sizes:
        need_size = torch.tensor([need_size_v], dtype=torch.int32, device="cuda")
        loc = torch.randint(
            1,
            device_pool_size,
            (need_size_v,),
            dtype=torch.int32,
            device="cuda",
            generator=rng,
        )
        cache_k_nope = torch.randn(
            (need_size_v, kv_lora_rank), device="cuda", dtype=torch.float16, generator=rng
        )
        cache_k_rope = torch.randn(
            (need_size_v, qk_rope_head_dim),
            device="cuda",
            dtype=torch.float16,
            generator=rng,
        )

        def run():
            set_mla_kv_buffer_cuda_graph(
                need_size,
                kv_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
            )

        us = _bench_us(run, rep=rep, warmup=warmup)
        rows.append(
            (
                "set_mla_kv_buffer_cuda_graph",
                {"need_size": need_size_v, "loc_capacity": need_size_v},
                us,
            )
        )

        # CUDA-graph replay path: loc points at the full allocator buffer while
        # only the first `need_size` entries are live. This is the shape seen in
        # memory_pool_host.set_mla_kv_buffer().
        loc_graph = torch.zeros((device_pool_size,), dtype=torch.int32, device="cuda")
        loc_graph[:need_size_v] = loc

        def run_graph():
            set_mla_kv_buffer_cuda_graph(
                need_size,
                kv_buffer,
                loc_graph,
                cache_k_nope,
                cache_k_rope,
            )

        us_graph = _bench_us(run_graph, rep=rep, warmup=warmup)
        rows.append(
            (
                "set_mla_kv_buffer_cuda_graph",
                {"need_size": need_size_v, "loc_capacity": device_pool_size},
                us_graph,
            )
        )
    return rows


def _bench_recall_update(rep: int, warmup: int, quick: bool) -> List[Tuple[str, Dict[str, int], float]]:
    host_pool_size = 524288 if quick else 1048576
    device_pool_size = 131072 if quick else 262144
    cfgs = [(128, 64), (512, 128)] if quick else [(128, 64), (512, 128), (1024, 256)]
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    rng = _make_rng(6)
    rows = []

    recall_device_indices = torch.arange(
        1, device_pool_size + 1, dtype=torch.int32, device="cuda"
    )
    host_token_to_device = torch.full(
        (host_pool_size + 1,), I32_MAX, dtype=torch.int32, device="cuda"
    )
    device_token_to_host = torch.full(
        (device_pool_size + 1,), I32_MAX, dtype=torch.int32, device="cuda"
    )
    layer_device_kv = torch.zeros(
        (device_pool_size + 1, kv_lora_rank + qk_rope_head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    layer_host_kv = torch.randn(
        (host_pool_size + 1, kv_lora_rank + qk_rope_head_dim),
        device="cuda",
        dtype=torch.float16,
        generator=rng,
    )
    recall_counter = torch.zeros((1,), dtype=torch.uint32, device="cuda")

    for q_len, topk in cfgs:
        recall_host_indices = torch.randint(
            1,
            host_pool_size,
            (q_len, topk),
            dtype=torch.int32,
            device="cuda",
            generator=rng,
        )
        if topk >= 4:
            recall_host_indices[:, : topk // 4] = -1
        BLOCK = 32
        grid = (q_len, triton.cdiv(topk, BLOCK))

        def run():
            recall_update(
                recall_host_indices,
                recall_device_indices,
                host_token_to_device,
                device_token_to_host,
                layer_device_kv,
                layer_host_kv,
                recall_counter,
                kv_lora_rank,
                qk_rope_head_dim,
            )

        us = _bench_us(run, rep=rep, warmup=warmup)
        rows.append(("recall_update", {"q_len": q_len, "topk": topk}, us))

        def run_kernel():
            recall_counter.fill_(0)
            _recall_update_kernel[grid](
                recall_counter,
                recall_host_indices,
                recall_device_indices,
                host_token_to_device,
                device_token_to_host,
                layer_device_kv,
                layer_host_kv,
                recall_host_indices.stride(0),
                kv_lora_rank,
                qk_rope_head_dim,
                BLOCK,
            )

        us_kernel = _bench_us(run_kernel, rep=rep, warmup=warmup)
        rows.append(("recall_update_kernel", {"q_len": q_len, "topk": topk}, us_kernel))
    return rows


def _bench_recall_update_extend(rep: int, warmup: int, quick: bool) -> List[Tuple[str, Dict[str, int], float]]:
    host_pool_size = 524288 if quick else 1048576
    device_pool_size = 131072 if quick else 262144
    host_sizes = [131072, 262144] if quick else [131072, 262144, 524288]
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    rng = _make_rng(7)
    rows = []

    recall_device_indices = torch.arange(
        1, device_pool_size + 1, dtype=torch.int32, device="cuda"
    )
    host_token_to_device = torch.full(
        (host_pool_size + 1,), I32_MAX, dtype=torch.int32, device="cuda"
    )
    device_token_to_host = torch.full(
        (device_pool_size + 1,), I32_MAX, dtype=torch.int32, device="cuda"
    )
    layer_device_kv = torch.zeros(
        (device_pool_size + 1, kv_lora_rank + qk_rope_head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    layer_host_kv = torch.randn(
        (host_pool_size + 1, kv_lora_rank + qk_rope_head_dim),
        device="cuda",
        dtype=torch.float16,
        generator=rng,
    )
    recall_counter = torch.zeros((1,), dtype=torch.uint32, device="cuda")

    for host_size in host_sizes:
        host_need_recall = torch.zeros((host_size,), dtype=torch.bool, device="cuda")
        # Recall around 10% tokens for a realistic sparse pattern.
        num_recall = max(1, host_size // 10)
        recall_pos = torch.randperm(host_size, device="cuda", generator=rng)[:num_recall]
        host_need_recall[recall_pos] = 1

        def run():
            recall_update_extend(
                host_need_recall,
                recall_device_indices,
                host_token_to_device,
                device_token_to_host,
                layer_device_kv,
                layer_host_kv[:host_size],
                recall_counter,
                kv_lora_rank,
                qk_rope_head_dim,
            )

        us = _bench_us(run, rep=rep, warmup=warmup)
        rows.append(("recall_update_extend", {"host_size": host_size, "recall_ratio_pct": 10}, us))
    return rows


def run_benchmarks(rep: int, warmup: int, quick: bool) -> List[Tuple[str, Dict[str, int], float]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton benchmark.")

    torch.set_default_device("cuda")
    results = []
    results.extend(_bench_protect(rep, warmup, quick))
    results.extend(_bench_get_free_loc(rep, warmup, quick))
    results.extend(_bench_free_update(rep, warmup, quick))
    results.extend(_bench_free_kernel(rep, warmup, quick))
    results.extend(_bench_update_alloc_kernel(rep, warmup, quick))
    results.extend(_bench_set_update(rep, warmup, quick))
    results.extend(_bench_set_mla_kv_buffer(rep, warmup, quick))
    results.extend(_bench_recall_update(rep, warmup, quick))
    results.extend(_bench_recall_update_extend(rep, warmup, quick))
    return results


def print_results(results: List[Tuple[str, Dict[str, int], float]]) -> None:
    print("kernel, config, us")
    for kernel, cfg, us in results:
        cfg_str = ",".join(f"{k}={v}" for k, v in cfg.items())
        print(f"{kernel}, {cfg_str}, {us:.2f}")


def save_csv(results: List[Tuple[str, Dict[str, int], float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "config", "us"])
        for kernel, cfg, us in results:
            cfg_str = ",".join(f"{k}={v}" for k, v in cfg.items())
            writer.writerow([kernel, cfg_str, f"{us:.4f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark newly added Triton kernels in srt mem_cache recall_ops"
    )
    parser.add_argument("--rep", type=int, default=200, help="Benchmark repetitions")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller configs for a fast smoke benchmark",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional csv output path for benchmark results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmarks(rep=args.rep, warmup=args.warmup, quick=args.quick)
    print_results(results)
    if args.csv is not None:
        save_csv(results, args.csv)
        print(f"Saved benchmark csv to {args.csv}")


if __name__ == "__main__":
    main()
