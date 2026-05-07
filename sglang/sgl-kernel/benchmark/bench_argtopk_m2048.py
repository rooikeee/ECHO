"""Benchmark fast_argtopk vs torch.topk for int32 scores."""

import argparse
import time

import torch
from sgl_kernel import fast_argtopk


def validate(N: int, topk: int, device: str = "cuda", largest: bool = True):
    """Check that fast_argtopk returns the same top-k set as torch.topk."""
    score = torch.randint(-(2**30), 2**30, (N,), dtype=torch.int32, device=device)
    indices = torch.empty(N, dtype=torch.int32, device=device)
    topk_t = torch.tensor(topk, dtype=torch.int32, device=device)

    fast_argtopk(score, indices, topk_t, largest)
    torch.cuda.synchronize()

    our_indices = indices[:topk].long()
    our_values = score[our_indices].sort(descending=largest).values

    ref_values, _ = torch.topk(score.int(), topk, dim=0, largest=largest, sorted=True)
    ref_values = ref_values.sort(descending=largest).values

    match = torch.equal(our_values, ref_values)
    if not match:
        diff = (our_values != ref_values).sum().item()
        print(f"  FAIL  N={N}, topk={topk}, largest={largest}: {diff}/{topk} mismatches")
        print(f"    ours (first 10): {our_values[:10].tolist()}")
        print(f"    ref  (first 10): {ref_values[:10].tolist()}")
    return match


def benchmark_latency(N: int, topk: int, warmup: int = 50, iters: int = 200, device: str = "cuda"):
    """Measure latency of fast_argtopk and torch.topk."""
    score = torch.randint(-(2**30), 2**30, (N,), dtype=torch.int32, device=device)
    indices = torch.empty(N, dtype=torch.int32, device=device)
    topk_t = torch.tensor(topk, dtype=torch.int32, device=device)

    # --- our kernel ---
    for _ in range(warmup):
        fast_argtopk(score, indices, topk_t, True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fast_argtopk(score, indices, topk_t, True)
    torch.cuda.synchronize()
    our_us = (time.perf_counter() - start) / iters * 1e6

    # --- torch.topk baseline ---
    for _ in range(warmup):
        torch.topk(score.int(), topk, dim=0, largest=True, sorted=False)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.topk(score.int(), topk, dim=0, largest=True, sorted=False)
    torch.cuda.synchronize()
    ref_us = (time.perf_counter() - start) / iters * 1e6

    return our_us, ref_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark fast_argtopk")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    Ns = [262144]
    # Test arbitrary topk values: multiples of 2048, odd numbers, primes, small values
    topks = [1, 7, 100, 1000, 1023, 1024, 2048, 3000, 4096, 5000, 8192, 10000, 16384, 18432]

    # --- Correctness ---
    print("=" * 60)
    print("Correctness validation")
    print("=" * 60)
    all_pass = True
    for largest in (True, False):
        for topk in topks:
            for N in Ns:
                if N < topk:
                    continue
                ok = validate(N, topk, largest=largest)
                status = "PASS" if ok else "FAIL"
                print(f"  N={N:>7d}  topk={topk:>5d}  largest={str(largest):>5s}  {status}")
                all_pass = all_pass and ok
        # extra: edge case where N == topk
        for topk in topks:
            ok = validate(topk, topk, largest=largest)
            status = "PASS" if ok else "FAIL"
            print(f"  N={topk:>7d}  topk={topk:>5d}  largest={str(largest):>5s}  (edge) {status}")
            all_pass = all_pass and ok
    print()
    if all_pass:
        print("All correctness checks passed!")
    else:
        print("Some correctness checks FAILED!")
    print()

    # --- Performance ---
    print("=" * 60)
    print("Latency benchmark  (us)")
    print("=" * 60)
    print(f"{'N':>8s}  {'topk':>5s}  {'ours (us)':>10s}  {'torch (us)':>10s}  {'speedup':>8s}")
    print("-" * 50)
    for topk in topks:
        for N in Ns:
            if N < topk:
                continue
            our_us, ref_us = benchmark_latency(N, topk, warmup=args.warmup, iters=args.iters)
            speedup = ref_us / our_us if our_us > 0 else float("inf")
            print(f"{N:>8d}  {topk:>5d}  {our_us:>10.1f}  {ref_us:>10.1f}  {speedup:>7.2f}x")
        print()


if __name__ == "__main__":
    main()
