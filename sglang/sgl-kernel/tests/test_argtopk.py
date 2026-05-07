import pytest
import torch
from sgl_kernel import fast_argmin_bounded, fast_argtopk, fast_argtopk_m2048


def _reference_argtopk(score, topk, largest=True):
    """Reference implementation using torch.topk."""
    _, indices = torch.topk(score, topk, dim=0, largest=largest, sorted=False)
    return indices


def _check_argtopk(score, topk_val, fn=fast_argtopk, largest=True):
    """Run kernel and verify result matches torch.topk."""
    N = score.size(0)
    indices = torch.empty(N, dtype=torch.int32, device="cuda")
    topk_tensor = torch.tensor([topk_val], dtype=torch.int32, device="cuda")
    fn(score, indices, topk_tensor, largest=largest)
    torch.cuda.synchronize()

    result_indices = indices[:topk_val]
    ref_indices = _reference_argtopk(score, topk_val, largest=largest)

    # Compare by value sets (order may differ)
    result_vals = score[result_indices.long()].sort().values
    ref_vals = score[ref_indices.long()].sort().values
    assert torch.equal(result_vals, ref_vals), (
        f"Mismatch: topk={topk_val}, N={N}, largest={largest}\n"
        f"  result indices: {result_indices.tolist()}\n"
        f"  ref indices:    {ref_indices.tolist()}"
    )


def _check_argmin_bounded(score, topk_val):
    """Run the bounded smallest-k kernel and verify result values."""
    N = score.size(0)
    indices = torch.empty(N, dtype=torch.int32, device="cuda")
    topk_tensor = torch.tensor([topk_val], dtype=torch.int32, device="cuda")
    fast_argmin_bounded(score, indices, topk_tensor)
    torch.cuda.synchronize()

    result_indices = indices[:topk_val]
    ref_indices = _reference_argtopk(score, topk_val, largest=False)
    result_vals = score[result_indices.long()].sort().values
    ref_vals = score[ref_indices.long()].sort().values
    assert torch.equal(result_vals, ref_vals)


# ── Correctness: non-2048 topk values ──────────────────────────────────────


@pytest.mark.parametrize("topk", [1, 7, 100, 255, 512, 1023, 1024, 2048, 3000, 4096])
def test_various_topk(topk):
    N = 262144
    score = torch.randint(-2**30, 2**30, (N,), dtype=torch.int32, device="cuda")
    _check_argtopk(score, topk)


@pytest.mark.parametrize("topk", [1, 7, 100, 255, 512, 1023, 1024, 2048, 3000, 4096])
def test_various_topk_smallest(topk):
    """Smallest-k mode: reversed monotone uint32 mapping inside the kernel."""
    N = 262144
    score = torch.randint(-2**30, 2**30, (N,), dtype=torch.int32, device="cuda")
    _check_argtopk(score, topk, largest=False)


@pytest.mark.parametrize("topk", [1, 7, 100, 255, 512, 1024, 2048, 4096])
def test_bounded_argmin_various_topk(topk):
    N = 262144
    score = torch.randint(0, 10_000_001, (N,), dtype=torch.int32, device="cuda")
    _check_argmin_bounded(score, topk)


@pytest.mark.parametrize("topk", [1, 64, 512, 1024])
def test_bounded_argmin_with_sentinels(topk):
    N = 262144
    score = torch.randint(0, 10_000_001, (N,), dtype=torch.int32, device="cuda")
    score[:4096] = torch.iinfo(torch.int32).max
    score[4096:8192] = 0
    _check_argmin_bounded(score, topk)


@pytest.mark.parametrize("topk", [1, 7, 100, 255, 512, 1023, 1024, 2048, 3000, 4096])
def test_various_topk_m2048_compat(topk):
    """Backward-compatible fast_argtopk_m2048 should behave the same."""
    N = 262144
    score = torch.randint(-2**30, 2**30, (N,), dtype=torch.int32, device="cuda")
    _check_argtopk(score, topk, fn=fast_argtopk_m2048)


# ── Edge case: N == topk ───────────────────────────────────────────────────


@pytest.mark.parametrize("N", [1, 7, 128, 1024, 2048, 4096, 8192])
def test_n_equals_topk(N):
    score = torch.randint(-2**30, 2**30, (N,), dtype=torch.int32, device="cuda")
    indices = torch.empty(N, dtype=torch.int32, device="cuda")
    topk_tensor = torch.tensor([N], dtype=torch.int32, device="cuda")
    fast_argtopk(score, indices, topk_tensor)
    torch.cuda.synchronize()
    # All indices should be present (in any order)
    sorted_indices = indices.sort().values
    expected = torch.arange(N, dtype=torch.int32, device="cuda")
    assert torch.equal(sorted_indices, expected), (
        f"N==topk=={N}: expected all indices 0..{N-1}"
    )


# ── Edge case: topk with CPU tensor (enables host-side validation) ─────────


@pytest.mark.parametrize("topk", [1, 100, 2048])
def test_cpu_topk_tensor(topk):
    N = 8192
    score = torch.randint(-2**30, 2**30, (N,), dtype=torch.int32, device="cuda")
    indices = torch.empty(N, dtype=torch.int32, device="cuda")
    topk_tensor = torch.tensor([topk], dtype=torch.int32)  # CPU tensor
    fast_argtopk(score, indices, topk_tensor)
    torch.cuda.synchronize()

    result_indices = indices[:topk]
    ref_indices = _reference_argtopk(score, topk)
    result_vals = score[result_indices.long()].sort().values
    ref_vals = score[ref_indices.long()].sort().values
    assert torch.equal(result_vals, ref_vals)


# ── Invalid-input error semantics ──────────────────────────────────────────


def test_error_topk_negative_cpu():
    """Negative topk (CPU tensor) should raise an error."""
    score = torch.randint(0, 100, (1024,), dtype=torch.int32, device="cuda")
    indices = torch.empty(1024, dtype=torch.int32, device="cuda")
    topk = torch.tensor([-1], dtype=torch.int32)  # CPU, negative
    with pytest.raises(RuntimeError, match="non-negative"):
        fast_argtopk(score, indices, topk)


def test_error_topk_exceeds_n_cpu():
    """topk > score.size(0) (CPU tensor) should raise an error."""
    N = 100
    score = torch.randint(0, 100, (N,), dtype=torch.int32, device="cuda")
    indices = torch.empty(200, dtype=torch.int32, device="cuda")
    topk = torch.tensor([N + 1], dtype=torch.int32)  # CPU, exceeds N
    with pytest.raises(RuntimeError, match="must not exceed"):
        fast_argtopk(score, indices, topk)


def test_error_indices_too_small_cpu():
    """indices.size(0) < topk (CPU tensor) should raise an error."""
    N = 1024
    score = torch.randint(0, 100, (N,), dtype=torch.int32, device="cuda")
    indices = torch.empty(50, dtype=torch.int32, device="cuda")  # too small
    topk = torch.tensor([100], dtype=torch.int32)  # CPU
    with pytest.raises(RuntimeError, match="must be >= topk"):
        fast_argtopk(score, indices, topk)


def test_error_empty_indices():
    """Empty indices tensor should raise an error."""
    score = torch.randint(0, 100, (1024,), dtype=torch.int32, device="cuda")
    indices = torch.empty(0, dtype=torch.int32, device="cuda")
    topk = torch.tensor([10], dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="must not be empty"):
        fast_argtopk(score, indices, topk)


def test_error_wrong_dtype():
    """Non-int32 score should raise an error."""
    score = torch.randn(1024, device="cuda")  # float32
    indices = torch.empty(1024, dtype=torch.int32, device="cuda")
    topk = torch.tensor([10], dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="int32"):
        fast_argtopk(score, indices, topk)


def test_error_2d_score():
    """2-D score tensor should raise an error."""
    score = torch.randint(0, 100, (2, 512), dtype=torch.int32, device="cuda")
    indices = torch.empty(512, dtype=torch.int32, device="cuda")
    topk = torch.tensor([10], dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="1-D"):
        fast_argtopk(score, indices, topk)


# ── Determinism: duplicate values ──────────────────────────────────────────


@pytest.mark.parametrize("topk", [1, 10, 100, 1024])
def test_duplicate_values(topk):
    """Kernel should handle many duplicate score values correctly."""
    N = 4096
    # Only 10 distinct values → many duplicates
    score = torch.randint(0, 10, (N,), dtype=torch.int32, device="cuda")
    indices = torch.empty(N, dtype=torch.int32, device="cuda")
    topk_tensor = torch.tensor([topk], dtype=torch.int32, device="cuda")
    fast_argtopk(score, indices, topk_tensor)
    torch.cuda.synchronize()

    result_indices = indices[:topk]
    ref_indices = _reference_argtopk(score, topk)
    # Values at selected indices must match
    result_vals = score[result_indices.long()].sort().values
    ref_vals = score[ref_indices.long()].sort().values
    assert torch.equal(result_vals, ref_vals)
