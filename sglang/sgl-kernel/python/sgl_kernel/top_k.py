import torch


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        # TODO: implement faster cuda kernels for large vocab sizes
        return torch.topk(values, topk, dim=dim)


def fast_topk_v2(score: torch.Tensor, lengths: torch.Tensor, topk: int) -> torch.Tensor:
    assert (
        topk == 2048
    ), "fast_topk_v2 is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    topk_indices = score.new_empty((score.size(0), topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk(score, topk_indices, lengths)
    return topk_indices


def fast_argtopk_m2048(
    score: torch.Tensor,
    indices: torch.Tensor,
    topk: torch.Tensor,
    largest: bool = True,
) -> None:
    """Argtopk for int32 scores. Supports any topk value.
    Batch size 1 only. score and indices are 1-D int32 tensors.
    topk is a scalar int32 tensor (can be on GPU for cuda graph).
    When ``largest`` is True (default), selects the top-k LARGEST values,
    matching ``torch.topk(..., largest=True, sorted=False)``. When False,
    selects the top-k SMALLEST values instead (no external negation needed).
    The first topk entries of indices are filled with the top-k indices."""
    torch.ops.sgl_kernel.fast_argtopk_m2048(score, indices, topk, largest)


def fast_argtopk(
    score: torch.Tensor,
    indices: torch.Tensor,
    topk: torch.Tensor,
    largest: bool = True,
) -> None:
    """Argtopk for int32 scores. Supports any topk value.
    Batch size 1 only. score and indices are 1-D int32 tensors.
    topk is a scalar int32 tensor (can be on GPU for cuda graph).
    When ``largest`` is True (default), selects the top-k LARGEST values,
    matching ``torch.topk(..., largest=True, sorted=False)``. When False,
    selects the top-k SMALLEST values instead (no external negation needed).
    The first topk entries of indices are filled with the top-k indices."""
    torch.ops.sgl_kernel.fast_argtopk(score, indices, topk, largest)


def fast_argmin_bounded(
    score: torch.Tensor,
    indices: torch.Tensor,
    topk: torch.Tensor,
) -> None:
    """Smallest-k selection for bounded non-negative int32 scores.

    Intended for `get_free_loc`, where scores are in `[0, 10_000_000]`
    plus large sentinels such as `I32_MAX` that must never be selected
    unless all bounded candidates are exhausted.
    """
    torch.ops.sgl_kernel.fast_argmin_bounded(score, indices, topk)


def fast_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,  # NOTE: page size should be 1
    cu_seqlens_q: torch.Tensor,
    topk: int,
    return_topk_logits: bool = False,
):
    assert (
        topk == 2048
    ), "fast_topk_transform_fused is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    src_page_table = page_table_size_1
    dst_page_table = score.new_empty((score.size(0), topk), dtype=torch.int32)
    topk_logits = (
        torch.zeros((score.size(0),), dtype=torch.float32, device=score.device)
        if return_topk_logits
        else None
    )
    if return_topk_logits:
        torch.ops.sgl_kernel.fast_topk_transform_fused(
            score,
            lengths,
            dst_page_table,
            src_page_table,
            cu_seqlens_q,
            topk_logits,
        )
        return dst_page_table, topk_logits
    torch.ops.sgl_kernel.fast_topk_transform_fused(
        score, lengths, dst_page_table, src_page_table, cu_seqlens_q
    )
    return dst_page_table
