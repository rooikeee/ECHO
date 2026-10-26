from types import SimpleNamespace

from echo_memory_budget_worker import resident_bytes


def test_mtp_resident_pool_includes_dummy_rows_and_tails():
    cfg = SimpleNamespace(max_num_batched_tokens=256, max_num_seqs=6)
    expected = 79 * 24 * (12288 + 256) * 576 * 2
    assert resident_bytes(cfg, 79, 12288, 128, 576) == expected
    cfg.max_num_seqs = 1
    assert resident_bytes(cfg, 79, 12288, 128, 576) == 79 * 6 * 12544 * 576 * 2
