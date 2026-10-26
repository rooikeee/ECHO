#!/usr/bin/env python3
"""CPU tests for controlled MTP3 profiling inputs, without NPU initialization."""

import unittest

import torch

from echo_profile_workload import controlled_resident_sources


class ControlledResidencyTest(unittest.TestCase):
    def setUp(self):
        # Four query rows share two hits and own four exclusive sources each.
        self.topk = torch.tensor([[[0, 1, 2, 3, 4, 5], [0, 1, 6, 7, 8, 9],
                                  [0, 1, 10, 11, 12, 13], [0, 1, 14, 15, 16, 17]]])

    def test_exact_counts_dedup_and_seed(self):
        sources = controlled_resident_sources(self.topk, 32, 16, 2, 7)
        self.assertEqual(sources.shape, (1, 16))
        self.assertEqual(sources.unique().numel(), 16)
        torch.testing.assert_close(sources, controlled_resident_sources(self.topk, 32, 16, 2, 7))
        resident = torch.zeros(32, dtype=torch.bool)
        resident[sources[0]] = True
        torch.testing.assert_close((~resident[self.topk[0]]).sum(1), torch.full((4,), 2))
        self.assertEqual(self.topk[0][~resident[self.topk[0]]].unique().numel(), 8)
        self.assertTrue(bool(resident[:2].all()))

    def test_zero_misses(self):
        sources = controlled_resident_sources(self.topk, 32, 24, 0, 7)
        self.assertTrue(bool(torch.isin(self.topk, sources).all()))

    def test_request_union_counts(self):
        topk = self.topk.repeat(2, 1, 1)
        original = topk.clone()
        residents = controlled_resident_sources(topk, 32, 24, 4, 9, request_union=True)
        for request, resident in zip(topk, residents):
            missing = request[~torch.isin(request, resident)]
            self.assertEqual(missing.unique().numel(), 4)
            self.assertGreaterEqual(missing.numel(), 4)
        torch.testing.assert_close(topk, original)

    def test_shared_missing_source_is_counted_once(self):
        topk = self.topk[:, :1].repeat(1, 4, 1)
        residents = controlled_resident_sources(topk, 32, 24, 4, 7, request_union=True)
        missing = topk[~torch.isin(topk, residents)]
        self.assertEqual(missing.unique().numel(), 4)
        self.assertEqual(missing.numel(), 16)

    def test_impossible_cache_or_exclusive_count(self):
        with self.assertRaisesRegex(ValueError, "hot cache"):
            controlled_resident_sources(self.topk, 32, 4, 2, 7)
        with self.assertRaisesRegex(ValueError, "query-exclusive"):
            controlled_resident_sources(self.topk, 32, 16, 5, 7)

    def test_invalid_sources(self):
        bad = self.topk.clone()
        bad[0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "legal sources"):
            controlled_resident_sources(bad, 32, 16, 2, 7)


if __name__ == "__main__":
    unittest.main()
