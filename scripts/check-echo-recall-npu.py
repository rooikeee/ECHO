#!/usr/bin/env python3
"""Exercise exact host KV recall and actual NPU Graph replay on one NPU."""

import unittest
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.echo_recall import (
    EchoRecallCache, map_topk_to_host, recalled_sparse_attention,
)


class TestEchoRecall(unittest.TestCase):
    def setUp(self):
        torch.npu.set_device(0)
        self.cache = EchoRecallCache(host_size=64, capacity=8)
        self.graphs = []
        generator = torch.Generator().manual_seed(42)
        self.keys = torch.randn(32, 1, 512, generator=generator).to(torch.bfloat16)
        self.ropes = torch.randn(32, 1, 64, generator=generator).to(torch.bfloat16)
        self.write_all()

    def tearDown(self):
        torch.npu.synchronize()
        self.graphs.clear()
        self.cache.close()

    def write_all(self):
        self.cache.write(
            torch.arange(1, 33, dtype=torch.int32, device="npu"),
            self.keys.npu(), self.ropes.npu(),
        )
        torch.npu.synchronize()

    def assert_recall(self, ids, output, misses=None):
        self.cache.check_errors()
        slots = output.cpu()
        key = self.cache.key.cpu()
        rope = self.cache.rope.cpu()
        h2d = self.cache.host_to_device.cpu()
        d2h = self.cache.device_to_host.cpu()
        for host, slot in zip(ids.reshape(-1).tolist(), slots.reshape(-1).tolist()):
            if host <= 0:
                self.assertEqual(slot, -1)
            else:
                self.assertGreaterEqual(slot, 0, f"host={host}; mapping={h2d[host]}; device={d2h.tolist()}")
                self.assertEqual(h2d[host].item(), slot)
                self.assertEqual(d2h[slot].item(), host)
                torch.testing.assert_close(key[slot], self.keys[host - 1], rtol=0, atol=0)
                torch.testing.assert_close(rope[slot], self.ropes[host - 1], rtol=0, atol=0)
        for slot, host in enumerate(d2h.tolist()):
            if host > 0:
                self.assertEqual(h2d[host].item(), slot)
        count = self.cache.free_count.item()
        free = self.cache.free_slots[:count].cpu().long()
        self.assertEqual(free.unique().numel(), count)
        self.assertEqual(count + (d2h > 0).sum().item(), self.cache.capacity)
        self.assertTrue(torch.all(d2h[free] == -1))
        if misses is not None:
            self.assertEqual(self.cache.miss_count.item(), misses)

    def run_recall(self, ids, misses=None):
        ids = torch.tensor(ids, dtype=torch.int32)
        out = self.cache.recall(ids.npu())
        self.assert_recall(ids, out, misses)
        return out.cpu()

    def test_host_write_and_deduplication(self):
        torch.testing.assert_close(self.cache.host.tensor[1:33, :512], self.keys[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(self.cache.host.tensor[1:33, 512:], self.ropes[:, 0], rtol=0, atol=0)
        out = self.run_recall([[5, 9, 12, 5], [9, 0, -1, 12]], misses=3)
        self.assertEqual(out[0, 0].item(), out[0, 3].item())
        self.run_recall([[12, 9, 5, -1]], misses=0)

    def test_eviction_and_protection(self):
        self.run_recall([[1, 2, 3, 4], [5, 6, 7, 8]], misses=8)
        before = self.cache.host_to_device.cpu().clone()
        self.run_recall([[1, 2, 9, 10], [11, 12, 13, 14]], misses=6)
        after = self.cache.host_to_device.cpu()
        self.assertEqual(before[1].item(), after[1].item())
        self.assertEqual(before[2].item(), after[2].item())
        self.assertTrue(torch.all(after[3:9] == -1))
        self.run_recall([[3, 4, 5, 6], [7, 8, 15, 16]], misses=8)

    def test_release_and_reuse(self):
        self.run_recall([[5, 9, 12, -1]], misses=3)
        ids = torch.tensor([9], dtype=torch.int32, device="npu")
        self.cache.release(ids)
        self.assertEqual(self.cache.written[9].item(), 0)
        self.assertEqual(self.cache.host_to_device[9].item(), -1)
        self.keys[8].fill_(2)
        self.ropes[8].fill_(-3)
        self.cache.write(ids, self.keys[8:9].npu(), self.ropes[8:9].npu())
        self.run_recall([[5, 9, 12, -1]], misses=0)

    def test_write_hot_before_host_backup(self):
        self.cache.clear()
        torch.npu.synchronize()
        self.cache.host.tensor.zero_()
        ids = torch.tensor([1, 2, 3], dtype=torch.int32, device="npu")
        offload = self.cache._offload

        def inspect_hot_before_offload(logical_ids, slots):
            torch.npu.synchronize()
            selected = slots.cpu().long()
            torch.testing.assert_close(self.cache.key.cpu()[selected], self.keys[:3], rtol=0, atol=0)
            torch.testing.assert_close(self.cache.rope.cpu()[selected], self.ropes[:3], rtol=0, atol=0)
            self.assertTrue(torch.all(self.cache.written.cpu() == 0))
            self.assertTrue(torch.all(self.cache.host.tensor == 0))
            offload(logical_ids, slots)

        with patch.object(self.cache, "_offload", side_effect=inspect_hot_before_offload) as backup:
            self.cache.write(ids, self.keys[:3].npu(), self.ropes[:3].npu())
            backup.assert_called_once()
        torch.npu.synchronize()
        torch.testing.assert_close(self.cache.host.tensor[1:4, :512], self.keys[:3, 0], rtol=0, atol=0)
        torch.testing.assert_close(self.cache.host.tensor[1:4, 512:], self.ropes[:3, 0], rtol=0, atol=0)
        self.run_recall([[1, 2, 3]], misses=0)

    def test_hot_hit_does_not_reload_host(self):
        self.cache.clear()
        ids = torch.tensor([1, 2], dtype=torch.int32, device="npu")
        self.cache.write(ids, self.keys[:2].npu(), self.ropes[:2].npu())
        torch.npu.synchronize()
        # A poisoned host copy must not affect already-resident device KV.
        self.cache.host.tensor[1:3].fill_(99)
        self.run_recall([[1, 2, 1]], misses=0)

    def test_hot_overwrite_preserves_slot(self):
        self.cache.clear()
        ids = torch.tensor([1, 2], dtype=torch.int32, device="npu")
        self.cache.write(ids, self.keys[:2].npu(), self.ropes[:2].npu())
        before = self.cache.host_to_device.cpu().clone()
        self.keys[0].fill_(7)
        self.ropes[0].fill_(-6)
        self.cache.write(ids[:1], self.keys[:1].npu(), self.ropes[:1].npu())
        torch.npu.synchronize()
        torch.testing.assert_close(self.cache.host_to_device.cpu(), before)
        torch.testing.assert_close(self.cache.host.tensor[1, :512], self.keys[0, 0], rtol=0, atol=0)
        torch.testing.assert_close(self.cache.host.tensor[1, 512:], self.ropes[0, 0], rtol=0, atol=0)
        self.run_recall([[1, 2]], misses=0)

    def test_oversized_write_preserves_all_host_kv(self):
        # setUp writes 32 rows through an eight-slot cache in four chunks.
        resident = self.cache.host_to_device.cpu()
        self.assertTrue(torch.all(resident[1:25] == -1))
        self.assertTrue(torch.all(resident[25:33] >= 0))
        self.run_recall([list(range(25, 33))], misses=0)
        self.run_recall([list(range(1, 9))], misses=8)

    def test_write_protects_resident_input(self):
        self.cache.clear()
        ids = torch.arange(1, 9, dtype=torch.int32, device="npu")
        self.cache.write(ids, self.keys[:8].npu(), self.ropes[:8].npu())
        slot = self.cache.host_to_device[1].item()
        selected = torch.tensor([1, 9, 10], dtype=torch.int32, device="npu")
        rows = torch.tensor([0, 8, 9])
        self.cache.write(selected, self.keys[rows].npu(), self.ropes[rows].npu())
        self.assertEqual(self.cache.host_to_device[1].item(), slot)
        self.run_recall([[1, 9, 10]], misses=0)

    def test_small_write_does_not_call_full_planner(self):
        self.cache.clear()
        ids = torch.tensor([1, 2], dtype=torch.int32, device="npu")
        with patch.object(self.cache, "_plan_slots", side_effect=AssertionError("full planner")):
            self.cache.write(ids, self.keys[:2].npu(), self.ropes[:2].npu())
        self.run_recall([[1, 2]], misses=0)

    def test_release_deduplicates_free_stack(self):
        self.run_recall([[9, 12]], misses=2)
        ids = torch.tensor([9, 9, 12, 0, -1], dtype=torch.int32, device="npu")
        self.cache.release(ids)
        self.assertEqual(self.cache.free_count.item(), 2)
        self.cache.release(ids)
        self.assertEqual(self.cache.free_count.item(), 2)
        rows = torch.tensor([8, 11])
        self.cache.write(torch.tensor([9, 12], dtype=torch.int32, device="npu"),
                         self.keys[rows].npu(), self.ropes[rows].npu())
        self.run_recall([[9, 12]], misses=0)

    def test_all_hit_does_not_scan_unselected_host_ids(self):
        self.cache.clear()
        ids = torch.tensor([1, 2], dtype=torch.int32, device="npu")
        self.cache.write(ids, self.keys[:2].npu(), self.ropes[:2].npu())
        # A tag matching the next epoch outside the actual request would be
        # consumed by a whole-pool scan, but the all-hit path must not scan it.
        self.cache.required[63:64].copy_(self.cache.clock + 1)
        self.run_recall([[1, 2, 1]], misses=0)
        self.assertEqual(self.cache.host_to_device[63].item(), -1)

    def test_bounded_lru_preserves_recent_and_protected_slots(self):
        self.cache.clear(preserve_written=True)
        for host in range(1, 9):
            self.cache.write(torch.tensor([host], dtype=torch.int32, device="npu"),
                             self.keys[host - 1:host].npu(), self.ropes[host - 1:host].npu())
        self.run_recall([[1]], misses=0)
        self.run_recall([[2]], misses=0)
        self.cache.write(torch.tensor([9], dtype=torch.int32, device="npu"),
                         self.keys[8:9].npu(), self.ropes[8:9].npu())
        self.assertEqual(self.cache.host_to_device[3].item(), -1)
        self.run_recall([[9, 10]], misses=1)
        self.assertEqual(self.cache.host_to_device[4].item(), -1)
        self.assertGreaterEqual(self.cache.host_to_device[1].item(), 0)
        self.assertGreaterEqual(self.cache.host_to_device[2].item(), 0)

    def test_non_power_of_two_capacity(self):
        cache = EchoRecallCache(host_size=64, capacity=5)
        try:
            cache.write(torch.arange(1, 11, dtype=torch.int32, device="npu"),
                        self.keys[:10].npu(), self.ropes[:10].npu())
            ids = torch.tensor([1, 6, 7, 10], dtype=torch.int32, device="npu")
            slots = cache.recall(ids)
            cache.check_errors()
            self.assertEqual(cache.miss_count.item(), 1)
            torch.testing.assert_close(cache.key[slots.long()].cpu(),
                                       self.keys[ids.cpu().long() - 1], rtol=0, atol=0)
        finally:
            cache.close()

    def test_reset_preserves_host_validity(self):
        self.run_recall([[5, 9, 12, -1]], misses=3)
        self.cache.clear(preserve_written=True)
        self.assertTrue(torch.all(self.cache.host_to_device.cpu() == -1))
        self.assertTrue(torch.all(self.cache.written[1:33].cpu() == 1))
        self.run_recall([[5, 9, 12, -1]], misses=3)
        self.cache.clear()
        self.assertTrue(torch.all(self.cache.written.cpu() == 0))

    def test_fail_closed(self):
        self.run_recall([[1, 2, 3, 4]], misses=4)
        before = self.cache.host_to_device.cpu().clone()
        for values, expected in [([53], 2), ([64], 1), ([-2], 1), (list(range(1, 10)), 4)]:
            output = self.cache.recall(torch.tensor(values, dtype=torch.int32, device="npu"))
            self.assertEqual(self.cache.errors.item() & expected, expected)
            self.assertTrue(torch.all(output.cpu() == -1))
            torch.testing.assert_close(self.cache.host_to_device.cpu(), before)
            with self.assertRaises(RuntimeError):
                self.cache.check_errors()

    def test_int64_ids_do_not_wrap(self):
        invalid = torch.tensor([2 ** 32 + 1], dtype=torch.int64, device="npu")
        self.cache.write(invalid, (self.keys[:1] + 3).npu(), (self.ropes[:1] - 2).npu())
        self.run_recall([[1]], misses=1)
        output = self.cache.recall(invalid)
        self.assertEqual(output.item(), -1)
        self.assertEqual(self.cache.errors.item() & 1, 1)
        with self.assertRaises(RuntimeError):
            self.cache.check_errors()

    def test_graph_repeated_replay(self):
        for batch in (1, 2):
            ids = torch.ones((batch, 4), dtype=torch.int32, device="npu")
            self.cache.recall(ids)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = self.cache.recall(ids)
            self.graphs.append(graph)
            self.cache.clear()
            self.write_all()
            for step in range(8):
                values = torch.arange(1 + step * 2, 1 + step * 2 + batch * 4, dtype=torch.int32).view(batch, 4)
                if step == 3:
                    values[:, -1] = -1
                if step == 4:
                    values.fill_(5)
                ids.copy_(values)
                graph.replay()
                torch.npu.synchronize()
                self.assert_recall(values, output)
            graph.replay()
            torch.npu.synchronize()
            self.assert_recall(values, output, misses=0)

    def test_graph_write_recall_and_release(self):
        loc = torch.tensor([1], dtype=torch.int32, device="npu")
        key = self.keys[:1].npu()
        rope = self.ropes[:1].npu()
        ids = torch.tensor([[1, -1, 0, -1]], dtype=torch.int32, device="npu")
        self.cache.write(loc, key, rope)
        self.cache.recall(ids)
        self.cache.release(loc)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            self.cache.write(loc, key, rope)
            output = self.cache.recall(ids)
            gathered_key = self.cache.key[output.clamp_min(0)].clone()
            self.cache.release(loc)
        self.graphs.append(graph)
        self.cache.clear()
        for host in (3, 9, 17, 3):
            loc.fill_(host)
            ids[:, 0].fill_(host)
            key.copy_(self.keys[host - 1:host])
            rope.copy_(self.ropes[host - 1:host])
            graph.replay()
            torch.npu.synchronize()
            self.cache.check_errors()
            torch.testing.assert_close(gathered_key[0, 0].cpu(), self.keys[host - 1], rtol=0, atol=0)
            self.assertEqual(self.cache.host_to_device[host].item(), -1)
            self.assertEqual(self.cache.written[host].item(), 0)

    def test_graph_write_hot_and_recall_history(self):
        for batch in (1, 2):
            self.cache.clear(preserve_written=True)
            loc = torch.arange(1, batch + 1, dtype=torch.int32, device="npu")
            key = self.keys[:batch].npu()
            rope = self.ropes[:batch].npu()
            ids = torch.full((batch, 2), 31, dtype=torch.int32, device="npu")
            ids[:, 0].copy_(loc)

            def run():
                self.cache.write(loc, key, rope)
                return self.cache.recall(ids)

            run()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = run()
            self.graphs.append(graph)
            for step in range(4):
                values = torch.arange(1 + step * batch, 1 + (step + 1) * batch, dtype=torch.int32)
                loc.copy_(values)
                ids[:, 0].copy_(values)
                key.copy_(self.keys[values.long() - 1].npu())
                rope.copy_(self.ropes[values.long() - 1].npu())
                if batch == 2 and step == 2:
                    loc[-1] = 0
                    ids[-1, 0] = 0
                self.cache.clear(preserve_written=True)
                graph.replay()
                torch.npu.synchronize()
                self.assert_recall(ids.cpu(), output, misses=1)
                for index in loc.cpu().tolist():
                    if index > 0:
                        torch.testing.assert_close(self.cache.host.tensor[index, :512], self.keys[index - 1, 0], rtol=0, atol=0)
                graph.replay()
                torch.npu.synchronize()
                self.assert_recall(ids.cpu(), output, misses=0)

    def test_sparse_attention_hot_cache_addressing(self):
        cache = EchoRecallCache(host_size=64, capacity=128)
        try:
            cache.write(
                torch.arange(1, 33, dtype=torch.int32, device="npu"),
                self.keys.npu(), self.ropes.npu(),
            )
            indices = torch.full((1, 1, 2048), -1, dtype=torch.int32, device="npu")
            indices[0, 0, :3] = torch.tensor([4, 8, 11], dtype=torch.int32, device="npu")
            host_ids = torch.where(indices >= 0, indices + 1, -1).contiguous()
            slots = cache.recall(host_ids)
            cache.check_errors()
            full_key = torch.zeros((1, 128, 1, 512), dtype=torch.bfloat16, device="npu")
            full_rope = torch.zeros((1, 128, 1, 64), dtype=torch.bfloat16, device="npu")
            full_key[0, :32].copy_(self.keys)
            full_rope[0, :32].copy_(self.ropes)
            query = torch.randn(1, 16, 512, dtype=torch.bfloat16, device="npu")
            query_rope = torch.randn(1, 16, 64, dtype=torch.bfloat16, device="npu")

            def attention(key, rope, selected, length):
                return torch_npu.npu_sparse_flash_attention(
                    query=query, key=key, value=key,
                    query_rope=query_rope, key_rope=rope,
                    sparse_indices=selected, scale_value=576 ** -0.5,
                    actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32, device="npu"),
                    actual_seq_lengths_kv=torch.tensor([length], dtype=torch.int32, device="npu"),
                    block_table=torch.tensor([[0]], dtype=torch.int32, device="npu"),
                    sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND",
                    sparse_mode=3, attention_mode=2, return_softmax_lse=False,
                )[0]

            reference = attention(full_key, full_rope, indices, 32)
            actual = attention(cache.key.view(1, 128, 1, 512), cache.rope.view(1, 128, 1, 64), slots, 128)
            torch.npu.synchronize()
            torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=0.02, atol=0.02)
        finally:
            cache.close()

    def test_prefill_multi_request_adapter_and_graph(self):
        cache = EchoRecallCache(host_size=512, capacity=2048)
        graph = None
        try:
            keys = torch.cat((self.keys, self.keys + 1), 0)
            ropes = torch.cat((self.ropes, self.ropes - 1), 0)
            loc = torch.cat((torch.arange(128, 160), torch.arange(256, 288))).int().npu()
            cache.write(loc, keys.npu(), ropes.npu())
            full_key = torch.zeros((3, 128, 1, 512), dtype=torch.bfloat16, device="npu")
            full_rope = torch.zeros((3, 128, 1, 64), dtype=torch.bfloat16, device="npu")
            full_key[1, :32].copy_(keys[:32])
            full_key[2, :32].copy_(keys[32:])
            full_rope[1, :32].copy_(ropes[:32])
            full_rope[2, :32].copy_(ropes[32:])
            selected = torch.full((3, 1, 2048), -1, dtype=torch.int32, device="npu")
            selected[:, 0, :3] = torch.tensor([[0, 13, 14], [1, 10, 15], [2, 15, 31]], dtype=torch.int32, device="npu")
            cu_q = torch.tensor([2, 3], dtype=torch.int32, device="npu")
            kv_lengths = torch.tensor([16, 32], dtype=torch.int32, device="npu")
            table = torch.tensor([[1], [2]], dtype=torch.int32, device="npu")
            query = torch.randn(3, 16, 512, dtype=torch.bfloat16, device="npu")
            query_rope = torch.randn(3, 16, 64, dtype=torch.bfloat16, device="npu")
            expected_ids = torch.tensor([[128, 141, 142], [129, 138, 143], [258, 271, 287]], dtype=torch.int32)

            def reference():
                return torch_npu.npu_sparse_flash_attention(
                    query=query, key=full_key, value=full_key,
                    query_rope=query_rope, key_rope=full_rope,
                    sparse_indices=selected, scale_value=576 ** -0.5,
                    actual_seq_lengths_query=cu_q, actual_seq_lengths_kv=kv_lengths,
                    block_table=table, sparse_block_size=1,
                    layout_query="TND", layout_kv="PA_BSND", sparse_mode=3,
                    attention_mode=2, return_softmax_lse=False,
                )[0]

            def recalled():
                host_ids = map_topk_to_host(selected, table, cu_q, kv_lengths, 128)
                return host_ids, recalled_sparse_attention(cache, query, query_rope, host_ids, 576 ** -0.5)

            host_ids, actual = recalled()
            cache.check_errors()
            torch.testing.assert_close(host_ids.cpu()[:, :3], expected_ids)
            torch.testing.assert_close(actual.cpu(), reference().cpu(), rtol=0.02, atol=0.02)
            grouped = recalled_sparse_attention(
                cache, query, query_rope, host_ids, 576 ** -0.5,
                all_history_fits=True,
            )
            cache.check_errors()
            torch.testing.assert_close(grouped.cpu(), reference().cpu(), rtol=0.02, atol=0.02)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured_ids, captured_output = recalled()
            for _ in range(3):
                query.normal_()
                graph.replay()
                torch.npu.synchronize()
                cache.check_errors()
                torch.testing.assert_close(captured_ids.cpu()[:, :3], expected_ids)
                torch.testing.assert_close(captured_output.cpu(), reference().cpu(), rtol=0.02, atol=0.02)
        finally:
            del graph
            cache.close()

    def test_pool_and_page_release(self):
        from types import SimpleNamespace
        from sglang.srt.hardware_backend.npu.echo_memory_pool import EchoNPUMLATokenToKVPool
        from sglang.srt.hardware_backend.npu.allocator_npu import EchoNPUPagedTokenToKVPoolAllocator

        pool = EchoNPUMLATokenToKVPool(
            size=4096, page_size=128, dtype=torch.bfloat16, kv_lora_rank=512,
            qk_rope_head_dim=64, index_head_dim=128, layer_num=2,
            device="npu", enable_memory_saver=False, cache_size=2048,
            start_layer=5, end_layer=7,
        )
        try:
            allocator = EchoNPUPagedTokenToKVPoolAllocator(4096, 128, torch.bfloat16, "npu", pool, False)
            ids = torch.tensor([128, 129], dtype=torch.int32, device="npu")
            for layer_id in (5, 6):
                pool.set_kv_buffer(SimpleNamespace(layer_id=layer_id), ids, self.keys[:2].npu(), self.ropes[:2].npu())
                pool.get_echo_cache(layer_id).recall(ids)
            pool.check_echo_errors()
            allocator.free(ids[:1])
            for cache in pool.echo_caches:
                self.assertTrue(torch.all(cache.written[128:256].cpu() == 0))
                self.assertTrue(torch.all(cache.host_to_device[128:256].cpu() == -1))
            allocator.clear()
            pool.check_echo_errors()
        finally:
            pool.close()

    def test_graph_capture_padding_at_serving_capacity(self):
        cache = EchoRecallCache(host_size=8320, capacity=4096)
        graph = None
        try:
            selected = torch.full((2, 1, 2048), -1, dtype=torch.int32, device="npu")
            selected[:, :, 0] = 0
            table = torch.zeros((2, 32), dtype=torch.int32, device="npu")
            cu_q = torch.tensor([1, 2], dtype=torch.int32, device="npu")
            lengths = torch.ones(2, dtype=torch.int32, device="npu")
            query = torch.zeros((2, 16, 512), dtype=torch.bfloat16, device="npu")
            rope = torch.zeros((2, 16, 64), dtype=torch.bfloat16, device="npu")

            def recalled():
                ids = map_topk_to_host(selected, table, cu_q, lengths, 128)
                return recalled_sparse_attention(cache, query, rope, ids, 576 ** -0.5)

            recalled()
            torch.npu.synchronize()
            cache.check_errors()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = recalled()
            for _ in range(3):
                graph.replay()
                torch.npu.synchronize()
                cache.check_errors()
                self.assertTrue(torch.isfinite(output).all().item())
        finally:
            del graph
            cache.close()

    def test_idle_dp_rank_has_no_queries(self):
        before = self.cache.host_to_device.cpu().clone()
        selected = torch.empty((0, 1, 2048), dtype=torch.int32, device="npu")
        table = torch.empty((0, 2), dtype=torch.int32, device="npu")
        lengths = torch.empty(0, dtype=torch.int32, device="npu")
        ids = map_topk_to_host(selected, table, lengths, lengths, 128)
        query = torch.empty((0, 16, 512), dtype=torch.bfloat16, device="npu")
        rope = torch.empty((0, 16, 64), dtype=torch.bfloat16, device="npu")
        output = recalled_sparse_attention(self.cache, query, rope, ids, 576 ** -0.5)
        self.assertEqual(output.shape, query.shape)
        torch.testing.assert_close(self.cache.host_to_device.cpu(), before)

    def test_dp_namespaces_do_not_share_kv_or_release_state(self):
        from types import SimpleNamespace
        from sglang.srt.hardware_backend.npu.echo_memory_pool import EchoNPUMLATokenToKVPool

        pools = []
        try:
            ids = torch.tensor([128, 129], dtype=torch.int32, device="npu")
            for rank in (0, 1):
                pool = EchoNPUMLATokenToKVPool(
                    size=4096, page_size=128, dtype=torch.bfloat16,
                    kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128,
                    layer_num=1, device="npu", enable_memory_saver=False,
                    cache_size=2048, attention_dp_rank=rank, attention_dp_size=2,
                )
                pools.append(pool)
                pool.set_kv_buffer(SimpleNamespace(layer_id=0), ids,
                                   torch.full((2, 1, 512), rank + 1, dtype=torch.bfloat16, device="npu"),
                                   torch.full((2, 1, 64), -rank - 1, dtype=torch.bfloat16, device="npu"))
            for rank, pool in enumerate(pools):
                cache = pool.get_echo_cache(0)
                slots = cache.recall(ids)
                pool.check_echo_errors()
                self.assertTrue(torch.all(cache.key[slots.long()].cpu() == rank + 1))
            pools[0].release_echo_tokens(ids)
            self.assertTrue(torch.all(pools[0].get_echo_cache(0).written[ids.long()].cpu() == 0))
            self.assertTrue(torch.all(pools[1].get_echo_cache(0).written[ids.long()].cpu() == 1))
            pools[0].reset_echo_state()
            slots = pools[1].get_echo_cache(0).recall(ids)
            pools[1].check_echo_errors()
            self.assertTrue(torch.all(pools[1].get_echo_cache(0).key[slots.long()].cpu() == 2))
        finally:
            torch.npu.synchronize()
            for pool in pools:
                pool.close()

    def test_graph_dp_rank_active_idle_active(self):
        selected = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device="npu")
        table = torch.ones((2, 1), dtype=torch.int32, device="npu")
        lengths = torch.tensor([2, 2], dtype=torch.int32, device="npu")
        cu_q = torch.tensor([1, 2], dtype=torch.int32, device="npu")
        cache = EchoRecallCache(host_size=512, capacity=128)
        graph = None
        try:
            cache.write(torch.tensor([128, 129], dtype=torch.int32, device="npu"),
                        self.keys[:2].npu(), self.ropes[:2].npu())
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                ids = map_topk_to_host(selected, table, cu_q, lengths, 128)
                slots = cache.recall(ids)
            for active in (True, False, True, False, True):
                lengths.fill_(2 if active else 0)
                graph.replay()
                torch.npu.synchronize()
                cache.check_errors()
                if active:
                    torch.testing.assert_close(ids.cpu(), torch.tensor([[128, 129]] * 2,
                                                                     dtype=torch.int32))
                    self.assertTrue(torch.all(slots.cpu() >= 0))
                else:
                    self.assertTrue(torch.all(ids.cpu() == -1))
                    self.assertTrue(torch.all(slots.cpu() == -1))
                self.assertEqual(cache.miss_count.item(), 0)
        finally:
            del graph
            cache.close()

    def test_serving_capacity_populated_graph_hits(self):
        cache = EchoRecallCache(host_size=8320, capacity=4096)
        graph = None
        try:
            context = torch.arange(128, 1152, dtype=torch.int32, device="npu")
            keys = self.keys.repeat(32, 1, 1).npu()
            ropes = self.ropes.repeat(32, 1, 1).npu()
            cache.write(context, keys, ropes)
            torch.npu.synchronize()
            mapped = cache.host_to_device[context.long()]
            self.assertTrue(torch.all((mapped >= 0) & (mapped < cache.capacity)).item())
            torch.testing.assert_close(cache.key[mapped.long()].cpu(), keys.cpu(), rtol=0, atol=0)
            ids = torch.full((2, 2048), -1, dtype=torch.int32, device="npu")
            ids[:, :1024].copy_(context.expand(2, -1))
            loc = torch.tensor([1152, 1153], dtype=torch.int32, device="npu")
            new_keys, new_ropes = self.keys[:2].npu(), self.ropes[:2].npu()

            def run():
                cache.write(loc, new_keys, new_ropes)
                ids[:, 1024].copy_(loc)
                return cache.recall(ids)

            run()
            torch.npu.synchronize()
            cache.check_errors()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = run()
            for step in range(4):
                loc.copy_(torch.tensor([1154 + step * 2, 1155 + step * 2], dtype=torch.int32))
                graph.replay()
                torch.npu.synchronize()
                cache.check_errors()
                self.assertEqual(cache.miss_count.item(), 0)
                torch.testing.assert_close(cache.key[output[0, :1024].long()].cpu(),
                                           keys.cpu(), rtol=0, atol=0)
                torch.testing.assert_close(cache.key[output[:, 1024].long()].cpu(),
                                           new_keys.cpu(), rtol=0, atol=0)
        finally:
            del graph
            cache.close()

    def test_bulk_release_and_rewrite_at_serving_capacity(self):
        cache = EchoRecallCache(host_size=8320, capacity=4096)
        graph = None
        try:
            context = torch.arange(128, 1152, dtype=torch.int32, device="npu")
            generator = torch.Generator().manual_seed(43)
            keys = torch.randn(1024, 1, 512, generator=generator).to(torch.bfloat16).npu()
            ropes = torch.randn(1024, 1, 64, generator=generator).to(torch.bfloat16).npu()
            repeated = torch.cat((context, context))

            def run():
                cache.release(repeated)
                cache.write(context, keys, ropes)

            run()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run()
            for _ in range(3):
                graph.replay()
                torch.npu.synchronize()
                cache.check_errors()
                mapped = cache.host_to_device[context.long()].cpu().long()
                self.assertEqual(mapped.unique().numel(), 1024)
                self.assertEqual(cache.free_count.item(), 3072)
                free = cache.free_slots[:3072].cpu().long()
                self.assertEqual(free.unique().numel(), 3072)
                self.assertTrue(torch.all(cache.device_to_host.cpu()[free] == -1))
                torch.testing.assert_close(cache.key[mapped.npu()].cpu(), keys.cpu(), rtol=0, atol=0)
        finally:
            del graph
            cache.close()


if __name__ == "__main__":
    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    unittest.main(verbosity=2)
