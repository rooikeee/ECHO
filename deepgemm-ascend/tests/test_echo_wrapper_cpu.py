"""CPU-only checks of the real graph launch wrapper, without loading DeepGEMM."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


class WrapperTest(unittest.TestCase):
    def setUp(self):
        self.package = types.ModuleType("echo_wrapper_cpu_test")
        self.package.__path__ = []
        self.package.get_num_sms = lambda: 32
        self.patch = patch.dict(sys.modules, {self.package.__name__: self.package})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        path = Path(__file__).resolve().parents[1] / "deep_gemm/echo.py"
        name = self.package.__name__ + ".echo"
        spec = importlib.util.spec_from_file_location(name, path)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.module
        self.addCleanup(lambda: sys.modules.pop(name, None))
        spec.loader.exec_module(self.module)

    def test_repeated_run_resets_outputs_and_reuses_addresses(self):
        source = torch.zeros(17, 3, dtype=torch.uint8)
        buffers = self.module.EchoPrefetchBuffers(
            torch.empty_strided((4, 9), (256, 1), dtype=torch.bfloat16),
            torch.full((4,), 123, dtype=torch.int32), torch.full((4, 2), 7, dtype=torch.int32),
            torch.full((4, 2), 7, dtype=torch.int32), torch.full((4, 2, 3), 211, dtype=torch.uint8),
            torch.empty(33, 2, dtype=torch.int32), 123456, source, "HBM-diagnostic")
        inputs = [torch.empty(1) for _ in range(14)]
        inputs[12] = torch.full((17,), -1, dtype=torch.int32)
        addresses = [t.data_ptr() for t in (buffers.logits, buffers.counts, buffers.tokens, buffers.hosts, buffers.staging, buffers.metadata)]
        calls = []
        def native(*args):
            self.assertEqual(len(args), 21)
            self.assertIs(args[7], buffers.metadata)
            self.assertIs(args[13], inputs[12])
            self.assertEqual(args[-1], 123456)
            self.assertTrue(bool((buffers.counts == 0).all()))
            self.assertTrue(bool((buffers.tokens == -1).all()))
            self.assertTrue(bool((buffers.hosts == -1).all()))
            self.assertTrue(bool(torch.isneginf(buffers.logits).all()))
            self.assertTrue(bool((buffers.staging == 211).all()))
            buffers.counts.fill_(2); buffers.tokens.fill_(4); buffers.hosts.fill_(5);buffers.logits.fill_(3)
            calls.append(1)
        self.package._C = types.SimpleNamespace(echo_paged_mqa_prefetch_out=native)
        # A replay launch must never allocate or extract a device scalar in Python.
        with patch.object(torch, "empty", side_effect=AssertionError("allocation during run")), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar read")):
            for _ in range(3):
                self.assertIs(buffers.run(*inputs),buffers.logits)
        self.assertEqual(len(calls),3)
        self.assertEqual(addresses,[t.data_ptr() for t in (buffers.logits,buffers.counts,buffers.tokens,buffers.hosts,buffers.staging,buffers.metadata)])
        inputs[12] = torch.empty(18,dtype=torch.int32)
        with self.assertRaisesRegex(ValueError,"different token counts"):
            buffers.run(*inputs)

    def test_cpu_source_is_not_silently_used_as_device_address(self):
        source = torch.zeros(17, 3, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError,"NPU tensor"):
            self.module.EchoPrefetchBuffers.allocate(4,128,8,source)
        with self.assertRaisesRegex(ValueError,"DVA"):
            self.module.EchoPrefetchBuffers.allocate(4,128,8,source,source_kind="MemFabric-DDR")
        with self.assertRaisesRegex(ValueError,"capacity"):
            self.module.EchoPrefetchBuffers.allocate(4,128,2049,source)


if __name__=="__main__":unittest.main()
