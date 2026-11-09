"""CPU checks of preallocated launch/cleanup ordering, without loading DeepGEMM."""
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

    def buffers(self):
        return self.module.EchoPrefetchBuffers(
            torch.empty_strided((4, 9), (256, 1), dtype=torch.bfloat16),
            torch.zeros(4, dtype=torch.int32), torch.full((4, 2), -1, dtype=torch.int32),
            torch.full((4, 2), -1, dtype=torch.int32), torch.full((4, 2, 3), 211, dtype=torch.uint8),
            torch.empty(33, 2, dtype=torch.int32), 123456, torch.zeros(17, 3, dtype=torch.uint8),
            "HBM-diagnostic", 1000)

    def test_run_preserves_old_tags_until_native_cleanup(self):
        buffers = self.buffers()
        inputs = [torch.empty(1) for _ in range(14)]
        inputs[12] = torch.full((17,), -1, dtype=torch.int32)
        addresses = [t.data_ptr() for t in (buffers.logits,buffers.counts,buffers.tokens,buffers.hosts,buffers.staging,buffers.metadata)]
        calls=[]
        def clear(mapping, counts, tokens, hosts, resident_capacity):
            # Production cleanup algorithm is exercised separately in C++.
            for flat,h in enumerate(hosts.flatten().tolist()):
                if h>0 and int(mapping[h]) == resident_capacity + flat:
                    mapping[h] = -1
            counts.zero_();tokens.fill_(-1);hosts.fill_(-1)
        def native(*args):
            self.assertEqual(len(args),22)
            self.assertIs(args[7],buffers.metadata)
            self.assertIs(args[13],inputs[12])
            self.assertEqual(args[-2:],(123456,1000))
            if calls:
                # Python must not erase the previous list before the device
                # cleanup kernel can identify its exact map entries.
                self.assertTrue(bool((buffers.counts == 1).all()))
                self.assertEqual(int(buffers.hosts[0,0]),5)
                self.assertEqual(int(inputs[12][5]),1000)
            clear(inputs[12],buffers.counts,buffers.tokens,buffers.hosts,1000)
            self.assertEqual(int(inputs[12][5]),-1)
            self.assertTrue(bool(torch.isneginf(buffers.logits).all()))
            buffers.counts.fill_(1);buffers.tokens[0,0]=4;buffers.hosts[0,0]=5;buffers.logits.fill_(3)
            inputs[12][5]=1000
            calls.append(1)
        self.package._C=types.SimpleNamespace(echo_paged_mqa_prefetch_out=native,echo_clear_prefetch_mappings=clear)
        with patch.object(torch,"empty",side_effect=AssertionError("allocation during run")):
            for _ in range(3):self.assertIs(buffers.run(*inputs),buffers.logits)
        self.assertEqual(len(calls),3)
        self.assertEqual(addresses,[t.data_ptr() for t in (buffers.logits,buffers.counts,buffers.tokens,buffers.hosts,buffers.staging,buffers.metadata)])
        inputs[12][5]=7  # Promotion by a future recall consumer.
        buffers.clear_prefetch_mappings()
        self.assertEqual(int(inputs[12][5]),7)
        self.assertTrue(bool((buffers.hosts == -1).all()))
        buffers.clear_prefetch_mappings()
        self.assertEqual(int(inputs[12][5]),7)
        bound=inputs[12]
        inputs[12]=inputs[12].clone()
        with self.assertRaisesRegex(ValueError,"same host_to_device"):
            buffers.run(*inputs)
        # A user resizing the same Tensor object must not silently change the
        # pointer used by cleanup (the keepalive alone would not detect this).
        bound.resize_(18)
        with self.assertRaisesRegex(ValueError,"fixed"):
            buffers.clear_prefetch_mappings()

    def test_run_does_not_extract_scalars_or_allocate(self):
        buffers=self.buffers()
        inputs=[torch.empty(1) for _ in range(14)]
        inputs[12]=torch.full((17,),-1,dtype=torch.int32)
        self.package._C=types.SimpleNamespace(echo_paged_mqa_prefetch_out=lambda *args:None,
                                             echo_clear_prefetch_mappings=lambda *args:None)
        with patch.object(torch,"empty",side_effect=AssertionError("allocation during run")), \
             patch.object(torch.Tensor,"item",side_effect=AssertionError("host scalar read")):
            buffers.run(*inputs);buffers.run(*inputs);buffers.clear_prefetch_mappings()

    def test_mapping_span_and_fixed_buffer_contract(self):
        buffers=self.buffers()
        buffers.resident_capacity+=1
        with self.assertRaisesRegex(ValueError,"fixed"):
            buffers.clear_prefetch_mappings()
        buffers=self.buffers();buffers.hosts=buffers.hosts.clone()
        with self.assertRaisesRegex(ValueError,"fixed"):
            buffers.clear_prefetch_mappings()
        source=torch.zeros(17,3,dtype=torch.uint8)
        for invalid in (0,-1,2**31-1,2**31-5):
            with self.assertRaisesRegex(ValueError,"int32"):
                self.module.EchoPrefetchBuffers.allocate(4,128,8,source,resident_capacity=invalid)
        with self.assertRaises(TypeError):
            self.module.EchoPrefetchBuffers.allocate(4,128,8,source)

    def test_cpu_source_is_not_silently_used_as_device_address(self):
        source=torch.zeros(17,3,dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError,"NPU tensor"):
            self.module.EchoPrefetchBuffers.allocate(4,128,8,source,resident_capacity=1000)
        with self.assertRaisesRegex(ValueError,"DVA"):
            self.module.EchoPrefetchBuffers.allocate(4,128,8,source,resident_capacity=1000,source_kind="MemFabric-DDR")
        with self.assertRaisesRegex(ValueError,"capacity"):
            self.module.EchoPrefetchBuffers.allocate(4,128,2049,source,resident_capacity=1000)


if __name__=="__main__":unittest.main()
