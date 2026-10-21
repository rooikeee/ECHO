#!/usr/bin/env python3
"""CPU-only checks of ECHO TP/DP validation and the paired serving driver."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "echo_parallel_smoke", Path(__file__).with_name("smoke-echo-parallel-npu.py")
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class TestEchoParallel(unittest.TestCase):
    def check_echo_args(self, **overrides):
        args = SimpleNamespace(
            enable_echo=True, device="npu", attention_backend="ascend",
            disable_radix_cache=True, disable_overlap_schedule=True,
            max_total_tokens=8192, echo_device_cache_size=4096,
            enable_dp_attention=True, tp_size=8, dp_size=8, attn_cp_size=1,
            enable_hierarchical_cache=False, enable_hisparse=False,
            enable_lmcache=False, speculative_algorithm=None,
            enable_torch_compile=False, enable_memory_saver=False,
            pp_size=1, disaggregation_mode="null",
        )
        vars(args).update(overrides)
        source = ROOT / "sglang-ascend/python/sglang/srt/server_args.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "ServerArgs")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "check_server_args")
        block = next(node for node in method.body if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "self.enable_echo")
        exec(compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec"),
             {"self": args})

    def test_supported_attention_topologies(self):
        for tp, dp in ((1, 1), (2, 1), (2, 2), (8, 1), (8, 2), (8, 4), (8, 8)):
            with self.subTest(tp=tp, dp=dp):
                self.check_echo_args(tp_size=tp, dp_size=dp, enable_dp_attention=dp > 1)
                smoke.validate_topology(tp, dp, [str(i) for i in range(tp)])

    def test_invalid_dp_topologies(self):
        for dp in (0, 3, 16):
            with self.subTest(dp=dp), self.assertRaises(ValueError):
                self.check_echo_args(dp_size=dp)

    def test_unsupported_features_still_rejected(self):
        for overrides in (
            {"attn_cp_size": 2}, {"pp_size": 2}, {"disaggregation_mode": "decode"},
            {"enable_torch_compile": True}, {"enable_hierarchical_cache": True},
            {"disable_overlap_schedule": False}, {"disable_radix_cache": False},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.check_echo_args(**overrides)

    def test_rank_group_layout(self):
        source = ROOT / "sglang-ascend/python/sglang/srt/layers/dp_attention.py"
        tree = ast.parse(source.read_text())
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "compute_dp_attention_world_info")
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
        compute = namespace[helper.name]
        for tp, dp in ((2, 1), (2, 2), (8, 2), (8, 4), (8, 8)):
            actual = [compute(dp > 1, rank, tp, dp) for rank in range(tp)]
            expected = [(rank % (tp // dp), tp // dp, rank // (tp // dp)) for rank in range(tp)]
            self.assertEqual(actual, expected)

    def args(self, dp=2):
        return SimpleNamespace(model=Path("/tmp/model"), tp_size=2, dp_size=dp,
                               tiny=True, quantization=None)

    def test_paired_server_configuration(self):
        baseline = smoke.server_command(self.args(), 1234, False)
        echo = smoke.server_command(self.args(), 1234, True)
        self.assertEqual(echo[:-3], baseline)
        self.assertIn("--enable-dp-attention", baseline)
        self.assertNotIn("--disable-cuda-graph", baseline)
        self.assertNotIn("--disaggregation-mode", baseline)
        self.assertNotIn("--enable-dp-attention", smoke.server_command(self.args(dp=1), 1234, False))
        self.assertEqual(baseline[baseline.index("--chunked-prefill-size") + 1], "1024")
        full_args = self.args()
        full_args.tiny = False
        full_args.quantization = "awq"
        full = smoke.server_command(full_args, 1234, True)
        self.assertNotIn("--load-format", full)
        self.assertIn("--trust-remote-code", full)
        self.assertEqual(full[full.index("--quantization") + 1], "awq")
        self.assertEqual(full[full.index("--watchdog-timeout") + 1], "1800")
        self.assertEqual(full[full.index("--mem-fraction-static") + 1], "0.96")

    def test_tiny_request_inputs(self):
        inputs = smoke.request_inputs(self.args())
        self.assertEqual(len(inputs["single"]), 2)
        self.assertEqual(len(inputs["concurrent"]), 2)
        self.assertEqual(len(inputs["long"]), 2200)

    def test_device_count_and_occupancy(self):
        for devices in (["4"], ["4", "4"], ["x", "7"]):
            with self.assertRaises(ValueError):
                smoke.validate_topology(2, 2, devices)
        info = "Process id\n| 2 1 | 38304 | VLLMEngineCor |\n| 4 0 | 3791625 | VLLMWorker |\n"
        smoke.ensure_devices_idle(info, ["4", "7"])
        with self.assertRaises(RuntimeError):
            smoke.ensure_devices_idle(info, ["8", "9"])
        with self.assertRaises(RuntimeError):
            smoke.ensure_devices_idle("driver failed", ["4"])

    def test_server_port_checks_dp_block_and_closes_probes(self):
        occupied = Mock()
        occupied.bind.side_effect = OSError("occupied")
        free = [Mock() for _ in range(7)]
        with patch.object(smoke.random.SystemRandom, "sample", return_value=[16000, 16016]), \
                patch.object(smoke.socket, "socket", side_effect=[occupied, *free]):
            self.assertEqual(smoke.choose_server_port(), 16016)
        occupied.close.assert_called_once()
        for sock, offset in zip(free, (0, 233, 234, 235, 236, 237, 238)):
            sock.bind.assert_called_once_with(("127.0.0.1", 16016 + offset))
            sock.close.assert_called_once()

    def test_graph_evidence_and_layout(self):
        log = "Capture npu graph end\n" * 2
        for rank in (0, 1):
            for batch in (1, 2):
                log += f"[test DP{rank} TP{rank}] Decode batch, #running-req: {batch}, npu graph: True\n"
        for rank in (0, 1):
            log += f"ECHO parallel layout: attention_tp_rank=0/1, attention_dp_rank={rank}/2\n"
        smoke.validate_graph(log, 2, 2, True)
        mixed = log + "[test DP0 TP0] Decode batch, npu graph: False, dp global prefill: True\n"
        self.assertEqual(smoke.validate_graph(mixed, 2, 2, True)["mixed_prefill_decode_eager_log_entries"], 1)
        for invalid in (log.replace("attention_dp_rank=1/2", "attention_dp_rank=0/2"),
                        log.replace("#running-req: 2", "#running-req: 3"),
                        log.replace("DP1 TP1", "DP0 TP0"),
                        log + "Decode batch, npu graph: False\n"):
            with self.assertRaises(AssertionError):
                smoke.validate_graph(invalid, 2, 2, True)

    def test_comparison_does_not_hide_divergence(self):
        def record(token, logprob):
            return {"label": "single", "dp_rank": 0, "inputs": [[4, 5]], "responses": [
                {"meta_info": {"output_token_logprobs": [[logprob, token]] * 16}}
            ]}
        self.assertTrue(smoke.compare_results([record(5, -1.0)], [record(5, -1.01)])["passed"])
        self.assertFalse(smoke.compare_results([record(5, -1.0)], [record(6, -1.0)])["passed"])
        self.assertFalse(smoke.compare_results([record(5, -1.0)], [record(5, -1.1)])["passed"])
        with self.assertRaises(AssertionError):
            smoke.compare_results([record(5, -1.0)], [record(5, float("nan"))])


if __name__ == "__main__":
    unittest.main(verbosity=2)
