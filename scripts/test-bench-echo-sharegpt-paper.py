#!/usr/bin/env python3
"""CPU-only checks of the half-pool mixed-PD profile."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location(
    "paper_driver", Path(__file__).with_name("bench-echo-sharegpt-paper-npu.py"),
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class TestPaperProfile(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(model=Path("/tmp/model"), dataset=Path("/tmp/data"),
                                    output_dir=Path("/tmp/results"), mem_fraction_static=0.8)

    def test_aligned_half_pools(self):
        self.assertEqual(bench.LOGICAL_TOKENS, 499968)
        self.assertEqual(bench.HOT_TOKENS, 99968)

    def test_device_occupancy_follows_selected_boards(self):
        info = "\n".join(f"No running processes found in NPU {i}" for i in range(4))
        bench.ensure_idle(info, [str(i) for i in range(8)])
        with self.assertRaises(RuntimeError):
            bench.ensure_idle(info, [str(i) for i in range(8, 16)])
        with self.assertRaises(RuntimeError):
            bench.ensure_idle(info.replace("No running processes found in NPU 2", "occupied"),
                              [str(i) for i in range(8)])

    def test_paired_servers(self):
        baseline = bench.server_command(self.args, 18000, False)
        echo = bench.server_command(self.args, 18000, True)
        self.assertEqual(echo[:-5], baseline)
        for flag, value in (("--tp-size", "8"), ("--dp-size", "8"),
                            ("--cuda-graph-max-bs", "32"), ("--chunked-prefill-size", "-1")):
            self.assertEqual(baseline[baseline.index(flag) + 1], value)
        self.assertIn("--enable-dp-attention", baseline)
        for flag in ("--max-running-requests", "--context-length", "--disable-cuda-graph",
                     "--max-total-tokens", "--disaggregation-mode"):
            self.assertNotIn(flag, baseline)
        self.assertEqual(echo[echo.index("--max-total-tokens") + 1], "499968")
        self.assertEqual(echo[echo.index("--echo-device-cache-size") + 1], "99968")

    def test_request_rate_sweep_without_concurrency_or_output_caps(self):
        self.assertEqual(len(bench.REQUEST_RATES), 10)
        for rate in bench.REQUEST_RATES:
            command = bench.benchmark_command(self.args, 18000, "echo", rate)
            self.assertEqual(command[command.index("--request-rate") + 1], rate)
            self.assertEqual(command[command.index("--num-prompts") + 1], "100")
            self.assertEqual(command[command.index("--warmup-requests") + 1], "0")
            for flag in ("--max-concurrency", "--sharegpt-output-len", "--sharegpt-context-len",
                         "--disable-ignore-eos", "--apply-chat-template"):
                self.assertNotIn(flag, command)

    def test_variable_output_budgets(self):
        samples = [{"input_tokens": i + 2, "output_tokens": i + 3} for i in range(100)]
        result = {"completed": 100, "input_lens": [r["input_tokens"] for r in samples],
                  "output_lens": [r["output_tokens"] for r in samples],
                  "errors": [""] * 100, "generated_texts": ["text"] * 100}
        bench.validate_result(result, samples)
        for key, value in (("completed", 99), ("output_lens", [128] * 100),
                           ("input_lens", [2] * 100), ("errors", ["failure"] * 100)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                bench.validate_result(dict(result, **{key: value}), samples)

    def test_pool_clamping_rejected(self):
        log = (f"max_total_num_tokens={bench.LOGICAL_TOKENS}, "
               f"ECHO exact recall enabled: logical tokens={bench.LOGICAL_TOKENS}, "
               f"cache tokens/layer={bench.HOT_TOKENS}\n") * 8
        self.assertEqual(bench.validate_capacity(log, True), [bench.LOGICAL_TOKENS] * 8)
        for invalid in (log.replace(str(bench.LOGICAL_TOKENS), "8192"),
                        log.replace(str(bench.HOT_TOKENS), "4096"), ""):
            with self.assertRaises(AssertionError):
                bench.validate_capacity(invalid, True)

    def test_graph_evidence(self):
        log = "Capture npu graph end\n" * 8
        for rank in range(8):
            log += f"ECHO parallel layout: attention_tp_rank=0/1, attention_dp_rank={rank}/8\n"
            for batch in (1, 2, 16, 32):
                log += f"[TP{rank} DP{rank}] Decode batch, #running-req: {batch}, npu graph: True\n"
        evidence = bench.graph_summary(log, True)
        self.assertEqual(len(evidence["observed_local_decode_batch_log_counts"]), 32)
        with self.assertRaises(AssertionError):
            bench.graph_summary(log + "[TP0 DP0] Decode batch, #running-req: 4, npu graph: False\n", True)
        with self.assertRaises(AssertionError):
            bench.graph_summary(log.replace("[TP7 DP7] Decode batch, #running-req: 2,", "missing"), True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
