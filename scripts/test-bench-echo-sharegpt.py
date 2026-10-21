#!/usr/bin/env python3
"""CPU-only checks for the paired ShareGPT driver; no model/server is started."""

import ast
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    "bench_echo_sharegpt", Path(__file__).with_name("bench-echo-sharegpt-npu.py")
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class TestShareGPTDriver(unittest.TestCase):
    def test_incremental_request_journal_survives_missing_aggregate(self):
        source = Path(__file__).resolve().parents[1] / "sglang-ascend/python/sglang/bench_serving.py"
        tree = ast.parse(source.read_text())
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "append_request_result")
        namespace = {"json": json, "asdict": asdict}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)

        @dataclass
        class Output:
            success: bool
            latency: float
            start_time: float

        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "requests.jsonl"
            namespace["append_request_result"](journal, 3, Output(True, 1.2, 100.0))
            namespace["append_request_result"](journal, 0, [Output(False, 0.0, 101.0)])
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual([record["request_index"] for record in records], [3, 0])
            self.assertEqual(records[0]["latency"], 1.2)
            self.assertFalse(records[1]["success"])
            self.assertFalse((Path(directory) / "metrics.jsonl").exists())

    def args(self):
        return SimpleNamespace(
            model=Path("/tmp/model"), dataset=Path("/tmp/data.json"),
            output_dir=Path("/tmp/result"), seed=42, num_prompts=100,
            output_len=128, sample_context_len=2048, prefill_chunk_size=512,
        )

    def result(self):
        return {
            "completed": 2, "input_lens": [10, 20], "output_lens": [128, 128],
            "errors": ["", ""], "generated_texts": ["first", "second"],
            "output_throughput": 10.0, "mean_e2e_latency_ms": 2000.0,
            "mean_ttft_ms": 100.0, "mean_tpot_ms": 10.0, "duration": 20.0,
        }

    def test_paired_server_settings(self):
        baseline = bench.server_command(self.args(), 1234, False)
        echo = bench.server_command(self.args(), 1234, True)
        self.assertEqual(echo[:-3], baseline)
        self.assertEqual(echo[-3:], ["--enable-echo", "--echo-device-cache-size", "4096"])
        self.assertNotIn("--disable-cuda-graph", baseline)
        self.assertNotIn("--enable-dp-attention", baseline)
        self.assertNotIn("--disaggregation-mode", baseline)
        self.assertEqual(baseline[baseline.index("--chunked-prefill-size") + 1], "512")

    def test_paired_benchmark_settings(self):
        command = bench.benchmark_command(self.args(), 1234, "baseline")
        self.assertIn("--benchmark-worker", command)
        for name, value in [
            ("--dataset-name", "sharegpt"), ("--num-prompts", "100"),
            ("--sharegpt-output-len", "128"), ("--max-concurrency", "2"),
            ("--seed", "42"), ("--warmup-requests", "2"),
        ]:
            self.assertEqual(command[command.index(name) + 1], value)
        self.assertNotIn("--disable-stream", command)
        self.assertNotIn("--disable-ignore-eos", command)
        self.assertEqual(command[command.index("--output-request-file") + 1],
                         "/tmp/result/baseline-requests.jsonl")

    def test_result_validation(self):
        samples = [{"input_tokens": 10}, {"input_tokens": 20}]
        bench.validate_result(self.result(), samples, 128)
        for field, value in [
            ("completed", 1), ("input_lens", [20, 10]),
            ("output_lens", [128, 127]), ("errors", ["failure", ""]),
        ]:
            with self.subTest(field=field):
                result = self.result()
                result[field] = value
                with self.assertRaises(AssertionError):
                    bench.validate_result(result, samples, 128)

    def test_graph_validation(self):
        log = (
            "Capture npu graph end\nECHO exact recall enabled\n"
            "Decode batch, #running-req: 1, npu graph: True\n"
            "Decode batch, #running-req: 2, npu graph: True\n"
        )
        bench.validate_graph(log, True)
        for invalid in (
            log.replace("#running-req: 2", "#running-req: 3"),
            log.replace("ECHO exact recall enabled", ""),
            log + "Decode batch, #running-req: 2, npu graph: False\n",
        ):
            with self.assertRaises(AssertionError):
                bench.validate_graph(invalid, True)

    def test_comparison_direction_and_quality_scope(self):
        baseline = self.result()
        echo = dict(baseline, output_throughput=20.0, mean_e2e_latency_ms=1000.0,
                    generated_texts=["first", "changed"])
        comparison = bench.compare_results(baseline, echo)
        self.assertEqual(comparison["output_throughput_speedup"], 2.0)
        self.assertEqual(comparison["mean_e2e_latency_speedup"], 2.0)
        self.assertEqual(comparison["generated_text_agreement_fraction"], 0.5)
        self.assertFalse(comparison["quality_accuracy_measured"])

    def test_idle_check_does_not_stop_other_processes(self):
        info = "\n".join(f"No running processes found in NPU {i}" for i in range(4, 8))
        bench.ensure_idle(info)
        with self.assertRaises(RuntimeError):
            bench.ensure_idle(info.replace("No running processes found in NPU 6", "occupied"))

    def test_owned_process_cleanup(self):
        process = Mock(pid=1234)
        with patch.object(bench.os, "killpg") as killpg:
            bench.stop_process(process)
            killpg.assert_called_once_with(1234, bench.signal.SIGTERM)
        process = Mock(pid=1234)
        process.wait.side_effect = [subprocess.TimeoutExpired("test", 30), None]
        with patch.object(bench.os, "killpg") as killpg:
            bench.stop_process(process)
            self.assertEqual(killpg.call_args_list[-1].args, (1234, bench.signal.SIGKILL))

    def test_graph_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args()
            args.output_dir = Path(directory)
            (args.output_dir / "baseline-server.log").write_text(
                "Capture npu graph end\n"
                "Decode batch, #running-req: 1, npu graph: True\n"
                "Decode batch, #running-req: 2, npu graph: True\n"
            )
            with patch.object(bench, "post_json", return_value={
                "meta_info": {"completion_tokens": 32}
            }) as post:
                bench.graph_probe(args, 1234, "baseline")
                self.assertEqual(post.call_count, 3)
            with patch.object(bench, "post_json", return_value={
                "meta_info": {"completion_tokens": 31}
            }):
                with self.assertRaises(AssertionError):
                    bench.graph_probe(args, 1234, "baseline")

    def test_baseline_reuse_checks_inputs_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            args = self.args()
            args.output_dir = source
            args.reuse_baseline_dir = source
            samples = [{"input_tokens": 10}, {"input_tokens": 20}]
            bench.save_json(source / "samples.json", samples)
            bench.save_json(source / "baseline-server-command.json",
                            bench.server_command(args, 1234, False))
            bench.save_json(source / "baseline-benchmark-command.json",
                            bench.benchmark_command(args, 1234, "baseline"))
            (source / "baseline-metrics.jsonl").write_text(json.dumps(self.result()) + "\n")
            (source / "baseline-server.log").write_text(
                "Capture npu graph end\n"
                "Decode batch, #running-req: 2, npu graph: True\n"
            )
            self.assertEqual(bench.load_baseline(args, samples), self.result())
            legacy_command = bench.benchmark_command(args, 1234, "baseline")[:-2]
            bench.save_json(source / "baseline-benchmark-command.json", legacy_command)
            self.assertEqual(bench.load_baseline(args, samples), self.result())
            with self.assertRaises(AssertionError):
                bench.load_baseline(args, list(reversed(samples)))
            args.prefill_chunk_size = 256
            with self.assertRaises(AssertionError):
                bench.load_baseline(args, samples)

    def test_atomic_result_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            bench.save_json(path, {"status": "prepared"})
            self.assertIn("prepared", path.read_text())
            self.assertFalse(path.with_suffix(".json.partial").exists())
            self.assertEqual(len(bench.file_hash(path)), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
