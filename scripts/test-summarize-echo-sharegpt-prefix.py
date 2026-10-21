"""CPU-only checks for stopped-benchmark recovery boundaries."""

import importlib.util
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "prefix_summary", Path(__file__).with_name("summarize-echo-sharegpt-prefix.py")
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def admission(second, status=200):
    return (
        f'[2026-10-10 10:00:{second:02d}] INFO: 127.0.0.1:1234 '
        f'"POST /generate HTTP/1.1" {status} OK'
    )


class PrefixWindowTests(unittest.TestCase):
    def test_uses_next_paired_admission_as_estimated_endpoint(self):
        text = "\n".join(admission(i) for i in (0, 10, 10, 40, 40))
        window = summary.recover_window(text, count=2, concurrency=2, excluded=1)
        self.assertEqual(window["performance_admissions"], 4)
        self.assertEqual(window["completion_boundary_admission"], 4)
        self.assertEqual(window["log_window_seconds_approx"], 30)

    def test_n_admissions_do_not_prove_n_completions(self):
        text = "\n".join(admission(i) for i in range(3))
        with self.assertRaisesRegex(ValueError, "Insufficient admissions"):
            summary.recover_window(text, count=2, concurrency=2, excluded=0)

    def test_http_errors_are_not_successful_admissions(self):
        text = "\n".join([admission(0), admission(1, 503), admission(2), admission(3)])
        with self.assertRaisesRegex(ValueError, "Insufficient admissions"):
            summary.recover_window(text, count=2, concurrency=2, excluded=0)

    def test_only_graph_decode_inside_window_is_counted(self):
        admissions = [admission(i) for i in (0, 10, 10, 40, 40)]
        decode = [
            "[2026-10-10 10:00:05 TP0] npu graph: True, gen throughput (token/s): 99.0",
            "[2026-10-10 10:00:15 TP0] npu graph: True, gen throughput (token/s): 6.0",
            "[2026-10-10 10:00:20 TP0] npu graph: False, gen throughput (token/s): 98.0",
            "[2026-10-10 10:00:25 TP0] npu graph: True, gen throughput (token/s): 8.0",
            "[2026-10-10 10:00:40 TP0] npu graph: True, gen throughput (token/s): 97.0",
        ]
        window = summary.recover_window("\n".join(admissions + decode), 2, 2, 1)
        self.assertEqual(window["graph_decode_log_count"], 2)
        self.assertEqual(window["median_logged_decode_throughput"], 7.0)

    def test_missing_decode_logs_do_not_invent_a_measurement(self):
        text = "\n".join(admission(i) for i in (0, 0, 3, 3))
        window = summary.recover_window(text, 2, 2, 0)
        self.assertIsNone(window["median_logged_decode_throughput"])

    def test_nonpositive_window_is_rejected(self):
        text = "\n".join(admission(0) for _ in range(4))
        with self.assertRaisesRegex(ValueError, "Nonpositive"):
            summary.recover_window(text, 2, 2, 0)

    def test_unpaired_admissions_are_not_treated_as_prefix_completion(self):
        text = "\n".join(admission(i) for i in range(4))
        with self.assertRaisesRegex(ValueError, "not paired"):
            summary.recover_window(text, 2, 2, 0)

    def test_invalid_arguments_are_rejected(self):
        for count, concurrency, excluded in ((0, 2, 0), (2, 0, 0), (2, 2, -1)):
            with self.subTest(count=count, concurrency=concurrency, excluded=excluded):
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    summary.recover_window(admission(0), count, concurrency, excluded)


if __name__ == "__main__":
    unittest.main()
