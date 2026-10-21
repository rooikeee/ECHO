"""Recover an explicitly approximate paired prefix from a stopped benchmark.

HTTP 200 for a streaming request is an admission, not a completion. For this
paired workload, the next pair's admission approximates the prefix endpoint.
This assumes requests cannot complete within one log timestamp bucket and no
earlier request is overtaken. FIFO/concurrency alone do not prove completion.
"""

import argparse
import datetime
import json
import re
import statistics
from pathlib import Path


ADMISSION = re.compile(
    r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*'
    r'"POST /generate HTTP/1\.1" 200 OK'
)
DECODE = re.compile(r"gen throughput \(token/s\): ([0-9.]+)")


def recover_window(text, count, concurrency, excluded):
    admissions = []
    for line in text.splitlines():
        match = ADMISSION.search(line)
        if match:
            admissions.append(datetime.datetime.fromisoformat(match.group(1)))
    performance = admissions[excluded:]
    boundary = count + concurrency - 1
    if count <= 0 or concurrency <= 0 or excluded < 0:
        raise ValueError("Invalid prefix, concurrency or excluded admission count")
    if concurrency != 2 or count % 2:
        raise ValueError("Recovery requires an even prefix and concurrency 2")
    if len(performance) <= boundary:
        raise ValueError("Insufficient admissions to bound prefix completion")
    for i in range(0, count + concurrency, 2):
        if performance[i] != performance[i + 1]:
            raise ValueError("Admissions are not paired within the log timestamp bucket")
    start, end = performance[0], performance[boundary]
    duration = (end - start).total_seconds()
    if duration <= 0:
        raise ValueError("Nonpositive log window")
    decode = []
    for line in text.splitlines():
        match = DECODE.search(line)
        if not match or "npu graph: True" not in line:
            continue
        timestamp = datetime.datetime.fromisoformat(line[1:20])
        if start <= timestamp < end:
            decode.append(float(match.group(1)))
    return {
        "performance_admissions": len(performance),
        "window_start": start.isoformat(sep=" "),
        "completion_boundary_admission": count + concurrency,
        "window_end": end.isoformat(sep=" "),
        "log_window_seconds_approx": duration,
        "median_logged_decode_throughput": statistics.median(decode) if decode else None,
        "graph_decode_log_count": len(decode),
    }


def summarize(directory, count):
    report = json.loads((directory / "result.json").read_text())
    concurrency = report["max_concurrency"]
    excluded = 1 + 3 + report["warmup_requests"]  # Readiness + graph probes + warmup.
    if concurrency != 2 or report["graph_batch_sizes"] != [1, 2]:
        raise ValueError("Recovery currently requires this driver's batch 1/2 probe policy")
    lines = (directory / "baseline-metrics.jsonl").read_text().splitlines()
    if len(lines) != 1:
        raise ValueError("Expected exactly one baseline metric record")
    baseline = json.loads(lines[0])
    expected_lengths = report["prompt_lengths"][:count]
    if baseline["input_lens"][:count] != expected_lengths:
        raise ValueError("Baseline prefix does not match the sample manifest")
    if any(baseline["errors"][:count]):
        raise ValueError("Baseline prefix contains errors")
    if baseline["output_lens"][:count] != [report["output_len"]] * count:
        raise ValueError("Baseline prefix has unexpected output lengths")
    output = {
        "status": "user_stopped_partial_log_recovery",
        "original_driver_status": report["status"],
        "comparison_count": count,
        "model": report["model"],
        "max_concurrency": concurrency,
        "output_tokens_per_request": report["output_len"],
        "prefix_input_tokens": sum(expected_lengths),
        "prefix_output_tokens": count * report["output_len"],
        "excluded_admissions_per_mode": excluded,
        "hardware_health_ok": report["hardware_health_ok"],
        "limitations": [
            "Server admissions are not completions; the next paired admission is an estimated endpoint.",
            "FIFO and concurrency alone do not prove that an exact prefix has completed.",
            "All observed pairs share a timestamp; assumes requests take over one second and no prefix overtaking.",
            "Assumes FIFO semaphore admission, no prefix request failures and fixed concurrency 2.",
            "Log timing is second-granularity and includes next-admission overhead.",
            "ECHO per-request client metrics and generated texts were not saved.",
            "No ECHO TTFT, TPOT, exact E2E latency or text agreement can be recovered.",
            "Logged decode throughput is not end-to-end output throughput.",
            "Original result.json remains failed; recovery is not a completed benchmark.",
        ],
    }
    for mode in ("baseline", "echo"):
        text = (directory / f"{mode}-server.log").read_text()
        recovered = recover_window(text, count, concurrency, excluded)
        recovered["output_throughput_log_estimate"] = (
            output["prefix_output_tokens"] / recovered["log_window_seconds_approx"]
        )
        output[mode] = recovered
    output["baseline"]["mean_client_ttft_ms_prefix"] = (
        statistics.mean(baseline["ttfts"][:count]) * 1000
    )
    output["baseline"]["mean_client_itl_ms_prefix"] = statistics.mean(
        gap for gaps in baseline["itls"][:count] for gap in gaps
    ) * 1000
    # The final DONE event adds latency not represented by TTFT/ITL details.
    output["baseline"]["mean_time_to_last_observed_token_ms_prefix_approx"] = (
        statistics.mean(
            ttft + sum(gaps)
            for ttft, gaps in zip(baseline["ttfts"][:count], baseline["itls"][:count])
        ) * 1000
    )
    client_log = (directory / "echo-benchmark.log").read_text()
    failures = sum(line.startswith("output.error=") for line in client_log.splitlines())
    output["echo"]["interrupted_client_error_count"] = failures
    output["echo"]["successful_requests_inferred_before_shutdown"] = (
        report["num_prompts"] - failures
    )
    if (
        "url: /server_info" not in client_log
        or failures != report["num_prompts"] - (
            output["echo"]["performance_admissions"] - concurrency
        )
    ):
        raise ValueError("Unexpected stop log; cannot establish the recovery assumptions")
    output["echo_vs_baseline_throughput_ratio_log_estimate"] = (
        output["echo"]["output_throughput_log_estimate"]
        / output["baseline"]["output_throughput_log_estimate"]
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()
    summary = summarize(args.directory, args.count)
    path = args.directory / f"stopped-prefix{args.count}-summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
