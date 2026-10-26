"""Export compact benchmark evidence without model weights or dataset text."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    status = json.loads((root / "controller-status.json").read_text())
    cases = []
    expected = ["shared30-rate-0.3"] + [
        f"{v}60-rate-{r}" for v in ["optimized", "original"] for r in ["0.3", "0.5", "0.7", "inf"]
    ]
    for label in expected:
        p = root / f"{label}.json"
        if not p.exists():
            partial = []
            log = root / f"{label}.log"
            if log.exists():
                for line in log.read_text(errors="replace").splitlines():
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if "index" in entry:
                        partial.append({k: v for k, v in entry.items() if k != "stream_intervals_s"})
            cases.append(
                {
                    "case": label,
                    "status": "not_completed",
                    "completed_requests": len(partial),
                    "successful_completed_requests": sum(bool(x["success"]) for x in partial),
                    "partial_requests": partial,
                }
            )
            continue
        data = json.loads(p.read_text())
        samples = data.get("scheduler_samples", [])
        summary = {k: v for k, v in data.items() if k not in ["requests_detail", "scheduler_samples"]}
        summary.update(
            case=label,
            status="passed" if data["successful"] == data["requests"] else "failed",
            raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            max_running=max((x.get("num_requests_running", 0) for x in samples), default=0),
            max_waiting=max((x.get("num_requests_waiting", 0) for x in samples), default=0),
            scheduler_samples=samples,
            per_request=[{k: v for k, v in x.items() if k != "stream_intervals_s"} for x in data["requests_detail"]],
        )
        cases.append(summary)
    result = {
        "date": "2026-10-26",
        "model": "Eco-Tech/GLM-5.2-w4a8",
        "revision": "29e0e32383ce2a95f6cbac7322f3b830a70301eb",
        "task": "longbook_sum_eng",
        "tp": 16,
        "ep": 16,
        "mtp": 3,
        "synthetic_kv": True,
        "accuracy_claim": False,
        "input_tokens": 64000,
        "output_tokens": 256,
        "weight_cache_budget_fraction": 0.9,
        "resident_tokens_per_request": 12288,
        "shared_cache_capacity": 16,
        "prefetch_enabled": False,
        "queue_sampling_seconds": 1,
        "deadline": "2026-10-26T21:00:00+08:00",
        "cleanup_time": "2026-10-26T20:58:00+08:00",
        "run_status": status["status"],
        "run_error": status.get("error"),
        "finished_at": status.get("finished_at"),
        "cases": cases,
    }
    evidence = {}
    for variant in ["optimized", "original"]:
        path = root / f"{variant}-server.log"
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        budgets = []
        for line in text.splitlines():
            if "ECHO_MEMORY_BUDGET " in line:
                budgets.append(json.loads(line.split("ECHO_MEMORY_BUDGET ", 1)[1]))
        counts = re.findall(r"Accepted: (\d+) tokens, Drafted: (\d+) tokens", text)
        accepted = sum(int(a) for a, _ in counts)
        drafted = sum(int(b) for _, b in counts)
        evidence[variant] = {
            "graph_replay_logged": "ACL graph replay is active" in text,
            "per_rank_memory_budgets": budgets,
            "logged_draft_accepted": accepted,
            "logged_draft_proposed": drafted,
            "approx_output_per_spec_step": 1 + 3 * accepted / drafted if drafted else None,
            "acceptance_scope": "all server metric windows, including smoke and all rates; not per-rate",
        }
    result["runtime_evidence"] = evidence
    for name in ["devices-after.log", "deadline-device-check.log"]:
        p = root / name
        if p.exists():
            result[name] = p.read_text()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 910C synthetic-KV English book summary benchmark",
        "",
        "16 devices, GLM-5.2 W4A8, TP16/EP16, MTP3, 64k input / 256 output tokens. "
        "Output text is not an accuracy result. Maximum concurrency is sampled every second.",
        "",
        "| Case | Success | Output token/s | Mean TTFT (s) | Mean TPOT (ms) | Max running | Max waiting |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        if c["status"] == "not_completed":
            lines.append(f"| {c['case']} | Not completed | — | — | — | — | — |")
        else:
            lines.append(
                f"| {c['case']} | {c['successful']}/{c['requests']} | {c['output_tokens_per_s']:.2f} | {c.get('mean_ttft_s', 0):.2f} | {1000 * c.get('mean_tpot_s', 0):.2f} | {c['max_running']:.0f} | {c['max_waiting']:.0f} |"
            )
    lines += [
        "",
        f"Run status: {status['status']}. Error: {status.get('error', 'none')}.",
        "",
        "See ../ECHO_INFINITEBENCH_910C.md for setup, methodology, memory accounting and limitations.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
