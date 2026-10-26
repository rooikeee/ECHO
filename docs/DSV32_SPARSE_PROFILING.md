# DS-V3.2 Sparse Operator Profiling

## Scope

2026-10-23: this experiment profiles the PR 17392 Indexer/management and
CopySFA operators on one physical Ascend 910C device (device 6, board query
reports chip Ascend910 / NPU name 9382). It is a synthetic single-layer,
single-rank experiment, not DeepSeek-V3.2-AWQ full-model inference.

The real AWQ checkpoint was checked with the current vLLM Ascend model
configuration, TP8, max model length 41024. Configuration fails before weight
loading with:

```text
Value error, auto_awq quantization is currently not supported in npu.
```

The current Ascend platform's supported quantization list does not include
AWQ/auto_awq. Existing SGLang AWQ full-model records do not establish vLLM
support, and SGLang does not call these two PR fusion operators.

The installed MemFabric 1.2.0 also lacks the PR manager's
`OffloadConfig.store_url` field. A separate single-rank shared allocator probe
without that field fails initialization because the extension/hcom library is
not configured. No serving configuration or quantization support was bypassed.

## Workload

- Context: 40960 tokens per request, interpreted as 40 Ki tokens.
- Batch: 1, 2, 4, 8, 16 independent one-token decode requests, no MTP.
- BF16 operator inputs, irrespective of the checkpoint's AWQ weight format.
- Indexer: 64 replicated query heads, head dimension 128; independent paged
  index-key storage per request; scores cover the offloaded prefix of 40832
  tokens, with the remaining 128 tokens forming the recent tail.
- Attention: 16 local heads, corresponding to 128 total heads / TP8; the
  experiment executes only one rank and does not include HCCL or other ranks.
- Hot cache: 8192 tokens per request; TopK 2048; recent tail 128.
- CopySFA: 0, 1024, or 2048 misses per request, with distinct random source
  token indices distributed across the host prefix. These are controlled
  workloads, not measured full-model hit rates.
- Host storage is real mmap-backed DRAM registered with `aclrtHostRegister`.
  Since CPU and device virtual addresses differ, the native adapter receives
  non-owning metadata views of the device aliases. These views must never be
  dereferenced on CPU. This is not the service's MemFabric allocation path.

Each batch validates capture and repeated replay. Indexer output/cache state
is compared exactly with eager execution and TopK with the native indexer.
CopySFA is captured in the hit state and replayed with changing miss metadata;
host reads and persistent cache writes are compared exactly, and attention
output is compared with an independent PyTorch reference. Registered buffers
remain alive through synchronization and graph destruction. Cache state is
restored before each timed replay on the same stream, outside timing events.

## Reproduce

First verify the selected device is healthy and unused. Use a new output
directory; the script refuses to overwrite an existing experiment directory.

```bash
ECHO_NPU_DEVICE=6 HF_HUB_OFFLINE=1 ./scripts/vllm-ascend-run.sh \
  python -u scripts/collect-vllm-sparse-profile-npu.py \
  --batches 1 2 4 8 16 --context 40960 --tp 8 --hot-tokens 8192 \
  --repeats 30 --profile-repeats 10 \
  --output build/profiling/dsv32-40k-b1-16-r1
```

Results include per-batch metrics, an aggregate `result.json`, device kernel
CSV files, traces and pipe-utilization counters. Profiling uses the isolated
vLLM/CANN 9.1 environment, not the SGLang/CANN 8.5 environment. Importing Triton
can take 100 seconds because it queries all devices with `npu-smi info`.

## Recorded Results

Completed on 2026-10-23 in
`build/profiling/dsv32-40k-b1-16-r1/`, status
`completed_operator_only`: 20 workload measurements and 207 target-kernel
records. Each workload has 30 unprofiled event-timed replays and 10 or 11
profiled kernel records (an in-flight warmup can cross the collection boundary).
All five batches passed the correctness checks described above. Device 6 was
healthy and had no remaining process after completion.

Mean device target-kernel duration in microseconds, excluding cache resets:

| Batch | Indexer + Manage | CopySFA 0 miss | CopySFA 1024 miss | CopySFA 2048 miss | Full-Miss Minus Hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 312.957 | 77.664 | 108.806 | 134.583 | 56.919 |
| 2 | 376.856 | 77.729 | 107.392 | 139.699 | 61.971 |
| 4 | 399.588 | 80.266 | 111.424 | 174.825 | 94.558 |
| 8 | 420.699 | 83.624 | 154.349 | 227.133 | 143.509 |
| 16 | 492.679 | 97.743 | 231.211 | 411.148 | 313.405 |

Miss counts are per request, not per batch. The Indexer workload has roughly
1611--1674 misses per request under its random, seeded resident mapping; it is
measured independently of the three controlled CopySFA workloads. It is not a
chained Indexer-to-CopySFA graph. Host source indices and queries are fixed
within each workload's replays; this is a warmed repeated workload, not a
cold-cache or streaming-bandwidth experiment.

At batch 16, the CopySFA AIV MTE2 ratio rises from approximately 46.3% for hits
to 87.4% for full miss. This is an operator pipe counter, not measured host-link
bandwidth or proof of saturation. The visible full-miss increment grows from
approximately 57 us at batch 1 to 313 us at batch 16. These results justify a
follow-up overlap experiment but do not establish an achievable prefetch gain.

## Interpretation

Use the profiler's target-kernel `Duration(us)` for kernel latency. Event
measurements also include replay submission gaps, which can dominate a short
kernel. Pipe-counter collection can perturb execution; the values are not
end-to-end latency measurements. Cache-reset kernels are excluded from the
target-kernel statistics.

CopySFA(full miss) minus CopySFA(all hit) estimates the exposed cost of changing
the source to host DRAM for this workload. It is not standalone transfer time:
the operator already pipelines reads, writeback and attention. Likewise,
Indexer duration is not the available prefetch window. Aggregate AIC/AIV pipe
counters do not prove temporal overlap or establish when a candidate becomes
available. A real prefetch implementation and a controlled concurrent-load
experiment remain necessary to establish how much latency can be hidden.
