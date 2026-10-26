# ECHO Prefetch Operator Profiling

For controlled workloads matching a supplied miss count rather than random
residency, see `ECHO_PREFETCH_MISS400_PROFILING.md` and the collector's
`--misses-per-request 400` option. The requested count is the request union
after deduplicating the four MTP3 rows. These remain synthetic workloads,
not real traces.

## Required Workload: Batch 16, MTP3

Going forward ECHO operator profiling uses **MTP=3**, as recorded in `AGENTS.md`.
The single-query results below are retained as historical measurements, not the
default acceptance workload. The collector now defaults to `--batches 16 --mtp 3`
and compares no prefetch with history-based bounded prefetch.

The current sparse-offload manager allocates target decode width as
`1 + speculative_config.num_speculative_tokens`. Thus MTP3 **target verification**
has four query rows per request: 16 requests, 64 query rows in total. Draft
forwards are separate calls; this experiment does not time the three draft
forwards, acceptance/rejection, model layers or a complete speculative round.
Do not divide by three or four and report a generated-token latency: accepted
tokens have not been measured.

### MTP3 Implementation

The earlier prefetch prototype required T=B and could not measure this workload
with actual prefetch. For this experiment it was extended, not silently disabled:

- LIM accepts internal int64 context `[T,14]`; per-query ready flags, cutoff
  feedback, budgets and staging replace the single-query indexing assumption.
- Each query can prefetch 64 tokens. MTP3 means up to 256 transfers per request,
  4096 per batch, not a 64-token total request budget.
- Both AIV subblocks compute query scores in MTP3. Prefetch still starts after
  each owner's local main loop; this is not a newly optimized pipeline schedule.
- `EchoMTPIndexerPrefetch` pools staged candidates within each request, picks
  one staged copy per unique source, commits each request-union source once,
  and partitions both query-level and request-union miss metadata. Original LIM
  outputs and CopySFA native code remain unchanged.
- Candidate matching uses 32-row tiles to fit the AIV UB. The functional
  consumer adds 17 kernels; it is not optimized for performance.

Host binding headers needed explicit recompilation: invoking the direct Make
target had missed a header change. `build-echo-indexer-prefetch.sh` now forces
the binding translation unit. The complete native rebuild and the forced
binding rebuild succeeded. Initial stale-binding and UB-overflow probes are
retained separately; neither is counted as a successful measurement.

### MTP3 Results

Completed 2026-10-25 on physical NPU 6 with CANN 9.1, Torch 2.10.0+cpu and
Torch-NPU 2.10.0.post4. Other settings remain 40960-token context, 40832-token
offloaded prefix, 8192-token hot cache, TopK 2048, 128-token recent tail, BF16,
64 indexer heads of dimension 128, and 16 local attention heads (TP8 shape only).
The actual execution uses one device with real registered Host DRAM.

Three modes have 30 unprofiled event-timed replays each and 11 profiled task
pairs each (10 active steps plus an in-flight warmup record). Queries remain
fixed, cache mapping and KV/KPE are restored before every measured replay,
and history is warmed from the same query. This is an optimistic predictor,
not consecutive model-generated queries.

Microseconds, per invocation processing all 16 requests and all 64 query rows:

| Mode | LIM Kernel | Consumer Kernel Sum | CopySFA Kernel | Complete Graph Event Mean |
| --- | ---: | ---: | ---: | ---: |
| No prefetch | 1606.530 | 0 | 1289.241 | 2902.643 |
| Prefetch context, budget 0 | 1632.833 | 3474.560 | 1286.145 | 6406.472 |
| History cutoff, budget 64/query | 2401.701 | 3466.576 | 1222.588 | 7120.769 |

Stage columns come from profiler `Duration(us)`; complete-graph event timing
is measured separately without profiler collection. They must not be summed
and treated as one measurement. Complete graph medians are 2900.730, 6400.220,
and 7118.770 us, respectively.

CopySFA decreases by 66.653 us (approximately 5.2%), but LIM increases by
795.171 us, and the consumer adds approximately 3.47 ms. The complete graph is
approximately 2.45x slower with prefetch. The budget-zero result demonstrates
large consumer/integration overhead even without Host prefetch. In history
mode, the two `_mtp_match_rows` launches total approximately 1761.770 us and
the four scatter-element kernels total approximately 1360.023 us. These are
the main exposed consumer costs, not additional CopySFA kernel time.

The baseline has 1598--1671 misses per query and 5918--6187 unique request-union
misses per request. Enabled prefetch copies 4096 tokens in total (4.5 MiB of
CKV/KPE), with 3801 unique useful sources after within-request deduplication.
Those sources cover 4145 query-level misses because some are shared across
query rows. Covered query misses are not the number of unique transfers.

### MTP3 Validation And Artifacts

All modes passed GLOBAL graph capture and three restored-state comparison
replays before timing. Both context modes also passed seven additional
changing/persistent replay scenarios: changed queries, shorter lengths with
budget 7, zero budget, identical query rows producing duplicate candidates,
mixed first-fill/steady requests, steady state again, and persistent all-hit
replay without cache reset. For the budget-zero mode these dynamic checks
intentionally also update device budgets to exercise the same captured graph.

Checks include exact cache mapping and complete KV/KPE against the no-context
reference, unchanged raw LIM outputs and valid union prefixes, exact staging
bytes from registered Host buffers, bounded counts, query/union miss reductions,
finite cutoff feedback, stable buffer addresses, and attention agreement with
rtol=0.02, atol=0.002. Full-model inference, draft-stage integration,
accept/reject rollback and multi-device execution are not validated.

Results: `build/profiling/echo-prefetch-b16-mtp3-20261025-r1/result.json`.
Per-mode kernel CSV, task timing and profiler timeline databases are in that
directory's `batch-16/` subdirectories. Log:
`build/reference/echo-prefetch-b16-mtp3-20261025-r1.log`.
Native build logs: `build/reference/echo-prefetch-mtp3-rebuild-20261025.log` and
`build/reference/echo-prefetch-mtp3-binding-rebuild-20261025.log`.
The required isolated SGLang tiny-model environment baseline also passed
batch-1/2 decode capture/replay after this build. It is not the MTP profiling
workload. Log: `build/reference/echo-prefetch-mtp3-sglang-baseline-20261025.log`.
Probe directories have the prefix `echo-prefetch-b16-mtp3-probe-20261025-`;
only probe r3 completed successfully.

```bash
ECHO_NPU_DEVICE=6 HF_HUB_OFFLINE=1 ./scripts/vllm-ascend-run.sh \
  python -u scripts/collect-echo-prefetch-profile-npu.py \
  --batches 16 --mtp 3 \
  --modes baseline context-budget-0 prefetch-64-history \
  --context 40960 --hot-tokens 8192 --tp 8 \
  --repeats 30 --profile-repeats 10 \
  --output build/profiling/echo-prefetch-b16-mtp3-20261025-r1
```

Use a fresh output directory when repeating; the collector refuses overwrite.

## Scope And Configuration

2026-10-25: profile the currently installed AscendC LIM/CopySFA kernels, including
the experimental optional prefetch path. This is synthetic, single-layer,
single-rank operator execution, not full-model inference or service latency.

The earlier 2026-10-23 experiment is recorded in `DSV32_SPARSE_PROFILING.md`.
Its configuration is retained here:

| Setting | Value |
| --- | --- |
| Physical device | 6, Ascend 910C; `npu-smi` reports Ascend910 |
| Runtime | Isolated vLLM environment, project-local CANN 9.1.0 |
| Python / Torch / Torch-NPU | 3.11.6 / 2.10.0+cpu / 2.10.0.post4 |
| Operator dtype | BF16 |
| Context / offloaded prefix / recent tail | 40960 / 40832 / 128 tokens per request |
| Batch sizes | 1, 2, 4, 8, 16; one decode query per request |
| Indexer | 64 heads, dimension 128, independent index-key pages per request |
| Attention | 16 local heads, CKV=512, KPE=64 |
| TP assumption | TP8 shape only: 128/8=16 heads; one actual device, no HCCL |
| Hot cache / TopK / page size | 8192 / 2048 / 128 tokens |
| Host storage | mmap DRAM registered with `aclrtHostRegister`, device aliases |
| Timing | Five warmup replays, 30 unprofiled event-timed replays per workload |
| Profiler | Level1, PipeUtilization, one warmup step and 10 active steps |

No model weights are loaded. AWQ refers only to the earlier attempted model
configuration, not to the BF16 operators measured here. Host registration does
not exercise the service's MemFabric allocator. Fixed queries and Host source
data are reused; this is not a cold-cache or streaming-bandwidth experiment.

## Two Experiments

### Original Workload Repeated

`build/profiling/dsv32-40k-b1-16-current-20261025-r1/` repeats the earlier script
without passing a prefetch context to LIM. LIM is measured separately from
CopySFA. CopySFA uses controlled miss counts 0, 1024 and 2048 **per request**;
these are not model-derived hit rates or the LIM output of a chained run.

Graph capture, repeated replay, native Indexer TopK comparison, exact cache
comparison and the independent attention reference checks all passed.

Mean profiler target-kernel duration, microseconds:

| Batch | LIM | CopySFA 0 Miss | CopySFA 1024 Miss | CopySFA 2048 Miss |
| --- | ---: | ---: | ---: | ---: |
| 1 | 308.879 | 75.785 | 103.478 | 148.833 |
| 2 | 366.125 | 75.216 | 106.184 | 156.015 |
| 4 | 390.465 | 77.218 | 112.997 | 217.622 |
| 8 | 412.685 | 80.846 | 155.337 | 250.527 |
| 16 | 489.905 | 92.696 | 235.830 | 409.139 |

These are new measurements, not replacements for the historical record.
Full-miss CopySFA at batches 1--8 differs noticeably from the earlier run;
there is no controlled attribution of that difference to source, build or
machine-state changes. Use same-run paired workloads for the prefetch comparison.

### Current Chained Workload

`scripts/collect-echo-prefetch-profile-npu.py` captures the actual chain:

```text
LIM -> optional staging match/commit/metadata partition -> CopySFA
```

It compares four modes with the same input tensors and restored cache state:

- `baseline`: no prefetch context or consumer; existing LIM -> CopySFA.
- `context-budget-0`: optional context and consumer enabled, no Host prefetch.
- `prefetch-64-history`: at most 64 tokens per query, using the cutoff from the
  previous invocation of the **same fixed query**. This is an optimistic warmed
  predictor example, not a measurement of real-model prediction accuracy.
- `prefetch-64-unfiltered`: cutoff reset to negative infinity outside timing;
  the first 64 eligible nonresident tokens are copied, irrespective of rank.

Hot cache remains 8192 to match the earlier profiling, not the 2048-token hot
cache of the previous prefetch functional check. Each token has 1152 BF16 bytes
of CKV/KPE; 64 prefetched tokens mean 73728 bytes (72 KiB) per request.

Each mode validates three restored-state replays before timing. Raw LIM outputs
and complete cache mapping/KV/KPE match the no-prefetch reference exactly;
attention uses rtol=0.02, atol=0.002 because metadata partitioning can reorder
accumulation. Budgets and remaining misses are checked, and watched buffer
addresses remain stable. The separate functional checker covers changing
queries, lengths, budgets, first-fill/steady states and persistent all-hit
replays; this timing script intentionally repeats a fixed workload.

Cache mapping and KV/KPE are restored on the same ordered stream before each
timed replay, outside the event interval. Prefetch readiness/count/source resets
are inside the graph and inside event timing. Host owners outlive graph
destruction. Capture and repeated replay are mandatory, not eager substitutes.

## Timing Interpretation

- `event_mean_us` measures the complete graph, including pre-LIM readiness
  resets and possible replay-submission gaps, without profiler collection.
- `lim_us` and `copy_sfa_us` are mean profiler `Duration(us)` for the target ops.
- `consumer_kernel_sum_us` sums device kernel durations between paired LIM and
  CopySFA tasks. It does not include every runtime memcpy or inter-task gap.
- `target_kernel_sum_us` adds those three stage values. It excludes pre-LIM
  readiness resets and is **not** a complete graph elapsed time.
- `lim_to_sfa_span_us` spans LIM start through CopySFA completion, including
  gaps, but excludes operations before LIM. Profiler counter collection may
  perturb execution; it is not identical to the unprofiled event measurement.

Graph-external cache reset tasks remain in raw traces but are excluded from
the paired stage statistics. Profiler exports may contain an extra in-flight
warmup record. Pipe ratios do not measure Host-link bandwidth, and a change in
LIM duration is not an isolated measurement of transfer time or proof of overlap.

## Chained Results

Completed in `build/profiling/echo-prefetch-40k-b1-16-20261025-r1/`, status
`completed_operator_chain_only`. All 20 workloads passed capture and repeated
replay checks. Each has 30 unprofiled timed replays and 10 or 11 paired profiled
records. No native kernel source was changed during this profiling experiment.

Complete graph event mean, microseconds, **not full-model decode latency**:

| Batch | No Prefetch | Context, Budget 0 | History, Budget 64 | Unfiltered, Budget 64 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 452.043 | 678.659 | 914.900 | 985.515 |
| 2 | 525.601 | 719.208 | 963.273 | 1046.361 |
| 4 | 556.383 | 772.642 | 1032.118 | 1082.625 |
| 8 | 671.799 | 919.281 | 1180.140 | 1217.217 |
| 16 | 950.525 | 1199.017 | 1453.193 | 1500.201 |

The batch-16 baseline median is 932.490 us versus a 950.525 us mean, indicating
an outlier. Other medians and p90 values are retained in `result.json`; one run
does not establish a stable cross-run latency distribution.

Profiler stage means for the history-budget-64 comparison, microseconds:

| Batch | Baseline LIM | Prefetch LIM | Consumer Kernel Sum | Baseline CopySFA | Prefetch CopySFA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 315.137 | 560.456 | 157.522 | 127.639 | 138.570 |
| 2 | 372.896 | 619.593 | 175.211 | 142.408 | 142.685 |
| 4 | 391.294 | 632.864 | 186.425 | 155.206 | 190.164 |
| 8 | 468.422 | 734.073 | 209.890 | 194.611 | 215.645 |
| 16 | 583.632 | 840.007 | 260.735 | 338.357 | 330.822 |

LIM's exposed duration increases by approximately 242--266 us. This includes
the entire changed execution schedule; it is not a standalone copy-time
measurement. The consumer adds 11 device kernels at batches 1/2/4 and 13 at
batches 8/16. Batch-1 examples include `_match` (18.516 us), `_commit`
(22.253 us), the two native cumsum kernels (24.095 us combined), and the four
native scatter kernels (81.224 us combined). CopySFA does not improve
consistently, despite fewer Host misses; differences may include changed
metadata order and cache effects, and are not causally isolated here.

Every history-mode request copies 64 tokens and consumes all 64. The original
miss count is 1611--1674 per request, so only about 4% of Host misses are covered.
Unfiltered mode copies the same amount but consumes only 1--6 tokens per
request. These usefulness results apply to this fixed synthetic workload only.

The current functional prototype is slower, even with optimistic prediction.
Neither profitable compute/transfer overlap nor an end-to-end speedup is
demonstrated. Before expanding the budget, the single-owner copy/scanning
schedule and the multi-kernel consumer need performance work. Budget-zero
results also show substantial integration overhead without any Host prefetch.

Logs are in `build/reference/dsv32-profile-current-20261025-r1.log` and
`build/reference/echo-prefetch-profile-20261025-r1.log`. Each output root has
`result.json`, per-batch `metrics.json`, and per-workload profiler directories
containing `ASCEND_PROFILER_OUTPUT/kernel_details.csv`, `task_time.csv`, and the
`ascend_pytorch_profiler.db` / `analysis.db` timeline databases. This environment
does not emit a `trace_view.json` by default.
The chained manifest also records source hashes for the script, consumer and
native patch. The initial small probe is retained separately at
`build/profiling/echo-prefetch-40k-probe-20261025-r1/`.

## Reproduce

Use a healthy, unused device and a new output directory for each run. Do not
run the two experiments concurrently on the same device.

```bash
ECHO_NPU_DEVICE=6 ./scripts/npu-run.sh python scripts/smoke-sglang-npu.py

ECHO_NPU_DEVICE=6 HF_HUB_OFFLINE=1 ./scripts/vllm-ascend-run.sh \
  python -u scripts/collect-vllm-sparse-profile-npu.py \
  --batches 1 2 4 8 16 --context 40960 --tp 8 --hot-tokens 8192 \
  --repeats 30 --profile-repeats 10 \
  --output build/profiling/dsv32-40k-b1-16-current-20261025-r1

ECHO_NPU_DEVICE=6 HF_HUB_OFFLINE=1 ./scripts/vllm-ascend-run.sh \
  python -u scripts/collect-echo-prefetch-profile-npu.py \
  --mtp 0 --modes baseline context-budget-0 prefetch-64-history prefetch-64-unfiltered \
  --batches 1 2 4 8 16 --context 40960 --tp 8 --hot-tokens 8192 \
  --repeats 30 --profile-repeats 10 \
  --output build/profiling/echo-prefetch-40k-b1-16-20261025-r1
```

The SGLang baseline uses the project's CANN 8.5 launcher. The native vLLM PR
operators require the isolated CANN 9.1 launcher; the environments are not mixed.
The SGLang baseline passed batch-1/2 decode capture/replay for this run; this
does not establish DeepSeek or ECHO full-model support. Log:
`build/reference/echo-profiling-sglang-baseline-20261025.log`.
