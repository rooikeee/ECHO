# LIM AIC / AIV Bottleneck Profile

Measured October 25, 2026. For **batch 16, MTP3, context 40960, and 400
request-union misses**, the current no-prefetch LIM is **AIV-side limited**.
Evidence points to vector processing plus an AIV cache-management tail,
not saturated AIC matrix arithmetic. This conclusion is shape-specific.

The current prefetch implementation adds much larger AIV scalar/control and
synchronization costs. Vector work barely changes. Moving the separate
consumer inside LIM will not by itself remove this producer cost.

## Workload And Coverage

- One physical NPU, device 6, 24 AICs and 48 AIVs; operator diagnostic frequency
  1800 MHz. BF16, 64 indexer heads, head dimension 128.
- Batch 16, four target-verification query rows/request, 64 rows total.
  MTP3 excludes draft forwards, acceptance/rejection, and model layers.
- Context 40960, offloaded prefix 40832, hot cache 8192/request, TopK 2048/row.
- Exactly 400 unique missing sources/request across four rows; per-row misses
  88--135. Same scoring/residency seeds as the corrected chained profiling run
  in `ECHO_PREFETCH_MISS400_PROFILING.md`.
- Synthetic fixed queries and controlled residency, not a model trace. The
  history cutoff is warmed on the same query.
- Only LIM runs in the measured graph: no CopySFA or prefetch consumer.
  Prefetch modes retain their captured ready/count/source resets. Cache mapping
  resets run on the same stream outside the timed graph.
- Actual graph capture/replay, three exact-output comparisons/mode, 30 event
  timings/mode, 10 recorded LIM tasks per PMU group. Persistent all-hit replay
  also passes against a fresh reference. Graph buffer addresses remain stable.
- No native operator or consumer implementation was changed for this study.

## Graph Hardware Counters

Main run:
`build/profiling/echo-lim-bottleneck-b16-mtp3-union400-20261025-r1/result.json`.
Kernel elapsed time covers the whole batch. Pipe times are core-normalized
aggregates and overlap; they must not be added together.

| Metric | Original LIM | Context, Budget 0 | Prefetch 64/Query |
| --- | ---: | ---: | ---: |
| Kernel elapsed (us) | 471.320 | 496.258 | 2199.992 |
| Unprofiled graph event mean (us) | 478.464 | 515.763 | 2218.825 |
| AIC core time (us) | 321.259 | 342.198 | 344.138 |
| AIC MAC time (us) | 122.978 | 122.940 | 122.978 |
| AIC MAC ratio | 38.27% | 35.92% | 35.74% |
| AIC Fixpipe time (us) | 228.324 | 227.510 | 227.834 |
| AIV core time (us) | 386.201 | 409.618 | 2116.030 |
| AIV Vector time (us) | 261.569 | 261.590 | 261.592 |
| AIV Vector ratio | 67.72% | 63.85% | 12.35% |
| AIV Scalar time (us) | 147.345 | 147.827 | 720.362 |
| AIV MTE2 time (us) | 164.105 | 154.547 | 258.546 |
| AIV MTE3 time (us) | 18.997 | 32.014 | 54.048 |

Original LIM AIV bank-group conflict ratio is 6.77%, bank conflict 1.10%,
and resource conflict 3.94%. These can overlap and are not additive latency
buckets. The `Memory` PMU group is retained, but its per-core bandwidth is
not used as a board-level saturation claim.

The approximately 1.70 ms increase from budget-zero to actual prefetch comes
with approximately 1.71 ms more AIV core time and only approximately 2 us more
AIC core time. MAC and Vector times stay constant. This localizes the
regression to the AIV path, but does not uniquely apportion candidate scanning,
Host latency, per-token synchronization, and cross-core waiting.

The older chained run gave 496.221 us for original LIM and 2219.365 us with
prefetch. Its preceding cache activity differs. Both runs support the same
conclusion; isolated-LIM numbers do not replace the chained measurements.

## Per-Core Diagnostic

`msprof op --aic-metrics=Default` collected the unchanged kernel using eager
invocation with identical inputs. This is supplementary diagnosis; accepted
latency and correctness above use actual graph replay.

Artifact root:
`build/profiling/echo-lim-op-b16-mtp3-union400-20261025-r4/`.
Use invocation directory **`1`**. Directory `0` is the preliminary original
prefix-residency pass that discovers TopK before constructing miss400. It is
not the controlled workload.

| Core Group | Mean Core Duration | Range |
| --- | ---: | ---: |
| All 24 AICs | 330.089 us | 323.491--336.396 us |
| Even AIVs, blocks 0--15 | 466.573 us | 453.371--477.656 us |
| Even AIVs, blocks 16--23 | 358.464 us | 357.752--359.088 us |
| All 24 odd AIVs | 357.133 us | 355.908--358.681 us |

Task duration is 480.930 us. The slow cores are the even AIVs owning the 16
requests' final union/cache management. In `fused_lightning_indexer_manage.cpp`,
the even AIV of block b executes `unionOp.Process(b, GetBlockNum())`.
Only blocks 0--15 receive this work for 16 requests and 24 blocks.

AIC `aic_scalar_wait_id0_time(us)` averages **245.390 us**, range
238.826--254.345 us. Both directional handshake IDs in
`lightning_indexer_common.h` are zero. `ProcessBaseBlock()` in
`lightning_indexer_kernel.h` waits on the AIV-to-AIC flag before QK, and AIV
publishes it after `ProcessVec()`. The large ID0 wait agrees with AIV-to-AIC
backpressure. It overlaps asynchronous pipeline execution; it is not 245 us
that can simply be subtracted from elapsed time.

Odd AIVs perform substantial work: mean Vector time **246.909 us**, versus
**276.195 us** for even AIVs. They are not idle prefetch workers in MTP3.
Scalar vector-stall time averages 195.413 us on even AIVs and 194.338 us on
odd AIVs. These dependency counters are not isolated TopK function timings.

No native phase timestamps were inserted. Per-core duration differences
identify the tail and its ownership, but do not time `unionOp.Process()` alone.

## All-Hit Control

Run:
`build/profiling/echo-lim-bottleneck-b16-mtp3-union400-20261025-r2/result.json`.
All-hit state uses the mapping after one miss400 pass. Scoring tensors, query
lengths and the captured graph remain the same.

| Metric | Miss400 | All Hit |
| --- | ---: | ---: |
| Kernel elapsed | 463.291 us | 350.473 us |
| Unprofiled graph event mean | 470.943 us | 356.937 us |
| AIC core time | 315.637 us | 317.003 us |
| AIC MAC time | 122.954 us | 122.973 us |
| AIV core time | 380.057 us | 346.128 us |
| AIV Vector time | 261.570 us | 246.947 us |
| AIV Scalar time | 146.023 us | 126.322 us |

Removing misses saves **112.818 us (24.4%)**, with essentially unchanged AIC
work. This delta includes all cache-dependent paths, such as miss metadata,
eviction and union handling. It is not exactly one function's duration.

## Implications And Counter Semantics

The prefetch owner runs after local `ProcessMain()` and before global TopK
reduction synchronization. `echo_indexer_prefetch.h` scans scores in scalar
source order, loads Host rows into one UB buffer, drains transfers to staging,
and publishes source/count metadata. In MTP3 this odd AIV also performs normal
scoring/TopK before reaching the prefetch loop.

Priorities supported by these results:

1. Reuse existing vector filtering and cache-management results; avoid another
   scalar full-context scan.
2. Address request-owner AIV load balance and the cache-management tail.
3. Batch/overlap transfers and measure their exposed tail. Low MAC occupancy
   does not imply an AIV is available for prefetch.
4. Fuse useful-prefetch handling where it removes duplicate work. Re-measure
   total LIM time; reducing kernel count alone does not establish a gain.

The torch report's `cube_utilization(%)` is **not peak MAC FLOP utilization**.
Installed CANN `viewer/runtime_report.py:cube_usage()` divides total AIC cycles
by frequency, core count and task duration. Core time is normalized across
blocks by `mscalculate/stars/ffts_pmu_calculator.py` and
`common_func/utils.py:cal_total_time()`.

Per-core `msprof op` ratios and torch aggregate ratios have different
denominators. Compare named times and interpret ratios within their own
report. High Fixpipe occupancy also warrants examining the AIC-to-AIV
interface; these results do not imply that every AIC-side optimization is
irrelevant.

## Reproduction

```bash
ECHO_NPU_DEVICE=6 HF_HUB_OFFLINE=1 ./scripts/vllm-ascend-run.sh \
  python scripts/profile-echo-lim-bottleneck-npu.py \
  --output build/profiling/lim-bottleneck-new-run

ECHO_NPU_DEVICE=6 HF_HUB_OFFLINE=1 ./scripts/vllm-ascend-run.sh \
  python scripts/profile-echo-lim-bottleneck-npu.py \
  --output build/profiling/lim-all-hit-new-run \
  --modes baseline --metrics PipeUtilization --profile-all-hit

ECHO_NPU_DEVICE=6 HF_HUB_OFFLINE=1 ./scripts/vllm-ascend-run.sh \
  msprof op --output=build/profiling/lim-op-new-run \
  --aic-metrics=Default --kernel-name='FusedLightningIndexerManage*' \
  --replay-mode=kernel --warm-up=0 --launch-count=2 \
  --application='python scripts/profile-echo-lim-bottleneck-npu.py --output build/profiling/lim-op-app-new-run --modes baseline --metrics --repeats 5'
```

Output directories must be new. Results retain individual rows, counters,
correctness checks and source hashes. The final script also records the
command/configuration. Changes after the recorded runs only refine CLI
validation and reporting, not the workload.

Initial MSTX-filtered attempts exported no tasks; a snake-case kernel filter
also did not match. These are archived as `echo-lim-op-...-r1/r2/r3`. Successful
r4 uses the generated CamelCase prefix. No conclusion uses empty attempts or
the preliminary high-miss invocation.

`scripts/summarize-echo-lim-bottleneck.py` generates the report data and figure
from `--graph`, `--all-hit`, `--op` and `--output` directories:

- `build/profiling/echo-lim-bottleneck-analysis-20261025/summary.json`
- `build/profiling/echo-lim-bottleneck-analysis-20261025/lim-bottleneck.png`
- `build/profiling/echo-lim-bottleneck-analysis-20261025/lim-bottleneck.svg`

The separate required environment check passed:
`./scripts/npu-run.sh python scripts/smoke-sglang-npu.py`, including tiny Qwen2
graph capture/decode replay for batches 1 and 2. Log:
`build/reference/echo-lim-bottleneck-sglang-baseline-20261025.log`.
It does not validate ECHO or full MTP serving.
