# Controlled Miss-400 MTP3 Profiling

For isolated LIM AIC/AIV counters, per-core diagnosis, and the all-hit control,
see [LIM bottleneck profiling](ECHO_LIM_BOTTLENECK_PROFILING.md).

## Workload

The earlier random-residency MTP3 benchmark produced about 1600 misses per query.
That is not a real-model hit-rate measurement. This experiment instead controls
the cache residency to match a supplied miss count, while executing the actual
LIM -> optional prefetch consumer -> CopySFA graph.

Settings remain batch 16, MTP3 target verification (four query rows per request,
64 rows total), context 40960, hot cache 8192 per request, TopK 2048, recent tail
128, BF16, 64 indexer heads of dimension 128 and 16 local attention heads. This
is one physical NPU (device 6), TP8 shapes only, not a distributed model run.

**The requested 400 is the request-level union after deduplicating all four
MTP3 query rows**, not 400 in each row. `--misses-per-request 400` constructs
exactly this count. A preliminary untimed LIM call identifies the exact selected
sources. The CPU workload builder randomly samples 400 distinct sources from
each request's four-row TopK union and makes those sources nonresident. All
other selected sources remain resident. It fills the remaining hot-cache slots
with unselected sources and randomizes slot assignment. Query-level miss counts
follow the selected sources' membership in the four rows; they are not forced
to 100 or 400. A source shared by query rows is counted only once in the union.
Resident CKV/KPE are populated from the corresponding registered Host data,
not unrelated random values. The scoring tensors and timed native operators
are unchanged, and observed LIM query/union counts must match the requested
counts before profiling can proceed.

This construction uses knowledge of the current selection solely to control
the benchmark workload. It is **not** a deployable cache policy, an authentic
MTP access trace, or evidence of real prediction accuracy. Query correlation
still comes from the synthetic scoring inputs, not measured model queries.
Matching the union total does not establish that per-row miss counts, sharing,
source locality or prefetch usefulness match a real model.

An initial misinterpretation used `--misses-per-query 400`, giving a request
union of 1600. Those runs remain archived under
`echo-prefetch-b16-mtp3-miss400-20261025-r1` (and its probe), but **do not represent
the user's requested workload**. Use the `union400` result directory below.

History-based prefetch uses the preceding invocation of the same fixed query
and remains an optimistic predictor. Cache mapping and KV/KPE are restored
before each timed replay, outside the timing interval. This is repeated-state
operator measurement, not a continuous model generation trace.

## Validation

The workload constructor has six CPU unit tests covering exact per-query and
deduplicated counts, deterministic seed, zero misses, invalid source IDs,
insufficient hot-cache capacity, insufficient query-exclusive candidates,
multiple requests and shared missing sources counted only once in the union.

Each mode captures a GLOBAL NPU graph and checks three restored-state replays.
Prefetch modes additionally exercise seven dynamic/persistent scenarios:
changing query, changing length with budget 7, budget zero, duplicate candidates
across query rows, mixed first-fill/steady requests, steady again and persistent
all-hit replay without resetting the cache. The controlled 400 count applies
to the fixed timed input, not these deliberately changed validation inputs.

Checks compare complete cache mapping and KV/KPE exactly against the no-prefetch
reference, preserve raw LIM outputs and valid union prefixes, validate staging
against Host data, check budgets and query/union miss reductions, retain stable
graph buffer addresses, and compare attention at rtol=0.02, atol=0.002.

## Reproduce

Use an unused healthy device, run experiments sequentially and choose fresh
output directories. Each formal mode uses 30 event-timed replays and 10 active
Level1/PipeUtilization profiler steps. See `ECHO_PREFETCH_PROFILING.md` for
timing interpretation, native environment and the original MTP3 experiment.

```bash
ECHO_NPU_DEVICE=6 HF_HUB_OFFLINE=1 ./scripts/vllm-ascend-run.sh \
  python -u scripts/collect-echo-prefetch-profile-npu.py \
  --batches 16 --mtp 3 --misses-per-request 400 \
  --modes baseline context-budget-0 prefetch-64-history \
  --context 40960 --hot-tokens 8192 --tp 8 \
  --repeats 30 --profile-repeats 10 \
  --output build/profiling/echo-prefetch-b16-mtp3-union400-20261025-r1
```

A 64-token budget is per query, not per request. Full-model execution, draft
forwards and acceptance/rejection are outside this experiment.

## Results: Request-Union 400

Completed on 2026-10-25 with status `completed_operator_chain_only` in
`build/profiling/echo-prefetch-b16-mtp3-union400-20261025-r1/`. This is the
authoritative run for the corrected user requirement. All 16 requests have
exactly 400 request-union misses; their query-level counts range from 88 to 135.

Mean microseconds per target-verification invocation for the entire batch:

| Mode | LIM Kernel | Consumer Kernel Sum | CopySFA Kernel | Complete Graph Event Mean |
| --- | ---: | ---: | ---: | ---: |
| No prefetch | 496.221 | 0 | 274.695 | 780.751 |
| Prefetch context, budget 0 | 515.628 | 2134.500 | 308.299 | 2987.853 |
| History cutoff, budget 64/query | 2219.365 | 2137.607 | 288.996 | 4671.471 |

Each mode uses 30 unprofiled event-timed replays and 11 recorded profiler task
pairs (10 active steps plus a boundary warmup record). Stage columns are
profiled kernel durations; the complete-graph column is a separate unprofiled
measurement, not their sum. Every mode passed fixed-input comparison replays;
both context modes passed all seven dynamic/persistent replay scenarios.

Each query actually stages 64 tokens. Within each request, deduplication leaves
230--244 useful sources, reducing union misses from 400 to 156--170. Query-level
remaining misses range from 24 to 68. The history predictor is still warmed on
the identical query and is not a real-model prediction measurement.

Prefetch is slower overall: approximately 4.67 ms versus 0.78 ms. Relative to
the no-context baseline, LIM adds approximately 1.72 ms and the consumer adds
2.14 ms. CopySFA is 289.0 versus 274.7 us, so its isolated duration does not
improve over that baseline; it is lower than the budget-zero path's 308.3 us.
Changing metadata order and preceding cache activity remain confounders, not
proven causes. Lower logical Host miss counts alone do not guarantee shorter
kernel duration. The native prefetch scans scores in source order until 64
eligible candidates are found; a sparse eligible set can also expose a longer
scalar scan. This experiment does not isolate scan cost from copy cost.

The result file retains per-query misses and prefetch counts, unique covered
sources per request, source hashes, event statistics, stage statistics and
dynamic validation records. Raw kernel CSVs, task times and timeline databases
are in `batch-16/{baseline,context-budget-0,prefetch-64-history}/`. Log:
`build/reference/echo-prefetch-b16-mtp3-union400-20261025-r1.log`.

CPU unit tests passed with:

```bash
./scripts/vllm-ascend-run.sh python scripts/test-echo-profile-workload.py
```

No native kernel or consumer implementation was changed for these controlled
miss-count experiments; only profiling input construction and validation changed.
