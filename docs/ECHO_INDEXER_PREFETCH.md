# Bounded ECHO Indexer Prefetch

## Scope

Experimental operator-level implementation for the pinned vLLM Ascend PR 17392
source. It is opt-in and is not wired into model serving. The original consumer
supports one query per request; `EchoMTPIndexerPrefetch` adds uniform four-query
MTP3 target verification with capacity 64 per query. This does not cover a full
speculative round, draft-model integration or acceptance/rejection. Prefill is
not supported. BF16/FP16 CKV=512, KPE=64 and
128-token pages follow the existing CopySFA layout.

The CUDA reference is commit `bc1b75c1000010d0ac6f032ebaac283255c050b1`,
`DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh`,
function `sm90_fp8_paged_mqa_logits_fused_v2`:

- Scan complete per-token block scores, without a new block-local TopK sort.
- Require score strictly greater than the caller's predicted cutoff.
- Skip resident tokens and cap successful transfers per query, across blocks.
- Stage real Host KV, then use exact TopK to consume useful prefetch and recall
  the remaining misses. Speculation does not change the exact selected tokens.

Capacity defaults to 64; device-side budgets can vary between zero and capacity
on every replay. The cutoff starts at infinity (prediction disabled), then the
Indexer publishes its current TopK cutoff to the same fixed device tensor for
the next step. Callers may override it in place. History is per bound query row;
reset it to infinity when a row is assigned to a different request.

## Ascend Scheduling

The existing scoring AIVs publish per-tile ready flags after raw-score MTE3
writes. One odd AIV per query owns its entire budget, scanning source order.
It starts consuming after its local main-compute work, before the existing
global TopK reduction barrier. It may overlap producers on other cores, but
does not reproduce CUDA's dedicated prefetch warp scheduling. Neither latency
hiding nor a performance gain is established by this implementation.

The owner reads registered Host CKV/KPE through their device aliases and writes
separate fixed-address HBM staging tensors. It never overwrites speculative
formal cache slots. Transfers and count publication drain before the owner
reaches the global reduction/cache-management barrier. Every published tile
has one producer; every query budget and staging row has one consumer owner.

After exact cache management, device kernels match staged sources against the
actual Miss prefix, commit useful KV to the allocated physical HBM slots, and
stably partition remaining Host misses first. The CopySFA consumer receives
separate metadata; original Indexer outputs retain their exact meaning.
First-fill lists are preserved for requests not consuming prefetch, including
mixed first-fill/steady batches. Unused speculative entries are discarded.

## Graph And Lifetime Contract

`EchoIndexerPrefetch.indexer()` records reset of readiness, counts and source
IDs before the Indexer launch. Stale staging bytes are invalid outside the
current valid prefix. Buffers, output metadata and source aliases are allocated
before capture and remain alive until graph destruction. Thresholds, budgets
and sequence lengths are updated in place between replays. All producer,
commit and CopySFA launches use the same ordered stream.

The optional `[T,14]` int64 context is an **internal pointer descriptor**, not
a public tensor protocol. The Python owner retains every target and the
registered Host buffer owners; never construct it manually, unregister Host
memory early, replace its buffers, or use this path under `torch.compile`.
Dispatcher alias analysis cannot see descriptor-referenced writes. The
unchanged no-context operator remains the default.

Triton 3.2.2 lowers scans and discrete scatter here through locks initialized
by a synchronous Host copy. The consumer instead uses preallocated native
`torch.cumsum(out=)` and `scatter_` outputs for those operations, allowing
GLOBAL-mode capture without relaxing capture restrictions. This functional
consumer adds launches and metadata copies; it is not a performance-optimized
fused SFA implementation.

## Build And Check

The upstream archive is ignored by the outer repository. Its changes are
preserved in `patches/vllm-ascend-echo-indexer-prefetch.patch`. With the existing
environment configured, run:

```bash
bash scripts/build-echo-indexer-prefetch.sh
ECHO_NPU_DEVICE=8 ./scripts/vllm-ascend-run.sh python \
  scripts/collect-echo-prefetch-profile-npu.py --batches 16 --mtp 3 \
  --context 40960 --hot-tokens 8192 --repeats 30 --profile-repeats 10 \
  --output build/profiling/echo-prefetch-b16-mtp3-new-run
```

Select a healthy, unused device. This uses the isolated vLLM/CANN 9.1 launcher,
not the SGLang/CANN 8.5 launcher. The project baseline still uses
`scripts/npu-run.sh`.

## Validation

2026-10-25:

- The project SGLang tiny-model baseline passed batch 1/2 decode Graph capture
  and replay. Log: `build/reference/echo-prefetch-sglang-baseline.log`.
- Existing no-context Indexer passed batch 1/2 and four input/state-changing
  Graph replays each. Log: `build/reference/echo-prefetch-indexer-regression.log`.
- The new 8320-token, BF16, batch-1 pipeline passed GLOBAL capture and ten
  dynamic/persistent replays on physical NPU 8. Checks cover budget 64/7/0,
  disabled cutoff, strict equality cutoff, changing query/length, first-fill,
  steady eviction and a persistent all-hit step. Real registered Host reads,
  staging bytes and complete formal caches match the reference exactly;
  Attention outputs match within rtol=0.02, atol=0.002 (partitioning reorders
  attention accumulation). Log/JSON: `build/reference/echo-indexer-prefetch-8k.*`.
- Scan and scatter capture failures encountered during development are retained
  as `echo-indexer-prefetch-8k-scan-failure.log` and
  `echo-indexer-prefetch-8k-scatter-failure.log` in the same directory.
- The final 40960-token BF16 pipeline passed batch 1/2, GLOBAL capture and
  eleven replays each, including automatic device cutoff feedback and use on
  the next changed query, mixed first-fill/steady requests, and persistent
  all-hit state. Log/JSON: `build/reference/echo-indexer-prefetch-40k.*`.
- In the deliberately disabled-filter scenario (cutoff=-infinity), each query
  copied 64 tokens but only 2 were useful at 40K. The cutoff-controlled and
  automatic-history examples used all 64. These are seeded synthetic cases,
  not measured model prediction accuracy or evidence of latency improvement.
- `scripts/build-echo-indexer-prefetch.sh` completed a full rebuild, package and
  project-local install. Log: `build/reference/echo-prefetch-rebuild.log`.
- Subsequent batch-16 MTP3 target verification passed capture, three fixed-input
  comparison replays per mode, and seven dynamic/persistent scenarios for both
  context modes. Thirty timed replays and ten active profiler steps per mode
  completed. Results: `build/profiling/echo-prefetch-b16-mtp3-20261025-r1/`.
  See `ECHO_PREFETCH_PROFILING.md` for the slower measured prefetch path and its
  stage breakdown. Subsequent ECHO profiling defaults to MTP3, not single query.

No full-model, multi-device, complete MTP round, prefill, FP16 or performance acceptance is
claimed. Correct data dependencies do not establish actual transfer/compute
overlap, especially at batch 1; a profiler timeline is still required.

The MTP3 extension uses one prefetch owner per query after its local main loop;
odd AIVs also compute query scores in this configuration, not dedicated copies.
Ready flags, staging, budgets and cutoff feedback are indexed by query row.
The consumer pools up to 256 candidates per request, selects the first staged
copy of a duplicate source, commits each useful request-union source once, and
partitions both the four query miss lists and the request-union miss list.
See `ECHO_PREFETCH_PROFILING.md` for MTP3 profiling and coverage records.
