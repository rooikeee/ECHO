# InfiniteBench English book summaries on 910C

This is an adaptation of the CUDA ECHO artifact's synthetic-prefill decode
benchmark, not an accuracy benchmark or an exact reproduction of its hardware,
model, or sequence-length distribution. The 2026-10-26 experiment uses the pinned
full GLM-5.2 W4A8 model, 16 Ascend devices, TP16/EP16, MTP3, target
`FULL_DECODE_ONLY` graphs, and an eager draft. ECHO prefetch is disabled.

## Dataset and request protocol

Source: ModelScope `AI-ModelScope/InfiniteBench`, file `longbook_sum_eng.jsonl`,
revision `cbbcf3d7716ad636915a0ab842e31342ce546085`.
SHA256: `4e4279cc06c07778f646cda1335755cf5ccdcc38fe9c8a62f1631642fc031027`.
The task has 103 distinct records. Only five fall in the original artifact's
100K–110K interval under the GLM tokenizer. We use the first records, keeping
32,000 token IDs from the beginning and 32,000 from the end of each prompt,
including the summarization instruction. No sample is duplicated.

Each request asks for 256 output tokens, temperature zero, `ignore_eos=true`.
A fixed random seed generates exponential inter-arrival times for rates
0.3, 0.5, and 0.7 requests/s; `inf` submits without arrival delays.
The first new test uses 30 records at 0.3 requests/s; subsequent four-rate
comparisons use 60 records per rate for optimized and original baseline only.
The intermediate unoptimized PR comparison was cancelled by the user.

The connector initializes actual cache storage with zeros once and skips
prompt computation/KV transfer. Newly decoded tokens still update cache.
The reported 100% external-prefix hit is a synthetic connector property, NOT
LIM hot-cache hit rate. Output text and MTP acceptance are not evidence of
model accuracy. Outputs from this synthetic state must not be used to assess
real-prefill quality or predict real-workload miss rates.

## Request-owned hot-cache layout

The pinned allocator reserves `max(BS*(1+MTP), 2*(BS+2))` full hot-cache rows.
At BS16/C12288 that is about 68 GiB per device before model weights; the
initial run therefore failed during cache profiling. This is not an intrinsic
MTP storage requirement: target queries already share request-owned slots.

`patches/vllm-ascend-request-shared-hot-cache.patch` adds an opt-in layout:

- one C12288 hot region plus two 128-token tails per resident request;
- separate private compact Graph scratch regions (2048+256 tokens per row);
- fixed addresses across replay, with logical dummy slots mapped to scratch;
- persistent request ownership and a memory-capacity admission gate. A request
  waits if resident slots are exhausted; finishing releases its slot.

The experiment scheduler can schedule all 30 submitted records. The physical
pool capacity is 16, calculated from the device budget, not an imposed BS6
benchmark limit. For 30 scheduler slots the compact layout allocates 22 full-row
equivalents (23.39 GiB over 79 target/draft layers), versus the old per-query
reservation. The same server handles subsequent sets of 60 arrivals; there is
no client concurrency limiter, and resident capacity controls admission.

Weights + hot cache + indexer/device KV are capped at 90% of runtime-reported
per-device memory. The remaining 10% serves Graph/activation/HCCL overhead.
The custom worker prints actual per-rank byte budgets. Standard
`gpu-memory-utilization=0.9` alone has a different accounting policy.
Baseline has no hot pool and a scheduler cap of 60 (all submitted requests);
its block availability determines active concurrency and queueing.

## Reproduction

Prepare the fixed sources and environment described in `README.vllm-ascend.md`.
Apply the shared-cache patch after the four existing upstream patches:

```bash
git apply patches/vllm-ascend-request-shared-hot-cache.patch
source scripts/vllm-ascend-env.sh
python scripts/prepare-echo-longbook-summaries.py \
  --data /path/to/longbook_sum_eng.jsonl \
  --tokenizer build/models/GLM-5.2-w4a8 \
  --count 60 --output build/summary-requests.jsonl
python scripts/serve-echo-infinitebench.py optimized --scheduler-cap 30 --resident-capacity 16
# In another shell, with the server ready:
python scripts/bench-echo-infinitebench-serving.py \
  --data build/summary-requests.jsonl --variant optimized \
  --rate 0.3 --limit 30 --output build/shared30-rate-0.3.json
# Subsequent rates: --limit 60, --rate 0.3 / 0.5 / 0.7 / inf, unique output paths.
# Stop the optimized server before launching the baseline:
python scripts/serve-echo-infinitebench.py original --scheduler-cap 60
```

These commands are specific to the recorded model/hot-cache/MTP configuration.
Recalculate physical pool capacity before changing them. The shared layout is
restricted to this synthetic connector experiment; a real PD connector needs
matching admission/slot-lifetime integration before enabling this layout.

## Validation and measurement

Tests cover CPU address bounds/admission, distinct request/dummy storage,
changed-request Graph replay and cache writes at fixed addresses. Full-model
startup, successful output token counts and actual Graph replay are recorded
separately in the results. Independent operator correctness checks from the
previous environment setup remain distinct from this synthetic serving test.

Throughput = successfully completed output tokens / wall-clock benchmark time,
including arrival delays and final drain. TTFT includes waiting; TPOT uses
observed output timing and actual token count. Every second the client samples
`num_requests_running`, `num_requests_waiting`, and `kv_cache_usage_perc`.
Reported maximum concurrency is the maximum observed running gauge, with
one-second sampling resolution. It is not inferred from configured batch size.
Neither whole-request timing nor derived TPOT is target-only kernel time.

The uploaded `lim_copysfa.md` table computes TPOT as ITL/3 and TPS as BS/TPOT.
Our actual MTP acceptance and observed batch occupancy differ; its 438.2 TPS
must not be directly compared to the earlier BS6 HTTP result of 77.41 TPS as
an operator slowdown. Raw per-step profiling would be needed for attribution.

## Environment fixes required on this host

- Complete system CANN 9.1, Python 3.11.6, torch 2.10.0,
  torch-npu 2.10.0.post4, Triton-Ascend 3.2.2, CMake 3.31.10.
- Install Triton-Ascend after base Triton: their overlapping files otherwise
  leave only CUDA/ROCm backend discovery. Test actual NPU Triton execution.
- MemFabric release/1.2 source produced version 1.2.1; build with HCOM enabled
  and RDMA disabled for this local shared-memory experiment. Unlike the old
  PyPI 1.2.0, this exposes `OffloadConfig.store_url`. Update the local
  vLLM-Ascend requirement accordingly; it is a recorded dependency change.
- Build its device helper with `mfcli kernel install --soc-version A3`.
  CANN's recursive object glob ignores hidden `.venv` paths, so kernel sources
  were staged into a visible temporary directory before building. The helper
  must be nonempty and loadable; all 16 shared allocator ranks passed.
- New processes need the `HwHiAiUser` group. Existing sessions used `sg`.

The task has a hard cleanup deadline of 2026-10-26 20:58 Asia/Shanghai,
ahead of the user's 21:00 deadline. Only this experiment's recorded process
groups are terminated. Incomplete tests are recorded as such, never as passes.

Local build fixes are retained as `patches/vllm-ascend-memfabric121.patch`,
`patches/memfabric-hybrid-kernel-staging.patch` (apply relative to the Python
site-packages directory), and `patches/vllm-ascend-two-op-build-dependency.patch`.
They are separate from the numerical operator and shared-cache changes.
The unmodified source lock continues to identify the original upstream snapshots.

## Validation recorded on this run

- 4 CPU synthetic-cache/layout/admission checks passed.
- Changed-request NPU Graph scatter/replay check passed; repeated with the
  final connector configuration (3 selected tests passed).
- Full shared-cache GLM Graph capture and replay succeeded.
- The 30-request trial and all four 60-request optimized sweeps completed
  without failed requests. This validates completion and state/address
  execution on the synthetic workload, not real-prefill numerical accuracy.
- Syntax checks, selected Ruff correctness rules, and reverse applicability
  of the retained shared-cache patch passed.
