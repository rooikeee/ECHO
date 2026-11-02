# 910 LIM ECHO prefetch: ForgeKernel experiment

本轮已在 `perf/910-lim-echo-forge` 分支完成 910 实机优化。主配置为
B16、MTP3（每请求 4 行）、L=40960、热池 C=8192、每请求 union miss=400，
每行实际预取 64 个 token。最终 40 次重放的中位耗时：

| 模式 | 整条 Graph 算子链 |
| --- | ---: |
| 同轮无预取 | 0.769 ms |
| 原预取实现（15 次重放） | 4.669 ms |
| 优化后预取 | 1.953 ms |
| 优化后预算为 0 | 1.757 ms |

相对原预取实现，耗时降低 **58.2%**，约 **2.39 倍加速**；
但仍是同轮无预取的 **2.54 倍耗时**，尚未取得打开预取后的净收益。
预取默认仍关闭。结果只代表目标验证阶段的融合算子链，不代表完整模型或完整 MTP 轮次。

最终 profiling 平均值约为 LIM **0.695 ms**、消费端 **0.944 ms**、CopySFA
**0.288 ms**。主要剩余瓶颈是消费端的匹配和 metadata 重排，不是 TopK 算法；
本轮没有修改实际打分、TopK 或外部 LIM/CopySFA ABI。

逐轮结果、源码与二进制哈希见 [结果记录](../forge/lim910/results-20261102.json)。

Only the fused 910 operator chain is optimized and benchmarked here:
`FusedLightningIndexerManage → prefetch consume → FusedScatterCopySparseFlashAttention`.
No standalone indexer or separate slot-plan operator is an optimization target.
MTP3 means four target query rows per request. This report does not measure an
entire speculative round or a full-model serving throughput improvement.

The branch starts at ECHO commit `6462c41a76bbde25402266038966c4498e2961dc`.
It already includes a real, slow ECHO-style prefetch prototype. This work improves
that implementation, with registered Host KV reads and actual HBM staging.
The 950 branch is a different implementation and is not changed by this experiment.

## Agent workflow

A local fork of OpenBMB/ForgeKernel at `5030503d3ff71851aa34c14b9caf13ebfa9d9a78`
adds an experimental `forge ascend` adapter. It reuses ForgeKernel's trajectory,
parsers, device-time performance gate and plateau detection, with separate
architecture and coding turns through the authenticated Codex CLI. The original
model/backend configuration is inherited. The final adapter is commit `cc1ee52`;
a self-contained patch and base ID are included under `forge/lim910/`. Apply it
on that base with `git apply --unidiff-zero forgekernel-ascend-adapter.patch`. This is not upstream native Ascend
support, and does not implement the CUDA NCU/SASS or hardware knowledge-base path.

The controller freezes implementation scope and the existing Graph validator.
Each accepted implementation must pass correctness and improve median latency by
at least 1.5%; rejected source edits are restored. Logs preserve prompts, code
snapshots, patches, evaluator results, and each KEEP/REVERT decision. No speedup
against no-prefetch is claimed merely for improving an older prefetch version.

## Workload and timing

Primary configuration: Ascend910_9391, CANN9.1, B16, Q4, context40960,
hot cache8192/request, TopK2048, 400 distinct request-union misses, BF16 CKV512
and KPE64, budget64/query. Each timed prefetch invocation copies 4096 token payloads,
4,718,592 bytes before accounting for deduplication and fallback.

Device events surround the complete captured operator chain. The prefetch reset
of ready/count/source state is inside the captured indexer wrapper and is timed.
Restoring the benchmark's seed KV/cache before each replay is **outside** the
events; it is identical for all modes. There is no CPU tensor read inside capture.
History thresholds are seeded outside measurement from an earlier call. The
primary repeated-query measurement is favorable to prediction quality; dynamic
changed-query checks establish correctness, not realistic prediction recall.

The no-prefetch and budget-zero controls are measured in the same process as each
candidate. Budget zero still executes the dynamic consumer path because budgets
may change on later replays. It cannot be eliminated by a Python branch on the
initial budget. All-hit and cold-fill scenarios are correctness tests unless a
separate timing is explicitly listed.

## B24 全命中对照

同样使用 MTP3、L=40960、热池 C=8192，改为 24 个请求、union miss=0。
最终 40 次 Graph 重放、独立 attention 参考与全部动态场景校验通过：

| 模式 | 整条 Graph 算子链 | 实际预取 token |
| --- | ---: | ---: |
| 无预取 | 0.914 ms | 0 |
| 预取上下文存在，预算为 0 | 1.608 ms | 0 |
| 正常开启预取，预算为 64 | 1.773 ms | 0 |

全命中时没有 Host KV 搬运，但正常预取路径仍额外增加约 **0.859 ms**。
最终 profiling 的消费端约 **0.625 ms**，LIM 约 **0.723 ms**（无预取约
**0.526 ms**）。固定消费流程和预测扫描仍需进一步融合或按设备状态快速跳过。

在 B16 部分 miss 的最终 profiling 中，CopySFA 从无预取约 0.279 ms 变为
约 0.288 ms，没有测到足够的阶段耗时下降来抵消新增处理；搬运数量减少不等于
整链延迟会降低。后续应优先压掉消费端的固定成本，再评估真正的重叠收益。

## Measured iteration history

| Change | Prefetch chain median | Decision |
| --- | ---: | --- |
| Original implementation | 4.669 ms | Baseline |
| Union scratch 32768 → 8192 | 3.777 ms | KEEP |
| Direct commit by union entry | 3.671 ms | KEEP |
| Native vector miss filtering | 2.290 ms | KEEP |
| Narrow union output scatter | 2.058 ms | KEEP |
| Candidate binary search through global memory | 4.576 ms | REVERT |
| Batched Host reads/publication | 1.929 ms | KEEP |
| Final 40-replay confirmation | 1.953 ms | Pass |

Iteration medians are separate jobs on the same physical NPU. The final result
uses a fresh paired no-prefetch control and independent CPU attention reference.
No result claims that the compiler-rejected gather proposal was benchmarked.

## Changes

- Narrow union scratch from 32768 to the proven maximum of `max(Q*TopK, hot_pool)`
  rounded to a power of two, while preserving the full-width external metadata ABI.
- Commit prefetched KV directly by unique union entry, instead of each candidate
  scanning the whole union. Lowest pooled candidate wins across MTP rows.
- DMA a tile of reverse-table entries into UB, vectorize strict score-threshold
  and negative-slot filtering, and use ordered GatherMask compaction. Signed
  slot sentinels -1 and INT_MIN are both recognized by the unsigned sign bit.
- Batch each query's real Host reads into disjoint CKV/rope UB rows, then publish
  CKV, rope and IDs in three bulk transfers before publishing count. The maximum
  65,536-byte BF16 CKV write uses the wide DMA parameter type.
- Scatter union metadata into narrow persistent output scratch, then copy back to
  the fixed external output allocation, including raw unused ABI tails.

## Correctness coverage

The existing frozen collector checks exact raw LIM outputs, cache mapping, full
KV/KPE tensors, remaining miss counts, and attention with rtol=.02/atol=.002.
Each context mode replays seven changing scenarios: changed query, shorter length
and budget7, budget0, cross-route duplicate candidates, mixed first-fill, steady
again, and persistent all-hit. Fixed tensor addresses are verified.

The separate existing B1/B2 checker tests nonidentity Host/HBM page tables,
strict/equal threshold, disabled threshold, budgets, first-fill, and persistent
all-hit. The additional filter regression checks ordered IDs and actual Host
payloads at 40 Graph replays, including INT_MIN/-1 slots and non-64-aligned lengths.
The required tiny Qwen2 SGLang smoke test passed B1/B2/B1 decode Graph replay; it
is only an environment check and not a LIM performance result.

## Reproduction

The worktree owns its native package and build outputs. Environment/dependencies
may be linked to the existing prepared ECHO installation, but do not rebuild or
install into that original checkout. See `scripts/vllm-ascend-env.sh` for the local
CANN and venv setup. Use device6 only after confirming it is free.

```bash
ECHO_NPU_DEVICE=6 ./scripts/npu-run.sh python scripts/smoke-sglang-npu.py
python3 scripts/forge-lim-eval.py --output build/forge-lim/new-eval --device 6
```

Run the agent controller from the ForgeKernel fork, after adapting absolute
workspace paths and baseline-result paths in the JSON task files:

```bash
PYTHONPATH=src python3 -m forgekernel ascend \
  /path/to/ECHO/forge/lim910/consumer-task.json --output /path/to/new-run
```

Native rebuilds invalidate generated LIM compile stamps and copied source, then
install only in the local worktree. `scripts/forge-build-lim.sh` assumes the local
CANN build has already been configured. A clean two-operator configuration needs
`ASCEND_OP_NAME` as a semicolon-separated CMake list. This pinned build framework
also assumes nonempty public and excluded operator INI files even when only inner
operators are selected. The experiment bootstrapped the inner host library with
`op_build`, created empty public/excluded placeholders, and omitted those empty
inputs only in the **generated** Ninja `parse_ini_to_json.py` command. Tracked
build-system source was not changed. Reconfiguring requires repeating that
bootstrap or using the normal complete project configuration.

The evaluator hashes all LIM headers and rebuilds if their hash changed. It records
consumer/native source hashes and installed LIM/CopySFA object hashes. A restored
native source must be rebuilt before subsequent validation; merely restoring a
header is insufficient to restore its loaded binary.

## Rejected approaches and limits

A four-route seven-step binary search was correct but slower: full-chain time
increased from 2.058 to 4.576 ms. The generated dependent global reads cost more
than the dense comparison; the controller restored the previous implementation.

A proposed inverse-partition gather was not implemented: the installed Triton
frontend rejects int32 gather sources, and a naive full-width 8192-element search
has an estimated 224-KiB live set before extra buffers, above the usable UB budget.
This rules out that proposed implementation, not every possible gather solution.
A tiled AscendC metadata path or a carefully validated float bit-cast gather could
be separate follow-up designs. Neither has a measured speedup here.

The consumer still launches matching, commit, native prefix sums, partition,
native scatter and copy kernels. Prefetch has not beaten the no-prefetch control.
The existing default-off setting remains appropriate for this measured workload.
This experiment does not establish real-workload prediction recall or model E2E
latency/accuracy, and is not the 950 fused operator's performance result.

## Validation of tooling and rebuild setup

Ruff and Python/shell syntax checks pass for the added runner scripts.
ForgeKernel's adapter/parser/trajectory/CLI selection passed 165 tests. The broader
upstream suite was interrupted after 292 passes because its external GPU-monitor
cases were slow; a complete upstream suite pass is not claimed.

`scripts/forge-configure-lim.sh` records the clean-build bootstrap described above.
`scripts/forge-build-lim.sh` rebuilds and installs a changed producer in the isolated
checkout. The bootstrap steps were exercised manually during this experiment;
syntax checking the combined convenience script is not a second clean-build test.
All kernel edits that appear in the performance results were actually compiled.

The phase JSON files under `forge/lim910/` archive the local experiment configuration.
For a new campaign, adapt `workspace` and remove `baseline_result` to obtain a fresh
measurement of the current source. The historical `build/forge-lim/...` references
are local artifacts, not a portable baseline for a different source revision.

Final measurements can be reproduced directly without starting an agent:

```bash
python3 scripts/forge-lim-eval.py --output build/forge-lim/new-final-b16 \
  --device 6 --repeats 40 --profile-repeats 5 --attention-reference
ECHO_NPU_DEVICE=6 ./scripts/vllm-ascend-run.sh python \
  scripts/collect-echo-prefetch-profile-npu.py \
  --batches 24 --mtp 3 --context 40960 --hot-tokens 8192 \
  --misses-per-request 0 --modes baseline context-budget-0 prefetch-64-history \
  --repeats 40 --profile-repeats 5 --attention-reference \
  --output build/forge-lim/new-final-b24-all-hit
ECHO_NPU_DEVICE=6 ./scripts/vllm-ascend-run.sh python \
  scripts/check-forge-prefetch-filter-npu.py
```
