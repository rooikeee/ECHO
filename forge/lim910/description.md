# 910 LIM ECHO prefetch optimization contract

Optimize the EXISTING ECHO prefetch implementation on Ascend910_9391 / CANN9.1.
The task is full Graph chain latency: LIM with real Host KV prefetch -> useful
prefetch commit and remaining miss metadata -> CopySFA. This is not a CUDA task.
B16, MTP3 means four query rows/request, context40960, hot cache8192/request,
TopK2048, 400 DISTINCT request-union misses. BF16 CKV512+KPE64. Per-query budget64.

Initial repeat measurement on Nov2: no prefetch ~0.776ms; context budget0 ~2.977ms;
real budget64 prefetch ~4.669ms. Correctness and all 7 dynamic Graph scenarios pass.
Goal: reduce full prefetch-chain median; ultimate target beat no-prefetch baseline.
A faster but still slower-than-baseline prefetch must be labelled as such.

Frozen validator: scripts/collect-echo-prefetch-profile-npu.py (DO NOT MODIFY).
It uses actual registered Host buffer aliases, exact formal KV/cache comparisons,
unchanged raw LIM outputs, attention rtol=.02 atol=.002, fixed addresses, different
queries/lengths/budgets, duplicate MTP candidates, mixed first-fill and all-hit replay.
The consumer must commit a source once per request union and move only prefetched
items into formal cache; all remaining misses must be passed correctly to CopySFA.
Do not bypass real prefetch, shrink budgets, skip tests, change seeds/inputs,
precompute the answer, move work outside Graph timing, or use CPU tensor.item() in
captured work. No changes to actual score/TopK or the external LIM/CopySFA ABI.

Native producer presently scans sources scalar after local ProcessMain and copies
one selected KV at a time. Python/Triton MTP consumer performs two dense match grids,
commit, two scans, two permutation steps and four scatters over fixed-size buffers.
Triton3.2.2 on this 910 compiler lowers cumsum and discrete scatter with host-initialized
locks that break GLOBAL graph capture. Native torch.cumsum(out=) and scatter_ avoid
that problem but add significant launches. Any replacement must actually replay.

First iteration focus: consumer-only cost. Keep Graph dynamic budget support. Do
not incorrectly skip consume on Python budget0 because the same captured graph is
validated with budget7/64 later. Device-side branches may skip work if outputs still
have correct values. Output tensor addresses must remain fixed across replay.
