# ECHO Development Requirements

NPU Graph compatibility is required for the ECHO Ascend implementation.

- New device paths for KV offloading, recall, cache updates, and prefetching must
  support graph capture and replay. Eager execution is useful for debugging but
  does not satisfy implementation or performance acceptance by itself.
- Keep graph-referenced buffer addresses valid across replay. Define how dynamic
  indices, valid counts, cache mappings, counters, and temporary state are updated
  or reset on every replay. Verify stream/event ordering and host-buffer lifetime.
- Validate capture, actual replay, repeated replay, and correctness as request
  lengths, batch sizes, cache hits, misses, and evictions change.
- Use `scripts/npu-run.sh` for the project environment. The baseline check is
  `./scripts/npu-run.sh python scripts/smoke-sglang-npu.py`; it requires graph
  capture and decode replay for batch sizes 1 and 2.
- The baseline uses a tiny Qwen2 model on one NPU. It does not establish ECHO,
  DeepSeek NSA, prefill graph, or multi-device support; validate those paths when
  implementing them and report the actual coverage.
- ECHO operator profiling must use MTP=3 going forward. The target verification
  phase has four query rows per request (one current token plus three drafts);
  report target-phase time separately from the complete speculative round.
  Keep the required tiny-model environment baseline distinct from this workload.

See `docs/ASCEND_MIGRATION.md` and `docs/NPU_ENVIRONMENT.md` for the migration
requirements, environment, and validation records.
