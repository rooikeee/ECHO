# 910 LIM / CopySFA：小预算与紧凑预取标记

2026-11-03。本轮只优化、验证融合 LIM 和融合 CopySFA，分别采集单算子 profiling。
MTP=3 表示每请求 4 行 query；没有做整模型或端到端吞吐测试。

## 默认预算和扫描范围

- 每 query 每次 LIM 调用最多预取 **8 个 token**，MTP3 每请求最多 **32 个**。
  这是一次 decode step 的总预算，不是每个 512-token tile 都搬 8 个。
- staging 容量仍是每 query 64 个；实际预算由固定地址的 `budgets` tensor 控制，
  Graph replay 之间可以原地修改。profiling 支持 8、16、32、64 的对照。
- 默认最多检查 **64 个 score tile**；当前 tile=512，即前 **32768 个 token**。
  达到搬运预算则提前结束。没有足够 miss 时也不会继续扫描整个 140k。
- 超出窗口不再发布预取 ready 标记，但完整打分、TopK、精确 miss 判定、slot 分配均保留。
  未被预取的真实 miss 继续从 Host 读取，因此窗口影响预取覆盖率，不改变选中的 token。
- `--prefetch-scan-tiles 0` 保留原来的全长预测扫描；默认 64。预测阈值仍来自上一步 cutoff，
  合成测试使用同一 query 重放，不能代表真实模型的预测准确率。

## LIM 发布紧凑标记

原来需要先发布 `source -> staging` 表，再由 CopySFA 建立 `slot -> owner` 表。
现在利用 LIM 已有的 request union 和 occurrence 信息，在其尾部一次确定 staging 和写回归属。

常见路径（请求的 miss occurrence 总数不超过 4096）在 UB 内完成：

1. DMA 读入最多四行有界预取候选，在已排序的 union source 中查找。
2. 利用已有 occurrence 中的 query、miss 前缀位置、union 编号，生成每个位置的标记。
3. 每个 query 用一次 DMA 写出有效 miss 前缀，不逐 token 写 GM。

超过此边界，MTP3 的完整 union 仍能放进 UB：读入 union 后逐 query 合并其排序 miss 前缀，
沿用同一 staging/owner 标记，不再对每个 miss 做 GM 二分查找。更宽的兼容路径保留原回退。
原有 TopK source、slot、miss counts 和 request union 输出保持不变。

`copy_routes[T,2048]` 的有效范围是各行原始 `topk_miss_counts`：

- 低 31 位为 0：从 Host 取数据；为 `1 + flattened_staging_row`：从 staging 取数据。
- 最高位为 1：该 occurrence 不写正式 slot；同一个 request union source 只有一个写回 owner。
- 命中位置不读取标记；每次 replay 覆盖当前有效 miss 前缀，不消费上一步的无效尾部。
- 同一 source 在不同 query 被选中时，各 query 从稳定的 Host/staging 数据计算 attention，
  不等待其他 query 写回正式 slot，也不增加 hit CAS。

LIM 的公开输入接口保持原样。内部 `[T,14]` 描述符的 field12 低 32 位仍是 staging capacity，
bit32 表示紧凑标记，bits40..55 表示扫描 tile 上限。旧描述符高位为零，仍走旧行为。
ready 和紧凑标记来自同一固定分配，布局为 `[T,source_capacity]` 后接 `[T,2048]`。
扫描上限对整个 owner 一致；不能手工给描述符不同行设置不同窗口。

## CopySFA 的变化

CopySFA 识别宽度 2048 的紧凑输入后：

- 随 source/slot 元数据一起 DMA 到 UB，不再逐 miss 随机读取 GM source map 和 owner 表。
- 移除 slot owner 表初始化、约 780KiB 临时 workspace 和对应的全核同步（B24/C8192）。
- 两个 miss 同为 Host 来源或同为 staging 来源时，按物理源地址排列，用两块 DMA 合并搬运
  CKV 和 RoPE；写回目标跟随 payload。混合来源保留单行搬运。
- 正式 slot 写回复用 attention gather 的 UB 数据，没有独立预取消费 kernel。
- 全命中跳过预取消费。first-fill 保留原批量回搬逻辑，不消费 staging。

两块搬运可能改变一对 attention key 的累加顺序，因此 attention 按原来的
`rtol=0.02, atol=0.002` 验收；缓存 KV/RoPE、原始 LIM 输出要求完全一致。

## 调用与复现

```python
prefetch = EchoMTPIndexerPrefetch(case, host_kv, host_rope, dram_table)
prefetch.indexer(cache, outputs)  # 默认 compact_routes=True
# CopySFA 的其他参数保持原样：
# ... **prefetch.copy_sfa_inputs()
```

owner、registered Host 缓冲及其固定地址 tensor 必须存活到引用它们的 Graph 销毁。
旧 source-map 对照需要同时给 `indexer()` 和 `copy_sfa_inputs()` 传 `compact_routes=False`。

```bash
ECHO_NPU_DEVICE=7 ./scripts/vllm-ascend-run.sh python \
  scripts/collect-echo-prefetch-profile-npu.py \
  --batches 24 --mtp 3 --context 143360 --hot-tokens 8192 \
  --misses-per-request 400 --modes baseline context-budget-0 prefetch-64-history \
  --prefetch-consumer copysfa --profile-scope operators \
  --route-metadata compact --prefetch-budget 8 --prefetch-scan-tiles 64 \
  --repeats 40 --profile-repeats 12 --attention-reference --check-staging-source \
  --output build/prefetch-v3-run
```

`prefetch-64-history` 是保留的历史模式名，64 指旧实验名称；实际预算看
`prefetch_budget` / `prefetch_budget_per_query`，本轮默认值为 8。

CopySFA 的 LIM 执行和 staging 准备都在计时之外；每次 CopySFA 计时前恢复相同正式缓存。
LIM 的 event 时间包含 ready/count/source 重置；表格的 kernel 时间来自 profiler 中目标任务。
每个配置独立测 baseline，以避免跨运行频率/负载变化造成误判。

## 最终实机结果

B24、MTP3、143360 tokens、热池8192、TopK2048、BF16、单张910（物理7号）。
每请求400个union miss；attention每rank16头。40次event重放，profiler目标任务13次。
下表为单算子kernel平均耗时（µs），两行分别独立捕获；不能据此宣称端到端加速。

| 场景 | baseline LIM | v3 LIM | baseline CopySFA | v3 CopySFA |
| --- | ---: | ---: | ---: | ---: |
| 每请求400 union miss | 2078.397 | 2230.248 | 418.046 | 414.442 |
| 全命中 | 1634.712 | 1848.999 | 418.044 | 419.855 |

默认8-token配置实际搬768个token（24×4×8），覆盖754个不同的request/source。
CopySFA已恢复到baseline附近；约1%的差别可能受测量波动影响，不保证所有输入都更快。
LIM仍有额外的预测、Host搬运、标记发布成本，不能认为预取已经没有成本。

预算对照都使用最终二进制和64-tile扫描窗口。数字是上限；窗口里没有足够候选时不会补满：

| 每query预算 | 实际搬运token | 有效不同source | LIM µs | CopySFA µs |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 768 | 754 | 2230.248 | 414.442 |
| 16 | 1529 | 1501 | 2331.030 | 419.952 |
| 32 | 2196 | 2156 | 2394.230 | 411.489 |
| 64 | 2205 | 2165 | 2399.003 | 418.395 |

对照的每次baseline和budget=0开销保存在 `budget_comparison.csv`。
v2历史记录为LIM2956.716µs、CopySFA540.104µs（每query64、全长预测扫描）；
这是旧版本结果，不是与本轮同时运行的baseline。

## 验证与数据位置

- 缓存、KV/RoPE、原始LIM输出完全一致；attention对照独立CPU参考。
- 每个预取配置验证7种动态Graph情况：query变化、长度/预算变化、零预算、重复候选、
  mixed first-fill、恢复steady，以及连续step后的全命中。
- 修改已预取token的Host数据后再执行CopySFA，输出保持正确，证明使用了staging。
- 120次边界重放覆盖负slot编码、非对齐长度、严格阈值、1/5/64-tile窗口、
  packed staging映射和每个miss仅一个写回owner。
- 大miss回退（每请求4000个union miss、occurrence>4096）、128个attention head和
  8项无效输入检查通过；仅做融合算子验证。

profiling数据在主工作区 `profiling/prefetch_v3/`：

- `baseline_lim/msprof/PROF_*` 与 `prefetch_lim/msprof/PROF_*`：LIM。
- `baseline_copysfa/msprof/PROF_*` 与 `prefetch_copysfa/msprof/PROF_*`：CopySFA。
- `all_hit/` 下相同目录：全命中对照。
- 在MindStudio中选择对应 `msprof/PROF_*` 目录；各算子目录也提供 `msprof.db`、
  `msprof.json`、`trace.json` 和 `kernel_details.csv`。
- `bounded_budget8/16/32/64`、`bounded_all_hit` 为最终原始采集，包含参数与源码哈希。
  其他 `budget*`、`pair_budget*`、`final_*` 目录是开发中间实验，不作为最终结果。
