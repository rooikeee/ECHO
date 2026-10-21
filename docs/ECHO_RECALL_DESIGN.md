# Ascend ECHO 第一阶段：Recall 实现与测试进展

更新日期：2026-10-10。本文描述当前工作区的设备优先写入实现，不是原 CUDA
ECHO 的完整性能复现。历史验证记录见 [实现与验证记录](ECHO_RECALL_IMPLEMENTATION.md)。

## 阶段结论

**第一阶段先实现按需精确 recall 及其必要的 KV 存储、热缓存 slot 管理和 Graph
接入；尚未实现预取。** 本阶段目标是打通功能路径，不是完成原 ECHO 的全部优化。

- 已实现：新 KV 设备优先写入、主机备份、Top-K 精确回搬、设备热缓存管理、
  注意力寻址适配，以及 decode NPU Graph capture/replay。
- 未实现：提前预测或提前搬入 KV 的预取、融合预取、独立传输流水线及传输/计算重叠。
- 验证进展：25 项 NPU 回归和微型 Qwen2 Graph 检查通过；完整 DeepSeek 模型的
  baseline/ECHO 均完成 40 条请求，decode Graph batch 1/2 通过。
- 性能结论：组件优化有效，但模型端到端**尚无加速，当前反而慢于 baseline**。
  最新输出吞吐为 baseline 26.68、ECHO 6.33 token/s，详见第 13 节。
- 验收边界：完整模型精度尚未通过独立验收；硬件 Alarm 仍在，性能为临时诊断结果。

没有预取意味着冷 miss 仍需在当前 attention 消费 KV 前按需回搬，当前同 stream
备份/回搬也没有与计算重叠。这解释了为何本阶段不能期待完整预取流水线的收益，
但**不能据此把本轮变慢全部归因于缺少预取**：prefill、备份、注意力适配及调度
开销尚未完成分项 profiling，当前短上下文测试也不保证存在大量冷 miss。

## 1. 当前实现解决什么问题

DeepSeek-V3.2 的 NSA 使用 index K 和 query 选择历史 token 的 Top-K，再对选中的
latent KV / RoPE KV 计算注意力。第一阶段保留原来的选择算法，只改变 KV 的存储位置
及访问路径：完整 KV 有主机副本，NPU 只保留有限容量的热缓存；选中但不在缓存中的
KV 必须精确回搬，不能为了省传输跳过它们。

这里的 recall 是“回搬选中的 KV”，不是修改检索召回率，也不预测下一步 Top-K。
当前路径不依赖 DeepGEMM；继续使用原来的 `npu_lightning_indexer` 和
`torch_npu.npu_sparse_flash_attention`。

本阶段已经接通了写入、命中检测、保护、淘汰、回搬、注意力寻址、请求释放和
decode NPU Graph。尚未实现融合预取、异步传输与计算重叠，也没有建立端到端收益。

## 2. 一层的完整执行顺序

```text
当前层在 NPU 计算出新 token 的 latent KV / RoPE KV
    -> 为新 KV 分配或复用热缓存槽位
    -> 写入 NPU 热缓存
    -> 从热缓存备份到注册的主机完整 KV

完整 index K（仍在 NPU）+ 当前 query
    -> 原 NSA indexer 给出逻辑 Top-K
    -> 结合请求页表，转换成完整 KV 的逻辑 ID，过滤未来位置
    -> 去重并检测缓存命中
    -> 保护本次需要的驻留 KV
    -> 为缺失 KV 选择空闲槽位或淘汰非保护项
    -> 从主机回搬缺失 KV 到热缓存
    -> 把原 Top-K 改写为热缓存物理槽位
    -> 原 NPU sparse attention 消费热缓存
```

关键区别：**新 KV 的主机副本来自已经写好的设备热缓存，而不是先写主机、再回搬
刚生成的 KV。** 已驻留的新 KV 被选中时直接命中。备份到主机不代表同时删除设备副本。

当前备份发生在 `set_kv_buffer()` 内，在同一 NPU stream 上排在热缓存写入之后、
本层后续注意力之前。它不是等整个模型 forward 完成以后才后台备份；也没有独立
传输 stream 或计算/传输重叠。设备首先产生并持有新 KV 的要求已经落实，延迟备份
和异步流水线仍是后续工作。

## 3. 数据结构和地址含义

下面的状态是**每层、每个 TP rank 独立**的。不同请求共享逻辑 KV 池，但通过请求
页表访问不同的逻辑位置。

| 数据 | 所在位置 | 含义 |
| --- | --- | --- |
| `index_k_buffer` | NPU | 完整 index K，不随 latent KV 淘汰 |
| `host.tensor` | 注册主机内存 | `[host_size, latent_dim + rope_dim]` 的完整 KV 副本 |
| `key`, `rope` | NPU | 容量 `C` 的热缓存，latent 和 RoPE 分开存放 |
| `host_to_device[id]` | NPU | 逻辑 KV ID 对应的设备槽位，`-1` 表示未驻留 |
| `device_to_host[slot]` | NPU | 设备槽位对应的逻辑 KV ID，`-1` 表示空闲 |
| `written[id]` | NPU | 主机 KV 是否完成写入；释放逻辑 ID 时清零 |
| `priority[slot]`, `clock` | NPU | 基于缓存操作的访问优先级，不是模型 token 位置 |
| `required`, `plan_flags`, `_miss_count` | NPU | 所需 ID 的 epoch 标签、输入块状态、当前缺失数 |
| `candidates` | NPU | 仅真实 miss 时使用的 LRU 淘汰候选优先级，无全量排序缓冲 |
| `free_slots`, `free_count` | NPU | 显式空闲槽位栈，设备端分配和回收；不是 CPU 管理的列表 |
| `recall_hosts`, `recall_slots` | NPU | 缺失 KV 的源 ID 和目标槽位 |
| `write_slots`, `resolved_slots` | NPU | 写入和索引转换的固定容量工作缓冲 |
| `errors`, `sticky_errors` | NPU | 当前操作错误和跨分块保留的错误标记 |

需要区分三个编号：

1. **请求内位置**：indexer 的逻辑 Top-K，例如该请求的第 130 个 token。
2. **完整 KV 池 ID**：请求页表转换后的全局存储 ID，作为主机 KV 的行号。
3. **热缓存槽位**：有限设备缓存内的位置，范围为 `[0, C)`。

主机内存用页对齐的 `mmap` 分配，通过 `aclrtHostRegister` 得到 NPU 可访问地址。
`written` 在 NPU 上，不是主机侧轮询或 CPU 调度条件。传输 kernel 直接读写映射主机
指针，不是每个 miss 都由 Python 发起一次动态 CPU memcpy。

服务池为 `size + page_size` 行预留主机 KV，保留框架 padding 页。直接 recall API
将 ID `0` 视为 padding，`-1` 视为未选中；服务页表转换会排除整个 page 0。

DeepSeek-V3.2 当前配置的 latent / RoPE 维度为 `512 / 64`，KV 为 BF16，
每个 token 每层的这两部分合计 `1152` 字节。热缓存只缩小这两部分，并不卸载
模型权重或完整 index K；Graph 工作空间和元数据也仍然占用 HBM。

## 4. 新 KV 怎么写

入口是 `EchoNPUMLATokenToKVPool.set_kv_buffer()`，最终调用
`EchoRecallCache.write(loc, key, rope)`：

1. 检查输入 dtype、设备、维度和连续性。
2. 写入数量超过 `C` 时，按静态容量分块；每块不超过热缓存容量。
3. 每块不超过 8 个 ID 时，使用 `_plan_small_write`：直接查映射，保护本次写入的
   已驻留项，直接从设备空闲栈取槽；没有空槽才通过最小优先级选择 LRU 槽位。不扫描主机
   逻辑 ID 空间，不做排序，直接输出到预分配的 `write_slots`。
4. 较大的写入使用 `_plan_slots(..., require_written=False)` 和 `_resolve`。
   新 ID 尚无主机副本，不要求 `written=1`；不再使用全量 `torch.sort`。
5. `_write_hot` 将模型刚产生的 NPU KV 写到 `key[slot]`、`rope[slot]`。
6. `_offload_hot` **从热缓存读取**，写入 `host[id]`，然后设置 `written[id]=1`。
7. 同一 stream 顺序保证本块备份完成后，下一块才能复用槽位。

覆盖已经驻留的同一个 ID 时保留槽位，原位更新设备 KV，并更新主机副本。
每次直接 `write` 的正 ID 必须唯一；没有为重复写入定义竞争顺序。

## 5. 缺失检测、保护与淘汰

当前小写入和 recall 已分开；较大写入仍复用 recall 规划，区别是 recall 要求主机
KV 已写入。Graph 内固定启动形状不变，分支和循环次数由设备数据决定：

1. `_begin_plan` 清零当前错误和 miss 数，递增 `clock`，不清空完整所需位图。
2. `_mark_required` 直接检查输入 ID 的映射和主机有效标记，把所需 ID 标记为当前
   epoch，并更新命中槽位的访问优先级。每个输入块输出 invalid/unwritten/miss 标记。
3. `_plan_misses` 汇总块状态。**全部命中时跳过逻辑池扫描、槽位扫描、分配和淘汰**。
   该判断发生在设备 kernel 内，不读回 CPU，也不使用 Python 动态分支。
4. 只有存在 miss 时，分块扫描带当前 epoch 的逻辑 ID，用块内 `tl.cumsum` 紧凑
   收集唯一缺失项，检查所需项总数不超过容量；优先直接从空闲栈批量取槽。
5. 只有空槽不足才用分块最小值归约选择 LRU 淘汰项。没有全量 `torch.sort`；
   容量检查在映射改写前完成，不允许淘汰本次命中项或输出不完整注意力集合。
6. `_assign_slots` 只处理紧凑缺失任务列表，更新双向映射。

分块限制片上 UB 工作空间，当前数据块最多 512 个元素。实际淘汰数很大时，逐项
LRU 归约的成本仍可能高，不能把全命中快路径的收益推广到高 miss 压力场景。

淘汰的是设备副本，主机副本仍有效。当前写入都会立即备份，因此不需要在淘汰时
额外写回脏 KV。如果以后改为延迟备份，必须额外设计脏位和写回依赖，不能直接复用
现在“主机副本已经可用”的假设。

如果某一组 query 所需 KV 的并集超过 `C`，当前实现不会只保留一部分继续计算。
它必须分组或报容量错误；否则会改变注意力集合。

## 6. Recall 和注意力适配

### 6.1 逻辑 Top-K 转换

`map_topk_to_host()` 在设备上结合原请求页表转换地址：

```text
page_column = request_position // page_size
pool_page   = block_table[request, page_column]
host_id     = pool_page * page_size + request_position % page_size
```

同时校验请求长度和 query 的原始因果位置。对 chunked prefill，因果位置是该 chunk
在完整序列内的位置，而不是热缓存槽位。越界、未来位置和 padding 输出 `-1`。

### 6.2 真正的回搬

`EchoRecallCache.recall(host_ids)` 先做上述规划，然后 `_copy_misses` 从注册主机
内存读取缺失 ID 的 latent / RoPE KV，写入新分配的热缓存槽位。

回搬 kernel 固定启动 `min(C, 32)` 个 worker，在图内按设备 miss 数量循环分工。
没有 miss 时直接跳过，不再为了全命中请求启动 `C` 个逐行 program。
命中项不传输 KV。最后 `_resolve` 按原输入顺序输出物理槽位，因此去重只减少回搬，
并不重排或删除原 Top-K 条目。

`miss_count` 是预分配的单元素设备计数，只表示**最近一次规划**的缺失数，
下一次 write/recall 会覆盖它；不是累计 offload/recall 流量统计。

### 6.3 为什么还要改注意力输入

原 sparse attention 的页表和稀疏索引描述完整逻辑 KV。直接换成任意热槽位可能
让算子按槽位号码重新做因果裁剪，例如把过去的 token 错当成未来位置。

当前适配方法是：先在原逻辑位置上完成因果过滤，再将每个 query 作为一个独立的
单 query 序列，KV 输入指向热缓存，使用 identity 页表和物理槽位稀疏索引。
热缓存按 `128` token 页组织；latent 同时作为当前 MLA 稀疏算子的 key/value，
RoPE 使用对应的独立缓存。

原来的 query、KV 内容、已经计算好的 RoPE 和选择集合保持不变；但算子输入分组
发生变化，浮点计算结果仍需模型级对照，不能只凭逻辑等价宣称数值验收通过。
当前适配不支持额外位置偏置。

默认每组 query 数为 `C // TopK`，保证最坏情况下选中 KV 的并集不超过 `C`。
当前 `C=4096`、`TopK=2048`，最坏情况一次最多处理两个 query。
如果 prefill 的 CPU 长度元数据能证明所有请求的完整历史总量不超过 `C`，则允许
一次处理全部 query。这是容量证明，不是从设备读回 miss 数量做 Python 分支。

## 7. 一个完整例子

下面只演示缓存管理，容量 `C=4`；实际 sparse attention 要求容量为 128 的倍数且
至少为 Top-K，因此不能把这个小例子直接作为完整服务参数。

初始状态：

| 设备槽位 | 驻留逻辑 ID | 访问优先级 |
| ---: | ---: | ---: |
| 0 | 101 | 5 |
| 1 | 102 | 2 |
| 2 | 103 | 4 |
| 3 | 空闲 | -1 |

主机已有有效 KV：`101、102、103、104`；其中 `104` 当前不驻留。

**先写新 token 105：** 分配空槽 3，先写 NPU KV，再从槽 3 备份到主机 105。
`host_to_device[105]=3` 保持有效，不因为主机备份而清除。

**当前 query 选择 `[101, 104, 105]`：**

1. 101 和 105 命中，保护槽 0 和槽 3。
2. 104 缺失。未保护的 102、103 中，102 的优先级更低，选择槽 1。
3. 清除 `host_to_device[102]`，设置 `104 <-> 槽 1`。
4. 只从主机回搬 104；101、105 不需要传输。
5. 返回物理稀疏索引 `[0, 1, 3]`，注意力读取这三个槽位。

主机仍有 102 的有效副本，后续再选中它时可以回搬。新 token 105 在本次直接命中，
并没有“先送去主机、又马上从主机取回来”。

如果下一次选择 `[101, 104, 105, 106, 107]`，并集为 5，超过容量 4；不能牺牲
其中一个命中项偷跑。对可分组的多个 query 先分组，单次规划仍不满足容量则报错。

## 8. NPU Graph 与资源生命周期

当前是 Graph 可捕获的缓存管理路径，不是一个重新实现的通用 Graph 显存分配器。

### 8.1 KV slot 管理是否在 NPU Graph 内

**是，但准确范围是 decode forward 内的设备热缓存 slot 管理。** 管理算法由
NPU 上的 Triton kernel 实现，这些 kernel 被捕获到 NPU Graph，并在每次 replay
按新的设备索引和缓存状态重新执行；不是 capture 时把槽位决策算一次后固定重用。
Graph 提供捕获和重放机制，实际槽位决策来自设备 kernel，而非 Graph 自动分配器。

| 操作 | 实现入口 | 完整服务中的执行边界 |
| --- | --- | --- |
| 小批新 KV 分配/复用槽位 | `_plan_small_write` | decode Graph 内；不超过 8 个新 KV 使用快路径 |
| 命中检测、去重标签、保护驻留 KV | `_mark_required` | decode Graph 内 |
| miss 汇总、空闲栈取槽、LRU victim 选择 | `_plan_misses` | decode Graph 内；全命中由设备分支跳过扫描和分配 |
| 清理旧映射、建立逻辑 ID 与物理槽位映射 | `_assign_slots` | decode Graph 内 |
| 新 KV 写入及主机备份 | `_write_hot`、`_offload_hot` | decode Graph 内，同一 stream 保证先写设备再备份 |
| 缺失 KV 回搬及物理索引解析 | `_copy_misses`、`_resolve` | decode Graph 内，完成后 attention 才消费 KV |
| 请求结束时释放热槽位 | `release`、`_tag_release`、`_release_slots` | NPU kernel 执行，但完整服务由图外 allocator 触发 |
| 请求逻辑页分配/释放、flush 和 capture 后重置 | `EchoNPUPagedTokenToKVPoolAllocator`、`clear` | 框架/Graph 生命周期管理，不属于 decode replay 内决策 |
| 所有层错误标记检查 | `check_echo_errors` | Graph replay 或 eager forward 后读回 CPU，存在同步 |

`free_slots/free_count`、双向映射、epoch 和优先级均驻留 NPU；CPU 不读取命中或
miss 数来选择槽位，也不逐项决定淘汰对象。CPU 仍负责框架请求调度、逻辑页生命周期
和 kernel/Graph 提交，不能将其表述为整个服务完全没有 CPU 参与。

当前 **prefill 是 eager**，使用同一套设备 kernel，但未验证完整服务 prefill Graph。
释放 kernel 在组件测试中有 Graph 验证，不代表完整服务的请求释放已纳入 decode 图。

### 8.2 状态、顺序和生命周期

| 阶段 | 状态处理 |
| --- | --- |
| 初始化 | 分配固定地址的热缓存、映射、工作缓冲和注册主机 KV |
| capture 前 | dummy 请求页表清零，避免 indexer 读取未初始化页 |
| capture 后 | 清除热缓存映射、时钟和错误；保留有效主机 KV 标记 |
| 每次规划 | 更新 clock；覆盖输入块状态，清零当前计数/错误；用 epoch 区分所需 ID |
| 每次 replay | 使用新的请求元数据；实际运行设备规划、写入、回搬和注意力 |
| replay 后 | Graph 外检查所有层的 sticky 错误，不在图内 `.item()` |
| 请求释放 | 各层解除双向映射并清零对应 `written`，避免 ID 复用读到旧请求 |
| flush | 清除热缓存状态和全部主机有效标记 |
| 关闭 | 确保没有图继续引用，同步设备后取消主机注册 |

固定形状的临时张量可能在 capture 时由 Torch Graph 内存池持有，不是所有临时
张量都由 ECHO 显式预分配。capture/replay 保证其地址有效，ECHO 自己的常驻状态
也必须与图共同存活。不能在 replay 间随意重分配主机或设备缓存。

操作在同一 NPU stream 上排序：设备写入完成后备份主机，备份完成后才允许后续
操作复用槽位；miss 回搬完成后注意力才能读取。当前不依靠 CPU 读取设备计数来
决定传输次数，注册主机内存不可在 stream 未同步时被 CPU 并发改写。

错误位：`1` 表示非法 recall ID，`2` 表示主机 KV 未写入，`4` 表示容量不足。
当前错误在每次规划时清零，但 sticky 错误跨分块保留，Graph 外统一检查。
服务 eager 路径在整个 forward 后、消费 logits 前检查一次所有层；decode 在 Graph
replay 后检查一次，不再在 prefill 的每层 attention 后同步读取错误标记。
不支持注册主机内存的环境直接报错，没有只跑 eager 的自动降级。

这与原 ECHO 的 Graph-aware KV 管理有相同的基本职责。当前已移除无条件全量扫描
和排序，并实现了设备空闲栈，但仍没有完整移植原实现的融合预取路径。
也没有解决 Graph 临时工作空间的整体预算优化，因此不能把“支持 Graph 的 KV
管理”称为“已完成原 ECHO 全部内存和性能优化”。

## 9. 修改文件和调用关系

生产代码路径相对 `sglang-ascend/python/sglang/srt/`：

| 文件 | 当前职责 |
| --- | --- |
| `hardware_backend/npu/echo_recall.py` | 主机注册、缓存状态、设备优先写入、规划/回搬 kernel、Top-K 转换及注意力适配 |
| `hardware_backend/npu/echo_memory_pool.py` | 完整 index K、每层热缓存、KV 池接口、所有层错误汇总 |
| `hardware_backend/npu/attention/ascend_backend.py` | 原 `forward_sparse` 内接入 recall 分支，不替换整个 attention backend |
| `hardware_backend/npu/allocator_npu.py` | 请求页释放与 flush 接入 ECHO 状态清理 |
| `hardware_backend/npu/graph_runner/npu_graph_runner.py` | dummy 页表初始化、capture 后重置、replay 后错误检查 |
| `model_executor/model_runner_kv_cache_mixin.py` | 选择 ECHO KV 池、校验 NSA Top-K、计算设备预算 |
| `model_executor/model_runner.py` | eager forward 结束后统一检查错误，避免逐层 CPU 同步 |
| `server_args.py` | `--enable-echo`、`--echo-device-cache-size` 及组合限制 |

验证入口：

- `scripts/check-echo-recall-npu.py`：设备 kernel、缓存行为和实际 Graph 单元测试。
- `scripts/smoke-echo-deepseek-npu.py`：完整模型短请求集成及数值对照。
- `scripts/bench-echo-sharegpt-npu.py`：完整 ShareGPT baseline/ECHO 对照。
- `scripts/summarize-echo-sharegpt-prefix.py`：本次停止后的日志窗口估算，不启动 NPU 服务。
- `scripts/bench-echo-recall-planner-npu.py`：优化前后 KV 管理的实际 Graph 分项计时，不包含完整模型。

因此，第一阶段确实要接入 attention 后端，但不是只改后端接口就完成；还需要
KV 池、逻辑页释放、主机注册、设备缓存管理和 Graph 生命周期一起配合。

## 10. 如何确认真的发生了 Offload / Recall

启动日志包含：

```text
ECHO exact recall enabled: ... mapped host KV=...;
no fused prefetch; write_policy=device_first, host_backup=same_stream,
planner=epoch_hit_fastpath, small_write=free_stack, eviction=bounded_lru
```

这能说明启用了哪条实现路径、主机分配大小和写入策略，**不能单独证明实际传输了
多少字节或发生了多少 cache miss**。decode 日志中的 `npu graph: True` 同样只
证明采用 Graph，不是传输计数。

当前更强的行为验证在单元测试中：

- 拦截主机备份调用，确认设备 KV 已正确写入、主机尚未写入。
- 完成备份后比较主机 KV；人为修改主机副本，确认热命中不会回搬覆盖正确设备 KV。
- 小容量写入超过容量的数据，确认全部主机副本有效；选择已淘汰项，检查 miss
  数及回搬后的设备内容。
- Graph batch 1/2 实际捕获并反复重放，在动态 ID、命中、缺失、淘汰及释放下检查结果。

2026-10-09 的设备优先版本 18 项测试通过，运行在物理设备 14，包含新 kernel 编译，
总用时 98.044 秒。完整模型也完成 decode Graph batch 1/2 检查。
这些是行为及 Graph 验证，不是传输带宽测量或模型精度验收。

2026-10-10 优化后增加了小写入不调用完整规划器、全命中不扫描未选中 ID、
LRU 顺序及非二次幂容量测试。最终空闲栈版本增加了 1024 个历史 KV、批量重复释放
和重写的服务容量 Graph 测试；25 项全部通过，用时 6.806 秒，已有编译缓存，
不是性能指标。服务容量 padding 和动态新 KV 的 batch 1/2 Graph 重放均通过。

服务尚没有逐层累计备份字节、回搬字节、hit/miss、淘汰次数及分阶段耗时日志。
因此本次 ShareGPT 日志无法直接给出真实传输流量；后续 profiling/遥测应补充这些
指标，并把 Graph 内计数与 Graph 外采集分开，避免同步破坏性能。

## 11. 当前配置限制

目前服务要求 NPU + Ascend 后端、NSA、BF16/FP16 KV、显式逻辑 token 容量，
热缓存容量为 128 的正整数倍且不小于 Top-K。需要关闭 radix cache 和 overlap
schedule。

第一阶段测量时 ECHO 分支禁止 DP attention；随后已接入实验性 TP/DP attention，
实际多卡验收状态见 [TP/DP 接入记录](ECHO_PARALLEL_SUPPORT.md)，不得用第一阶段
TP8/DP1 的结果代替 DP 验证。当前仍禁止 PP、PD 分离、推测解码、HiCache/HiSparse/LMCache、
torch compile 和 memory saver 等组合。这是该实验分支的限制，不能据此断言整个
sglang-ascend 不支持这些功能。普通多副本 DP 也没有在本轮验证；第一阶段完整模型实测只有 TP8/DP1、
PD 合部，decode Graph batch 1/2。Prefill 执行仍是 eager，并没有验证 prefill Graph。

原来的严格完整模型 logprob 对照未通过；短测试的文本一致率不等于任务准确率。
本阶段可称为“精确所选 KV 的功能路径已实现并有 Graph 测试”，不能称为完整性能
或模型精度验收完成。

## 12. 优化前的前 40 条 ShareGPT 对照

2026-10-10 使用 DeepSeek-V3.2-AWQ，物理设备 8–15，TP8/DP1、PD 合部、
并发 2，每条固定生成 128 token，seed 42，输入加输出预算不超过 2048 token。
完整 index K 及逻辑 token 容量为 8192，设备热缓存每层 4096 token。

用户要求测到 40 条停止；处理停止指令时日志表明 ECHO 已完成 42 条、开始 44 条。
已停止并清理测试进程，不继续测剩余请求。比较只取与 baseline 相同的前 40 条，
输入合计 14628 token，预期输出合计 5120 token。

本次停止处理有一个记录缺口：停止服务后，官方 benchmark 客户端在请求
`/server_info` 时抛异常，未写出 ECHO 的逐请求结果。原始 `result.json` 因此仍是
`failed`；没有改成成功。单独的停止报告是 `user_stopped_partial_log_recovery`。

| 前 40 条对照 | Baseline | ECHO |
| --- | ---: | ---: |
| 服务日志估算时间窗口 | 192 s | 1115 s |
| 估算输出吞吐，包含 prefill | 26.67 token/s | 4.59 token/s |
| 窗口内 Graph decode 日志吞吐中位数 | 29.54 token/s | 6.32 token/s |
| 客户端平均 TTFT | 902.39 ms | 未保存，无法恢复 |
| 客户端平均 ITL | 68.92 ms | 未保存，无法恢复 |

时间窗口采用本轮性能请求第一次 HTTP 接收时间，到第 42 条性能请求接收时间：
两边前 42 条均观察到成对请求在同一秒开始；在固定并发 2、FIFO 提交、请求执行
超过一个日志时间桶、无前序失败或跨批超越的假设下，用下一对开始估算前 40 条完成。
仅凭并发和 FIFO 不能一般性地证明某个前缀已完成，这个窗口仍然是日志恢复估算。
HTTP 200 本身只代表流式响应开始，不能把 40 次 HTTP 200 当作完成 40 条。
该方法有秒级日志粒度和下一对接收开销，**不是客户端精确计时**。
稳态 decode 日志吞吐也不能替代端到端吞吐。

按这个一致的日志口径，ECHO 输出吞吐约为 baseline 的 17.2%，没有速度收益。
不提供缺失的 ECHO TTFT、TPOT、精确 E2E、P99 或文本一致率，也不混用旧版本
100 条结果补齐。8–15 卡在测试前后均有 `80C98001` Alarm，数据为临时诊断。

并发 2、每条历史预算不超过 2048，与热缓存 4096 的配置可能容纳全部活跃历史，
因此该工作负载不是证明大量淘汰/冷 miss 的压力测试。强制 miss 的正确性证据来自
小容量单元测试；模型服务里的实际 miss 流量尚没有累计日志。

产物目录：`build/echo-sharegpt/20261010-091033/`。
日志恢复结果：`stopped-prefix40-summary.json`；baseline 完整原始指标：
`baseline-metrics.jsonl`；停止及缺口原因：`echo-benchmark.log`。

离线重新生成估算报告，不会启动测试：

```bash
python scripts/summarize-echo-sharegpt-prefix.py \
  build/echo-sharegpt/20261010-091033 --count 40
```

## 13. 第一阶段最新测试结果：优化后完整 40 条对照

### 13.1 测试配置与口径

2026-10-10 11:19:11 启动的完整测试目录为
`build/echo-sharegpt/20261010-111911/`。本节是最新完成结果，第 12 节保留的是
优化前被停止测试的日志估算，两者不能混称同口径加速比。

| 项目 | 配置 |
| --- | --- |
| 模型 | DeepSeek-V3.2-AWQ，4-bit 权重量化 |
| 设备/并行 | 物理 NPU 8–15，TP8/DP1，PD 合部 |
| 数据 | 同一批 40 条 ShareGPT，seed 42，输入共 14628 token |
| 输出 | 每条固定 128 token，忽略 EOS，两边各输出 5120 token |
| 请求配置 | 客户端并发 2，输入加输出采样预算不超过 2048 token |
| 容量 | 逻辑 token 容量 8192，每层热缓存 4096，服务 context 4096 |
| Prefill | chunk 512，eager |
| Graph | 两边 decode 均开启，batch 1/2 capture 和 probe 通过，无 decode eager fallback |
| 计时 | 两边分别启动 fresh 服务；加载、编译、capture、probe 和 warmup 不计入正式计时 |
| 重复次数 | 每种模式 1 次，尚无多轮置信区间 |

### 13.2 组件验证与单层耗时

优化后最终 25 项 NPU 回归全部通过，覆盖设备优先写入、热命中、覆盖更新、去重、
LRU 淘汰、冷回搬、错误保护、释放复用、非二次幂容量、稀疏注意力及动态 batch 1/2
Graph capture/replay。微型随机 Qwen2 的 batch 1/2 Graph 检查也通过。

单层 Graph 组件计时：逻辑容量 8320、热缓存 4096、历史 KV 1024，50 次 replay
一组、重复 3 组，以下为 batch 2 的 NPU event 耗时中位数。

| 路径 | 优化前设备优先版本 | 优化后 | 加速比 |
| --- | ---: | ---: | ---: |
| 新 KV 写入 | 1.742 ms | 0.0345 ms | 50.5x |
| 全命中 recall | 2.090 ms | 0.3176 ms | 6.58x |
| 写入 + recall | 3.848 ms | 0.3351 ms | 11.48x |

这里比较的是设备优先版本优化前后，不是 host-first；不包括完整模型、indexer、
sparse attention、服务调度和图外错误检查，不能将 11.48x 作为模型吞吐提升。
原始组件结果为 `build/echo-recall-optimization/planner-metrics.json`。

### 13.3 完整模型端到端结果

| 客户端指标 | Baseline | 第一阶段 ECHO（优化后） |
| --- | ---: | ---: |
| 成功请求数 | 40/40 | 40/40 |
| 正式测量时长 | 191.93 s | 809.39 s |
| 输出吞吐 | 26.6759 token/s | 6.3257 token/s |
| 平均端到端请求延迟 | 9595.07 ms | 40456.95 ms |
| 平均首 token 延迟（TTFT） | 886.03 ms | 24117.37 ms |
| 平均每输出 token 耗时（TPOT） | 68.58 ms | 128.66 ms |
| 平均 token 间隔（ITL） | 68.90 ms | 129.29 ms |
| P99 端到端请求延迟 | 10782.07 ms | 50741.99 ms |

**第一阶段已打通 recall，但尚未实现预取和传输/计算重叠，端到端没有获得加速。**
本轮 ECHO 输出吞吐是 baseline 的 23.71%，测试耗时约为 4.22 倍，不能描述成
“已有少量端到端收益”。相较 baseline，平均请求增加约 30.86 秒，其中首 token
前增加约 23.23 秒，占增量约 75.3%；TTFT 还包含排队与调度，不能直接定位某个 kernel。

小写入快路径只覆盖不超过 8 个 KV，不覆盖 512-token prefill。下一阶段需要对
prefill 的批量规划、主机备份、注意力输入构造、稀疏 attention 和调度分别计时，
再设计预取及异步重叠，不能假定仅加入预取就能解决全部开销。

### 13.4 正确性与结论限制

- 两边生成文本完全一致 12/40（30%）。这不是任务准确率，不能据此认定模型精度
  通过或失败；此前严格完整模型数值对照尚未通过，本轮没有完成独立精度验收。
- 原有 `80C98001` 硬件 Alarm 在测试前后均存在，结果状态为
  `completed_with_hardware_alarm`，只能作为带告警的临时诊断。
- 当前活跃历史可能全部容纳在 4096 热槽位内，尚不能证明长上下文、持续冷 miss
  或大量淘汰时的性能与预取收益；强制 miss 的行为验证来自组件测试。
- 旧 host-first 的 7.07 token/s 来自 100 条测试，不能与本轮 40 条作严格同样本对比。
- 本轮脚本正常退出，两边测试服务已退出；测试结束时确认 8–15 卡已释放，不代表
  后续时刻仍然空闲。

可追溯产物（均在上述完整测试目录）：`result.json`、`baseline-metrics.jsonl`、
`echo-metrics.jsonl`、`baseline-requests.jsonl`、`echo-requests.jsonl` 以及两边
server/benchmark 日志。详细优化过程见 [Recall 优化记录](ECHO_RECALL_OPTIMIZATION.md)。

## 14. 后续优化需要验证的方向

- 逐请求增量结果落盘已经接入；仍需停止接收新请求并排空在途请求的机制。
- 对新 KV 写入、全容量扫描/排序、主机备份、miss 回搬、attention 分组分别 profiling，
  不仅凭 TTFT 或整体吞吐推断瓶颈。
- 小写入和全命中路径已优化；后续重点验证高 miss 场景，避免大量逐项 LRU 归约
  抵消空闲栈和命中快路径收益。
- 评估固定工作空间复用和 Graph 内存增量，统计包含权重、index K、KV、元数据、
  Graph 池的完整 HBM 占用，不能只报热缓存缩小。
- 在明确缓存写回依赖后评估异步备份、计算/回搬重叠和融合预取。
- 单独验证完整模型数值误差，以及长上下文、高并发、热缓存小于活跃历史的场景。

第 12 节那次测试已按要求停止。随后用户要求继续优化；新的性能测试和结果单独记录
于 [优化记录](ECHO_RECALL_OPTIMIZATION.md)，不覆盖或改写此前停止测试的数据。

优化后已在 `build/echo-sharegpt/20261010-111911/` 完成两边各 40 条 fresh 测量：
baseline 26.68 token/s，ECHO 6.33 token/s；ECHO 平均 TTFT 24.12 秒、TPOT 128.66 ms。
两边请求均成功，Graph batch 1/2 验证通过，但没有端到端性能收益。
后续必须覆盖批量 prefill 路径，不仅测不超过 8 个新 KV 的小写入组件。
硬件 Alarm 仍在，文本一致 12/40 不代表严格数值或任务精度验收通过。
