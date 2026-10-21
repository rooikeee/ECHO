# ECHO CUDA 预取实现详解：Decode 与 Prefill

## 1. 阅读范围与版本

本文记录 ECHO 的 CUDA 融合预取实现，回答三个问题：预取何时发生、具体搬什么、对应代码在哪里。

- 代码基线：`bc1b75c1000010d0ac6f032ebaac283255c050b1`，简称 `bc1b75c`。
- 重点：DeepGEMM 的 SM90 融合 kernel，以及 SGLang 的调用、缓存管理和 Attention 衔接。
- 论文对应：第 5 节，Figure 10 对应 decode，Figure 11 对应 prefill。
- 本文是静态代码分析，不代表重新运行并验证了 CUDA kernel 的正确性、Graph replay 或性能。
- 当前工作区已迁移到 `sglang-ascend/` 和 `deepgemm-ascend/`。本文中的 `sglang/`、`DeepGEMM/` 路径均指上述历史提交，不是当前 NPU 源码。

可以在仓库根目录查看历史源码，无需恢复或覆盖当前工作区：

```bash
git show bc1b75c:sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py
git show bc1b75c:DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh
git show bc1b75c:DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh
```

下文代码定位中的行号都属于 `bc1b75c`。代码块标为“原文节选”时来自对应源码；标为“伪代码”时仅表达逻辑，不是可直接运行的实现。

目录：

- [整体模型](#2-先建立整体模型)
- [代码地图](#3-代码地图)
- [开启条件与入口](#4-开启条件与入口)
- [Decode 实现](#5-decode同一个-query-内的预取)
- [Prefill 实现](#6-prefill不同-query-块之间的预取)
- [Host KV 搬运](#7-kv-是怎样从-host-搬到-gpu-的)
- [容量与各层关系](#8-池大小2048-和各层的关系)
- [正确性与实现边界](#9-正确性同步和实现边界)
- [当前 NPU 实现](#10-与当前-npu-实现的区别)
- [Decode/Prefill 对照](#11-最后对照)

## 2. 先建立整体模型

### 2.1 Indexer 和最终 Attention 不是一回事

Indexer 为历史 Token 计算选择分数，精确 Top-K 决定最终使用哪些 Token；最终 Attention 使用这些 Token 的 Attention KV 计算输出。

```text
当前层的 Indexer Q、Indexer K
               |
               v
Indexer 算分 + 融合预取
               |
               v
精确 Top-K
               |
               v
整理缓存映射、补搬缺失的 Attention KV
               |
               v
最终 Attention
```

两种融合预取都没有把最终 Attention 放进 Indexer 的三段流水。这里隐藏的是 KV 搬运相对于 Indexer 算分的部分延迟。

“KV 全部就绪”指该次 Attention 调用需要的选中 KV 已就绪，不是把整个历史上下文都搬回 GPU。短上下文中有效选择数也可能少于配置的 K。

### 2.2 必须区分三种数据

| 数据 | 保存位置和用途 | 融合 kernel 中的工作 |
| --- | --- | --- |
| Indexer K 和量化 scale | 保留在 GPU，用来给历史 Token 算分 | TMA 从 GPU 全局显存加载到共享内存 |
| Attention KV | Host 保存完整副本，GPU 正式池保存其中一部分 | Prefetch 组从 Host 读取缺失项 |
| Indexer logits | Token 的选择分数 | GEMM 产出，Prefetch 组消费，最终 Top-K 再使用 |

本实现的 Attention KV 实际是 MLA 压缩表示和 RoPE 部分，不应简单理解为常规 MHA 的两份完整 K、V。

### 2.3 TMA、GEMM、Prefetch 的职责

| 阶段 | 主要操作 | 不是什么 |
| --- | --- | --- |
| TMA | 加载 Indexer Q/K、scale、weight 到共享内存 | 不是本文 Host Attention KV 的主要搬运路径 |
| GEMM | FP8 WGMMA、缩放、ReLU、加权归约，生成 Indexer 分数 | 不是最终 Attention 的矩阵计算 |
| Prefetch | 筛候选、查缓存映射、申请预算、读取 Host KV 并写 GPU | 不是精确 Top-K 的替代品 |

不同 Warp 组负责这些工作，利用多槽共享内存缓冲和 barrier 形成生产者/消费者流水。流水能重叠不等于没有竞争；共享内存缓冲耗尽或预取太慢时，GEMM 仍可能被反压。

### 2.4 分数的含义

省略量化 scale 已吸收的位置后，Indexer 分数可概括为：

```text
score(q, t) = sum_h weight(q, h) * ReLU(dot(index_q(q, h), index_k(t)))
```

真实代码还处理 Q/K 量化 scale 和跨线程归约。不要把这个 score 当作最终 Attention 的 softmax 概率。

## 3. 代码地图

为缩短后文定位，用以下代号表示历史文件：

| 代号 | `bc1b75c` 中的路径 | 重要位置 |
| --- | --- | --- |
| IDX | `sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py` | 开关 L45；状态 L226；decode L324；prefill L445；分发 L737 |
| ATT | `sglang/python/sglang/srt/layers/attention/nsa_backend.py` | 精确 Top-K L136；prefill 补搬 L613；decode 补搬 L777 |
| POOL | `sglang/python/sglang/srt/mem_cache/memory_pool_host.py` | 映射 L856；预取缓冲 L934；prefill Recall L1303；decode Recall L1379 |
| RECALL | `sglang/python/sglang/srt/mem_cache/recall_ops.py` | decode 预取落池 L385；清理 L525；prefill 补搬 L573 |
| D-KERNEL | `DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh` | decode 融合 kernel L413 |
| P-KERNEL | `DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh` | 桶映射 L319；prefill 融合 kernel L889 |
| D-JIT | `DeepGEMM/csrc/jit_kernels/impls/smxx_fp8_paged_mqa_logits.hpp` | decode 线程/流水配置 L333 |
| P-JIT | `DeepGEMM/csrc/jit_kernels/impls/smxx_fp8_mqa_logits.hpp` | prefill 线程/流水配置 L410 |
| API | `DeepGEMM/csrc/apis/attention.hpp` | prefill Python/C++ 绑定 L182；容量和形状校验 L245 |
| COPY | `DeepGEMM/deep_gemm/include/deep_gemm/common/utils.cuh` | Warp 搬运 L301；1152 字节特化 L335；L2 预取 L370 |

## 4. 开启条件与入口

### 4.1 两个预取开关默认都关闭

IDX L45-L50，原文节选：

```python
NSA_FUSE_LOGITS_RECALL_EXTEND = get_bool_env_var(
    "SGLANG_NSA_FUSE_LOGITS_RECALL_EXTEND", "false"
)
NSA_FUSE_LOGITS_RECALL = get_bool_env_var(
    "SGLANG_NSA_FUSE_LOGITS_RECALL_DECODE", "false"
)
```

只有 Host offload 池和相应开关同时满足时，才走下文的融合预取分支。仅开启 KV offload 不等于开启预取。

### 4.2 Decode 与 Prefill 的分发

IDX L737-L744，原文节选：

```python
if forward_batch.forward_mode.is_decode_or_idle():
    topk_result = self._get_topk_paged(
        forward_batch, layer_id, q_fp8, weights, metadata
    )
else:
    topk_result = self._get_topk_ragged(
        forward_batch, layer_id, q_fp8, weights, metadata
    )
```

Decode 用分页 Indexer K；prefill 的 ragged 路径先整理各请求的连续 Indexer K 和有效区间。

## 5. Decode：同一个 Query 内的预取

### 5.1 为什么不等精确 Top-K

精确 Top-K 需要看完整个 Query 的有效分数。若先算完所有分数、再选 Token、最后搬 KV，搬运无法与该 Query 的算分重叠。

Decode 改用预测阈值：一个 K 块分数刚生成，就检查其中哪些 Token 高于阈值，立即尝试搬运。最终仍执行精确 Top-K，因此预测只影响搬运时机和额外开销，不决定最终 Attention 的选择。

这里的“Top-p”指 `score > p`，不是生成采样中的累计概率 nucleus sampling，也不是每个 K 块独立选 Top-K。

### 5.2 用 EMA 更新阈值

IDX L233 设置 `decode_logits_ema_decay = 0.5`，预测数组初始化为 0。融合算分结束后，IDX L410-L416 原文：

```python
topk_result, topk_logits = metadata.topk_transform(
    logits, self.index_topk, return_topk_logits=True
)
self.decode_topk_logits[:q_len] = (
    self.decode_logits_ema_decay * self.decode_topk_logits[:q_len]
    + (1 - self.decode_logits_ema_decay) * topk_logits
)
```

其中 `topk_logits` 是精确选择得到的第 K 大分数。因而：

```text
本轮预取使用旧预测 p_t
本轮精确 Top-K 得到 s_t
p_(t+1) = 0.5 * p_t + 0.5 * s_t
```

预测阈值偏低会增加候选，偏高会漏掉应预取项。两者都通过预算限制和最终补搬处理；不能宣称预测候选恰好为 2048 个。

代码按 `q_len` 切片维护预测状态，本文不把它额外解释成一个有请求 ID、层 ID 键值的独立预测表。

### 5.3 Python 准备和调用

IDX L382-L389 会将本次 `query_recall_counter` 清零，并取出预取缓冲切片。IDX L391-L409 原文：

```python
logits = deep_gemm.fp8_paged_mqa_logits_fused_v2(
    q_fp8,
    kv_cache_fp8,
    weights,
    seqlens_32,
    block_tables,
    schedule_metadata,
    max_seq_len,
    # kv data
    page_table_1,
    device_pool_buf,
    host_pool_buf,
    prefetch_host_loc,
    prefetch_kv_buf,
    host_token_to_device,
    query_recall_counter,
    self.decode_topk_logits[:q_len],
    clean_logits=False,
)
```

参数分两组：`q_fp8/kv_cache_fp8/weights` 供算分使用；Host/GPU 池、页表、映射、计数器、预测阈值供 KV 预取使用。一个 kernel 同时拿到两类数据，才能把搬运融合进去。

### 5.4 任务划分与 Warp 分工

D-KERNEL L553 创建 `PagedMQALogitsScheduler`。任务按 Query 及 K 区间分配，不要求一个 SM 负责该 Query 的全部历史 K。

D-JIT L360-L365 原文：

```cpp
const int num_specialized_threads = 128;
const int num_math_threads = num_math_warp_groups * 128;
const int num_prefetch_threads = 128;
const int num_extra_threads = device_runtime->get_arch_major() == 10 ? 128 : 0;
const int num_q_stages = 3, num_kv_stages = 3;
const int split_kv = num_math_warp_groups * block_kv;
```

对本文 SM90 路径，TMA 组配置 128 线程，Prefetch 组 128 线程，即各 4 个 Warp；Math 组大小由 `num_math_warp_groups` 决定。配置线程数不意味着这些线程每周期都执行有效工作。

TMA 组加载 Indexer Q、weights、分页 K 和 scale。对于连续处理的同一 Query，Q 在共享内存中复用，K 块不断替换。Math 组在 D-KERNEL L680-L743 执行 WGMMA 和后处理。

### 5.5 分数通过共享内存交给预取组

D-KERNEL L747-L755 原文：

```cpp
CUTE_TIE_DECL(get_pf_pipeline(pf_iter_idx++), pf_stage_idx, pf_phase);
empty_logits_barriers[pf_stage_idx]->wait(pf_phase ^ 1);    // !!! IMPORTANT !!!

#pragma unroll
for (uint32_t i = 0; i < kNextN; ++ i) {
    smem_logits[pf_stage_idx][i * kNumMathWarpGroups * BLOCK_KV + kv_group_idx * BLOCK_KV + sub_warp_offset + v_0_offset] = v_0[i];
    smem_logits[pf_stage_idx][i * kNumMathWarpGroups * BLOCK_KV + kv_group_idx * BLOCK_KV + sub_warp_offset + v_1_offset] = v_1[i];
}
full_logits_barriers[pf_stage_idx]->arrive();
```

GEMM 先等待一个可写槽位，写入分数，再通知消费者。同一批分数还写入全局 `logits`，供后续精确 Top-K 使用。

Prefetch 组等待对应 `full_logits_barriers` 后读取分数。读取并保存候选布尔标记后，先在 L880 执行 `empty_logits_barriers[pf_stage_idx]->arrive()`，释放共享内存槽位，再做耗时的 KV 搬运。

这点很重要：不需要一直占着分数缓冲等待 Host KV 返回。D-JIT L396-L419 还尽量利用剩余共享内存增加分数流水槽数，要求至少 3 槽，但有限槽数仍可能产生反压。

### 5.6 从候选到缓存缺失项

D-KERNEL L864-L866，原文节选：

```cpp
if (logit > topk_logits) {
    need_copy = true;
}
```

这里的 `topk_logits` 来自传入的预测值，不是当前块单独计算的第 K 大分数。

随后 L901-L907 原文节选：

```cpp
if (need_copy_buf[iter]) {
    const uint32_t idx = pf_tx + iter * kNumScanThreads;
    const int kv_token_offset = kv_idx * BLOCK_KV + idx;
    kv_token_host_loc = __ldg(
        page_table_1 + q_idx * page_table_1_stride + kv_token_offset
    );
    is_miss = host_token_to_device[kv_token_host_loc] == INT32_MAX;
}
```

所以高于阈值只是候选；正式 GPU 缓存已有映射的 Token 不必再从 Host 搬。L858 还限制 `kv_token_offset < context_len - 1`，不把当前最后一个 Token 当作历史缺失项预取。

### 5.7 上限是每个 Query 64 个，不是每块 64 个

D-KERNEL L571 写死 `req_pf_max = 64`。多个 SM 通过同一 Query 的原子计数器竞争预算。L796-L799 原文节选：

```cpp
if (static_cast<int>(lane_idx) == leader) {
    slot_base = atomicAdd(query_recall_counter + q_idx, miss_count);
    valid_count = slot_base < req_pf_max ? min(req_pf_max - slot_base, miss_count) : 0;
}
```

计数器的原子累加结果可能超过 64，但真正写缓冲和搬运的数量由 `valid_count` 截断，不会因此扩大缓冲区。

获得名额后，L811-L812 写入临时位置记录和临时映射：

```cpp
prefetch_host_loc[q_idx * req_pf_max + slot_x] = host_loc;
host_token_to_device[host_loc] = static_cast<int>(device_pool_size_m1 + 1 + slot_x);
```

此处映射值不是一个正式缓存槽位，而是“正式池范围之后的临时编码”。

### 5.8 预取 KV 先进入单独的 GPU 缓冲区

D-KERNEL L823-L826 原文：

```cpp
transfer_item_warp_v2<kTransferItemBytes>(
    &host_pool_buf[(size_t)host_loc * kMLAHeadDim],
    &prefetch_kv_buf[((size_t)q_idx * req_pf_max + (size_t)slot_x) * kMLAHeadDim]
);
```

POOL L940-L955 的实际分配为：

```text
decode_prefetch_max      = 64
decode_prefetch_host_loc : [max_num_reqs, 64]，int32
decode_prefetch_kv_buf   : [max_num_reqs, 64, kv_lora_rank + qk_rope_head_dim]，BF16
```

该临时 KV 缓冲没有 layer 维度，在层间按执行顺序复用；正式 KV 缓存、映射和优先级则按层维护。

### 5.9 精确 Top-K 之后：落入正式池与补搬

ATT L780-L815 读取最终选中 Token 的映射。融合预取分支用以下条件判断哪些项还需要正式槽位，原文节选：

```python
need_device = device_page_table_1 >= kv_cache.shape[0]
recall_host_indices = torch.where(
    need_device, page_table_1, -1
)   # [q_len, topk]
recall_size = torch.sum(need_device).to(torch.int32)
```

它同时涵盖两类 Token：真正缺失的 `I32_MAX`，以及位于临时预取缓冲、尚未落入正式池的 Token。

POOL L1379-L1430 的处理顺序：

1. 计算需要分配多少个正式槽位。
2. 保护本轮选中的已有正式缓存槽位，空间不足时淘汰其他槽位。
3. 分配正式槽位并更新优先级。
4. 调用 `recall_update_with_prefetch()`。
5. 清理未转成正式映射的临时预取映射。

RECALL L448-L475 先检查临时编码和 `prefetch_host_loc` 是否匹配当前 Host Token，再决定从哪里读。逻辑伪代码：

```python
for token in exact_topk:
    if token_is_already_in_regular_gpu_cache(token):
        continue
    dst = allocate_regular_gpu_slot()
    if temporary_prefetch_entry_matches(token):
        copy(prefetch_gpu_buffer[token], regular_gpu_cache[dst])
    else:
        copy(host_kv[token], regular_gpu_cache[dst])
    update_mapping_and_attention_indices(token, dst)
```

因此预取命中仍有一次 GPU 内部拷贝，但省掉了最终 Recall 阶段的那次 Host 读取。没有被精确 Top-K 选中的临时候选不会因此全部占据正式缓存池。

清理时只将仍等于临时编码的映射置回 `I32_MAX`，不会删除已经转成正式槽位的映射。RECALL L549-L551 原文：

```python
expected_temp_idx = prefetch_device_start + offsets
should_clear = mask & (host_idx > 0) & (current_device_idx == expected_temp_idx)
tl.store(host_token_to_device_ptr + host_idx, I32_MAX, mask=should_clear)
```

### 5.10 最后才执行 Attention

ATT L807 调用 `recall_miss_tokens_decode_cuda_graph_with_prefetch()`，L858 才调用 `_forward_fa3()`。

这是一条有设备执行依赖的链，不要求 CPU 在每一步调用 `synchronize()`，但最终 Attention 不能先消费尚未准备好的 KV。

```text
TMA：      加载 K0      加载 K1      加载 K2
GEMM：                 算分 K0      算分 K1      算分 K2
Prefetch：                          预取候选0    预取候选1
                                  ...
融合 kernel 结束 -> 精确 Top-K -> 临时缓冲落池/缺失补搬 -> FA3
```

以上仅为依赖示意，不代表各阶段耗时相同，也不表示 Host 搬运一定能完全隐藏。

## 6. Prefill：不同 Query 块之间的预取

### 6.1 与 Decode 的根本区别

Prefill 一次有很多 Query。当前 Q 块的所有分数算完后，可以为这些 Query 搬 KV，同时让 GEMM 继续算后续 Q 块。

不是“用 Q0 的分数预测 Q1 要哪些 KV”，而是“用 Q0 自己的分数给 Q0 选候选，借 Q1 算分的时间搬运”。

```text
GEMM：      计算 Q块0 的各 K块       计算 Q块1 的各 K块
Prefetch：  对 Q块0 累积直方图  ->   选择/搬 Q块0 的候选 KV
                                  对 Q块1 累积直方图 -> ...
```

实际是 Warp 组和多槽流水的交错推进，不是三个互不影响、无限缓冲的独立线程。

### 6.2 Python 侧准备连续 K 和元数据

IDX L484-L510 为各请求获取连续的 FP8 Indexer K 和 scale，拼接后构造有效区间：

```text
ks[q]：Query 对应请求的 K 起始位置
ke[q]：当前 Query 可见 K 的结束位置，包含因果边界
extend_seq_to_req[q]：这个 Query 属于哪个请求
extend_seq_lens[req]：当前请求本轮 extend 的长度
```

这些是 kernel 区分请求、过滤无效位置和限制候选扫描范围的依据。不能把 batch 中所有 Query 都看成可访问同一段 K。

IDX L520-L532 清空分配记录、按优先级从低到高排列正式池槽位、清零本次预取计数器。

IDX L534-L548 原文：

```python
logits = deep_gemm.fp8_mqa_logits_fuse_prefetch(
    q_fp8,
    kv_fp8,
    weights,
    ks,
    ke,
    page_table_1, extend_seq_lens, extend_seq_to_req,
    device_pool_buf, host_pool_buf,
    device_pool_loc_alloc_buf,
    device_pool_priority, device_pool_loc_small_priority,
    device_token_to_host, host_token_to_device,
    recall_counter,
    self.extend_logits_offsets,
    clean_logits=False,
)
```

与 decode 不同，这里传入正式池的优先级排序和双向映射，允许预取 kernel 直接选择正式槽位并更新映射；没有使用 decode 的 `[Query, 64, KV维度]` 临时 KV 缓冲。

### 6.3 线程配置和 Q/K 切分

P-JIT L434-L445 原文：

```cpp
constexpr int block_qh = 128;
constexpr int block_kv = 128;
constexpr int num_specialized_threads = 128;
constexpr int num_math_threads = 256;
// constexpr int num_topk_threads = 128;
constexpr int num_topk_threads = 256;
constexpr int smem_size_for_topk_input = 16 * 1024; // 16KB, to make [2, 2K] i32 buffer

constexpr int num_q_stages = 3, num_kv_stages = 3;
const int block_q = block_qh / num_heads;
DG_HOST_ASSERT(block_qh % num_heads == 0);
DG_HOST_ASSERT(seq_len_alignment % block_q == 0);
```

因此一个 SM90 线程块的配置是 640 个线程，即 20 个 Warp：TMA 组 4 个，Math 组 8 个，分桶/预取组 8 个。TMA 分支实际仅保留第一个 Warp 执行加载逻辑，不能把配置的 128 线程都算成持续工作的搬运线程。

Q 块大小为 `128 / Indexer头数`，不是固定 128 个 Query。K 块大小为 128。

P-KERNEL L1015-L1018 持久化调度 Q 块：从 `blockIdx.x` 开始，后续按 `gridDim.x` 跨步。每个 Q 块内部遍历其对应的 K 块。

### 6.4 TMA 和 GEMM 产出分数

P-KERNEL L1060-L1108 的 TMA 分支外层切换 Q，内层加载 K 和 scale；L1109-L1235 的 Math 分支执行 WGMMA、缩放、ReLU 和 head 加权归约。

每个分数块写两份：共享内存 `smem_logits` 供融合预取使用，全局 `logits` 供后续精确 Top-K 使用。通过 `empty_logits_barriers` 和 `full_logits_barriers` 管理可写/可读状态。

### 6.5 先对每个 Query 建 256 桶直方图

Prefill 不用 decode 那种仅比较预测阈值的筛选，而是执行一轮粗分桶选择。

P-KERNEL L1378-L1385 原文节选：

```cpp
const auto bin = convert_to_uint8(
    smem_logits[topk_stage_idx][local_q_idx * BLOCK_KV + local_kv_idx] -
    extend_logits_offsets_val[req_id]
);
const uint32_t global_kv_idx = kv_start + kv_block_idx * BLOCK_KV + local_kv_idx;
if (cu_seq_len_k_start[q_idx] <= global_kv_idx and global_kv_idx < cu_seq_len_k_end[q_idx]) {
    ::atomicAdd(&s_histogram[local_q_idx][bin], 1);
}
```

`s_histogram[local_q_idx]` 说明每个 Query 单独统计，不是把 Q 块中所有 Query 的分数混在一起。因果边界以有效区间检查实现。

桶映射函数 P-KERNEL L319-L324 原文：

```cpp
__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}
```

实际步骤是：先转 FP16，再把浮点位表示变成保序整数 key，最后取高 8 位。不是直接截取原始 FP32 的最高字节，也不是对分数做等宽十进制分桶。正常有限数值下，高桶对应更高分数。

### 6.6 分数平移：设计与实际代码要分开

减去同一 Query 使用的 offset 不改变原始分数的相对排序，但改变它们落入浮点粗桶的位置。目的是让 Top-K 边界附近的临界桶更小，从而提前识别更多 Top-K 成员。

论文第 5.2 节描述用前一个 prefill chunk 末尾 Token 的第 K 大分数 EMA 来预测 offset。但 IDX 在这个提交中的实际更新不同，L550-L556 原文：

```python
offset = 0
for i in range(forward_batch.batch_size):
    s = max(offset + extend_seq_lens_cpu[i]-self.extend_logits_window, offset)
    self.extend_logits_offsets[i].fill_(torch.mean(
        logits[s: offset + extend_seq_lens_cpu[i]]
    ))
    offset += extend_seq_lens_cpu[i]
```

这里 `extend_logits_window = 4`，取末尾最多 4 个 Query 对应 logits 切片的整体均值；没有在此调用中提取第 K 大分数，也没有做 EMA。首个 chunk 在 IDX L502-L503 将 offset 置 0。

因此应分别记住：论文说明预测 Top-K 阈值平移的设计，本文代码实际使用 logits 均值更新 offset。不要用论文的预测公式替代这段真实代码，也不能假定该均值只覆盖有效、未填充的 logits 区域。

### 6.7 找临界桶，先取高分完整桶

当前 Q 块的 K 块扫描完成后，P-KERNEL L1278 的 `run_cumsum()` 为每个桶计算后缀累计数量。设：

```text
H[b] = 第 b 桶的 Token 数
C[b] = H[b] + H[b+1] + ... + H[255]
```

L1418-L1421 原文：

```cpp
int topk = TOPK;
if (tx < RADIX && s_histogram[i][tx] > topk && s_histogram[i][tx + 1] <= topk) {
    s_threshold_bin_id = tx;
}
```

找到的临界桶 `b` 满足 `C[b] > K`，但 `C[b+1] <= K`。只取 `bin > b` 的 Token，暂时不精细处理临界桶本身。

示例：K=2048，临界桶以上有 1900 个 Token，临界桶内有 300 个。先搬高分桶中的候选；剩下 148 个需要精确选择确定。若完整高分桶恰好有 2048 个，也可能一次覆盖全部 K，并非永远严格少于 K。

在直方图正确、桶映射保序且输入有效的前提下，这种选择得到 Top-K 子集，而不是 decode 那种可能过选的阈值预测集合。实际搬运还会受到缓存命中、扫描范围和预取预算限制。

### 6.8 扫描候选范围并开始 Recall

P-KERNEL L1424-L1446 中的关键原文：

```cpp
const auto threshold_bin = s_threshold_bin_id;
// Prefetch for tokens larger than threshold_bin
if (kv_len < cur_extend_len) { // e.g., token 1 for prefill with 3 tokens
    continue;
}
const int total_kv_items = kv_len - cur_extend_len;
```

后续候选判断：

```cpp
if (bin > threshold_bin) {
    const auto page_table_ptr = page_table_1 + req_id * page_table_1_stride + kv_idx;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(page_table_ptr));
    kv_token_host_loc = __ldg(page_table_ptr);
    is_candidate = true;
}
```

这里并不扫描所有有效 K，而是按源码将扫描长度限制为 `kv_len - cur_extend_len`，不足则跳过。它体现了避免预取当前 extend 尾部的策略，但对因果边界处的具体覆盖范围，应以这条计算为准，不能简单说每个 Query 都完整扫描了所有历史前缀。

### 6.9 原子去重，直接写正式缓存池

同一历史 Token 可能被多个 Query 选中，不能为每个 Query 各搬一份。P-KERNEL L1295-L1299 原文：

```cpp
const auto& recall_func = [&](const bool is_candidate, const int kv_token_host_loc,
                              const uint32_t& max_recall_tasks) {
    bool is_miss = false;
    if (is_candidate)
        is_miss = atomicCAS(host_token_to_device + kv_token_host_loc, INT32_MAX, -1) == INT32_MAX;
```

只有把 `I32_MAX` 成功换成临时占位 `-1` 的线程获得处理资格，其他看到已有映射或占位的线程跳过。这是 prefill 融合 kernel 中的去重/竞争处理。

接下来用 Warp ballot 汇总缺失项，通过 `atomicAdd(recall_counter, miss_count)` 申请名额。预算内的项从 `device_pool_loc_small_priority` 选正式池槽位，撤销旧 Token 的反向映射，再建立新映射；超出预算的占位恢复为 `I32_MAX`。

P-KERNEL L1345-L1348 的实际搬运目标为正式池：

```cpp
transfer_item_warp_v2<kTransferItemBytes>(
    &host_pool_buf[(size_t)copy_host_loc * kMLAHeadDim],
    &device_pool_buf[(size_t)copy_device_loc * kMLAHeadDim]
);
```

与 decode 对比：decode 是 `Host -> 临时GPU缓冲 -> 正式池`；prefill 是 `Host -> 正式池`。映射在 kernel 内可能早于数据写完而更新，后续消费者必须受整个 kernel 的完成依赖约束，不能仅看见映射就跨流提前读取。

### 6.10 Prefill 的预算不是每个 Query 64 个

P-KERNEL L1238-L1239 原文：

```cpp
constexpr auto RADIX = 256, TOPK = 2048;
constexpr auto kMaxPrefetchTasks = 8192;
```

L1355-L1356 原文：

```cpp
const uint32_t left_pool_size = device_pool_size_m1 - seq_len;
const uint32_t max_recall_tasks = left_pool_size > kMaxPrefetchTasks ? kMaxPrefetchTasks : left_pool_size;
```

在 `device_pool_size_m1 >= seq_len` 的适用条件下，实际预算可读成：

```text
本次融合 kernel 的预取预算 = min(8192, 正式池容量 - 本次总 Query 数)
```

`recall_counter` 是本次调用共享的全局计数器，不是每个 Query 或每个 Q 块都各有 8192 个。

API L211 从分配记录缓冲长度取得 `device_pool_size_m1`，L245 校验正式池物理行数等于它加 1；这里减去的是可用容量 C，而不是把 padding 行计入容量。

例如配置容量 4096、本次总 Query 数 2048，按该公式预取预算最多为 2048。这个数不能解释成 2048 个 Query 各自都能再预取 2048 个。

注意这里是无符号减法，不是安全饱和减法；不满足容量前提时不能拿上述公式推导出正确预算。该实现也把粗分桶的 K 写死为 2048，不应直接当作任意 K 的通用 kernel。

### 6.11 Kernel 返回后仍做精确 Top-K 和补搬

IDX L558-L571 先将预取分配记录同步到 allocator 并更新优先级，然后执行：

```python
assert logits.shape[0] == len(seq_lens_expanded)
topk_result = metadata.topk_transform(logits, self.index_topk)
```

ATT L638-L646 原文：

```python
recall_host_indices = torch.where(
    device_page_table_1 == I32_MAX, page_table_1, -1
)   # [q_len, topk]
forward_batch.token_to_kv_pool.recall_miss_tokens_extend_cuda_graph(
    recall_host_indices, device_page_table_1, layer.layer_id
)
device_page_table_1 = forward_batch.token_to_kv_pool.host_token_to_device[
    layer.layer_id, page_table_1
]
```

POOL L1309-L1311 先将缺失项标记到 Host Token 维度的布尔表中：

```python
self.host_need_recall.fill_(0)
mark_host_need_recall(self.host_need_recall, recall_host_indices)
recall_size = self.host_need_recall[:-1].sum().to(torch.int32)
```

这样可对多个 Query 重复选择的缺失 Host Token 去重。随后保护选中项、准备槽位，由 `recall_update_extend()` 从 Host 搬剩余缺失项。

最后 ATT L655-L662 调用 `_forward_flashmla_prefill()`。即使某个早期 Q 块的 KV 已搬完，这条调用链也没有立即为它启动最终 Attention；它仍等待本次融合算分、精确选择和补搬阶段。

## 7. KV 是怎样从 Host 搬到 GPU 的

POOL 构造 Host 池时使用 `pin_memory=True` 和 `layout="layer_first"`，融合 kernel 直接接收 Host 数据指针。本文讨论的路径不是每个 Token 在 CPU 上调用一次 `cudaMemcpy`，也不是通过 TMA 描述符完成这次 KV 传输。

`transfer_item_warp_v2()` 由一个 Warp 协作搬一个 Token：按 16 字节切片，各 lane 先读进寄存器，再写到 GPU 目标地址。

对于 `512 + 64` 维 BF16 KV，一个 Token 为 1152 字节，COPY L335-L360 存在特化。以下为其中的两段原文节选：

```cpp
for (int i = 0; i < kChunksPerLane; i++) {
    const int idx = lane_idx + i * 32;
    if (idx < kChunks) {
        regs[i] = __ldg(src + idx);
    }
}
```

```cpp
for (int i = 0; i < kChunksPerLane; i++) {
    const int idx = lane_idx + i * 32;
    if (idx < kChunks) {
        const auto& val = regs[i];
        asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(dst + idx),
                        "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                     : "memory");
    }
}
```

1152 字节对应 72 个 16 字节片段，由 32 个 lane 分担。搬运本身没有矩阵乘加，不使用 Tensor Core；与它并行的 Indexer GEMM 才使用 WGMMA。

Prefill 还在搬当前 Token 时调用 `prefetch_item_warp()`，尝试将下一个 Host Token 的数据预取到 L2。不要混淆两种“预取”：

- 算法级预取：提前将候选 KV 从 Host 搬进 GPU 缓存。
- 指令级预取：`prefetch.global.L2` 提示提前加载页表或后续源数据到 L2，不代表已经写入最终 KV 槽位。

## 8. 池大小、2048 和各层的关系

### 8.1 三种容量不能混用

| 名称 | Decode CUDA 基线 | Prefill CUDA 基线 |
| --- | --- | --- |
| 最终 Attention 的选择数量 K | 由 `self.index_topk` 决定，本讨论按 2048 | 最终由 `self.index_topk` 决定；融合预取内部固定 TOPK=2048 |
| 预取缓冲/预算 | 每 Query 64 个临时 KV 槽位 | 直接用正式池，共享预算最多 8192，另受容量减 Query 数限制 |
| 正式 GPU KV 池容量 C | `NSA_DEV_CACHE_SIZE`，按层 | 同一个按层正式池 |

`model_runner.py` L203 的环境变量默认值是 4096；L1757 起将其作为 `MLATokenToKVPool` 的 size。CUDA 基准脚本可覆盖为 200000 或 260000，详见 `docs/reference/ECHO-CUDA-README.md`。

容量以每层的 Token 槽位计，分配器及双向映射也按层维护。基线还为 padding 保留槽位，物理 tensor 的行数可能是 `C+1`，不要把这个 padding 槽位计成可用 Token 容量。

### 8.2 4096 不是固定双缓冲

单 Query、K=2048、C=4096 时，Attention 只读取索引选中的约 2048 个有效 Token；其他槽位可以保留本轮未选中但后续可能复用的 KV。

但它不是两块固定的 2048 轮流使用，也不是算法要求 `C=2K`。多个请求或 prefill 的多个 Query 共用同一层的池；必须考虑这些 Query 选中 KV 的并集，而不是只考虑单个 Query 的 K。

缓存可以通过淘汰不再受保护的项腾空间，但若一次 Attention 调用必须同时保留的不同 KV 已超过容量，单靠淘汰无法解决。不能认为 4096 对任意 batch 或 prefill 长度都足够。

### 8.3 每层字节数

仅计算 MLA Attention KV 数据，忽略 padding 和管理元数据：

```text
每层正式池字节数 = C * (kv_lora_rank + qk_rope_head_dim) * element_size
4096 * (512 + 64) * 2 = 4,718,592 字节 = 4.5 MiB
每 Query 的 decode 临时预取缓冲 = 64 * 576 * 2 = 72 KiB
```

前者按层分配，后者按最大请求数预分配并跨层复用。全量 GPU Indexer K、量化 scale、logits、映射和优先级都另占空间。

当前模型各层使用相同的 KV 维度和 dtype，所以相同槽位数量对应相同字节数；各层实际选中 Token、命中率和 Recall 数量仍可能不同。

## 9. 正确性、同步和实现边界

### 9.1 Lossless 不等于没有白搬

保持最终精确 Top-K 和缺失补搬，才是预取不改变 Attention 选择的基础。Decode 预测候选可以包含最终未选中的 Token，Prefill 粗分桶也不保证覆盖所有 Top-K。

因此应分别统计：超过阈值的候选数、实际搬运数、最终选中的预取命中数，以及最终补搬数。四者不是同一个计数。

### 9.2 Graph 不会自动创造计算重叠

共享内存 barrier 保证融合 kernel 内生产者/消费者的顺序；同流 kernel 顺序或对应 Graph 依赖保证后续 Recall 和 Attention 不提前消费数据。真正的算分/搬运重叠来自 Warp 分工和软件流水，不来自把串行调用捕获进 Graph 本身。

ATT 的 Graph replay 元数据入口 L482 明确只支持 decode/idle。虽然 prefill 使用了带 `cuda_graph` 名字的设备侧 Recall 接口，这不代表本提交已经实现完整 prefill Graph capture/replay。

### 9.3 不能从静态阅读直接宣称已验证的事项

以下是复用或修改这份代码前需要单独验证的点，不是本文已修复的问题：

- Prefill 的 `s_histogram` 在 P-KERNEL L1272-L1275 初始化，而外层 Q 块循环在 L1353；循环末尾未看到为下一 Q 块重新清零。单 CTA 若持续处理多个 Q 块，需验证直方图复用是否正确，不能无条件套用理想算法的子集保证。
- Prefill 使用 `C - seq_len` 的无符号预算计算，需保证容量前提，并验证有效 Token 数少于或等于 K 时的临界桶行为。
- Prefill 的 offset 均值更新不等于论文的 Top-K EMA；`clean_logits=False` 时还需检查均值切片内无效位置对预测的影响。
- Decode 的预测状态以 batch 行切片使用，多个请求调度、换位或重置时的状态归属应独立检查。
- 两种 kernel 都需要测试重复 Token、多 Query 命中、预算耗尽、跨层复用，以及反复 Graph replay 下的临时映射清理。

这些限制是区分“论文算法意图”和“某个代码版本实际行为”的必要部分。精确 Top-K 和补搬能处理预取不足，但不自动证明所有映射竞争、越界或生命周期问题都不存在。

## 10. 与当前 NPU 实现的区别

当前 `sglang-ascend/python/sglang/srt/hardware_backend/npu/echo_memory_pool.py` 明确记录 `no fused prefetch`。当前 NPU 路径具有按层有界缓存和精确 Recall，但不能因此声称已有本文的 CUDA 三段融合预取。

`scripts/smoke-echo-deepseek-npu.py` 和 `scripts/bench-echo-sharegpt-npu.py` 中的 `--echo-device-cache-size 4096` 是当前 NPU 测试设置，不是论文所有 CUDA 实验的统一参数。

若迁移预取，除了算法外，还必须处理临时缓冲、去重、层间复用、映射发布、最终补搬以及 NPU Graph 地址和执行顺序约束。本文没有进行迁移或修改运行代码。

## 11. 最后对照

| 问题 | Decode | Prefill |
| --- | --- | --- |
| 什么时候开始 KV 搬运？ | 当前 Query 的部分 K 块分数产出后即可 | 当前 Q 块的相关 K 块扫描完成后 |
| 候选怎么选？ | 分数超过 EMA 预测阈值 | 一轮 256 桶统计，取临界桶以上的完整桶 |
| 与谁重叠？ | 同一 Query 的后续 K 块算分 | 后续 Q 块的算分 |
| 会不会过选？ | 会，预测集合不保证为精确 Top-K 子集 | 理想单轮粗分桶选择为 Top-K 子集；代码复用限制见第 9 节 |
| 先搬到哪里？ | 单独的 GPU 临时预取缓冲 | 正式 GPU KV 池 |
| 最终还做什么？ | 精确 Top-K、临时 KV 落池、缺失补搬 | 精确 Top-K、跨 Query 缺失去重、补搬 |
| 最终 Attention 何时执行？ | KV 就绪后调用 FA3 | KV 就绪后调用 FlashMLA prefill |

一句话总结：**Decode 通过阈值预测突破“必须先算完当前 Query”的限制；prefill 利用多个 Q 块自然形成的算分窗口提前搬运。两者都保留最终精确选择和必要的 KV 补搬。**

## 12. 参考来源与核验方式

- 论文：*ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs*，第 5.1-5.3 节、Figure 10-11。
- 论文地址：`https://www.usenix.org/system/files/osdi26-liu-guangda.pdf`。
- 实现：本仓库历史提交 `bc1b75c1000010d0ac6f032ebaac283255c050b1`，第 3 节列出的文件。
- 配置说明：`docs/reference/ECHO-CUDA-README.md`。

带行号查看任一历史文件的方法，例如：

```bash
git show bc1b75c:sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py | nl -ba | sed -n '350,430p'
git show bc1b75c:DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh | nl -ba | sed -n '1238,1453p'
```

本文仅新增说明文档，未修改 CUDA 或 NPU 代码，也未运行新的设备测试。
