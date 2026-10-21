# ECHO 适配 SGLang-Ascend 修改清单

## 1. 范围与基线

本文基于 2026-10-09 对本仓库源码的检查，描述不引入 DeepGEMM 的 ECHO
Ascend 适配方案。本文是设计与修改清单，不代表相关功能已经实现或验证。

第一阶段精确 recall 的实验实现和实际验证范围见
[Recall 实现说明](ECHO_RECALL_IMPLEMENTATION.md)。下文保留完整迁移方案。

- 当前 SGLang-Ascend 基线：`f30df723bf146d4a8037010b0b11939a601df423`。
- 原始 CUDA ECHO 参考提交：`bc1b75c1000010d0ac6f032ebaac283255c050b1`。
- 当前 NPU NSA 索引路径调用 `torch_npu.npu_lightning_indexer`。
- 当前 NPU 稀疏注意力路径调用 `torch_npu.npu_sparse_flash_attention`。
- 仓库保留 CUDA 分支的 DeepGEMM 代码，但上述 NPU 路径不调用 DeepGEMM。
- 不修改 `deepgemm-ascend/`，也不要求安装 DeepGEMM。

相关背景见 [Ascend 迁移说明](ASCEND_MIGRATION.md)、
[本机 NPU 环境](NPU_ENVIRONMENT.md) 和 [开发要求](../AGENTS.md)。

## 2. 目标数据流

```text
当前 token 的 latent KV / RoPE KV
    -> 写入主机完整 KV 和设备热缓存

完整设备 index K + 当前 query
    -> 当前 NSA indexer
    -> 逻辑 Top-K
    -> ECHO 缓存命中检测
    -> 保护命中项、淘汰、分配设备槽位
    -> 回搬缺失 KV
    -> 构造热缓存对应的稀疏注意力输入
    -> 当前 NPU sparse attention
```

完整 index K 留在设备上参与索引选择；注意力 KV 可以卸载到主机，只保留有限的
设备热缓存。两者的容量、页表和位置映射必须解耦。

先实现精确回搬版，再恢复预测预取。精确回搬版是验证基础，不等同于完整 ECHO
融合预取和性能实现。

## 3. 现有文件修改清单

下表路径均相对 `sglang-ascend/python/sglang/srt/`。这是建议的修改位置，
实际实现应复用现有接口和钩子，不要求为了清单而修改每个文件。

| 文件 | 修改内容 | 阶段 |
| --- | --- | --- |
| `server_args.py` | 增加 ECHO 开关、主机 KV 容量、设备热缓存容量和预取配置；校验功能组合 | 第一阶段；预取配置在第二阶段补齐 |
| `model_executor/model_runner_kv_cache_mixin.py` | 创建 ECHO 缓存池和分配器；分别计算逻辑容量、完整 index K、设备热缓存和工作缓冲的预算 | 第一阶段 |
| `hardware_backend/npu/memory_pool_npu.py` | 接入 ECHO NPU 缓存池，保留缓存读写接口，增加逐层双向映射、优先级和固定回搬缓冲 | 第一阶段 |
| `hardware_backend/npu/allocator_npu.py` | 保留逻辑槽位分配，增加独立的逐层热缓存分配；逻辑槽位释放时清理映射和缓存状态 | 第一阶段 |
| `mem_cache/memory_pool_host.py` | 接入主机完整 KV，适配 latent/RoPE 分离布局，实现逐层、选中 token 的卸载和回搬 | 第一阶段 |
| `hardware_backend/npu/attention/ascend_backend.py` | 在 `forward_sparse` 中加入命中检测、保护、淘汰、回搬和注意力输入适配 | 第一阶段 |
| `hardware_backend/npu/graph_runner/npu_graph_runner.py` | 接入 capture 后状态恢复、replay 元数据更新、临时状态重置和传输同步 | 第一阶段 |
| `model_executor/cuda_graph_runner.py` | 仅在 NPU Graph 基类缺少所需生命周期钩子时增加通用钩子，避免改变 CUDA 行为 | 按需 |
| `layers/attention/nsa/nsa_indexer.py` | 第一阶段保留 NPU 索引计算；第二阶段接入预测阈值状态及 Ascend 融合预取实现 | 第二阶段为主 |
| `hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` | 需要在 indexer 与 attention 之间提前启动传输或调整 KV 写入时序时修改 | 按需 |

### 3.1 缓存池与容量

当前 `NPUMLATokenToKVPool` 使用分离的 latent KV、RoPE KV 和 index K 缓冲。
旧 ECHO 按连续的 `512 + 64 = 576` 维 KV 进行回搬，旧地址计算不能直接复用。

ECHO 缓存池需要明确：

- 请求逻辑 token 容量与主机 KV 容量。
- 完整设备 index K 容量。
- 每层设备热缓存容量。
- 每层 `host_to_device` / `device_to_host` 映射。
- 每层优先级和淘汰状态。
- 固定容量的 miss、槽位分配、回搬及预取工作缓冲。
- padding、无效位置和空闲槽位的哨兵语义。

内存预算不能只缩小注意力 KV 而忽略完整 index K、映射表和临时缓冲。

### 3.2 主机 KV 传输

当前主机缓存已有 `kernel_ascend` 路径，值得复用其传输基础设施。
但当前 MLA 的该路径由第 0 层触发全层传输，不能直接当作 ECHO 按层、按选中
token 的回搬接口。

需要验证或补齐：

- 固定主机缓冲的分配、注册和生命周期。
- 按层、按 token 索引的 H2D / D2H 传输。
- latent KV 和 RoPE KV 的布局转换。
- 设备端动态有效数量与固定容量传输缓冲的配合。
- NPU Graph 下传输与消费者的同步。

旧 CUDA kernel 直接读取主机 KV 指针的实现不能默认迁移。应先验证目标 NPU
支持的主机内存访问机制；不支持时采用 staging buffer、异步传输和设备 scatter。
具体方案必须经过 Graph 捕获和重放测试。

### 3.3 注意力输入适配

当前 `npu_sparse_flash_attention` 使用逻辑稀疏索引、`block_table` 和
`PA_BSND` KV 布局。ECHO 热缓存采用逐层动态槽位，因此不能把物理槽位号直接
替换为当前 `sparse_indices`。

需要选择并验证以下方案之一：

- 构造符合注意力算子约束的热缓存页表和索引映射。
- 将选中 KV 整理到固定容量的紧凑缓冲，构造对应的合法索引和元数据。

无论采用哪一种方案，都必须保持请求隔离、因果位置、有效长度和 padding 语义。
先验证现有注意力算子能否表达目标输入；如不能，再评估扩展注意力算子的必要性。

### 3.4 请求与缓存生命周期

优先在分配器的 `free` / `clear` 等接口中统一管理 ECHO 清理，避免修改所有
scheduler 和 prefix cache 调用点。

需要覆盖请求完成、abort、retract、flush、逻辑位置复用和实际前缀淘汰。
启用前缀缓存时，不能在请求结束时无条件清理仍被前缀缓存持有的 KV。
还要确保相关异步传输完成后才复用主机和设备槽位。

只有现有生命周期接口不能覆盖这些语义时，才修改 `mem_cache/common.py`、
`managers/schedule_batch.py` 或 scheduler 相关文件。

## 4. 建议新增的模块与算子

建议新增 `hardware_backend/npu/echo_ops.py`，集中封装 ECHO 的 NPU 操作。
复杂缓存状态管理可另设专用模块，避免将全部实现塞进通用缓存类。
底层扩展源码应纳入仓库，不修改 `.venv-npu` 中的安装文件。

| 功能 | 原始 ECHO 参考 | NPU 适配要求 |
| --- | --- | --- |
| 命中检测与索引转换 | 原 `nsa_backend.py` 的映射和 miss 路径 | 输出固定容量索引及设备端有效数量，处理 padding 和重复索引 |
| 保护本次使用的槽位 | `protect` | 更新逐层优先级，确保本次命中项不被淘汰 |
| 淘汰选择 | `get_free_loc` / `fast_argmin_bounded` | 实现原优先级策略，支持设备端动态选择数量 |
| 映射更新 | `free_update` / `set_update` | 一致地维护双向映射、优先级和槽位状态 |
| 图兼容分配与释放 | `CudaGraphTokenToKVPoolAllocator` | 固定缓冲、设备端计数和空闲栈，支持重复 replay |
| 新 KV 写入与卸载 | `set_mla_kv_buffer_cuda_graph` 等 | 适配 NPU 分离布局，并保证主机完整副本可用 |
| 精确回搬 | `mark_host_need_recall` / `recall_update_extend` / `recall_update` | prefill 去重、decode 回搬、映射发布和传输同步 |
| 预取消费与补漏 | `recall_update_with_prefetch` / `clear_decode_prefetch_mappings` | 校验预取身份、补回遗漏 KV、清理临时映射 |

这些是功能划分，不要求一项对应一个独立 kernel。可以复用 NPU 现有算子，
并在正确性验证后融合映射、计数和 scatter 等操作。

## 5. 不使用 DeepGEMM 的融合预取

当前 NPU indexer 调用返回 Top-K 索引，调用方没有使用旧 ECHO 所需的完整
logits、预测阈值状态或预取副作用。因此，完整融合预取不能只靠 Python 接口包装。

原始接口仅作为算法参考：

- prefill：`fp8_mqa_logits_fuse_prefetch`。
- decode：`fp8_paged_mqa_logits_fused_v2`。

Ascend 实现需要新增或扩展索引算子，在打分过程中产生预取候选，并与 KV
传输路径配合。需要迁移预测阈值及其更新语义、候选容量限制、临时映射和
预取缓冲管理，但不要求保留旧函数名、FP8 布局或 DeepGEMM API。

如果先做独立预测与预取路径，应明确它不是原始融合实现，并单独评估能否与
索引计算重叠。Top-K 完成后才回搬属于精确回搬，不应称为融合预测预取。

## 6. NPU Graph 要求

NPU Graph 兼容是实现要求，eager 仅用于调试。

- 预分配图引用的缓存、映射表、计数器和工作缓冲，保持地址稳定。
- 动态请求长度、Top-K、命中与 miss 数量通过元数据、有效数量和掩码表达。
- 图内不依赖 `.item()`、设备结果驱动的 Python 分支或动态工作缓冲分配。
- 定义 capture 后的状态恢复，以及每次 replay 的临时状态重置。
- 跨 replay 保留正确的热缓存状态和预测状态，不要将所有持久状态每次清零。
- 验证主机缓冲生命周期和传输 Stream/Event 顺序。
- 明确 prefill 的执行方式；不能用 decode Graph 验证替代 prefill 验证。

## 7. 实施顺序

### 第一阶段：精确回搬版

1. 增加配置、内存预算和 ECHO 缓存池。
2. 实现双向映射、热缓存分配、保护和淘汰。
3. 实现主机 KV 写入与按层精确回搬。
4. 在 `forward_sparse` 接入缓存管理和注意力输入转换。
5. 完成请求生命周期和 NPU Graph 适配。

初期限制未验证的前缀缓存、CP、投机解码、HiCache、HiSparse 等组合，明确报错，
不要静默启用不兼容路径。当前 HiSparse 有相关缓存管理代码可供参考，但不能
视为已经接入 NPU ECHO。

### 第二阶段：预测与融合预取

1. 实现 decode 预测状态、候选生成和预取。
2. 接入预取消费、精确补漏和临时映射清理。
3. 实现 prefill 预测与融合预取。
4. 验证融合索引计算与传输的重叠和输出一致性。

### 第三阶段：性能与功能扩展

1. 测量长上下文显存占用、传输量、吞吐和延迟。
2. 扩展 batch、请求长度、热缓存容量和多设备验证。
3. 按需求逐项恢复前缀缓存、CP、投机解码等组合。

## 8. 验收清单

- 与当前无卸载 NPU 基线比较输出及注意力结果，设定明确的数值容差。
- 覆盖全命中、全 miss、混合命中、重复 Top-K、padding 和缓存容量边界。
- 覆盖连续淘汰、回搬、请求隔离和槽位复用。
- 覆盖请求完成、abort、retract、flush 和传输未完成时的释放。
- 验证 Graph 捕获、实际 replay、重复 replay，以及 batch 和长度变化。
- 验证预取正确、错误、容量耗尽和遗漏时精确回搬仍保证正确性。
- 验证模型级长上下文、多层和目标 TP 配置。
- 性能报告分别列出精确回搬版和融合预取版，不混用结果。

使用 `scripts/npu-run.sh` 运行项目环境。现有微型 Qwen2 Graph smoke test 只验证
基础环境，不证明 ECHO、DeepSeek NSA、prefill 或多设备支持。

## 9. 不在本次默认修改范围内

- `deepgemm-ascend/` 及 DeepGEMM 安装依赖。
- 普通 GEMM、MoE、RoPE、Norm 和模型权重加载。
- CUDA `layers/attention/nsa_backend.py` 的 ECHO 接入。
- 与迁移无关的 CUDA、HIP、CPU 后端重构。

上述模块只有出现明确的接口或正确性阻塞时才扩大修改范围。
