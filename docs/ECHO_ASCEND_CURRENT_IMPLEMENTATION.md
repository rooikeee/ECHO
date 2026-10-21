# Ascend ECHO 当前实现与交接

更新日期：2026-10-21。本文件描述当前源码，不代表性能或完整模型精度验收通过。
测试入口、资源准备和失败判定见 [跨机器复现](ECHO_ASCEND_REPRODUCTION.md)。

## 1. 实现范围

当前版本是 **exact KV offload + recall，无预测/融合 prefetch**。
目标为 DeepSeek NSA/DSA 模型、Ascend attention 后端、BF16/FP16 MLA KV。
不依赖安装 DeepGEMM-Ascend；该源码只保留为后续迁移参考。

每个 attention DP rank 独立持有：

| 数据 | 位置 | 容量口径 |
| --- | --- | --- |
| 完整 MLA KV（latent K/value + RoPE K） | 注册映射的主机内存 | 每层逻辑 token 池 |
| 热 MLA KV | NPU | 每层 `echo_device_cache_size` |
| Indexer K | NPU，当前为未压缩 BF16/FP16 | 每层完整逻辑池 |
| 双向映射、有效标记、epoch、LRU、空闲栈和错误状态 | NPU | 每层独立 |

逻辑池的 token 数是该 rank 全部活跃请求共享的总容量，不是请求数，
也不是单个请求的上下文长度。Attention DP8 不把一个请求的 KV 分成八份。
MoE TP8 分片模型权重，但不会再把每个 DP rank 的池容量除以八。

## 2. 调用链与代码位置

路径均相对 `sglang-ascend/python/sglang/srt/`：

| 代码 | 职责 |
| --- | --- |
| `server_args.py` | `--enable-echo`、容量参数、不支持组合的检查 |
| `model_executor/model_runner_kv_cache_mixin.py` | 选择池/分配器、检查 NSA 和 Top-K、传递 TP/DP rank |
| `hardware_backend/npu/echo_memory_pool.py` | `EchoNPUMLATokenToKVPool`，按层持有缓存，Indexer K 分配 |
| `hardware_backend/npu/echo_recall.py` | 主机注册、Triton 写入/备份/规划/回搬/释放、attention 地址转换 |
| `hardware_backend/npu/attention/ascend_backend.py` | Top-K 到逻辑 ID，再到热缓存物理槽位的 sparse attention 接入 |
| `hardware_backend/npu/allocator_npu.py` | 页释放同步清除 ECHO 状态 |
| `hardware_backend/npu/graph_runner/npu_graph_runner.py` | capture dummy 元数据、捕获后恢复、replay 后错误检查 |
| `model_executor/model_runner.py` | eager forward 后集中检查 ECHO 错误 |
| `observability/scheduler_metrics_mixin.py` | 标记 DP global prefill，区分混合步骤与纯 decode eager |

原 CUDA 实现保存在提交 `bc1b75c1000010d0ac6f032ebaac283255c050b1`，
不是当前工作目录中的旧 `sglang/`。上游导入来源见 `upstreams.lock.json`。

## 3. 写入、回搬和释放

1. 新 KV 先分配/复用热槽并写入设备，然后同一 stream 备份到主机，发布 `written`。
   这是 device-first，不是 host-first；没有独立异步备份 stream。
2. 不超过 8 个新 KV 的写入使用小批量快路径，优先从设备空闲栈取槽，必要时 LRU。
   较大写入按热缓存容量分块规划，备份后才允许下一块复用槽位。
3. Top-K 经请求页表、有效长度和因果掩码转换为逻辑 ID。ID 0 为 padding，-1 为未选中。
4. epoch 标记本轮集合，去重并保护命中项。全命中时图内设备分支跳过逻辑池扫描。
   miss 时扫描逻辑池、紧凑收集 ID，先用空闲栈，容量不足才做受保护的 LRU 淘汰。
5. 固定数量 worker 按设备 miss count 从映射主机内存读取 KV，恢复到热缓存。
   输出物理槽位交给 `npu_sparse_flash_attention`，没有近似删减最终 Top-K。
6. 为保证一轮所需 KV 能同时驻留，按最坏 union 大小拆分 query；prefill 全历史
   已知能放入热缓存时可批量执行。设备热缓存至少要容纳一个 query 的 Top-K。
7. 请求释放展开整个逻辑页，逐层清除有效性和映射并归还热槽；重复释放去重。

## 4. NPU Graph 契约

- 持久池、映射、空闲栈、规划缓冲和注册的主机地址保持稳定，生命周期覆盖所有图。
- 每次规划重置本轮错误/miss count 并递增 epoch；持久映射、LRU 和空闲栈跨 replay 保留。
- 命中/miss、计数和淘汰选择在设备 kernel 中决定，不依赖设备结果触发 Python 分支。
- 固定形状临时 tensor 在 capture 时由图内存池持有；不能据此使用设备数据动态分配。
- 图捕获前清零 dummy 页表，捕获后清除热映射并保留已备份主机数据的有效标记。
- 新 KV 备份、recall 和 attention 同 stream 有序；外部主机写入/解除注册前必须同步。
- sticky error 跨层累计，图外回读后 fail-closed，不通过关闭检查换取性能。
- 当前建立的是 decode Graph 路径。组件中 prefill adapter 的 capture/replay 测试
  不代表完整模型 prefill Graph 支持。

## 5. 已知限制与预算

必须关闭 radix cache、overlap schedule。不支持 CP、PP、PD 分离、HiCache、HiSparse、
LMCache、投机解码、torch compile、memory saver、MLA preprocessing、NSA prefill CP。
上游 Ascend 有 PD 实现，但 ECHO 的主机 KV 接收、Indexer K 传输和状态发布尚未接入。

重 miss 时的逻辑池扫描、大量淘汰时的逐项 LRU 归约、逐层主机备份及图外错误同步
仍可能较昂贵。当前没有逐请求硬件传输字节数/命中率的完整采集，不能从启动日志推算。

对本地 61 层模型，设每 rank 逻辑容量 H、热容量 C、BF16，每卡主数据预算约为：

```text
主机完整 KV = H * 61 * (512 + 64) * 2 bytes
设备热 KV   = C * 61 * (512 + 64) * 2 bytes
设备 Index K = H * 61 * 128 * 2 bytes
```

本轮目标 H=499,968、C=99,968：每卡热 KV 约 6.54 GiB，Indexer K 约 7.27 GiB；
八 rank 主机 KV 合计约 262 GiB。尚需映射、工作区、通信、图内存和模型权重。
历史权重约 45.14 GiB/卡不能作为新环境可用显存保证。

**当前服务的 profiling 仍使用保守 full-KV/token 预算，并可能将 `max-total-tokens`
静默降到较小值。新实验脚本会检查实际分配容量，不满足目标就失败。**
仅增大 `mem-fraction-static` 不是保证成功的方法；修正预算需要独立验证和记录。

## 6. 真实验证状态

- 历史单卡算子测试覆盖写入顺序、hit/miss、重复 ID、保护/淘汰、释放和实际重复 Graph replay。
- 历史微型 Qwen2 基线通过单卡 batch 1/2 decode capture/replay。
- 小型随机 NSA 模型的 TP2/DP1、TP2/DP2、TP4/DP2、TP8/DP8 对照通过；不是完整 AWQ 精度验收。
- 完整 AWQ TP8/DP1 有短请求和 40 条 ShareGPT 性能结果，严格 logprob 对照未通过。
  40 条旧配置下 ECHO 6.33 token/s，baseline 26.68 token/s，且有当时的硬件告警，
  不能作为本轮半容量 TP8/DP8 实验结果。
- 2026-10-21 半容量 TP8/DP8 实验仅完成样本准备和 CPU 脚本测试；启动被
  `npu-smi` 超时、随后其他进程占卡阻断。没有加载模型，没有任何新吞吐/延迟结果。
- 此文档交接没有重新运行 NPU 回归。详细历史见 `ECHO_RECALL_OPTIMIZATION.md`、
  `ECHO_PARALLEL_SUPPORT.md`；`build/` 历史日志不随 Git 分发。
