# Ascend ECHO 精确 Recall 实现

第一阶段当前设备优先版本的逐步算法、缓存示例、KV slot 的 NPU Graph 执行边界、
代码调用关系和最新完整 40 条测试结果，见
[第一阶段：Recall 实现与测试进展](ECHO_RECALL_DESIGN.md)。

本阶段先实现按需 recall 及必要的 KV 管理，尚未实现预取、融合预取或传输/计算
重叠。decode forward 内的热缓存 slot 分配、命中检测、LRU 淘汰和映射更新由
NPU kernel 在 Graph 内执行；请求逻辑页管理、释放触发及错误检查仍在图外，
prefill 当前使用 eager。不能将本阶段称为完整 ECHO 性能复现。

优化后两边各 40 条的 fresh 测量已完成，baseline/ECHO 输出吞吐分别为
26.68/6.33 token/s，没有端到端收益。结果、批量 prefill 的剩余性能问题和硬件告警
限制见 [优化记录](ECHO_RECALL_OPTIMIZATION.md)。本文后面的旧版本数据不替代该结果。

## 范围

本实现是第一阶段的实验性精确回搬路径，不依赖 DeepGEMM，不包含预测或融合预取。
当前面向 DeepSeek NSA、Ascend attention 后端和 BF16/FP16 KV。

完整 index K 仍留在 NPU，使用当前 `npu_lightning_indexer` 选择 Top-K。
每层完整的 latent KV / RoPE KV 保存在注册的主机内存中，设备仅保存有限热缓存。

## 数据与执行流程

1. 使用页对齐的 `mmap` 分配主机 KV，并通过 `aclrtHostRegister` 获取 NPU 可访问的指针。
2. 新 token 的 KV 先分配或复用设备热缓存槽位，写入 latent KV / RoPE KV；
   再从热缓存备份到主机。覆盖已有驻留逻辑位置时原位更新，不清除映射。
3. 根据原始 Top-K、逻辑页表和 query 长度，将选中位置转换为主机 KV ID，保留因果过滤。
4. 用设备端 epoch 标签去重并识别缺失项，全命中跳过完整规划。
5. 保护命中槽位，优先使用空槽，只有真实淘汰才通过分块 LRU 最小值选择槽位。
6. 分配槽位、维护双向映射，从映射主机指针精确复制缺失 KV 到设备。
7. 使用更新后的设备槽位调用现有 `npu_sparse_flash_attention`。

新 token 写入后已经驻留热缓存，被 Top-K 选中时直接命中，不经过主机回搬。
主机备份不意味着立即淘汰设备副本；只有后续容量不足时才复用槽位。
超过热缓存容量的写入按固定大小分块，每块完成设备写入和主机备份后才处理下一块。
每次写入的正逻辑 ID 必须唯一；最后一块可驻留，之前被淘汰的 KV 仍有完整主机副本。
当前设备写入、主机备份和消费者在同一 stream 上排序，尚未实现异步传输与计算重叠。

## 注意力适配

物理槽位不能直接替换原后端的逻辑稀疏索引。因此，本实现先保留原 query 的因果
筛选，再将每个 query 作为独立的单 query 序列：

- KV 输入是热缓存，按 128-token 页面组织。
- 页表是热缓存页面的 identity 映射。
- 稀疏索引是实际选中 KV 的物理槽位。
- RoPE 已由模型按原始位置计算；query、选中 KV 和 softmax 集合保持不变。
- 不支持额外位置偏置，该适配仅接入目前的 DeepSeek NSA 路径。

默认每块 query 数量不超过 `cache_size // Top-K`，确保最坏情况下选中 KV 的并集
不超过缓存容量。Prefill 如果能通过 CPU 长度元数据证明全部历史 KV 可放入缓存，
则一次处理全部 query。其余情况采用保守分块。

## 文件

路径相对 `sglang-ascend/python/sglang/srt/`：

| 文件 | 职责 |
| --- | --- |
| `hardware_backend/npu/echo_recall.py` | 主机注册、设备端缺失检测、保护、分配、淘汰、回搬、释放和注意力输入适配 |
| `hardware_backend/npu/echo_memory_pool.py` | 完整 index K、每层 recall 状态、缓存池读写接口和错误汇总 |
| `hardware_backend/npu/allocator_npu.py` | 逻辑页释放时清理所有层的 ECHO 映射，flush 时重置状态 |
| `hardware_backend/npu/attention/ascend_backend.py` | 在 `forward_sparse` 中调用精确 recall 和注意力适配 |
| `hardware_backend/npu/graph_runner/npu_graph_runner.py` | 初始化预热页表；capture 后重置热缓存状态；replay 后检查设备错误标记 |
| `model_executor/model_runner_kv_cache_mixin.py` | 缓存池选择和保守显存预算 |
| `model_executor/model_runner.py` | eager forward 后统一检查 ECHO 错误，避免逐层同步 |
| `server_args.py` | ECHO 开关、热缓存容量和功能组合校验 |

## Graph 与生命周期

- KV、映射、计数器和工作缓冲预先分配；设备结果不驱动 Python 分支或动态形状。
- 算子内部没有 `.item()`、索引读回 CPU或动态数量的主机传输命令。
- 固定形状的临时张量由 Graph 内存池持有；其地址由捕获和重放机制保持有效。
- 每次 recall 更新所需 ID 的 epoch，覆盖块状态并重置计数及当前错误；映射和优先级跨 replay 保留。
- Sticky 错误标记在后续 recall 中保留，避免分块操作覆盖之前的失败。
- capture 后清除热缓存状态，保留有效的主机 KV；flush 清除全部有效标记。
- capture 前清零 dummy 请求的页表，避免 indexer 在首次元数据 replay 前读取未初始化页面。
- 错误检查在 Graph 外执行。完整服务的 replay 会检查所有层，不能静默输出不完整结果。
- eager prefill 在整个 forward 后统一检查，不再每层同步读取错误标记。
- 当前所有写入、回搬和消费者按同一 NPU stream 排序。
- 写入槽位缓冲预分配；Graph 中先更新设备 KV，再由独立 kernel 读取热缓存备份主机，
  后续写入或 recall 不会在主机备份完成前复用对应槽位。
- 主机注册分配必须与图共同存活；仅在不再重放相关图时关闭，并先同步设备。
- 直接 API 的 `write` 要求正逻辑 ID 不重复；0 是 padding，recall 允许重复 ID 和 -1。

主机映射不支持的设备或环境会显式报错，没有只支持 eager 的自动降级。

2026-10-10 的小写入、全命中快路径和固定 worker 回搬优化见
[优化记录](ECHO_RECALL_OPTIMIZATION.md)。历史测量中的全容量排序描述仅适用于优化前代码。

## 使用

在原 DeepSeek-V3.2-AWQ 启动命令中增加：

```bash
--enable-echo \
--echo-device-cache-size 4096 \
--max-total-tokens 8192 \
--disable-radix-cache \
--disable-overlap-schedule
```

同时使用 `--device npu --attention-backend ascend`。热缓存容量必须是 128 的正整数倍，
至少能容纳一个 query 的 NSA Top-K，且不能大于逻辑 token 容量。
`--max-total-tokens` 必须显式给定，用于限制完整主机 KV 和 index K 的容量。

第一阶段测量时不支持 DP attention；后续实验性接入及验收边界见
[TP/DP 接入记录](ECHO_PARALLEL_SUPPORT.md)。当前仍不支持前缀缓存、重叠调度、
HiCache、HiSparse、LMCache、投机解码、torch.compile、memory saver、PP、KV disaggregation、NSA prefill CP
或 MLA preprocessing。TP8 需要模型级验证，不能从单卡测试推断其可用性。

## 验证

单卡 recall 与注意力检查：

```bash
ECHO_NPU_DEVICE=14 ./scripts/npu-run.sh python scripts/check-echo-recall-npu.py
```

未启用 ECHO 的框架回归：

```bash
ECHO_NPU_DEVICE=14 ./scripts/npu-run.sh python scripts/smoke-sglang-npu.py
```

完整模型基线对比，顺序启动两个 TP8 服务，结束后自动停止：

```bash
ECHO_NPU_DEVICE=8,9,10,11,12,13,14,15 \
  ./scripts/npu-run.sh python scripts/smoke-echo-deepseek-npu.py
```

模型测试覆盖 batch 1/2、重复请求、超过 2048 token 的提示、实际 decode Graph replay，
并比较基线与 ECHO 的生成 token 和 logprob。日志及结果保存在
`build/echo-deepseek-smoke/`。测试默认设置 `HCCL_BUFFSIZE=1536` 和
`HCCL_OP_EXPANSION_MODE=AIV`，已有环境变量可覆盖该默认值。
`--skip-long-prompt` 仅用于隔离短请求集成，不代表长上下文验证通过。
调试时可以通过 `--baseline-responses` 复用相同模型、输入和服务配置的基线输出；
结果会显式记录复用，调用者必须确保配置一致。

### 2026-10-09 已完成的检查

- 14 卡的 12 项 recall 测试通过：主机写入、去重、命中、淘汰保护、错误处理、
  释放复用、逐层页释放、保留主机 KV 的热缓存重置、稀疏注意力对比、
  int64 越界保护、多请求 prefill 映射及实际 Graph 重放。另对比了全部历史可缓存的
  prefill 批量路径及服务容量下的 padding 图捕获/重放。最终回归用时 5.466 秒
  （已存在编译缓存，不是性能基准）。
- recall Graph 测试覆盖 batch 1/2；另验证了写入、回搬、读取和释放的一体化 Graph。
- 微型随机 Qwen2 基线通过捕获及 batch 1/2 decode replay。
- DeepSeek-V3.2-AWQ 的短请求 TP8 基线和 ECHO 均完成 batch 1/2 图捕获、
  实际 decode replay 和 batch 1/2/1 生成。四个请求的 32 个生成 token 全部一致。
  ECHO 修正后重试复用了前一次同配置运行保存的基线结果。
- 最大 logprob 绝对差为 `0.2027294338`，超过 `0.05` 阈值；
  `build/echo-deepseek-smoke/result.json` 的严格对照状态为 `failed`，没有放宽阈值。
  基线重复数学请求本身的最大差为 `0.04008`，ECHO 为 `0.08945`。
  这些观测尚不能解释跨路径数值差异，不能宣称模型级精度验收通过。
- TP8 ECHO 首次在预热阶段发生设备异常，日志为 `echo-capture-failed-server.log`。
  补充 dummy 页表初始化和安全掩码地址处理后，八个 rank 均完成捕获及实际重放。
- 捕获日志中的图内存增量约为基线 `2.35 GB`、ECHO `4.97 GB`（日志单位）。
  当前临时工作空间和图内存开销仍需优化，不能据此宣称端到端显存或性能收益。
- 最初使用 200 MB HCCL 缓冲的基线通过短请求，在长请求中发生 allreduce
  AICPU 异常（507018）；日志保存在 `baseline-long-failed-server.log`。
  使用 1536 MB 缓冲重试仍失败，日志为 `baseline-long-1536-failed-server.log`。
  当前通过 `--skip-long-prompt` 单独验证短请求集成；长文本模型级验证尚未通过。

以上没有建立吞吐、延迟收益或完整 ECHO 融合预取性能结论。

### ShareGPT 初步性能对照

以下结果来自修正前的“新 KV 先存主机、按需回搬”写入策略，不能作为修正后的性能结果。

随后完成了 DeepSeek-V3.2-AWQ 的 100 条 ShareGPT 对照：TP8 / DP1、PD 合部，
并发 2，每条生成 128 token，输入与输出预算之和不超过 2048 token。
两组均通过 batch 1/2 decode Graph 检查，100 条请求全部成功。
baseline 输出吞吐为 26.34 token/s，第一阶段 ECHO 为 7.07 token/s，
当前配置下没有速度收益。设备仍有 Alarm，结果为带硬件告警的临时测量。
生成文本仅 25/100 完全一致，不构成模型级精度验收。
采样、baseline 复用说明和完整指标见 [ShareGPT 测试记录](ECHO_SHAREGPT_BENCHMARK.md)。

### 新 KV 写入顺序修正

当前写入改为设备热缓存优先，再从热缓存备份主机，不再清除已驻留的写入映射。
启动日志增加 `write_policy=device_first, host_backup=same_stream`，用于辨识实现策略，
该标记不是实际传输量统计。

2026-10-09 在物理设备 14 上完成 18 项测试，全部通过，用时 98.044 秒，
包含新 kernel 编译，不是性能基准。新增验证包括：

- 拦截主机备份阶段，确认备份前设备 KV 已正确写入、主机仍未写入。
- 临时改坏主机副本，确认已驻留的新 KV 直接命中、不会被主机回搬覆盖。
- 原位覆盖保留槽位，同时更新设备与主机 KV；保护本次写入中已有的驻留项。
- 超容量分块写入保留全部主机 KV，最后一块驻留，旧块能够精确回搬。
- batch 1/2 的实际 Graph 捕获及重复重放，动态新 KV、padding、历史 miss 和重复 hit。

随后完成 `build/echo-sharegpt/20261009-190539/` 的完整模型 TP8 集成检查，
于 2026-10-09 19:14 结束，脚本退出码为 0。两组各处理 4 条真实 ShareGPT 输入，
每条生成 8 token，全部成功；计时前的 batch 1/2 Graph 检查均通过，无 decode
eager 回退。8 个 rank 均记录了设备优先写入策略，测试服务已退出。
汇总状态为 `completed_with_hardware_alarm`，原有硬件告警未消除。

小请求检查仍没有建立性能收益：baseline 平均 TTFT / TPOT 为
697.13 ms / 77.58 ms，ECHO 为 11508.38 ms / 314.43 ms。
当前写入和 recall 均使用全容量扫描及排序，还没有完成分配热路径优化；
具体开销需要 profiling 验证，不能仅凭这些指标确定瓶颈。
生成文本 3/4 完全一致，不构成精度验收。这不是 100 请求性能复测，
不能与修正前的 100 条、每条 128 token 指标直接比较。

### 结束后的设备状态

本次测试进程已退出，`npu-smi` 在物理设备 8–15 上没有发现运行进程，
但设备健康状态显示 `Alarm`。只读抽查物理设备 8（NPU 4、chip 0）得到
`80C98001`，信息为 AIC RAS `module error can not be fixed`。
告警原因尚未确认，没有执行设备重置；后续长时间运行前需要单独检查硬件状态。
