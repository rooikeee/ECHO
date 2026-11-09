# ECHO DeepGEMM Ascend 实验实现

目标为 **Ascend 950 / CANN 9.20**。本次实现位于 `deepgemm-ascend/`，
复用 ECHO 中导入的 DeepSeek MQA logits 核；未替换当前 SGLang 的生产 indexer。
原 CUDA ECHO 参考提交为 `bc1b75c1000010d0ac6f032ebaac283255c050b1` 的
`DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh`。
对应原逻辑是：超过历史阈值 → 直接查 host→device 映射 → 分配有上限的 staging → 搬 KV。

## 实现范围

```text
DeepGEMM 原 paged MQA 打分（每个 KV split 512 tokens）
    → 分数在 UB 中完成归约，归还 AIC accumulation credit
    → 128-lane SIMT：score > threshold，检查 stable/active/host validity
    → 直接查询只读 host_to_device，保留 missing=-1 的 token
    → 每 tile/row 一次有上限的 CAS 预算预留
    → Host DDR DVA → UB 双缓冲 → query staging HBM
    → 继续后续 split；整个 launch 结束后才允许消费 staging
```

这里没有再嵌入原生 QLI 的整行候选 hash、LRU 划分或淘汰 Plan。
打分主体只增加一个默认空的 epilogue 接口；原 DeepGEMM 入口使用空实现。
ECHO 入口为 `deep_gemm._C.echo_paged_mqa_prefetch_out`，Python 封装为
`deep_gemm.echo.EchoPrefetchBuffers`。封装预分配 logits、metadata、计数、token/host ID 和 staging。

GPU ECHO 的专用 prefetch warp group 没有原样映射到 Ascend。本实现由当前 owner AIV
执行 SIMT 筛选和 DMA 提交，AIC 可以使用已经归还的 credit 继续工作；实际重叠、AIV stall
和吞吐必须在 950 上测量。没有单独的跨核生产者/消费者等待队列。

## 第一版输入契约

- 仅支持 paged MXFP8 E4M3：`q=[R,1,32,128]`，page size=128。
- Q scales 为 `int16[R*32,2]`，已打包 UE8M0 scale 对，MN-major stride=`[1,R*32]`。
  它不同于原生 indexer 的 FP32 qscale/kscale；没有偷偷重解释原始 scale。
- indexer KV 为 `uint8[P,128,1,132]` 连续张量，但每个 page 的物理内容是
  `[128*128 字节 FP8 数据 | 128*4 字节 scale]`，与当前 DeepGEMM paged API 一致。
- 每行长度、分页表、连续相同 request 的 `indices` 遵守 DeepGEMM scheduler 契约：
  同 request 行的页表相同、context length 非递减，长度不超过预分配的 max_context；
  QLI/indexer 页表必须合法。Python/host 校验 shape/layout，不把设备元数据拷到 CPU。
- `host_ids[R,max_context]` 独立于 indexer 页表；它把逻辑 token 映射到被卸载 KV 记录。
  host ID=0 是 ECHO padding；负值和超出 host pool 的 ID 被忽略。
- `host_to_device[host_tokens]` 为只读 int32 快照，**仅 -1 表示 miss**；0 是有效 cache slot。
  `written[host_tokens]` 非零才允许读取源记录。请求切换/写入/淘汰由现有缓存管理更新这两者。
- `thresholds[R]` 是 FP32 严格分数阈值，不是概率累计 top-p；NaN/+inf 不产生候选。
  `stable[R]`、`active[1]`、`budgets[R]` 均为 int32 device tensor。
- staging 容量每行 1–2048 项；每个 KV record 为 1–65536 个连续 uint8 字节。
  MLA 可把 latent K 和 RoPE 合在一个记录里；数据不与 indexer 的 key 混用。
- 真实 DDR 必须提供 MemFabric 注册池的 DVA，并保持源池存活。普通 CPU 地址不接受作默认源。
  `HBM-diagnostic` 模式明确只测 NPU 内复制，不能报告成 DDR 带宽。

## 并发、预算与一致性

DeepGEMM scheduler 会把同 query 的不同 KV split 分配给不同计算块，因此 staging slot
通过全局 atomic CAS 预留，`counts[row]` 永不超过 clamp 后的预算。
每个 tile 内先用四个 warp compaction，再由一个 lane 预留整个批次；容量已满时整块跳过查询。

预算饱和时不同块的先后顺序可能变化，选中的有效子集不保证每次完全一致。
同一逻辑 query/token 只由 scheduler 处理一次，输出不能重复该逻辑 token。
MTP 四行各有自己的预算与 staging，当前没有把四行做 host-ID 去重合并；
不同逻辑 token 映射到同一 host ID 时也可能重复搬运，这是明确的带宽优化后续项。

**counts 是预留游标，不是 ready 标志。**消费者只能在同 stream 的整个算子完成后运行，
或跨 stream 显式等完成事件。不能看到 count 增长就提前消费数据。
本算子不修改 resident 映射，不会把未完成搬运的数据宣布为 cache hit；也不执行淘汰。
最终精确 TopK 与 attention/recall 仍走后续路径。预测漏掉的 token 必须精确回补；
不能把 staging 当作完整 TopK KV。这是独立算子原型，目前未自动接入 SGLang 缓存消费者。

## Graph capture/replay

`EchoPrefetchBuffers.allocate` 在图外预分配。`run` 内只有固定的设备操作：

1. 清 counts、tokens/hosts，logits 置 -inf；staging 尾部不清零，只有 counts 前缀有效。
2. 在 caller-owned metadata 上重算 paged schedule（每次重放都更新）。
3. 发起一个融合 MQA+筛选+staging kernel。

因此整段图包含初始化和 metadata kernel，**不是整个工作负载只有一次 launch**。
所有输入/输出/源池地址保持有效；length、active、threshold、budget、host mapping、written
在重放之间原地更新。不能同时用同一份 outputs 在不同 stream 中重放。
先在图外运行两次完成 JIT，然后捕获；shape/容量/源 DVA 变化需要重新创建并捕获。

## 950 构建和验收

在 **950 的 CANN 9.20 + TorchNPU 环境**中：

```bash
bash scripts/build-echo-deepgemm-950.sh
python3 scripts/check-echo-deepgemm-950.py --requests 1 2 --seq-len 2048 --capacity 64
python3 scripts/check-echo-deepgemm-950.py --requests 1 2 --seq-len 2048 --capacity 64 \
  --host-ddr --output build/deepgemm-echo/950-ddr-validation.json
python3 scripts/check-echo-deepgemm-950.py --requests 24 --seq-len 143360 --capacity 64 \
  --host-ddr --repeats 100 --output build/deepgemm-echo/950-140k.json
```

原包依赖仍需按 `deepgemm-ascend/README.md` 配置，包括 C++20 `<format>`、DeepJIT 子模块和 tilelang。
构建脚本只重装独立 DeepGEMM 包，不改当前 SGLang 的算子路由。
目标脚本首先检查 950；不允许在 910 上通过改架构名称冒充目标设备。

验证覆盖：MTP=3（4 query/request）、batch 1/2、capture、实际多次 replay、动态长度、
预算/阈值、all-hit、mixed、invalid host、inactive、请求映射更新。
它对照未带 ECHO epilogue 的 DeepGEMM，检查有效 logits 完全一致、选择子集合法、无逻辑重复、
counts 上界、staging 每字节正确以及 resident/written 不被修改。

输出的 `target_mqa_prefetch_graph_us` 包含 reset + metadata + MQA + staging，
`baseline_mqa_with_metadata_graph_us` 是 metadata + MQA，无搬运；
两者都不含最终 TopK、attention 和完整推测解码轮次。`complete_speculative_round_us` 留空。
这些数字用于评估算子开销，不直接证明完整 ECHO 吞吐收益。

## 当前验证状态

本机只进行了 CPU 检查：

```bash
./scripts/npu-run.sh python scripts/test-echo-deepgemm.py
```

测试直接执行生产 SIMT 选择和 staging 控制逻辑，在 128 个 CPU 线程中模拟 block barrier/shuffle，
两个计算块竞争同一行预算。36 个并发/重放状态组合和 2 项 Python out-wrapper 测试通过，
并启用了 AddressSanitizer / UndefinedBehaviorSanitizer（包括退出时的 leak 检查）。
CPU DMA mock 检查字节、UB 对齐和事件配对；不模拟真实 cache、MTE 时序或 AIC/AIV 并发。

**950 的 native 编译、实际运行、Graph capture/replay 和性能均未验证。**
本机 CANN 9.1 连未修改的 DeepGEMM 基线也缺少 9.20 C API 类型；不通过修改上游类型或
关闭硬件检查声称支持 910。当前代码是待 950 验收的实验实现。
