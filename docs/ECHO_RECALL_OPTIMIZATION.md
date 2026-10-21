# ECHO Recall 热路径优化记录

日期：2026-10-10。保留设备优先写入、主机完整 KV 和精确 Top-K 集合，不退回
host-first，也不关闭 NPU Graph。完整算法见 [Recall 实现详解](ECHO_RECALL_DESIGN.md)。

## 已实现的改动

1. 不超过 8 个新 KV 的写入使用独立设备分配 kernel，直接复用驻留槽位，直接从
   设备空闲栈取槽；空槽不足才选择 LRU。没有完整主机逻辑池扫描、前缀和或排序。
2. recall 使用 epoch 标签，避免每次清零整个 `required`。输入块直接检测命中及
   有效性，全部命中时由图内设备分支跳过逻辑池扫描和槽位分配。
3. miss 路径分块紧凑收集缺失 ID、从设备空闲栈批量取槽。只有真实淘汰才做分块 LRU
   最小值选择；删除无条件 `torch.sort` 和 `torch.cumsum` 调用。
4. 回搬使用固定 `min(C, 32)` 个 worker 按设备计数循环，而不是固定启动 `C` 个
   逐行 program。全命中没有主机 KV 读取。
5. eager prefill 从每层错误读回改为整个 forward 后统一检查；Graph decode 仍在
   replay 后检查，输出被消费前仍然 fail-closed，没有关闭诊断来换速度。
6. benchmark 增加逐请求 JSONL 落盘，包含 start_time、latency、TTFT、ITL、生成文本
   和请求序号，不含 warmup。两边都启用，避免再因停止后的 `/server_info` 异常丢失结果。

设备写入和主机备份仍是两个有顺序依赖的 kernel，同一 stream 排在 attention 前。
未实现异步备份或融合预取，不能将本次修改描述成完整 ECHO 性能优化。

## 正确性验证

物理设备 14 上最终 25 项 NPU 测试全部通过，用时 6.806 秒，使用已有编译缓存，
不是性能指标。
覆盖写入顺序、热命中、覆盖更新、去重、保护/LRU 淘汰、冷回搬、错误保护、逻辑
页释放、非二次幂容量、稀疏注意力对照，以及实际 batch 1/2 Graph 捕获和重复重放。

首版融合 kernel 在服务容量遇到片上 UB 溢出，并有一项动态 Graph 测试失败。
改为最大 512 元素的数据块后，相关测试和完整回归均通过；未采用 eager 降级。
扩大到 1024 个历史 KV 后发现旧空槽紧凑生成路径出现重复槽位；改为显式设备空闲栈，
并加入大批量映射唯一性、KV 内容和重复释放后的 Graph 重放验证。
物理设备 15 上的微型随机 Qwen2 框架检查也通过，包含 batch 1/2 capture/replay。

## 测量口径

- 组件测试：`scripts/bench-echo-recall-planner-npu.py`，真实 NPU Graph，逻辑容量
  8320、热缓存 4096、历史 KV 1024、batch 1/2；分别测 write-only、all-hit recall
  和 write+recall。捕获、编译、warmup 在计时之外，使用 NPU event 计时并校验 KV 内容。
- 优化前组件代码保存在 `build/echo-recall-optimization/echo_recall_before.py`，
  是设备优先但全扫描/排序的版本，**不是 host-first 版本**。
- 模型测试：DeepSeek-V3.2-AWQ、物理 8–15、TP8/DP1、PD 合部、并发 2、固定
  128 输出 token、seed 42、40 条 ShareGPT；两边 fresh 服务和 fresh 指标，decode Graph
  batch 1/2、prefill chunk 512、逻辑容量 8192、热缓存 4096。
- 前一次 40 条的 4.59 token/s 只是日志估算；不能与新的客户端精确计时混称同口径。
  原 host-first 的 7.07 token/s 来自完整 100 条；与新 40 条直接比较也有样本范围差别。

## Graph 组件计时结果

物理设备 15，50 次 replay 为一组，重复 3 组，以下是 NPU event 单次耗时中位数。
捕获、编译和 warmup 不计入；每组结束后校验映射和 KV 内容。表中优化前是设备
优先但全扫描/排序的版本，不是 host-first。

| Batch | 路径 | 优化前 | 优化后 | 设备耗时加速比 |
| ---: | --- | ---: | ---: | ---: |
| 1 | 新 KV 写入 | 1.752 ms | 0.0317 ms | 55.3x |
| 1 | 全命中 recall | 2.081 ms | 0.3111 ms | 6.69x |
| 1 | 写入 + recall | 3.816 ms | 0.3196 ms | 11.94x |
| 2 | 新 KV 写入 | 1.742 ms | 0.0345 ms | 50.5x |
| 2 | 全命中 recall | 2.090 ms | 0.3176 ms | 6.58x |
| 2 | 写入 + recall | 3.848 ms | 0.3351 ms | 11.48x |

原始结果：`build/echo-recall-optimization/planner-metrics.json`。
这是单层 KV 管理计时，不包括 indexer、sparse attention、模型计算、逐步服务端
错误检查或完整请求调度，不能把 11.48x 当成模型吞吐提升。

## 完整模型测量：首次尝试被占用阻断

尝试启动 `build/echo-sharegpt/20261010-110825/`，计划两边各 40 条同配置新测量。
样本已生成，并确认完整文本、顺序及 token 预算与上一轮前 40 条完全一致。

2026-10-10 11:08:43 在启动服务前被空闲检查拦住，原始 `result.json` 为 `failed`：

```text
RuntimeError: NPU board 4 is occupied; no processes were stopped
```

8–15 卡上新出现一组 root vLLM TP8 服务，每卡进程占用 38473 MiB。
EngineCore PID 为 `2291759`；worker PID 为 `2292974, 2294759, 2299187, 2299615,
2300357, 2302746, 2304130, 2304966`，worker 于 11:06:45–11:07:57 启动。
这些进程不是本轮测试脚本启动的，没有终止它们。组件计时在设备 15 上于 11:06:48
完成，早于该卡 vLLM worker 的启动时间。

该次没有启动 DeepSeek 模型服务，也没有得到新 baseline/ECHO 的模型吞吐、TTFT、
TPOT 或精度对照。当时不能确认是否超过原 host-first 的 7.07 token/s。
现有证据是设备优先语义和 Graph 回归通过、单层全命中 KV 管理的设备耗时降低，
不是完整模型吞吐验收。

卡释放后用新目录重跑，不复用本次失败目录：

```bash
ECHO_NPU_DEVICE=8,9,10,11,12,13,14,15 ./scripts/npu-run.sh \
  python scripts/bench-echo-sharegpt-npu.py \
  --num-prompts 40 --output-len 128 --sample-context-len 2048 \
  --prefill-chunk-size 512 --seed 42
```

## 完整模型测量：释放设备后的 40 条结果

用户释放 8–15 卡后，2026-10-10 11:19:11 启动新目录
`build/echo-sharegpt/20261010-111911/`，沿用上述命令和配置。
baseline/ECHO 顺序启动独立 fresh TP8 服务，两边均成功完成 40 条，输入 14628 token、
输出 5120 token。完整样本文本和顺序与上一轮前 40 条一致，不复用旧 baseline 指标。
本轮没有修改实现或参数，加载、Graph 捕获、probe 和 warmup 均不计入正式计时。

| 客户端指标 | Baseline | 优化后 ECHO |
| --- | ---: | ---: |
| 成功请求数 | 40/40 | 40/40 |
| 正式测量时长 | 191.93 s | 809.39 s |
| 端到端输出吞吐 | 26.6759 token/s | 6.3257 token/s |
| 平均 E2E 延迟 | 9595.07 ms | 40456.95 ms |
| 平均 TTFT | 886.03 ms | 24117.37 ms |
| 平均 TPOT | 68.58 ms | 128.66 ms |
| 平均 ITL | 68.90 ms | 129.29 ms |
| P99 E2E 延迟 | 10782.07 ms | 50741.99 ms |

ECHO 吞吐为 baseline 的 23.71%，完整请求耗时约为 4.22 倍，**没有端到端收益**。
其 6.33 token/s 也低于原 host-first 完整 100 条的 7.07 token/s，数值约低 10.6%；
但后者样本范围和测量时段不同，不能把这个差值当作严格同配置同样本的性能回归结论。
优化前设备优先版本前 40 条的 4.59 token/s 是日志估算，也不能与本轮精确客户端计时
混称严格加速比。本轮证明组件加速没有解决完整模型性能问题。

ECHO 稳态 Graph decode 日志约 18 token/s，但平均 TTFT 达 24.12 秒。
相对 baseline，平均请求增加 30.86 秒，其中首 token 前增加约 23.23 秒，占增量约
75.3%。首 token 前还包含调度、排队等开销，不能仅据 TTFT 归因到某一个 kernel。
后续应单独 profiling prefill 的批量 KV 写入规划、主机备份、注意力输入构造及
稀疏注意力。当前小写入快路径只处理不超过 8 个 KV，不覆盖 512-token prefill。

两边 8 个 rank 均有 Graph 捕获完成日志，batch 1/2 probe 通过，decode 没有 eager
fallback；ECHO 8 个 rank 均记录 `device_first/epoch_hit_fastpath/free_stack`。
两边各 40 条逐请求 JSONL 已核对：序号完整唯一、全部成功、输出长度均为 128，
生成文本与最终 metrics 对应一致。baseline/ECHO 文本完全一致 12/40，**不是任务精度
或严格数值验收通过**，原有数值一致性问题仍需单独检查。

最终状态 `completed_with_hardware_alarm`，测试前后均保留 `80C98001` Alarm，未执行
重置，数据仍为临时诊断。测试脚本退出码为 0，两边服务已退出；结束后再次检查，
8–15 卡没有运行进程，其他卡上的进程没有终止。

主要产物：`result.json`、`baseline-metrics.jsonl`、`echo-metrics.jsonl`、
`baseline-requests.jsonl`、`echo-requests.jsonl`，以及两边 server/benchmark 日志。
不得用先前“启动阻断”状态描述这个已完成目录，也不得用组件 11.48x 代替上述结果。

## 保留的限制

大批量 miss 且空槽不足时，逐项 LRU 归约仍可能昂贵，必须另做压力测试。
当前短上下文可能全部容纳在热缓存内，不能证明长上下文下的传输或淘汰收益。
图外错误检查仍有一次同步；性能分析不得把它与主机 KV 数据传输混为一谈。
原有 `80C98001` 硬件 Alarm 未清除，测量只能作为带告警的临时结果。
完整模型严格数值对照的历史未通过状态没有改变，本轮性能测试不是精度验收。
